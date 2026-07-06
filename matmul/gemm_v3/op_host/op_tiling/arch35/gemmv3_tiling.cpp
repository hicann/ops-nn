/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gemmv3_tiling.cc
 * \brief
 */

#include "gemmv3_tiling.h"

#include "gemmv3_tiling_strategy.h"
#include "gemmv3_tiling_key.h"
#include "register/op_impl_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_cfg.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_simplifiedkey.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "matmul/common/op_host/log_format_util.h"

namespace {
using namespace optiling;
using namespace optiling::gemmv3;
using namespace optiling::matmul_v3_advanced;

constexpr size_t HF32_ATTR_NUM = 5;
constexpr size_t HF32_ATTR_INDEX = 4;
constexpr size_t ALLOW_DIM = 2;
constexpr size_t ALLOW_DIM_BIAS = 1;
constexpr size_t A_IDX = 0;
constexpr size_t B_IDX = 1;
constexpr size_t BIAS_IDX = 2;
constexpr size_t C_IDX = 2;
constexpr size_t Y_IDX = 0;
constexpr size_t ALPHA_ATTR_INDEX = 0;
constexpr size_t BETA_ATTR_INDEX = 1;
constexpr size_t TRANSA_ATTR_INDEX = 2;
constexpr size_t TRANSB_ATTR_INDEX = 3;
constexpr uint64_t ONE_BATCH_DIM = 1UL;
constexpr float epsilon = 1e-6f;

static inline void GetDtype(const gert::TilingContext& context, MatMulV3Args& args)
{
    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    // using args.biasType as c type, biasType dimNum maybe 1 or 2
    if (args.hasBias) {
        args.biasType = context.GetInputDesc(C_IDX)->GetDataType();
    }
    // hf32
    if (context.GetAttrs()->GetAttrNum() == HF32_ATTR_NUM) {
        args.isHf32 = *context.GetAttrs()->GetAttrPointer<bool>(HF32_ATTR_INDEX);
    }
    args.aDtypeSize = ge::GetSizeByDataType(args.aType);
    args.bDtypeSize = ge::GetSizeByDataType(args.bType);
    OP_LOGD(args.opName, "GemmV3 Hf32 flag is: %d", args.isHf32);
}

ge::graphStatus IsValidDtype(const MatMulV3Args& args)
{
    std::vector<ge::DataType> dtype = {args.aType, args.bType, args.cType};
    // a, b, y
    const std::vector<std::vector<ge::DataType>> dtypeSuportList = {
        {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT}, // 累加模式，FP16进 32出
        {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT},       // 累加模式，BF16进 32出
        {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}      // 累加模式，FP32进 32出
    };
    for (auto& supported : dtypeSuportList) {
        if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
            return ge::GRAPH_SUCCESS;
        }
    }
    std::string incorrectDtypes = Ops::Base::ToString(args.aType) + ", " + Ops::Base::ToString(args.bType) + ", " +
                                  Ops::Base::ToString(args.cType);
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
        args.opName, "A, B, out", incorrectDtypes.c_str(),
        Ops::NN::FormatString("The dtypes of %s must be within the range %s", "A, B, out",
                              "(FLOAT16,FLOAT16,FLOAT), (BF16,BF16,FLOAT), (FLOAT,FLOAT,FLOAT)")
            .c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus GemmV3OpSpecificCheck(MatMulV3Args& args)
{
    // dtype check
    return IsValidDtype(args);
}

ge::graphStatus GetGemmV3ShapeMKN(const gert::Shape& aShape, const gert::Shape& bShape, MatMulV3Args& args)
{
    int64_t ka = aShape[args.isATrans ? 0 : 1];
    int64_t kb = bShape[args.isBTrans ? 1 : 0];
    if (ka != kb) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args.opName, "A, B",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(aShape).c_str(), Ops::Base::ToString(bShape).c_str())
                .c_str(),
            Ops::NN::FormatString("%s of %s must be equal", "K-axis", "A, B").c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t m = aShape[args.isATrans ? 1 : 0];
    int64_t n = bShape[args.isBTrans ? 0 : 1];
    args.mValue = static_cast<uint64_t>(m);
    args.kValue = static_cast<uint64_t>(ka);
    args.nValue = static_cast<uint64_t>(n);
    // check dim value overlimit int32_max
    auto isValidDimValue = [](int64_t dim) -> bool { return (dim > 0) && (dim <= INT32_MAX); };
    if (!isValidDimValue(args.mValue) || !isValidDimValue(args.kValue) || !isValidDimValue(args.nValue)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args.opName, "A, B",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(aShape).c_str(), Ops::Base::ToString(bShape).c_str())
                .c_str(),
            Ops::NN::FormatString("%s of %s must be within the range %s", "m, k, n", "A, B", "(0, INT32_MAX]").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace optiling {
namespace gemmv3 {
ge::graphStatus GemmV3Tiling::GetArgs()
{
    GetFormat();
    GetDtype(*context_, args_);
    if (GetShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return GemmV3OpSpecificCheck(args_);
}

void GemmV3Tiling::GetFormat()
{
    // GemmV3 only support ND format
    ge::Format formatA = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(0)->GetStorageFormat()));
    ge::Format formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(1)->GetStorageFormat()));
    ge::Format formatOut = static_cast<ge::Format>(
        ge::GetPrimaryFormat(context_->GetOutputDesc(0)->GetStorageFormat()));
    args_.aFormat = (formatA != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatA;
    args_.bFormat = (formatB != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatB;

    args_.outFormat = (formatOut != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatOut;
}

ge::graphStatus GemmV3Tiling::GetShape()
{
    const gert::Shape& aShape = context_->GetInputShape(A_IDX)->GetOriginShape();
    const gert::Shape& bShape = context_->GetInputShape(B_IDX)->GetOriginShape();
    const gert::Shape& yShape = context_->GetOutputShape(Y_IDX)->GetOriginShape();
    const size_t aDimNum = aShape.GetDimNum();
    const size_t bDimNum = bShape.GetDimNum();
    const size_t yDimNum = yShape.GetDimNum();
    // 输入a&b维度校验
    if ((aDimNum != ALLOW_DIM) || (bDimNum != ALLOW_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            args_.opName, "A, B", Ops::NN::FormatString("[%zu, %zu]", aDimNum, bDimNum).c_str(),
            Ops::NN::FormatString("The shape dims of %s must be %zu", "A, B", ALLOW_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    // 输出y维度校验
    if (yDimNum != ALLOW_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            args_.opName, "out", std::to_string(yDimNum).c_str(),
            Ops::NN::FormatString("The shape dim of %s must be %zu", "out", ALLOW_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    // get transpose
    args_.isATrans = *context_->GetAttrs()->GetAttrPointer<bool>(TRANSA_ATTR_INDEX);
    args_.isBTrans = *context_->GetAttrs()->GetAttrPointer<bool>(TRANSB_ATTR_INDEX);
    // 校验mnk
    OP_TILING_CHECK(GetGemmV3ShapeMKN(aShape, bShape, args_) != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(args_.opName, "get m/k/n failed"), return ge::GRAPH_FAILED);
    // 输入c维度校验
    if (context_->GetInputDesc(C_IDX) != nullptr) {
        const gert::Shape& cShape = context_->GetInputShape(C_IDX)->GetOriginShape();
        const size_t cDimNum = cShape.GetDimNum();
        if (cDimNum != ALLOW_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                args_.opName, "C", std::to_string(cDimNum).c_str(),
                Ops::NN::FormatString("The shape dim of %s must be %zu", "C", ALLOW_DIM).c_str());
            return ge::GRAPH_FAILED;
        }
        // c和y相同shape
        if (cShape[0] != yShape[0] || cShape[1] != yShape[1]) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                args_.opName, "C, out",
                Ops::NN::FormatString("%s, %s", Ops::Base::ToString(cShape).c_str(),
                                      Ops::Base::ToString(yShape).c_str())
                    .c_str(),
                Ops::NN::FormatString("The shapes of %s must be the same", "C, out").c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmV3Tiling::CheckArgs()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // check input
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(A_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(A_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(B_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(B_IDX));
    // check output
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(Y_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_IDX));
    // 判断c维度，当前tiling只支持累加场景
    if (context_->GetInputDesc(C_IDX) != nullptr) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(C_IDX));
    }

    // check attrs
    const float* alphaValue = attrs->GetAttrPointer<float>(ALPHA_ATTR_INDEX);
    const float* betaValue = attrs->GetAttrPointer<float>(BETA_ATTR_INDEX);
    const bool* transAValue = attrs->GetAttrPointer<bool>(TRANSA_ATTR_INDEX);
    const bool* transBValue = attrs->GetAttrPointer<bool>(TRANSB_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, alphaValue);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, betaValue);
    // alpha和beta必须为1，默认1.0
    if (std::abs(*alphaValue - 1.0f) > epsilon || std::abs(*betaValue - 1.0f) > epsilon) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            args_.opName, "alpha, beta", Ops::NN::FormatString("%f, %f", *alphaValue, *betaValue).c_str(),
            Ops::NN::FormatString("The values of %s must be %s", "alpha, beta", "1.0").c_str());
        return ge::GRAPH_FAILED;
    }
    // transA和transB必须有bool值，默认false
    OPS_CHECK_NULL_WITH_CONTEXT(context_, transAValue);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, transBValue);

    // 校验enable_hf32
    if (attrs->GetAttrNum() >= HF32_ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(HF32_ATTR_INDEX));
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmV3Tiling::GetShapeAttrsInfo()
{
    args_.opName = context_->GetNodeName();
    OP_TILING_CHECK(args_.opName == nullptr, CUBE_INNER_ERR_REPORT("GemmV3", "get op name invalid"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(args_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK((CheckArgs() != ge::GRAPH_SUCCESS) || (GetArgs() != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args_.opName, "invalid context"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmV3Tiling::DoTiling()
{
    if (GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GemmV3TilingKey GemmV3TilingKey_;
    MatMulTilingCfg tilingCfg(false, context_->GetCompileInfo(), static_cast<void*>(&args_), &GemmV3TilingKey_);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingCfg.compileInfo);
    NpuArch npuArch = static_cast<const MatmulV3CompileInfo*>(tilingCfg.compileInfo)->npuArch;
    MMRegisterCfg registerCfg{"MatMulV3", npuArch, strategy::GetGemmV3Priorities(npuArch)};
    return MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg);
}
} // namespace gemmv3
} // namespace optiling