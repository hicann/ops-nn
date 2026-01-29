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
 * \file fused_matmul_builtin_tiling.cpp
 * \brief
 */

#include "fused_matmul_builtin_tiling.h"

#include "fused_matmul_common.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_tiling_key.h"
#include "matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_tiling_strategy.h"
#include "matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_common_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"

namespace {
using namespace optiling;
using namespace optiling::fused_matmul;

static const std::vector<std::vector<ge::DataType>> DTYPE_SUPPORT_LIST_RESERVED = {
    // x1,              x2,             y,              bias
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
};

static const std::vector<std::vector<ge::DataType>> DTYPE_SUPPORT_LIST_91095 = {
    // x1,              x2,             y,               bias
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT},
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}};

inline void GetDtype(const gert::TilingContext& context, MatMulV3Args& args, platform_ascendc::SocVersion socVersion)
{
    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    if (args.hasBias) {
        args.biasType = context.GetInputDesc(INPUT_BIAS_IDX)->GetDataType();
    }
    // op_impl_mode_enum: 0x1: default 0x2: high_performance 0x4: high_precision 0x8: super_performance
    // 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
    args.isHf32 = *((context.GetAttrs())->GetAttrPointer<bool>(ATTR_ENABLE_HF32_IDX));
    args.aDtypeSize = ge::GetSizeByDataType(args.aType);
    args.bDtypeSize = ge::GetSizeByDataType(args.bType);
    if (args.isHf32 && socVersion != platform_ascendc::SocVersion::ASCEND950) {
        OP_LOGW(args.opName, "Hf32 flag is: %d, which is not support yet", args.isHf32);
    }
}

ge::graphStatus IsValidDtype(const MatMulV3Args& args, platform_ascendc::SocVersion socVersion)
{
    std::vector<ge::DataType> dtype = {args.aType, args.bType, args.cType};
    if (args.hasBias) {
        dtype.push_back(args.biasType);
    }

    // check dtype
    auto supportList = socVersion == platform_ascendc::SocVersion::ASCEND950 ? DTYPE_SUPPORT_LIST_91095 :
                                                                                  DTYPE_SUPPORT_LIST_RESERVED;
    for (auto& supported : supportList) {
        if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
            return ge::GRAPH_SUCCESS;
        }
    }

    if (args.hasBias) {
        OP_LOGE(
            args.opName, "Unsupported data type: x1[%s], x2[%s], y[%s], bias[%s]", std::to_string(args.aType).c_str(),
            std::to_string(args.bType).c_str(), std::to_string(args.cType).c_str(),
            std::to_string(args.biasType).c_str());
        return ge::GRAPH_FAILED;
    } else {
        OP_LOGE(
            args.opName, "Unsupported data type: x1[%s], x2[%s], y[%s]", std::to_string(args.aType).c_str(),
            std::to_string(args.bType).c_str(), std::to_string(args.cType).c_str());
        return ge::GRAPH_FAILED;
    }
}

ge::graphStatus OpSpecificCheck(
    const gert::TilingContext& context, const MatMulV3Args& args, platform_ascendc::SocVersion socVersion)
{
    // check bias
    if (args.hasBias) {
        const gert::Shape& biasShape = context.GetInputShape(INPUT_BIAS_IDX)->GetOriginShape();
        const gert::Shape& cShape = context.GetOutputShape(0)->GetOriginShape();
        const int64_t biasValue = biasShape[biasShape.GetDimNum() - 1];
        const int64_t nOriValue = cShape[cShape.GetDimNum() - 1];
        if (biasValue != nOriValue) {
            OP_LOGE(args.opName, "illegal value: bias[%ld], n[%ld]", biasValue, nOriValue);
            return ge::GRAPH_FAILED;
        }
    }

    // dtype check
    return IsValidDtype(args, socVersion);
}
} // namespace

namespace optiling {
namespace fused_matmul {

ge::graphStatus FusedMatMulBuiltInTiling::GetBmmBiasInfo(const gert::TilingContext &context, MatMulV3Args& args,
                                                    MatMulV3BatchInfo& batchInfo)
{   
    // 本质上是由于matmul判断hasBias有OptionalInput无法占位问题
    bool hasBias =
        (context.GetOptionalInputDesc(INPUT_BIAS_IDX) != nullptr &&
         context.GetOptionalInputShape(INPUT_BIAS_IDX)->GetOriginShape().GetDimNum() > 0);
    if (!hasBias) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasShape = context.GetOptionalInputShape(2)->GetOriginShape(); // bias == 2
    auto outputShape = context.GetOutputShape(0)->GetOriginShape();
    size_t biasDims = biasShape.GetDimNum();
    size_t cDims = outputShape.GetDimNum();
    uint64_t batchBias3 = 1UL;
    uint64_t batchBias2 = 1UL;
    uint64_t batchBias1 = 1UL;
    uint64_t batchBias0 = 1UL;
    // 先校验bias的尾值是否与output尾值相等
    if (biasShape[biasDims - FINAL_SHAPE_DIM] != outputShape[cDims - FINAL_SHAPE_DIM]) {
        OP_LOGE(args.opName, "Last dim of bias is not equal to last dim of output.");
        return ge::GRAPH_FAILED;
    }
    // 不支持batchbias
    if (biasDims > NUM_TWO) {
        OP_LOGE(args.opName, "Bias dim of fusedMatmul must be lower than 3.");
        return ge::GRAPH_FAILED;
    }
    if (biasDims == NUM_TWO && biasShape[0] != 1) { // BIAS[0]必须为1
        OP_LOGE(args.opName, "M of bias must be 1.");
        return ge::GRAPH_FAILED;
    }
    batchInfo.batchBias = batchBias3 * batchBias2 * batchBias1 * batchBias0;
    OP_LOGI(args.opName, "Check bias success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedMatMulBuiltInTiling::GetBatchInfo(
    const gert::TilingContext& context, MatMulV3Args& args, MatMulV3BatchInfo& batchInfo)
{
    auto aShape = context.GetInputShape(0)->GetOriginShape();
    auto bShape = context.GetInputShape(1)->GetOriginShape();
    auto cShape = context.GetOutputShape(0)->GetOriginShape();

    size_t aDims = aShape.GetDimNum();
    size_t bDims = bShape.GetDimNum();
    size_t cDims = cShape.GetDimNum();
    if (aDims > BATCH_DIM_MAX || bDims > BATCH_DIM_MAX) {
        OP_LOGE(
            args.opName, "The current input dimensions is greater than 6 where x1_dims is (%zu) and x2_dims is (%zu)",
            aDims, bDims);
        return ge::GRAPH_FAILED;
    }
    batchInfo.batchA3 = aDims > NO_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - ONE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchA2 = aDims > ONE_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - TWO_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchA1 = aDims > TWO_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - THREE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchA0 = aDims > THREE_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - FOUR_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchB3 = bDims > NO_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - ONE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchB2 = bDims > ONE_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - TWO_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchB1 = bDims > TWO_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - THREE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchB0 = bDims > THREE_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - FOUR_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchC3 = cDims > NO_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - ONE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchC2 = cDims > ONE_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - TWO_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchC1 = cDims > TWO_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - THREE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchC0 = cDims > THREE_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - FOUR_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchA = batchInfo.batchA0 * batchInfo.batchA1 * batchInfo.batchA2 * batchInfo.batchA3;
    batchInfo.batchB = batchInfo.batchB0 * batchInfo.batchB1 * batchInfo.batchB2 * batchInfo.batchB3;
    batchInfo.batchC = batchInfo.batchC0 * batchInfo.batchC1 * batchInfo.batchC2 * batchInfo.batchC3;

    // Check if one of the batch size is zero
    bool isBatchZero = (batchInfo.batchA == 0UL || batchInfo.batchB == 0UL);
    if (isBatchZero) {
        OP_LOGE(args.opName, "One of the batch size is zero");
        return ge::GRAPH_FAILED;
    }

    // when BatchB == 1, adjust M = batchA * M, batchA = 1
    MergeBatchAndMAxis(args, batchInfo); // check if batch merge to M

    // Check if batch info is valid, if batch is M broadcast to N, return failed.
    bool batch3Invalid = batchInfo.batchA3 != batchInfo.batchB3 && batchInfo.batchA3 != 1UL && batchInfo.batchB3 != 1UL;
    bool batch2Invalid = batchInfo.batchA2 != batchInfo.batchB2 && batchInfo.batchA2 != 1UL && batchInfo.batchB2 != 1UL;
    bool batch1Invalid = batchInfo.batchA1 != batchInfo.batchB1 && batchInfo.batchA1 != 1UL && batchInfo.batchB1 != 1UL;
    bool batch0Invalid = batchInfo.batchA0 != batchInfo.batchB0 && batchInfo.batchA0 != 1UL && batchInfo.batchB0 != 1UL;
    if (batch3Invalid || batch2Invalid || batch1Invalid || batch0Invalid) {
        OP_LOGE(args.opName, "Is M broadcast to N situation, do not support!");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        (GetBmmBiasInfo(context, args, batchInfo) != ge::GRAPH_SUCCESS),
        CUBE_INNER_ERR_REPORT(args_.opName, "GetBmmBiasInfo failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedMatMulBuiltInTiling::DoTiling()
{
    if (GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 重写BMM校验Batch和bias
    MatMulV3BatchInfo tempBatchInfo;
    OP_TILING_CHECK(
        (GetBatchInfo(*context_, args_, tempBatchInfo) != ge::GRAPH_SUCCESS),
        CUBE_INNER_ERR_REPORT(args_.opName, "GetBatchInfo failed"), return ge::GRAPH_FAILED);
    args_.batchInfo = &tempBatchInfo;
    FusedMatmulTilingKey fusedMatmulTilingKey;
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    fusedMatmulTilingKey.SetFusedOpType(FUSED_OP_TYPE_MAP.at(opType));
    MatMulTilingCfg tilingCfg(
        false, context_->GetCompileInfo(), reinterpret_cast<void*>(&args_), &fusedMatmulTilingKey);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingCfg.compileInfo);
    MMRegisterCfg registerCfg{"FusedMatMul", socVersion_, strategy::GetFusedMatMulPriorities(socVersion_)};
    return MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg);
}

ge::graphStatus FusedMatMulBuiltInTiling::GetArgs()
{
    GetFormat();
    GetDtype(*context_, args_, socVersion_);
    if (GetShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return OpSpecificCheck(*context_, args_, socVersion_);
}

ge::graphStatus FusedMatMulBuiltInTiling::CheckArgs()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;

    if (context_->GetInputDesc(idx) != nullptr) {
        args_.hasBias = true;
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(ATTR_ENABLE_HF32_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(ATTR_OP_TYPE_IDX));

    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedMatMulBuiltInTiling::GetShapeAttrsInfo()
{
    OP_TILING_CHECK(
        GetSocVersion(context_, socVersion_) == ge::GRAPH_FAILED,
        CUBE_INNER_ERR_REPORT("FusedMatMul", "fail to get soc version"), return ge::GRAPH_FAILED);
    return MatMulV3Tiling::GetShapeAttrsInfo();
}
} // namespace fused_matmul
} // namespace optiling