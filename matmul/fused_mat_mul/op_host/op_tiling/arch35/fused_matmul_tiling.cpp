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
 * \file fused_matmul_tiling.cpp
 * \brief
 */
#include "fused_matmul_tiling.h"

#include "../../../op_kernel/arch35/fused_mat_mul_tiling_data.h"
#include "fused_matmul_builtin_tiling.h"
#include "fused_matmul_common.h"
#include "fused_matmul_simplifiedkey.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "tiling_base/tiling_key.h"
#include "register/op_impl_registry.h"

namespace {
using namespace optiling;
using Ops::NN::Optiling::GET_TILINGKEY;
constexpr uint64_t ONE_BATCH_DIM = 1;
constexpr uint64_t TWO_BATCH_DIM = 2;
constexpr uint64_t THREE_BATCH_DIM = 3;
constexpr uint64_t FOUR_BATCH_DIM = 4;
constexpr uint64_t ALIGN_NUM = 16;
constexpr uint64_t WORKSPACE_SIZE = 1024;

// soc_version, built-in, others
const std::unordered_map<platform_ascendc::SocVersion, std::array<std::vector<std::string>, 2>> SocFusedOpSupport = {
    {platform_ascendc::SocVersion::ASCEND910_95,
     std::array<std::vector<std::string>, 2>{
         std::vector<std::string>{"", "relu", "add", "mul",}, std::vector<std::string>{"gelu_erf", "gelu_tanh"}}},
    {platform_ascendc::SocVersion::RESERVED_VERSION,
     std::array<std::vector<std::string>, 2>{std::vector<std::string>{"relu"}, std::vector<std::string>{}}}};

const std::initializer_list<std::string> FusedOpTypeSupportF32 = {"", "relu", "add", "mul"};

const std::vector<std::vector<ge::DataType>> DavidDtypeSupportListB16 = {
    // x1,              x2,             bias,              x3,           y
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16} // david supports bias-bf16
};

const std::vector<ge::DataType> DtypeSupportF32 = {
    // x1,              x2,             bias,              x3,           y
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}};

FusedMatmulTrans GetTrans(bool aTrans, bool bTrans)
{
    FusedMatmulTrans trans = FusedMatmulTrans::NO_TRANS;
    if (aTrans && bTrans) {
        trans = FusedMatmulTrans::AB_TRANS;
    } else if (aTrans) {
        trans = FusedMatmulTrans::A_TRANS;
    } else if (bTrans) {
        trans = FusedMatmulTrans::B_TRANS;
    }
    return trans;
}

bool CheckFormat(const gert::TilingContext& context, bool hasX3Input)
{
    ge::Format formatA =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(INPUT_X1_IDX)->GetStorageFormat()));
    ge::Format formatB =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(INPUT_X2_IDX)->GetStorageFormat()));
    ge::Format formatOut = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetOutputDesc(0)->GetStorageFormat()));
    ge::Format formatX3 = hasX3Input ? static_cast<ge::Format>(ge::GetPrimaryFormat(
                                           context.GetOptionalInputDesc(INPUT_X3_IDX)->GetStorageFormat())) :
                                       ge::FORMAT_ND;

    OP_TILING_CHECK(
        (formatA == ge::FORMAT_FRACTAL_NZ || formatB == ge::FORMAT_FRACTAL_NZ || formatOut == ge::FORMAT_FRACTAL_NZ ||
         formatX3 == ge::FORMAT_FRACTAL_NZ),
        CUBE_INNER_ERR_REPORT("FusedMatMul", "invalid context"), return false);
    return true;
}

bool CheckDtype(const gert::TilingContext& context, bool hasX3Input, bool hasBias)
{
    std::vector<ge::DataType> dtype;
    dtype.push_back(context.GetInputDesc(INPUT_X1_IDX)->GetDataType());
    dtype.push_back(context.GetInputDesc(INPUT_X2_IDX)->GetDataType());
    if (hasBias) {
        dtype.push_back(context.GetOptionalInputDesc(INPUT_BIAS_IDX)->GetDataType());
    } else {
        dtype.push_back(context.GetInputDesc(INPUT_X1_IDX)->GetDataType());
    }
    if (hasX3Input) {
        dtype.push_back(context.GetOptionalInputDesc(INPUT_X3_IDX)->GetDataType());
    } else {
        dtype.push_back(context.GetOutputDesc(0)->GetDataType());
    }
    dtype.push_back(context.GetOutputDesc(0)->GetDataType());

    auto attrs = context.GetAttrs();
    std::string fusedOpType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    bool isSupportF32 = std::find(FusedOpTypeSupportF32.begin(), FusedOpTypeSupportF32.end(), fusedOpType) !=
                        FusedOpTypeSupportF32.end();

    for (auto& supported : DavidDtypeSupportListB16) {
        if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
            return true;
        }
    }
    if (isSupportF32 && std::equal(dtype.begin(), dtype.end(), DtypeSupportF32.begin())) {
        return true;
    }

    OP_LOGE(
        "FusedMatMul", "Unsupported data type: x1[%s], x2[%s], bias[%s], x3[%s], y[%s]",
        Ops::Base::ToString(dtype[INPUT_X1_IDX]).c_str(), Ops::Base::ToString(dtype[INPUT_X2_IDX]).c_str(),
        Ops::Base::ToString(dtype[INPUT_BIAS_IDX]).c_str(), Ops::Base::ToString(dtype[INPUT_X3_IDX]).c_str(),
        Ops::Base::ToString(dtype[OUTPUT_Y_IDX]).c_str());
    return false;
}

ge::graphStatus GetInputDims(const gert::Shape& storageShape, ge::Format format, int64_t (&dims)[TWO_BATCH_DIM])
{
    const size_t dimNum = storageShape.GetDimNum();
    if (format == ge::FORMAT_ND) {
        if (dimNum != TWO_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        dims[0] = storageShape[dimNum - TWO_BATCH_DIM];
        dims[1] = storageShape[dimNum - ONE_BATCH_DIM];
    } else {
        if (dimNum != FOUR_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        dims[0] = storageShape[dimNum - THREE_BATCH_DIM] * storageShape[dimNum - TWO_BATCH_DIM];
        dims[1] = storageShape[dimNum - FOUR_BATCH_DIM] * storageShape[dimNum - ONE_BATCH_DIM];
    }
    return ge::GRAPH_SUCCESS;
}

bool OpSpecificCheck(const gert::TilingContext& context, int64_t m, int64_t n, bool hasX3Input, bool hasBias)
{
    auto isMatrix = [](const gert::Shape& oriShape) { return oriShape.GetDimNum() == TWO_BATCH_DIM; };
    if (!isMatrix(context.GetInputShape(0)->GetOriginShape()) ||
        !isMatrix(context.GetInputShape(1)->GetOriginShape())) {
        OP_LOGE("FusedMatMul", "invalid input dim num");
        return false;
    }

    if (hasX3Input) {
        const gert::Shape& x3Shape = context.GetOptionalInputShape(INPUT_X3_IDX)->GetOriginShape();
        const size_t x3DimNum = x3Shape.GetDimNum();
        if (x3DimNum < TWO_BATCH_DIM) {
            OP_LOGE("FusedMatMul", "illegal value: output dim num (%zu)", x3DimNum);
            return false;
        }
        if (x3Shape[0] != m || x3Shape[1] != n) {
            OP_LOGE(
                "FusedMatMul", "illegal value: shape x3Shape[0]:%ld, x3Shape[1]:%ld, m:%ld, n:%ld", x3Shape[0],
                x3Shape[1], m, n);
            return false;
        }
    }
    // check bias
    if (hasBias) {
        const gert::Shape& biasShape = context.GetOptionalInputShape(INPUT_BIAS_IDX)->GetOriginShape();
        const int64_t biasValue = biasShape[biasShape.GetDimNum() - 1];
        if (biasValue != n) {
            OP_LOGE("FusedMatMul", "illegal value: bias[%ld], n[%ld]", biasValue, n);
            return false;
        }
    }
    return true;
}

bool GetShape(const gert::TilingContext& context, int64_t& m, int64_t& n, int64_t& k)
{
    // get transpose
    bool isATrans = *(context.GetAttrs()->GetAttrPointer<bool>(0));
    bool isBTrans = *(context.GetAttrs()->GetAttrPointer<bool>(1));

    // get (m, k, n)
    int64_t mkDims[TWO_BATCH_DIM];
    int64_t knDims[TWO_BATCH_DIM];
    if ((GetInputDims(context.GetInputShape(0)->GetStorageShape(), ge::FORMAT_ND, mkDims) != ge::GRAPH_SUCCESS) ||
        (GetInputDims(context.GetInputShape(1)->GetStorageShape(), ge::FORMAT_ND, knDims) != ge::GRAPH_SUCCESS)) {
        OP_LOGE("FusedMatMul", "invalid input dim num");
        return false;
    }
    uint64_t kIdxA = isATrans ? 0ULL : 1ULL;
    uint64_t kIdxB = isBTrans ? 1ULL : 0ULL;
    k = mkDims[kIdxA];
    if (k != knDims[kIdxB]) {
        OP_LOGE("FusedMatMul", "unequal input kDim values: k_left[%ld], k_right[%ld]", k, knDims[kIdxB]);
        return false;
    }
    uint64_t mIdx = isATrans ? 1ULL : 0ULL;
    uint64_t nIdx = isBTrans ? 0ULL : 1ULL;
    m = mkDims[mIdx];
    n = knDims[nIdx];
    return true;
}

bool CheckShape(const gert::TilingContext& context, bool hasX3Input, bool hasBias)
{
    int64_t m;
    int64_t n;
    int64_t k;
    if (!GetShape(context, m, n, k)) {
        return false;
    }

    auto isValidDimValue = [](int64_t dim) -> bool { return (dim > 0) && (dim <= INT32_MAX); };
    if (!isValidDimValue(m) || !isValidDimValue(k) || !isValidDimValue(n)) {
        OP_LOGE("FusedMatMul", "illegal value: m[%ld], k[%ld], n[%ld]", m, k, n);
        return false;
    }
    // check output dim num
    const gert::Shape& cShape = context.GetOutputShape(0)->GetOriginShape();
    const size_t cDimNum = cShape.GetDimNum();
    if (cDimNum < TWO_BATCH_DIM) {
        OP_LOGE("FusedMatMul", "illegal value: output dim num (%zu)", cDimNum);
        return false;
    }
    return OpSpecificCheck(context, m, n, hasX3Input, hasBias);
}

bool CheckFusedOpType(const gert::TilingContext& context)
{
    // get fused op type
    auto attrs = context.GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(&context, attrs);
    std::string fusedOpType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);

    // get available fused op type
    platform_ascendc::SocVersion socVersion;
    OP_TILING_CHECK(
        GetSocVersion(&context, socVersion) == ge::GRAPH_FAILED,
        CUBE_INNER_ERR_REPORT(context.GetNodeName(), "fail to get soc version"), return false);
    auto it = SocFusedOpSupport.find(socVersion);
    OP_TILING_CHECK(
        it == SocFusedOpSupport.end(),
        CUBE_INNER_ERR_REPORT(context.GetNodeName(), "unsupported platform(impossible situation)"), return false);

    // check op type support
    const auto& builtInSupportedOps = (it->second)[1];
    return std::find(builtInSupportedOps.begin(), builtInSupportedOps.end(), fusedOpType) != builtInSupportedOps.end();
}

/*
    Check if the current fused op type can use built-in tiling of BatchMatMulV3
*/
ge::graphStatus BuiltInTilingCheck(gert::TilingContext* context, bool& useBuiltInTiling)
{
    // get fused op type
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    std::string fusedOpType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);

    // get available fused op type
    platform_ascendc::SocVersion socVersion;
    OP_TILING_CHECK(
        GetSocVersion(context, socVersion) == ge::GRAPH_FAILED,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "fail to get soc version"), return ge::GRAPH_FAILED);
    auto it = SocFusedOpSupport.find(socVersion);
    OP_TILING_CHECK(
        it == SocFusedOpSupport.end(),
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "unsupported platform(impossible situation)"),
        return ge::GRAPH_FAILED);

    // check op type support
    const auto& builtInSupportedOps = (it->second)[0];
    useBuiltInTiling =
        std::find(builtInSupportedOps.begin(), builtInSupportedOps.end(), fusedOpType) != builtInSupportedOps.end();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedMatMulTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(
        context == nullptr, CUBE_INNER_ERR_REPORT("FusedMatMul", "context is null"), return ge::GRAPH_FAILED);
    if (!IsAdvancedSocVersion(context)) {
        OP_LOGE("FusedMatMul", "not support soc version");
        return ge::GRAPH_FAILED;
    }
    bool useBuiltInTiling = false;
    OP_TILING_CHECK(
        BuiltInTilingCheck(context, useBuiltInTiling) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to check built-in tiling"), return ge::GRAPH_FAILED);
    // 继承BMM Tiling
    if (useBuiltInTiling) {
        return fused_matmul::FusedMatMulBuiltInTiling(context).DoTiling();
    }
    return fused_matmul::FusedMatMulTiling(context).DoTiling();
}

} // namespace

namespace optiling {
IMPL_OP_OPTILING(FusedMatMul)
    .Tiling(FusedMatMulTilingFunc)
    .TilingParse<MatmulV3CompileInfo>(matmul_v3_advanced::InitCompileInfo)
    .GenSimplifiedKey(fused_matmul::GenSimplifiedKey);
namespace fused_matmul {

bool FusedMatMulTiling::CheckArgs() const
{
    return (
        CheckFormat(*context_, hasX3Input_) && CheckDtype(*context_, hasX3Input_, hasBias_) &&
        CheckShape(*context_, hasX3Input_, hasBias_) && CheckFusedOpType(*context_));
}

ge::graphStatus FusedMatMulTiling::CheckArgsPtr()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_X1_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_X1_IDX));

    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_X2_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_X2_IDX));

    hasBias_ =
        (context_->GetOptionalInputDesc(INPUT_BIAS_IDX) != nullptr &&
         context_->GetOptionalInputShape(INPUT_BIAS_IDX)->GetOriginShape().GetDimNum() > 0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(0));

    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(ATTR_TRANS_X1_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(ATTR_TRANS_X2_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(ATTR_ENABLE_HF32_IDX));

    if (*attrs->GetAttrPointer<bool>(ATTR_ENABLE_HF32_IDX)) {
        OP_LOGW(opName_, "invalid context not support hf32");
    }
    opType_ = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (opType_ == "mul" || opType_ == "add") {
        hasX3Input_ = true;
        OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputDesc(INPUT_X3_IDX));
    }
    return ge::GRAPH_SUCCESS;
}

bool FusedMatMulTiling::GetShapeAttrsInfo() // 检查输入属性是否支持
{
    opName_ = context_->GetNodeName();
    OP_TILING_CHECK(
        opName_ == nullptr, CUBE_INNER_ERR_REPORT("fusedMatmul", "get op name invalid context"), return false);
    OP_LOGI(opName_, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(
        (CheckArgsPtr() == ge::GRAPH_FAILED || !CheckArgs()), CUBE_INNER_ERR_REPORT(opName_, "invalid context"),
        return false);
    return true;
}

ge::graphStatus FusedMatMulTiling::DoTiling()
{
    if (!GetShapeAttrsInfo()) {
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        (FUSED_OP_TYPE_MAP.find(opType_) == FUSED_OP_TYPE_MAP.end()),
        CUBE_INNER_ERR_REPORT(opName_, "invalid opType_ %s", opType_.c_str()), return ge::GRAPH_FAILED);

    bool isATrans = *(context_->GetAttrs()->GetAttrPointer<bool>(0));
    bool isBTrans = *(context_->GetAttrs()->GetAttrPointer<bool>(1));

    // high level, trans, mm basic, no fullload, on the fly, fuse_op_type
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        0, static_cast<uint64_t>(GetTrans(isATrans, isBTrans)), 0, 0, 0, 0,
        static_cast<uint64_t>(FUSED_OP_TYPE_MAP.at(opType_)));

    int64_t m = 0L;
    int64_t n = 0L;
    int64_t k = 0L;
    GetShape(*context_, m, n, k);

    FusedMatMulTilingData tilingData{
        static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k), static_cast<uint32_t>(hasBias_)};
    OP_TILING_CHECK(
        context_->GetRawTilingData() == nullptr,
        CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "context rawtilingData is nullptr"), return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), &tilingData,
        sizeof(tilingData));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData));
    auto compileInfoPtr = reinterpret_cast<const MatmulV3CompileInfo*>(context_->GetCompileInfo());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);

    context_->SetBlockDim(compileInfoPtr->aicNum);
    context_->SetTilingKey(tilingKey);
    OP_LOGI(opName_, "TilingKey: %lu", tilingKey);
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workapce
    OP_TILING_CHECK(
        workspaces == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "workspace is nullptr"),
        return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(ALIGN_NUM * WORKSPACE_SIZE * WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
};
} // namespace fused_matmul
} // namespace optiling