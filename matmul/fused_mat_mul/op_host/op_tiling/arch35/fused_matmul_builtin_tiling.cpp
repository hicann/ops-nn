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
 * \brief FusedMatMul built-in tiling implementation.
 */

#include "fused_matmul_builtin_tiling.h"

#include "fused_matmul_common.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_tiling_key.h"
#include "matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_tiling_strategy.h"
#include "matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_tiling_advanced.h"
#include "matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_common_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "matmul/common/op_host/log_format_util.h"

namespace {
using namespace optiling;
using namespace optiling::fused_matmul;

// Layout: [x1, x2, y, bias, x3], DT_UNDEFINED = optional input absent

// opType group: "" / relu (no x3)
static const std::vector<std::vector<ge::DataType>> DTYPE_LIST_RELU_DAV_3510 = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_UNDEFINED},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_UNDEFINED, ge::DT_UNDEFINED},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_UNDEFINED},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_UNDEFINED},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_UNDEFINED},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_UNDEFINED, ge::DT_UNDEFINED},
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_UNDEFINED},
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_UNDEFINED, ge::DT_UNDEFINED},
};
static const std::vector<std::vector<ge::DataType>> DTYPE_LIST_RELU_DAV_RESV = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_UNDEFINED},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_UNDEFINED, ge::DT_UNDEFINED},
};

// opType group: add / mul (x3 required, x3Type == aType)
static const std::vector<std::vector<ge::DataType>> DTYPE_LIST_ADDMUL_DAV_3510 = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_UNDEFINED, ge::DT_FLOAT16},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_UNDEFINED, ge::DT_BF16},
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_UNDEFINED, ge::DT_FLOAT},
};

// opType: scale_add (x3 required, no bias, x1/x2/x3/y use the same fp16/bf16 dtype)
static const std::vector<std::vector<ge::DataType>> DTYPE_LIST_SCALE_ADD_DAV_3510 = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_UNDEFINED, ge::DT_FLOAT16},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_UNDEFINED, ge::DT_BF16},
};

// opType group: 16cast32 (no x3, only DAV_3510)
static const std::vector<std::vector<ge::DataType>> DTYPE_LIST_16CAST32_DAV_3510 = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_UNDEFINED},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_UNDEFINED},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_UNDEFINED, ge::DT_UNDEFINED},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_UNDEFINED},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_UNDEFINED},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_UNDEFINED, ge::DT_UNDEFINED},
};

// opType group: gelu_erf / gelu_tanh (no bias, no x3, only DAV_3510)
static const std::vector<std::vector<ge::DataType>> DTYPE_LIST_GELU_DAV_3510 = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_UNDEFINED, ge::DT_UNDEFINED},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_UNDEFINED, ge::DT_UNDEFINED},
};

// opType group: quant / relu_quant (x3 required, only DAV_RESV)
static const std::vector<std::vector<ge::DataType>> DTYPE_LIST_QUANT_DAV_RESV = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_UINT64},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_UNDEFINED, ge::DT_UINT64},
};

constexpr size_t DTYPE_BIAS_INDEX = 3UL;
constexpr size_t DTYPE_X3_INDEX = 4UL;
constexpr size_t DTYPE_LIST_SIZE = 5UL;
constexpr size_t BATCH_DIM_INDEX = 0UL;
constexpr size_t MATRIX_ROW_DIM_INDEX = 1UL;
constexpr size_t MATRIX_COLUMN_DIM_INDEX = 2UL;

} // namespace

namespace optiling {
namespace fused_matmul {

// ====== Phase 2: ValidateInputsNotNull (own attr layout, no base call) ======
ge::graphStatus FusedMatMulBuiltInTiling::ValidateInputsNotNull()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(ATTR_TRANS_X1_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(ATTR_TRANS_X2_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(ATTR_ENABLE_HF32_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(0));
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 3: DetectOptionalInputs (bias + x3, with null-check) ======
ge::graphStatus FusedMatMulBuiltInTiling::DetectOptionalInputs()
{
    if (context_->GetOptionalInputDesc(INPUT_BIAS_IDX) != nullptr &&
        context_->GetOptionalInputShape(INPUT_BIAS_IDX)->GetOriginShape().GetDimNum() > 0) {
        args_.hasBias = true;
    }
    opType_ = context_->GetAttrs()->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (IsAddMulOpType(opType_) || IsQuantOpType(opType_) || opType_ == "scale_add") {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputDesc(INPUT_X3_IDX));
        OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputShape(INPUT_X3_IDX));
        args_.hasX3Input = true;
    }
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 5: ExtractDtype (base + x3Type) ======
void FusedMatMulBuiltInTiling::ExtractDtype()
{
    MatMulV3Tiling::ExtractDtype();
    if (args_.hasX3Input) {
        args_.x3Type = context_->GetOptionalInputDesc(INPUT_X3_IDX)->GetDataType();
    }
}

// ====== Phase 5: ExtractAttrFlags (hf32 from ATTR_ENABLE_HF32_IDX, innerPrecise) ======
void FusedMatMulBuiltInTiling::ExtractAttrFlags()
{
    args_.isHf32 = *context_->GetAttrs()->GetAttrPointer<bool>(ATTR_ENABLE_HF32_IDX);
    if (args_.isHf32 && arch_ != NpuArch::DAV_3510) {
        OP_LOGW(args_.opName, "Hf32 flag is: %d, which is not supported yet", args_.isHf32);
    }
    innerPrecise_ = *context_->GetAttrs()->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE_IDX);
    OP_LOGI(args_.opName, "FusedMatMul built-in tiling inner_precise is %ld", innerPrecise_);
    OP_LOGD(args_.opName, "Hf32 flag is: %d", args_.isHf32);
}

// ====== Phase 7: ValidateOpSpecific (constraints not needing batchInfo) ======
ge::graphStatus FusedMatMulBuiltInTiling::ValidateOpSpecific()
{
    OP_TILING_CHECK(innerPrecise_ != INNER_PRECISE_HIGH_PRECISION && innerPrecise_ != INNER_PRECISE_HIGH_PERFORMANCE,
                    CUBE_INNER_ERR_REPORT(args_.opName, "inner_precise only supports 0 or 1"), return ge::GRAPH_FAILED);

    const auto& aShape = context_->GetInputShape(0)->GetOriginShape();
    const auto& bShape = context_->GetInputShape(1)->GetOriginShape();

    // gelu: input dims must be 2
    if (IsGeluOpType(opType_)) {
        const size_t aDimNum = aShape.GetDimNum();
        const size_t bDimNum = bShape.GetDimNum();
        const size_t cDimNum = context_->GetOutputShape(0)->GetOriginShape().GetDimNum();
        if (aDimNum != NUM_TWO || bDimNum != NUM_TWO || cDimNum != NUM_TWO) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                args_.opName, "x1, x2, y", Ops::NN::FormatString("%zu, %zu, %zu", aDimNum, bDimNum, cDimNum).c_str(),
                Ops::NN::FormatString("The shape dims of %s must be %zu for gelu op type", "x1, x2, y", NUM_TWO)
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }

    // add/mul/scale_add: basic API capability + x3 format + x3 M/N shape
    if (IsAddMulOpType(opType_) || opType_ == "scale_add") {
        auto compileInfo = reinterpret_cast<const MatmulV3CompileInfo*>(context_->GetCompileInfo());
        OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
        if (compileInfo->aivNum != (compileInfo->aicNum * NUM_TWO)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                args_.opName, "aivNum, aicNum",
                Ops::NN::FormatString("%lu, %lu", compileInfo->aivNum, compileInfo->aicNum).c_str(),
                "FusedMatMul add/mul/scale_add basic API requires aivNum == aicNum * 2");
            return ge::GRAPH_FAILED;
        }
        auto x3Desc = context_->GetOptionalInputDesc(INPUT_X3_IDX);
        if (args_.aFormat != ge::FORMAT_ND || args_.bFormat != ge::FORMAT_ND ||
            x3Desc->GetStorageFormat() != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                args_.opName, "x1, x2, x3",
                Ops::NN::FormatString("%s, %s, %s", ge::TypeUtils::FormatToSerialString(args_.aFormat).c_str(),
                                      ge::TypeUtils::FormatToSerialString(args_.bFormat).c_str(),
                                      ge::TypeUtils::FormatToSerialString(x3Desc->GetStorageFormat()).c_str())
                    .c_str(),
                "The storage formats of x1, x2 and x3 must be ND for add/mul/scale_add op type");
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& x3Shape = context_->GetOptionalInputShape(INPUT_X3_IDX)->GetOriginShape();
        const size_t x3DimNum = x3Shape.GetDimNum();
        if (x3DimNum < NUM_TWO || x3DimNum > NUM_THREE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                args_.opName, "x3", Ops::NN::FormatString("%zu", x3DimNum).c_str(),
                Ops::NN::FormatString("The shape dim of %s must be within the range %s", "x3", "{2, 3}").c_str());
            return ge::GRAPH_FAILED;
        }
        if (x3Shape[x3DimNum - NUM_TWO] != static_cast<int64_t>(args_.mValue) ||
            x3Shape[x3DimNum - 1] != static_cast<int64_t>(args_.nValue)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                args_.opName, "x3", Ops::Base::ToString(x3Shape).c_str(),
                Ops::NN::FormatString(
                    "%s of %s must be equal to %s of %s (%lu), %s of %s must be equal to %s of %s (%lu)", "Shape[-2]",
                    "x3", "Shape[-2]", "y", args_.mValue, "Shape[-1]", "x3", "Shape[-1]", "y", args_.nValue)
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }

    if (opType_ == "scale_add") {
        const auto* compileInfo = reinterpret_cast<const MatmulV3CompileInfo*>(context_->GetCompileInfo());
        OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
        const auto& x3Shape = context_->GetOptionalInputShape(INPUT_X3_IDX)->GetOriginShape();
        const auto& yShape = context_->GetOutputShape(0)->GetOriginShape();
        const auto* x1Desc = context_->GetInputDesc(INPUT_X1_IDX);
        const auto* x2Desc = context_->GetInputDesc(INPUT_X2_IDX);
        const auto* x3Desc = context_->GetOptionalInputDesc(INPUT_X3_IDX);
        const auto* yDesc = context_->GetOutputDesc(0);
        if (compileInfo->aicNum == 0UL || args_.isATrans || args_.isBTrans) {
            OP_LOGE(args_.opName,
                    "FusedMatMul scale_add requires aicNum greater than 0 and does not support transpose");
            return ge::GRAPH_FAILED;
        }
        if (x1Desc->GetStorageFormat() != ge::FORMAT_ND || x2Desc->GetStorageFormat() != ge::FORMAT_ND ||
            x3Desc->GetStorageFormat() != ge::FORMAT_ND || yDesc->GetStorageFormat() != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                args_.opName, "x1, x2, x3, y",
                Ops::NN::FormatString("%s, %s, %s, %s",
                                      ge::TypeUtils::FormatToSerialString(x1Desc->GetStorageFormat()).c_str(),
                                      ge::TypeUtils::FormatToSerialString(x2Desc->GetStorageFormat()).c_str(),
                                      ge::TypeUtils::FormatToSerialString(x3Desc->GetStorageFormat()).c_str(),
                                      ge::TypeUtils::FormatToSerialString(yDesc->GetStorageFormat()).c_str())
                    .c_str(),
                "The storage formats of x1, x2, x3 and y must be ND for scale_add op type");
            return ge::GRAPH_FAILED;
        }
        if (aShape.GetDimNum() != FUSED_MATMUL_BATCH_MATMUL_DIM_NUM ||
            bShape.GetDimNum() != FUSED_MATMUL_BATCH_MATMUL_DIM_NUM ||
            x3Shape.GetDimNum() != FUSED_MATMUL_BATCH_MATMUL_DIM_NUM ||
            yShape.GetDimNum() != FUSED_MATMUL_BATCH_MATMUL_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                args_.opName, "x1, x2, x3, y",
                Ops::NN::FormatString("%zu, %zu, %zu, %zu", aShape.GetDimNum(), bShape.GetDimNum(), x3Shape.GetDimNum(),
                                      yShape.GetDimNum())
                    .c_str(),
                "The shape dims of x1, x2, x3 and y must be 3 for scale_add op type");
            return ge::GRAPH_FAILED;
        }

        const int64_t batch = aShape.GetDim(BATCH_DIM_INDEX);
        const int64_t m = aShape.GetDim(MATRIX_ROW_DIM_INDEX);
        const int64_t k = aShape.GetDim(MATRIX_COLUMN_DIM_INDEX);
        const int64_t n = bShape.GetDim(MATRIX_COLUMN_DIM_INDEX);
        if (batch <= 0 || m <= 0 || n <= 0 || k <= 0 || bShape.GetDim(BATCH_DIM_INDEX) != batch ||
            bShape.GetDim(MATRIX_ROW_DIM_INDEX) != k || x3Shape.GetDim(BATCH_DIM_INDEX) != batch ||
            x3Shape.GetDim(MATRIX_ROW_DIM_INDEX) != m || x3Shape.GetDim(MATRIX_COLUMN_DIM_INDEX) != n ||
            yShape.GetDim(BATCH_DIM_INDEX) != batch || yShape.GetDim(MATRIX_ROW_DIM_INDEX) != m ||
            yShape.GetDim(MATRIX_COLUMN_DIM_INDEX) != n) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                args_.opName, "x1, x2, x3, y",
                Ops::NN::FormatString("%s, %s, %s, %s", Ops::Base::ToString(aShape).c_str(),
                                      Ops::Base::ToString(bShape).c_str(), Ops::Base::ToString(x3Shape).c_str(),
                                      Ops::Base::ToString(yShape).c_str())
                    .c_str(),
                "scale_add requires x1[B,M,K] * x2[B,K,N] + x3[B,M,N] -> y[B,M,N] with positive dimensions");
            return ge::GRAPH_FAILED;
        }
    }

    // quant: x3 must be [1]
    if (IsQuantOpType(opType_)) {
        const gert::Shape& x3Shape = context_->GetOptionalInputShape(INPUT_X3_IDX)->GetOriginShape();
        if (x3Shape.GetDimNum() != 1UL || x3Shape.GetShapeSize() != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                args_.opName, "x3", Ops::Base::ToString(x3Shape).c_str(),
                Ops::NN::FormatString("The shape of %s must be [1] for quant/relu_quant", "x3").c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 7: ValidateBias (bias shape constraints: no batch bias) ======
ge::graphStatus FusedMatMulBuiltInTiling::ValidateBias()
{
    // gelu/scale_add op type does not support bias
    if ((IsGeluOpType(opType_) || opType_ == "scale_add") && args_.hasBias) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            args_.opName, "fusedOpType, bias", Ops::NN::FormatString("%s, not null", opType_.c_str()).c_str(),
            Ops::NN::FormatString("The input %s is not supported for %s op type", "bias", opType_.c_str()).c_str());
        return ge::GRAPH_FAILED;
    }
    if (!args_.hasBias) {
        return ge::GRAPH_SUCCESS;
    }
    // bias[-1] == c[-1] (base class check)
    if (MatMulV3Tiling::ValidateBias() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto biasShape = context_->GetOptionalInputShape(INPUT_BIAS_IDX)->GetOriginShape();
    size_t biasDims = biasShape.GetDimNum();
    if (IsQuantOpType(opType_)) {
        // quant: bias must be 1D
        if (biasDims != 1UL) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                args_.opName, "bias", Ops::NN::FormatString("%zu", biasDims).c_str(),
                Ops::NN::FormatString("The shape dim of %s must be 1 or empty for quant/relu_quant", "bias").c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        // non-quant: bias must be <= 2D (no batch bias), biasShape[0]==1 when 2D
        if (biasDims > NUM_TWO) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                args_.opName, "bias", Ops::NN::FormatString("%zu", biasDims).c_str(),
                Ops::NN::FormatString("The shape dim of %s must be less than %llu", "bias", MAX_BIAS_DIM).c_str());
            return ge::GRAPH_FAILED;
        }
        if (biasDims == NUM_TWO && biasShape[0] != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                args_.opName, "bias", Ops::Base::ToString(biasShape).c_str(),
                Ops::NN::FormatString("%s of %s must be equal to %d", "M-axis", "bias", 1).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 7: GetDtypeSupportList (opType group + npuArch) ======
std::vector<std::vector<ge::DataType>> FusedMatMulBuiltInTiling::GetDtypeSupportList() const
{
    if (opType_ == "scale_add") {
        return DTYPE_LIST_SCALE_ADD_DAV_3510;
    }
    if (IsQuantOpType(opType_)) {
        return DTYPE_LIST_QUANT_DAV_RESV;
    }
    if (IsAddMulOpType(opType_)) {
        return DTYPE_LIST_ADDMUL_DAV_3510;
    }
    if (opType_ == "16cast32") {
        return DTYPE_LIST_16CAST32_DAV_3510;
    }
    if (IsGeluOpType(opType_)) {
        return DTYPE_LIST_GELU_DAV_3510;
    }
    // "" / relu
    return (arch_ == NpuArch::DAV_3510) ? DTYPE_LIST_RELU_DAV_3510 : DTYPE_LIST_RELU_DAV_RESV;
}

// ====== Phase 7: ValidateDtype (unified 5-element exact match) ======
ge::graphStatus FusedMatMulBuiltInTiling::ValidateDtype()
{
    // build fixed 5-element dtypes
    std::vector<std::string> names = {"x1", "x2", "y", "bias", "x3"};
    std::vector<ge::DataType> dtypes = {args_.aType, args_.bType, args_.cType, ge::DT_UNDEFINED, ge::DT_UNDEFINED};
    if (args_.hasBias) {
        dtypes[DTYPE_BIAS_INDEX] = args_.biasType;
    }
    if (args_.hasX3Input) {
        dtypes[DTYPE_X3_INDEX] = args_.x3Type;
    }

    auto supportList = GetDtypeSupportList();
    for (auto& supported : supportList) {
        if (supported.size() >= DTYPE_LIST_SIZE && std::equal(dtypes.begin(), dtypes.end(), supported.begin())) {
            return ge::GRAPH_SUCCESS;
        }
    }

    // error message (exclude absent inputs)
    std::string nameStr;
    std::string valueStr;
    for (size_t i = 0; i < DTYPE_LIST_SIZE; ++i) {
        if (dtypes[i] == ge::DT_UNDEFINED) {
            continue;
        }
        if (!nameStr.empty()) {
            nameStr += ", ";
            valueStr += ", ";
        }
        nameStr += names[i];
        valueStr += Ops::Base::ToString(dtypes[i]);
    }
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
        args_.opName, nameStr.c_str(), valueStr.c_str(),
        Ops::NN::FormatString("The dtypes of %s must be within the range %s", nameStr.c_str(), "dtype support list")
            .c_str());
    return ge::GRAPH_FAILED;
}

// ====== Phase 8: ValidateMatrixBatchInfo (no broadcast on non-DAV_RESV) ======
ge::graphStatus FusedMatMulBuiltInTiling::ValidateMatrixBatchInfo()
{
    if (arch_ != NpuArch::DAV_RESV) {
        const auto& aShape = context_->GetInputShape(0)->GetOriginShape();
        const auto& bShape = context_->GetInputShape(1)->GetOriginShape();
        if (IsBatchBroadcast(aShape, bShape)) {
            OP_LOGE(args_.opName, "Batch broadcast is only supported on DAV_RESV, but current npu arch is %d.",
                    static_cast<int32_t>(arch_));
            return ge::GRAPH_FAILED;
        }
    }
    return BatchMatMulV3Tiling::ValidateMatrixBatchInfo();
}

// ====== Phase 8: ExtractOptionalBatchInfo (no batch bias, set batchBias=1) ======
ge::graphStatus FusedMatMulBuiltInTiling::ExtractOptionalBatchInfo()
{
    batchInfo_.batchBias = 1;
    args_.batchInfo = &batchInfo_;
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 9: ValidateOptionalBatchInfo (x3 batch-axis broadcast only) ======
ge::graphStatus FusedMatMulBuiltInTiling::ValidateOptionalBatchInfo()
{
    if (!args_.hasX3Input) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape& x3Shape = context_->GetOptionalInputShape(INPUT_X3_IDX)->GetOriginShape();
    const size_t x3DimNum = x3Shape.GetDimNum();
    // x3 batch-axis broadcast (only 3D x3 has batch axis)
    if (x3DimNum == NUM_THREE) {
        if (x3Shape[0] != 1 && x3Shape[0] != static_cast<int>(args_.batchInfo->batchC)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                args_.opName, "x3", Ops::Base::ToString(x3Shape).c_str(),
                Ops::NN::FormatString(
                    "The batch-axis of %s must meet the broadcast principle: The batch-axis in the corresponding "
                    "positions must be equal, or one of the batch-axis in the corresponding positions must be 1",
                    "x3, y")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        args_.batchX3 = x3Shape[0];
    }
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 10: Registry priorities ======
std::vector<int32_t> FusedMatMulBuiltInTiling::GetRegistryPriorities(NpuArch npuArch) const
{
    return strategy::GetFusedMatMulPriorities(npuArch);
}

// ====== Phase 10: Tiling key ======
MatMulV3TilingKey* FusedMatMulBuiltInTiling::GetTilingKeyObj()
{
    auto it = FUSED_OP_TYPE_MAP.find(opType_);
    if (it == FUSED_OP_TYPE_MAP.end()) {
        OP_LOGE(args_.opName, "invalid opType: %s", opType_.c_str());
        return nullptr;
    }
    fusedMatmulTilingKey_.SetFusedOpType(it->second);
    fusedMatmulTilingKey_.SetInnerPrecise(static_cast<FusedInnerPrecise>(innerPrecise_));
    return &fusedMatmulTilingKey_;
}
} // namespace fused_matmul
} // namespace optiling
