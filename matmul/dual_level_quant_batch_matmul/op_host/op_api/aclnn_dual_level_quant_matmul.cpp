/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_dual_level_quant_matmul_nz.h"
#include "dual_level_quant_matmul.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"

#include <limits>
#include "graph/types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "log/log.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
using Ops::NN::SwapLastTwoDimValue;
using TupleRequiredTensor = std::tuple<const aclTensor*&, const aclTensor*&, const aclTensor*&, const aclTensor*&,
                                       const aclTensor*&, const aclTensor*&, const aclTensor*&, aclTensor*&>;
using TupleOptionalTensor = std::tuple<const aclTensor*&>;
using TupleAttr = std::tuple<bool&, bool&, int64_t&, int64_t&>;

static constexpr int INDEX_X1_IN_MANDATORY_TUPLE = 0;
static constexpr int INDEX_X2_IN_MANDATORY_TUPLE = 1;
static constexpr int INDEX_X2_REF_IN_MANDATORY_TUPLE = 2;
static constexpr int INDEX_X1_L0_SCALE_IN_MANDATORY_TUPLE = 3;
static constexpr int INDEX_X1_L1_SCALE_IN_MANDATORY_TUPLE = 4;
static constexpr int INDEX_X2_L0_SCALE_IN_MANDATORY_TUPLE = 5;
static constexpr int INDEX_X2_L1_SCALE_IN_MANDATORY_TUPLE = 6;
static constexpr int INDEX_Y_IN_MANDATORY_TUPLE = 7;
static constexpr int INDEX_BIAS_IN_OPTIONAL_TUPLE = 0;
static constexpr int INDEX_TRANSPOSE_X1_IN_ATTR_TUPLE = 0;
static constexpr int INDEX_TRANSPOSE_X2_IN_ATTR_TUPLE = 1;
static constexpr int INDEX_L0_GROUP_SIZE_IN_ATTR_TUPLE = 2;
static constexpr int INDEX_L1_GROUP_SIZE_IN_ATTR_TUPLE = 3;

namespace {
const size_t FIRST_DIM = 0;
const size_t SECOND_DIM = 1;
const size_t BIAS_DIM_MAX_VALUE = 2;
const uint64_t ONE_DIM_VALUE = 1;
const uint64_t TWO_DIM_VALUE = 2;
const uint64_t THREE_DIM_VALUE = 3;
const uint64_t FP4_NUMS_IN_INT8 = 2;
const int64_t SECOND_LAST_DIM = 2;

static constexpr const char* kOpName = "aclnnDualLevelQuantMatmulWeightNz";

static const std::initializer_list<DataType> X1_X2_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<DataType> X1_X2_INPUT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<DataType> L0_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<DataType> L1_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<DataType> Y_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::vector<uint64_t> DIM_RANGE_ONLY_TWO_DIM = {TWO_DIM_VALUE, TWO_DIM_VALUE};
static const std::vector<uint64_t> DIM_RANGE_ONLY_THREE_DIM = {THREE_DIM_VALUE, THREE_DIM_VALUE};
static const std::vector<uint64_t> DIM_RANGE_OPTIONAL_INPUT = {ONE_DIM_VALUE, TWO_DIM_VALUE};

static bool IsFormatSupport(const aclTensor* input, Format format, const std::string& inputName)
{
    if (input != nullptr && input->GetStorageFormat() != format) {
        OP_LOGE_FOR_INVALID_FORMAT(kOpName, inputName.c_str(), op::ToString(input->GetStorageFormat()).GetString(),
                                   op::ToString(format).GetString());
        return false;
    }
    return true;
}

static bool IsDimSupport(const aclTensor* input, const std::vector<uint64_t>& dimRange, const std::string& inputName)
{
    if (input != nullptr &&
        (input->GetViewShape().GetDimNum() < dimRange[0] || input->GetViewShape().GetDimNum() > dimRange[1])) {
        auto dimNum = input->GetViewShape().GetDimNum();
        std::string dimReason = "The shape dim of " + inputName + " must be in range [" + std::to_string(dimRange[0]) +
                                "D, " + std::to_string(dimRange[1]) + "D]";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, inputName.c_str(), std::to_string(dimNum).c_str(),
                                                 dimReason.c_str());
        return false;
    }
    return true;
}

static aclnnStatus CheckSocValid()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    NpuArch npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_3510:
            break;
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
            return ACLNN_ERR_RUNTIME_ERROR;
        }
    }
    return ACLNN_SUCCESS;
}

static bool IsNzFormat(const aclTensor* weight)
{
    return weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ ||
           weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ_C0_2;
}

static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Level0Scale,
                         const aclTensor* x2Level0Scale, const aclTensor* x1Level1Scale, const aclTensor* x2Level1Scale,
                         const aclTensor* y)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(x1Level0Scale, return false);
    OP_CHECK_NULL(x1Level1Scale, return false);
    OP_CHECK_NULL(x2Level0Scale, return false);
    OP_CHECK_NULL(x2Level1Scale, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool SetShapeStrideForNZ(const aclTensor* weight, aclTensor* weightTemp, bool transposeX2)
{
    if (!transposeX2) {
        op::Strides newStrides = weight->GetViewStrides();
        auto strideSize = newStrides.size();
        OP_CHECK(strideSize >= DIM_RANGE_ONLY_TWO_DIM.front() && strideSize <= DIM_RANGE_ONLY_TWO_DIM.back(),
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "weight", std::to_string(strideSize).c_str(),
                                                          "The shape dim of weight must be 2D"),
                 return false);
        // 当transposeX2=false时，viewStride的倒数第二维要放大2倍， 即(n/2，1) -> (n, 1)
        newStrides[strideSize - SECOND_LAST_DIM] *= FP4_NUMS_IN_INT8;
        weightTemp->SetViewStrides(newStrides);
    }
    auto newOriginalShape = transposeX2 ? SwapLastTwoDimValue(weightTemp->GetViewShape()) : weightTemp->GetViewShape();
    weightTemp->SetOriginalShape(newOriginalShape);
    // storageShape的倒数第一维要放大2倍， 即(n/64,k/16,16,32) -> (n/64,k/16,16,64)
    auto storageShape = weight->GetStorageShape();
    auto storageShapeDim = storageShape.GetDimNum();
    storageShape[storageShapeDim - 1] *= FP4_NUMS_IN_INT8;
    weightTemp->SetStorageShape(storageShape);
    return true;
}

static aclnnStatus ModifyTensorDtype(const aclTensor*& tensorRef, aclTensor* tensorMod, DataType dtype,
                                     aclOpExecutor* executor)
{
    auto tensorTmp = tensorMod == nullptr ?
                         executor->CreateView(tensorRef, tensorRef->GetViewShape(), tensorRef->GetViewOffset()) :
                         tensorMod;
    CHECK_RET(tensorTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    tensorTmp->SetDataType(dtype);
    tensorRef = tensorTmp;
    return ACLNN_SUCCESS;
}

static aclnnStatus SetNZC0FormatToNZFormat(aclTensor* input)
{
    if (input->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ_C0_2) {
        input->SetViewFormat(op::Format::FORMAT_ND);
        input->SetOriginalFormat(op::Format::FORMAT_ND);
        input->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus InputPreProcess(const aclTensor* input, const aclTensor*& inputRef, const char* inputName,
                                   aclOpExecutor* executor)
{
    CHECK_RET(IsDimSupport(input, DIM_RANGE_ONLY_TWO_DIM, inputName), ACLNN_ERR_PARAM_INVALID);
    auto viewShape = input->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    // shape 的最后一维要放大 2 倍， 即 (k, n/2) -> (k, n)
    viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * FP4_NUMS_IN_INT8;
    auto inputTemp = executor->CreateView(input, viewShape, input->GetViewOffset());
    CHECK_RET(inputTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (IsNzFormat(input)) {
        CHECK_RET(SetShapeStrideForNZ(input, inputTemp, false), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(SetNZC0FormatToNZFormat(inputTemp) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }
    if (input->GetDataType() == DataType::DT_INT8) {
        CHECK_RET(ModifyTensorDtype(inputRef, inputTemp, DataType::DT_FLOAT4_E2M1, executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
        OP_LOGD("The conversion of input from int8 to fp4_e2m1 is completed.");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus InputTensorProcess(TupleRequiredTensor mandatoryTensors, aclOpExecutor* executor)
{
    auto& x2 = std::get<INDEX_X2_IN_MANDATORY_TUPLE>(mandatoryTensors);
    auto& x2Ref = std::get<INDEX_X2_REF_IN_MANDATORY_TUPLE>(mandatoryTensors);
    // 将 int8 的输入 x2 dtype 修改为 mxfp4, 同时 ViewShape, ViewStrides 也从 int8 修改为 mxfp4 所对应的
    OP_CHECK(x2->GetDataType() == DataType::DT_INT8,
             OP_LOGE_FOR_INVALID_DTYPE(kOpName, "x2", op::ToString(x2->GetDataType()).GetString(), "INT8"),
             return ACLNN_ERR_PARAM_INVALID);
    // 当前 InputPreProcess 的处理逻辑是以 x2 为转置且连续为前提
    CHECK_RET(InputPreProcess(x2, x2Ref, "x2", executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX1(const aclTensor* x1)
{
    int64_t kX1 = x1->GetViewShape().GetDim(SECOND_DIM);

    OP_CHECK(
        IsContiguous(x1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "x1", "not contiguous", "The value of x1 must be contiguous"),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x1, DIM_RANGE_ONLY_TWO_DIM, "x1"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsFormatSupport(x1, Format::FORMAT_ND, "x1"), ACLNN_ERR_PARAM_INVALID);
    if (!CheckType(x1->GetDataType(), X1_X2_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "x1", op::ToString(x1->GetDataType()).GetString(),
                                              std::string("The dtype of x1 must be within the range ") +
                                                  op::ToString(X1_X2_DTYPE_SUPPORT_LIST).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 当 x1 为非转置时，k 作为内轴必须为偶数
    OP_CHECK(
        (static_cast<uint64_t>(kX1) & 1) == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "x1.k", std::to_string(kX1).c_str(),
                                              "When the value of transposeX1 is false, the value of x1.k must be even"),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX2(const aclTensor* x2)
{
    OP_CHECK(
        IsContiguous(x2),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "x2", "not contiguous", "The value of x2 must be contiguous"),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x2, DIM_RANGE_ONLY_TWO_DIM, "x2"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsFormatSupport(x2, Format::FORMAT_FRACTAL_NZ, "x2"), ACLNN_ERR_PARAM_INVALID);
    if (!CheckType(x2->GetDataType(), X1_X2_INPUT_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "x2", op::ToString(x2->GetDataType()).GetString(),
                                              std::string("The dtype of x2 must be within the range ") +
                                                  op::ToString(X1_X2_INPUT_DTYPE_SUPPORT_LIST).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX1L0Scale(const aclTensor* x1Level0Scale)
{
    OP_CHECK(IsContiguous(x1Level0Scale),
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "x1Level0Scale", "not contiguous",
                                                   "The value of x1Level0Scale must be contiguous"),
             return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x1Level0Scale, DIM_RANGE_ONLY_TWO_DIM, "x1Level0Scale"), ACLNN_ERR_PARAM_INVALID);
    if (!CheckType(x1Level0Scale->GetDataType(), L0_SCALE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "x1Level0Scale",
                                              op::ToString(x1Level0Scale->GetDataType()).GetString(),
                                              std::string("The dtype of x1Level0Scale must be within the range ") +
                                                  op::ToString(L0_SCALE_DTYPE_SUPPORT_LIST).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(IsFormatSupport(x1Level0Scale, Format::FORMAT_ND, "x1Level0Scale"), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX1L1Scale(const aclTensor* x1Level1Scale)
{
    OP_CHECK(IsContiguous(x1Level1Scale),
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "x1Level1Scale", "not contiguous",
                                                   "The value of x1Level1Scale must be contiguous"),
             return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x1Level1Scale, DIM_RANGE_ONLY_THREE_DIM, "x1Level1Scale"), ACLNN_ERR_PARAM_INVALID);
    if (!CheckType(x1Level1Scale->GetDataType(), L1_SCALE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "x1Level1Scale",
                                              op::ToString(x1Level1Scale->GetDataType()).GetString(),
                                              std::string("The dtype of x1Level1Scale must be within the range ") +
                                                  op::ToString(L1_SCALE_DTYPE_SUPPORT_LIST).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x1Level1Scale->GetStorageFormat() != Format::FORMAT_ND &&
        x1Level1Scale->GetStorageFormat() != Format::FORMAT_NCL) {
        OP_LOGE_FOR_INVALID_FORMAT(kOpName, "x1Level1Scale",
                                   op::ToString(x1Level1Scale->GetStorageFormat()).GetString(), "ND or NCL");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX2L0Scale(const aclTensor* x2Level0Scale)
{
    OP_CHECK(IsContiguous(x2Level0Scale),
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "x2Level0Scale", "not contiguous",
                                                   "The value of x2Level0Scale must be contiguous"),
             return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x2Level0Scale, DIM_RANGE_ONLY_TWO_DIM, "x2Level0Scale"), ACLNN_ERR_PARAM_INVALID);
    if (!CheckType(x2Level0Scale->GetDataType(), L0_SCALE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "x2Level0Scale",
                                              op::ToString(x2Level0Scale->GetDataType()).GetString(),
                                              std::string("The dtype of x2Level0Scale must be within the range ") +
                                                  op::ToString(L0_SCALE_DTYPE_SUPPORT_LIST).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(IsFormatSupport(x2Level0Scale, Format::FORMAT_ND, "x2Level0Scale"), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX2L1Scale(const aclTensor* x2Level1Scale)
{
    OP_CHECK(IsContiguous(x2Level1Scale),
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "x2Level1Scale", "not contiguous",
                                                   "The value of x2Level1Scale must be contiguous"),
             return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x2Level1Scale, DIM_RANGE_ONLY_THREE_DIM, "x2Level1Scale"), ACLNN_ERR_PARAM_INVALID);
    if (!CheckType(x2Level1Scale->GetDataType(), L1_SCALE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "x2Level1Scale",
                                              op::ToString(x2Level1Scale->GetDataType()).GetString(),
                                              std::string("The dtype of x2Level1Scale must be within the range ") +
                                                  op::ToString(L1_SCALE_DTYPE_SUPPORT_LIST).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2Level1Scale->GetStorageFormat() != Format::FORMAT_ND &&
        x2Level1Scale->GetStorageFormat() != Format::FORMAT_NCL) {
        OP_LOGE_FOR_INVALID_FORMAT(kOpName, "x2Level1Scale",
                                   op::ToString(x2Level1Scale->GetStorageFormat()).GetString(), "ND or NCL");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorY(const aclTensor* y)
{
    OP_CHECK(IsContiguous(y),
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "y", "not contiguous", "The value of y must be contiguous"),
             return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(y, DIM_RANGE_ONLY_TWO_DIM, "y"), ACLNN_ERR_PARAM_INVALID);
    if (!CheckType(y->GetDataType(), Y_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            kOpName, "y", op::ToString(y->GetDataType()).GetString(),
            std::string("The dtype of y must be within the range ") + op::ToString(Y_DTYPE_SUPPORT_LIST).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorBias(const aclTensor* bias)
{
    if (bias == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(IsContiguous(bias),
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "bias", "not contiguous",
                                                   "The value of bias must be contiguous"),
             return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(bias, DIM_RANGE_OPTIONAL_INPUT, "bias"), ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(bias->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE_FOR_INVALID_DTYPE(kOpName, "bias", op::ToString(bias->GetDataType()).GetString(), "FLOAT"),
             return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsFormatSupport(bias, Format::FORMAT_ND, "bias"), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInputOutputShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                         const aclTensor* y)
{
    int64_t mX1 = x1->GetViewShape().GetDim(FIRST_DIM);
    int64_t kX1 = x1->GetViewShape().GetDim(SECOND_DIM);
    int64_t nX2 = x2->GetViewShape().GetDim(FIRST_DIM);
    int64_t kX2 = x2->GetViewShape().GetDim(SECOND_DIM);
    int64_t mY = y->GetViewShape().GetDim(FIRST_DIM);
    int64_t nY = y->GetViewShape().GetDim(SECOND_DIM);

    if (kX1 != kX2) {
        std::string shapesStr = std::to_string(kX1) + ", " + std::to_string(kX2);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(kOpName, "x1, x2", shapesStr.c_str(),
                                               "The shape sizes of x1.k and x2.k must be equal");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (kX1 < 1 || mX1 < 1 || nX2 < 1) {
        std::string valuesStr = std::to_string(mX1) + ", " + std::to_string(kX1) + ", " + std::to_string(nX2);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(kOpName, "m, k, n", valuesStr.c_str(),
                                               "The values of m, k, n must be >= 1");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (mY != mX1) {
        std::string shapesStr = std::to_string(mY) + ", " + std::to_string(mX1);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(kOpName, "y, x1", shapesStr.c_str(),
                                               "The shape sizes of y.m and x1.m must be equal");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (nY != nX2) {
        std::string shapesStr = std::to_string(nY) + ", " + std::to_string(nX2);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(kOpName, "y, x2", shapesStr.c_str(),
                                               "The shape sizes of y.n and x2.n must be equal");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (bias == nullptr) {
        return ACLNN_SUCCESS;
    }
    // bias shape 支持 (n)
    if (bias->GetViewShape().GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "bias", std::to_string(bias->GetViewShape().GetDimNum()).c_str(), "1D");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (bias->GetViewShape().GetDim(0) != nX2) {
        std::string expectedShapeStr = std::to_string(nX2);
        OP_LOGE_FOR_INVALID_SHAPE(kOpName, "bias", op::ToString(bias->GetViewShape()).GetString(),
                                  expectedShapeStr.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX2Ref(const aclTensor* x2Ref)
{
    if (!CheckType(x2Ref->GetDataType(), X1_X2_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "x2Ref", op::ToString(x2Ref->GetDataType()).GetString(),
                                              std::string("The dtype of x2Ref must be within the range ") +
                                                  op::ToString(X1_X2_DTYPE_SUPPORT_LIST).GetString());
        return false;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAttrs(const bool transposeX1Attr, const bool transposeX2Attr, const int64_t level0GroupSize,
                              const int64_t level1GroupSize)
{
    OP_CHECK(!transposeX1Attr,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "transposeX1", "true",
                                                   "The value of transposeX1 can not be true"),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        transposeX2Attr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "transposeX2", "false", "The value of transposeX2 must be true"),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(level0GroupSize == 512,
             OP_LOGE_FOR_INVALID_VALUE(kOpName, "level0GroupSize", std::to_string(level0GroupSize).c_str(), "512"),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(level1GroupSize == 32,
             OP_LOGE_FOR_INVALID_VALUE(kOpName, "level1GroupSize", std::to_string(level1GroupSize).c_str(), "32"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static inline bool CreateContiguous(const aclTensor*& contiguousTensor, aclOpExecutor* executor)
{
    // 根据输入Tensor的ViewShape，创建一个连续的Tensor
    if (contiguousTensor == nullptr) {
        return true;
    }
    contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

aclnnStatus EnableContiguous(const aclTensor*& x1, const aclTensor*& x1Level0Scale, const aclTensor*& x2Level0Scale,
                             const aclTensor*& x1Level1Scale, const aclTensor*& x2Level1Scale, const aclTensor*& bias,
                             aclOpExecutor* executor)
{
    CHECK_RET(CreateContiguous(x1, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(x1Level0Scale, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(x2Level0Scale, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(x1Level1Scale, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(x2Level1Scale, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(bias, executor), ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnDualLevelQuantMatmulGetWorkspaceSizeProcess(TupleRequiredTensor mandatoryTensors,
                                                             TupleOptionalTensor optionalTensors, TupleAttr attrs,
                                                             aclOpExecutor* executor)
{
    auto& x1 = std::get<INDEX_X1_IN_MANDATORY_TUPLE>(mandatoryTensors);
    auto& x2 = std::get<INDEX_X2_IN_MANDATORY_TUPLE>(mandatoryTensors);
    auto& x2Ref = std::get<INDEX_X2_REF_IN_MANDATORY_TUPLE>(mandatoryTensors);
    auto& x1Level0Scale = std::get<INDEX_X1_L0_SCALE_IN_MANDATORY_TUPLE>(mandatoryTensors);
    auto& x1Level1Scale = std::get<INDEX_X1_L1_SCALE_IN_MANDATORY_TUPLE>(mandatoryTensors);
    auto& x2Level0Scale = std::get<INDEX_X2_L0_SCALE_IN_MANDATORY_TUPLE>(mandatoryTensors);
    auto& x2Level1Scale = std::get<INDEX_X2_L1_SCALE_IN_MANDATORY_TUPLE>(mandatoryTensors);
    auto& y = std::get<INDEX_Y_IN_MANDATORY_TUPLE>(mandatoryTensors);
    auto& bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& transposeX1Attr = std::get<INDEX_TRANSPOSE_X1_IN_ATTR_TUPLE>(attrs);
    auto& transposeX2Attr = std::get<INDEX_TRANSPOSE_X2_IN_ATTR_TUPLE>(attrs);
    auto& level0GroupSize = std::get<INDEX_L0_GROUP_SIZE_IN_ATTR_TUPLE>(attrs);
    auto& level1GroupSize = std::get<INDEX_L1_GROUP_SIZE_IN_ATTR_TUPLE>(attrs);

    aclnnStatus res = CheckTensorX1(x1);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    res = CheckTensorX2(x2);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    res = CheckTensorX1L0Scale(x1Level0Scale);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    res = CheckTensorX1L1Scale(x1Level1Scale);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    res = CheckTensorX2L0Scale(x2Level0Scale);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    res = CheckTensorX2L1Scale(x2Level1Scale);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    res = CheckTensorY(y);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    res = CheckTensorBias(bias);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    res = InputTensorProcess(mandatoryTensors, executor);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    res = CheckTensorX2Ref(x2Ref);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    res = CheckInputOutputShape(x1, x2Ref, bias, y);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    res = CheckAttrs(transposeX1Attr, transposeX2Attr, level0GroupSize, level1GroupSize);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    res = EnableContiguous(x1, x1Level0Scale, x2Level0Scale, x1Level1Scale, x2Level1Scale, bias, executor);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Level0Scale, const aclTensor* x2Level0Scale,
    const aclTensor* x1Level1Scale, const aclTensor* x2Level1Scale, const aclTensor* optionalBias, bool transposeX1,
    bool transposeX2, int64_t level0GroupSize, int64_t level1GroupSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDualLevelQuantMatmulWeightNz,
                   DFX_IN(x1, x2, x1Level0Scale, x2Level0Scale, x1Level1Scale, x2Level1Scale, optionalBias),
                   DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclnnStatus socRes = CheckSocValid();
    CHECK_RET(socRes == ACLNN_SUCCESS, socRes);

    CHECK_RET(CheckNotNull(x1, x2, x1Level0Scale, x2Level0Scale, x1Level1Scale, x2Level1Scale, out),
              ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(IsNzFormat(x2),
             OP_LOGE_FOR_INVALID_FORMAT(kOpName, "x2", op::ToString(x2->GetStorageFormat()).GetString(),
                                        "FORMAT_FRACTAL_NZ"),
             return ACLNN_ERR_PARAM_INVALID);

    const aclTensor* x2Ref = x2;
    aclnnStatus processRes = AclnnDualLevelQuantMatmulGetWorkspaceSizeProcess(
        std::tie(x1, x2, x2Ref, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, out),
        std::tie(optionalBias), std::tie(transposeX1, transposeX2, level0GroupSize, level1GroupSize),
        uniqueExecutor.get());
    CHECK_RET(processRes == ACLNN_SUCCESS, processRes);

    int64_t dtype = static_cast<int64_t>(out->GetDataType());
    auto l0Result = l0op::DualLevelQuantBatchMatmul(x1, x2Ref, x1Level0Scale, x2Level0Scale, x1Level1Scale,
                                                    x2Level1Scale, optionalBias, dtype, transposeX1, transposeX2,
                                                    level0GroupSize, level1GroupSize, uniqueExecutor.get());
    CHECK_RET(l0Result != nullptr, ACLNN_ERR_PARAM_INVALID);

    auto viewCopyResult = l0op::ViewCopy(l0Result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_INVALID);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDualLevelQuantMatmulWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDualLevelQuantMatmulWeightNz)
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif