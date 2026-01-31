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
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
using Ops::NN::SwapLastTwoDimValue;
using TupleRequiredTensor = std::tuple<
    const aclTensor*&, const aclTensor*&, const aclTensor*&, const aclTensor*&, const aclTensor*&, const aclTensor*&,
    const aclTensor*&, aclTensor*&>;
using TupleOptionalTensor = std::tuple<const aclTensor*&>;
using TupleAttr = std::tuple<bool&, bool&, int64_t&, int64_t&>;

static constexpr int INDEX_X1_IN_MANDTORY_TUPLE = 0;
static constexpr int INDEX_X2_IN_MANDTORY_TUPLE = 1;
static constexpr int INDEX_X2_REF_IN_MANDTORY_TUPLE = 2;
static constexpr int INDEX_X1_L0_SCALE_IN_MANDTORY_TUPLE = 3;
static constexpr int INDEX_X1_L1_SCALE_IN_MANDTORY_TUPLE = 4;
static constexpr int INDEX_X2_L0_SCALE_IN_MANDTORY_TUPLE = 5;
static constexpr int INDEX_X2_L1_SCALE_IN_MANDTORY_TUPLE = 6;
static constexpr int INDEX_Y_IN_MANDTORY_TUPLE = 7;
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

static const std::initializer_list<DataType> X1_X2_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<DataType> X1_X2_INPUT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<DataType> L0_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<DataType> L1_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<DataType> Y_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::vector<uint64_t> DIM_RANGE_ONLY_TWO_DIM = {TWO_DIM_VALUE, TWO_DIM_VALUE};
static const std::vector<uint64_t> DIM_RANGE_ONLY_THREE_DIM = {THREE_DIM_VALUE, THREE_DIM_VALUE};
static const std::vector<uint64_t> DIM_RANGE_OPTIONAL_INPUT = {ONE_DIM_VALUE, TWO_DIM_VALUE};

static const aclTensor* SetTensorToNDFormat(const aclTensor* input)
{
    const aclTensor* output = nullptr;
    if (input == nullptr) {
        return output;
    }
    if (input->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGD("weightQuantBatchMatmul set tensor to ND format.");
        output = l0op::ReFormat(input, op::Format::FORMAT_ND);
    } else {
        output = input;
    }
    return output;
}

static bool IsFormatSupport(const aclTensor* input, Format format, const std::string& inputName)
{
    if (input != nullptr && input->GetStorageFormat() != format) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "%s's format should be [%s]. actual is [%s].", inputName.c_str(),
            op::ToString(format).GetString(),
            op::ToString(input->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool IsDimSupport(const aclTensor* input, const std::vector<uint64_t>& dimRange, const std::string& inputName)
{
    if (input != nullptr &&
        (input->GetViewShape().GetDimNum() < dimRange[0] || input->GetViewShape().GetDimNum() > dimRange[1])) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "%s's dim should be in range [%lu, %lu]. actual is [%zu].", inputName.c_str(),
            dimRange[0], dimRange[1], input->GetViewShape().GetDimNum());
        return false;
    }
    return true;
}

static aclnnStatus CheckSocValid()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND950:
            break;
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
            return ACLNN_ERR_RUNTIME_ERROR;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus IsNzFormat(const aclTensor* weight)
{
    return weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ ||
           weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ_C0_2;
}

static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Level0Scale, const aclTensor* x2Level0Scale,
    const aclTensor* x1Level1Scale, const aclTensor* x2Level1Scale, const aclTensor* y)
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

bool IsTransLastTwoDims(const aclTensor* tensor)
{
    // 相对于公共仓接口区别于，输入shape仅支持2维，在tensor输入shape为（1, 1）时返回true
    if (tensor->GetViewShape().GetDimNum() != 2) {
        return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        return true;
    }
    return false;
}

static bool SetShapeStrideForNZ(const aclTensor* weight, aclTensor* weightTemp, bool transposeX2)
{
    if (!transposeX2) {
        op::Strides newStrides = weight->GetViewStrides();
        auto strideSize = newStrides.size();
        OP_CHECK(
            strideSize >= DIM_RANGE_ONLY_TWO_DIM.front() && strideSize <= DIM_RANGE_ONLY_TWO_DIM.back(),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dim of weight's strides should be in range [%lu, %lu]. actual is [%lu].",
                DIM_RANGE_ONLY_TWO_DIM.front(), DIM_RANGE_ONLY_TWO_DIM.back(), strideSize),
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

static aclnnStatus ModifyTensorDtype(
    const aclTensor*& tensorRef, aclTensor* tensorMod, DataType dtype, aclOpExecutor* executor)
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

static aclnnStatus InputPreProcess(
    const aclTensor* input, const aclTensor*& inputRef, const char* inputName, aclOpExecutor* executor)
{
    CHECK_RET(IsDimSupport(input, DIM_RANGE_ONLY_TWO_DIM, inputName), ACLNN_ERR_PARAM_INVALID);
    auto viewShape = input->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    bool isTranspose = IsTransLastTwoDims(input);
    if (isTranspose) {
        // 当 isTranspose=true 时，shape 的倒数第 2 维要放大 2 倍， 即 (k/2, n) -> (k, n)
        viewShape[viewShapeDim - 2] = viewShape[viewShapeDim - 2] * FP4_NUMS_IN_INT8;
    } else {
        // 当 isTranspose=false 时，shape 的最后一维要放大 2 倍， 即 (k, n/2) -> (k, n)
        viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * FP4_NUMS_IN_INT8;
    }
    auto inputTemp = executor->CreateView(input, viewShape, input->GetViewOffset());
    CHECK_RET(inputTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (isTranspose) {
        op::Strides newStrides = input->GetViewStrides();
        auto strideSize = newStrides.size();
        OP_CHECK(
            strideSize >= DIM_RANGE_ONLY_TWO_DIM.front() && strideSize <= DIM_RANGE_ONLY_TWO_DIM.back(),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dim of input's strides should be in range [%lu, %lu]. actual is [%lu].",
                DIM_RANGE_ONLY_TWO_DIM.front(), DIM_RANGE_ONLY_TWO_DIM.back(), strideSize),
            return ACLNN_ERR_PARAM_INVALID);
        // 当 isTranspose=true 时，strides 的最后一维要放大 2 倍， 即(1, k/2) -> (1, k)
        newStrides[strideSize - 1] *= FP4_NUMS_IN_INT8;
        inputTemp->SetViewStrides(newStrides);
    }
    if (IsNzFormat(input)) {
        CHECK_RET(SetShapeStrideForNZ(input, inputTemp, isTranspose), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(SetNZC0FormatToNZFormat(inputTemp) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }
    if (input->GetDataType() == DataType::DT_INT8) {
        CHECK_RET(
            ModifyTensorDtype(inputRef, inputTemp, DataType::DT_FLOAT4_E2M1, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID);
        OP_LOGD("The conversion of input from int8 to fp4_e2m1 is completed.");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus InputTensorProcess(TupleRequiredTensor mandatoryTensors, aclOpExecutor* executor)
{
    auto& x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& x2Ref = std::get<INDEX_X2_REF_IN_MANDTORY_TUPLE>(mandatoryTensors);
    // 将 int8 的输入 x2 dtype 修改为 mxfp4, 同时 ViewShape, ViewStrides 也从 int8 修改为 mxfp4 所对应的
    OP_CHECK(
        x2->GetDataType() == DataType::DT_INT8, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x2 are int8."),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(InputPreProcess(x2, x2Ref, "x2", executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX1(const aclTensor* x1, const bool transposeX1Attr)
{
    int64_t kX1 = x1->GetViewShape().GetDim(SECOND_DIM);

    OP_CHECK(
        !transposeX1Attr, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x1 is not transposed."),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        IsContiguous(x1), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x1 is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x1, DIM_RANGE_ONLY_TWO_DIM, "x1"), false);
    CHECK_RET(IsFormatSupport(x1, Format::FORMAT_ND, "x1"), false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X1_X2_DTYPE_SUPPORT_LIST, return false);

    // 当 x1 为非转置时，k 作为内轴必须为偶数
    OP_CHECK(
        (static_cast<uint64_t>(kX1) & 1) == 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "If x1 is not transposed, k[%ld] should be an even number.", kX1),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX2(const aclTensor* x2, const bool transposeX2Attr)
{
    int64_t kX2 = x2->GetViewShape().GetDim(SECOND_DIM);

    OP_CHECK(
        transposeX2Attr, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x2 is transposed."),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        IsContiguous(x2), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x2 is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x2, DIM_RANGE_ONLY_TWO_DIM, "x2"), false);
    CHECK_RET(IsFormatSupport(x2, Format::FORMAT_FRACTAL_NZ, "x2"), false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X1_X2_INPUT_DTYPE_SUPPORT_LIST, return false);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX1L0Scale(const aclTensor* x1Level0Scale)
{
    OP_CHECK(
        IsContiguous(x1Level0Scale), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x1Level0Scale is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x1Level0Scale, DIM_RANGE_ONLY_TWO_DIM, "x1Level0Scale"), false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1Level0Scale, L0_SCALE_DTYPE_SUPPORT_LIST, return false);
    CHECK_RET(IsFormatSupport(x1Level0Scale, Format::FORMAT_ND, "x1Level0Scale"), false);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX1L1Scale(const aclTensor* x1Level1Scale)
{
    OP_CHECK(
        IsContiguous(x1Level1Scale), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x1Level1Scale is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x1Level1Scale, DIM_RANGE_ONLY_THREE_DIM, "x1Level1Scale"), false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1Level1Scale, L1_SCALE_DTYPE_SUPPORT_LIST, return false);
    if (x1Level1Scale->GetStorageFormat() != Format::FORMAT_ND &&
        x1Level1Scale->GetStorageFormat() != Format::FORMAT_NCL) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "x1Level1Scale's format should be ND or NCL. actual is [%s].",
            op::ToString(x1Level1Scale->GetStorageFormat()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX2L0Scale(const aclTensor* x2Level0Scale)
{
    OP_CHECK(
        IsContiguous(x2Level0Scale), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x2Level0Scale is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x2Level0Scale, DIM_RANGE_ONLY_TWO_DIM, "x2Level0Scale"), false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Level0Scale, L0_SCALE_DTYPE_SUPPORT_LIST, return false);
    CHECK_RET(IsFormatSupport(x2Level0Scale, Format::FORMAT_ND, "x2Level0Scale"), false);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX2L1Scale(const aclTensor* x2Level1Scale)
{
    OP_CHECK(
        IsContiguous(x2Level1Scale), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x2Level1Scale is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(x2Level1Scale, DIM_RANGE_ONLY_THREE_DIM, "x2Level1Scale"), false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Level1Scale, L1_SCALE_DTYPE_SUPPORT_LIST, return false);
    if (x2Level1Scale->GetStorageFormat() != Format::FORMAT_ND &&
        x2Level1Scale->GetStorageFormat() != Format::FORMAT_NCL) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "x2Level1Scale's format should be ND or NCL. actual is [%s].",
            op::ToString(x2Level1Scale->GetStorageFormat()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorY(const aclTensor* y)
{
    OP_CHECK(
        IsContiguous(y), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support y is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(y, DIM_RANGE_ONLY_TWO_DIM, "y"), false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, Y_DTYPE_SUPPORT_LIST, return false);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorBias(const aclTensor* bias)
{
    if (bias == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(
        IsContiguous(bias), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support bias is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsDimSupport(bias, DIM_RANGE_OPTIONAL_INPUT, "bias"), false);
    OP_CHECK(
        bias->GetDataType() == DataType::DT_FLOAT,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "bias's dtype should be [DT_FLOAT], actual is [%s].",
            op::ToString(bias->GetDataType()).GetString()),
        return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsFormatSupport(bias, Format::FORMAT_ND, "bias"), false);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInputOutputShape(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* y)
{
    int64_t mX1 = x1->GetViewShape().GetDim(FIRST_DIM);
    int64_t kX1 = x1->GetViewShape().GetDim(SECOND_DIM);
    int64_t nX2 = x2->GetViewShape().GetDim(FIRST_DIM);
    int64_t kX2 = x2->GetViewShape().GetDim(SECOND_DIM);
    int64_t mY = y->GetViewShape().GetDim(FIRST_DIM);
    int64_t nY = y->GetViewShape().GetDim(SECOND_DIM);

    if (kX1 != kX2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "x1's k and x2's k should be equal, actual x1's k is %ld, x2's k is %ld.", kX1,
            kX2);
        return false;
    }

    if (kX1 < 1 || mX1 < 1 || nX2 < 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "m, k, n shouldn't be smaller than %d, actual m is %ld, k is %ld, n is %ld.", 1,
            mX1, kX1, nX2);
        return false;
    }

    if (mY != mX1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "y's m and x1's m should be equal, actual y's m is %ld, x1's m is %ld.", mY, mX1);
        return false;
    }

    if (nY != nX2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "y's n and x2's n should be equal, actual y's n is %ld, x2's n is %ld.", nY, nX2);
        return false;
    }

    if (bias == nullptr) {
        return ACLNN_SUCCESS;
    }
    // bias shape 支持 (n)
    if (bias->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "input [bias] only support 1 dim, actual is %ld.",
            bias->GetViewShape().GetDimNum());
        return false;
    }
    if (bias->GetViewShape().GetDim(0) != nX2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "when bias's shape size is 1, it's shape should be [%ld], actual is %s.", nX2,
            op::ToString(bias->GetViewShape()).GetString());
        return false;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorX2Ref(const aclTensor* x2Ref)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Ref, X1_X2_DTYPE_SUPPORT_LIST, return false);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAttrs(
    const bool transposeX1Attr, const bool transposeX2Attr, const int level0GroupSize, const int level1GroupSize)
{
    OP_CHECK(
        !transposeX1Attr, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x1 is not transposed."),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        transposeX2Attr, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x2 is transposed."),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        level0GroupSize == 512, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "level0GroupSize must be 512."),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        level1GroupSize == 32, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "level1GroupSize must be 32."),
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

aclnnStatus EnableContiguous(
    const aclTensor*& x1, const aclTensor*& x1Level0Scale, const aclTensor*& x2Level0Scale,
    const aclTensor*& x1Level1Scale, const aclTensor*& x2Level1Scale, const aclTensor*& bias, aclOpExecutor* executor)
{
    CHECK_RET(CreateContiguous(x1, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(x1Level0Scale, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(x2Level0Scale, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(x1Level1Scale, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(x2Level1Scale, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(bias, executor), ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnDualLevelQuantMatmulGetWorkspaceSizeProcess(
    TupleRequiredTensor mandatoryTensors, TupleOptionalTensor optionalTensors, TupleAttr attrs, aclOpExecutor* executor)
{
    auto& x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& x2Ref = std::get<INDEX_X2_REF_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& x1Level0Scale = std::get<INDEX_X1_L0_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& x1Level1Scale = std::get<INDEX_X1_L1_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& x2Level0Scale = std::get<INDEX_X2_L0_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& x2Level1Scale = std::get<INDEX_X2_L1_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& y = std::get<INDEX_Y_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& transposeX1Attr = std::get<INDEX_TRANSPOSE_X1_IN_ATTR_TUPLE>(attrs);
    auto& transposeX2Attr = std::get<INDEX_TRANSPOSE_X2_IN_ATTR_TUPLE>(attrs);
    auto& level0GroupSize = std::get<INDEX_L0_GROUP_SIZE_IN_ATTR_TUPLE>(attrs);
    auto& level1GroupSize = std::get<INDEX_L1_GROUP_SIZE_IN_ATTR_TUPLE>(attrs);

    aclnnStatus res = CheckTensorX1(x1, transposeX1Attr);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    res = CheckTensorX2(x2, transposeX2Attr);
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
    L2_DFX_PHASE_1(
        aclnnDualLevelQuantMatmulWeightNz,
        DFX_IN(x1, x2, x1Level0Scale, x2Level0Scale, x1Level1Scale, x2Level1Scale, optionalBias), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclnnStatus socRes = CheckSocValid();
    CHECK_RET(socRes == ACLNN_SUCCESS, socRes);

    CHECK_RET(
        CheckNotNull(x1, x2, x1Level0Scale, x2Level0Scale, x1Level1Scale, x2Level1Scale, out), ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(
        IsNzFormat(x2), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support weight tensor is FRACTAL_NZ."),
        return ACLNN_ERR_PARAM_INVALID);

    const aclTensor* x2Ref = x2;
    aclnnStatus processRes = AclnnDualLevelQuantMatmulGetWorkspaceSizeProcess(
        std::tie(x1, x2, x2Ref, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, out),
        std::tie(optionalBias), std::tie(transposeX1, transposeX2, level0GroupSize, level1GroupSize),
        uniqueExecutor.get());
    CHECK_RET(processRes == ACLNN_SUCCESS, processRes);

    int64_t dtype = static_cast<int64_t>(out->GetDataType());
    auto l0Result = l0op::DualLevelQuantBatchMatmul(
        x1, x2Ref, x1Level0Scale, x2Level0Scale, x1Level1Scale, x2Level1Scale, optionalBias, dtype, transposeX1,
        transposeX2, level0GroupSize, level1GroupSize, uniqueExecutor.get());
    CHECK_RET(l0Result != nullptr, ACLNN_ERR_PARAM_INVALID);

    auto viewCopyResult = l0op::ViewCopy(l0Result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_INVALID);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDualLevelQuantMatmulWeightNz(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDualLevelQuantMatmulWeightNz)
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif