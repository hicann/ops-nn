/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_quant_batch_matmul_inplace_add.h"
#include "aclnn_quant_batch_matmul_inplace_add_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include <dlfcn.h>
#include "securec.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "quant_batch_matmul_inplace_add.h"
#include "../../quant_batch_matmul_v4/op_host/op_api/quant_matmul_common_check.h"
#include "util/math_util.h"

using namespace op;
using namespace QBMMInplaceAdd;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::NN::SwapLastTwoDimValue;

namespace {
static aclnnStatus CheckNotNull(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    OP_CHECK_NULL(params.x1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x2Scale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.yRef, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x1ScaleOptional, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    CHECK_COND(
        params.x1->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
        "Format of x1 should be ND, current format is %s, this format is not supported.",
        op::ToString(params.x1->GetStorageFormat()).GetString());
    CHECK_COND(
        params.x2->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
        "Format of x2 should be ND, current format is %s, this format is not supported.",
        op::ToString(params.x2->GetStorageFormat()).GetString());
    CHECK_COND(
        params.x2Scale->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
        "Format of x2Scale should be ND, current format is %s, this format is not supported.",
        op::ToString(params.x2Scale->GetStorageFormat()).GetString());
    CHECK_COND(
        params.yRef->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
        "Format of input y should be ND, current format is  %s, this format is not supported.",
        op::ToString(params.yRef->GetStorageFormat()).GetString());
    CHECK_COND(
        params.x1ScaleOptional->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
        "Format of x1_scale should be ND, current format is  %s, this format is not supported.",
        op::ToString(params.x1ScaleOptional->GetStorageFormat()).GetString());
    return ACLNN_SUCCESS;
}

static aclnnStatus IsMxQuantDim(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    auto x1ScaleDimNum = params.x1ScaleOptional->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = params.x2Scale->GetViewShape().GetDimNum();
    CHECK_COND(
        x2ScaleDimNum == MX_X2_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
        "In Mx Quant, the dimension of x2 should be equal to 3, but actual is %zu.", x2ScaleDimNum);
    CHECK_COND(
        x1ScaleDimNum == MX_X1_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
        "In Mx Quant, the dimension of x1 should be equal to 3, but actual is %zu.", x1ScaleDimNum);
    
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInputOutDims(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    auto x1DimNum = params.x1->GetViewShape().GetDimNum();
    auto x2DimNum = params.x2->GetViewShape().GetDimNum();
    auto yInputDimNum = params.yRef->GetViewShape().GetDimNum();
    CHECK_COND(
        x1DimNum == MX_X1_DIM, ACLNN_ERR_PARAM_INVALID,
        "The dimension of x1 should be equal to 2, but current dim is %zu.", x1DimNum);
    CHECK_COND(
        x2DimNum == MX_X2_DIM, ACLNN_ERR_PARAM_INVALID,
        "The dimension of x2 should be equal to 2, but current dim is %zu.", x2DimNum);
    CHECK_COND(
        yInputDimNum == Y_INPUT_DIM, ACLNN_ERR_PARAM_INVALID,
        "The dimension of yRef should be equal 2, but actual is %zu.", yInputDimNum);

    return ACLNN_SUCCESS;
}

static bool IsSpecialTranspose(const aclTensor* const inputTensor)
{
    const auto inputShape = inputTensor->GetViewShape();
    int64_t dim1 = inputShape.GetDimNum() - LAST_FIRST_DIM_INDEX;
    int64_t dim2 = inputShape.GetDimNum() - LAST_SECOND_DIM_INDEX;
    return inputShape.GetDim(dim1) == 1 && inputShape.GetDim(dim2) == 1;
}

static inline bool IsMicroScaling(const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    if (x1Scale == nullptr) {
        return false;
    }
    return x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
           x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static bool CheckMKN(int64_t m, int64_t k, int64_t n)
{
    if (k == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of k should not be equal to zero.");
        return false;
    }

    CHECK_COND(m > 0, ACLNN_ERR_PARAM_INVALID, "The M value[%ld] in x1 should be positive.", m);
    CHECK_COND(k > 0, ACLNN_ERR_PARAM_INVALID, "The K value[%ld] in x1 and x2 should be positive.", k);
    CHECK_COND(n > 0, ACLNN_ERR_PARAM_INVALID, "The N value[%ld] in x2 should be positive.", n);
    return true;
}

static bool CheckGroupSize(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    auto groupSize = params.groupSize;
    uint64_t groupSizeM = (static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
    if (IsMicroScaling(params.x1ScaleOptional, params.x2Scale)) {
        if (groupSizeK != static_cast<uint64_t>(PERGROUP_GROUP_SIZE) || groupSizeM != 1UL || groupSizeN != 1UL) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Unsupported groupSize. In mx quantification, input or infered groupSize should be \
4295032864(for torch api, group_sizes should be [1, 1, 32]). Actual groupSize: %ld(for torch api \
group_sizes is [%lu, %lu, %lu]).",
                groupSize, groupSizeM, groupSizeN, groupSizeK);
            return false;
        }
    } else if (groupSize != 0L) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Unsupported groupSize. When quantification mode is not mx quantification, \
groupSize should be 0(torch api group_sizes should be [0, 0, 0] or None). \
Actual groupSize: %ld(torch api group_sizes is [%lu, %lu, %lu]).",
            groupSize, groupSizeM, groupSizeN, groupSizeK);
        return false;
    }
    OP_LOGD("QuantBatchMatmulInplaceAdd check group_size success.");
    return true;
}

static aclnnStatus CheckShape(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    CHECK_COND(CheckInputOutDims(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Check CheckInputOutDims failed.");

    int64_t mDim = params.x1->GetViewShape().GetDim(1);  // 从x1的第1维获取m
    int64_t x1KDim = params.x1->GetViewShape().GetDim(0);  // 从x1的第0维获取k
    int64_t nDim = params.x2->GetViewShape().GetDim(1);  // 从x2的第1维获取n
    int64_t x2KDim = params.x2->GetViewShape().GetDim(0); // 从x2的第0维获取k
    int64_t yRefMDim = params.yRef->GetViewShape().GetDim(0); // 从y的第0维获取m
    int64_t yRefNDim = params.yRef->GetViewShape().GetDim(1); // 从y的第1维获取n
    CHECK_COND(
        x1KDim == x2KDim, ACLNN_ERR_PARAM_INVALID, "The k value of x1/x2 should be equal, but the actual is %ld/%ld.",
        x1KDim, x2KDim);
    CHECK_COND(
        mDim == yRefMDim, ACLNN_ERR_PARAM_INVALID, "The m value of x1/yRef should be equal, but the actual is %ld/%ld.",
        mDim, yRefMDim);
    CHECK_COND(
        nDim == yRefNDim, ACLNN_ERR_PARAM_INVALID, "The n value of x2/yRef should be equal, but the actual is %ld/%ld.",
        nDim, yRefNDim);

    if (!CheckMKN(mDim, x1KDim, nDim)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckMKN failed.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto scale1LastDimValue = params.x1ScaleOptional->GetViewShape().GetDim(MX_X1_SCALE_DIM - 1);
    auto scale2LastDimValue = params.x2Scale->GetViewShape().GetDim(MX_X2_SCALE_DIM - 1);
    CHECK_COND(
        scale1LastDimValue == 2, ACLNN_ERR_PARAM_INVALID, // last dim should be 2 in mx quant mode
        "The last dim of x1 scale should be 2 in mx quant mode, but actual is %ld.", scale1LastDimValue);
    CHECK_COND(
        scale2LastDimValue == 2, ACLNN_ERR_PARAM_INVALID, // last dim should be 2 in mx quant mode
        "The last dim of x2 scale should be 2 in mx quant mode, but actual is %ld.", scale2LastDimValue);

    // x1的shape期望为[K, M]
    op::Shape x1ExpectShape = {x1KDim, mDim};
    // x1Scale的shape期望为[CeilDiv(K, 64), M, 2]
    op::Shape x1ScaleExpectShape = {Ops::Base::CeilDiv(x1KDim, SPLIT_SIZE), mDim, 2};
    // x2的shape期望为[K, N]
    op::Shape x2ExpectShape = {x1KDim, nDim};
    // x2Scale的shape期望为[CeilDiv(K, 64), N, 2]
    op::Shape x2ScaleExpectShape = {Ops::Base::CeilDiv(x1KDim, SPLIT_SIZE), nDim, 2};
    // y_input的shape期望为[M, N]
    op::Shape yInputExpectShape = {yRefMDim, yRefNDim};
    // y_output的shape期望为[M, N]
    op::Shape yOutputExpectShape = {mDim, nDim};

    const aclTensor* x1 = params.x1;
    const aclTensor* x1Scale = params.x1ScaleOptional;
    const aclTensor* x2 = params.x2;
    const aclTensor* x2Scale = params.x2Scale;
    const aclTensor* yInput = params.yRef;
    const aclTensor* yOutput = params.yRef;

    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(x1, x1ExpectShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(x1Scale, x1ScaleExpectShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(x2, x2ExpectShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(x2Scale, x2ScaleExpectShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(yInput, yInputExpectShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(yOutput, yOutputExpectShape, return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static inline bool MxScaleContiguousProcess(const aclTensor*& mxScaleTensor, aclOpExecutor* executor)
{   
    if (mxScaleTensor == nullptr || mxScaleTensor->GetViewShape().GetDimNum() < MX_SCALE_MAX_DIM) {
        OP_LOGD("MX scale no need to do contiguous process.");
        return true;
    }
    auto transposeFlag = false;
    int64_t dimNum = mxScaleTensor->GetViewShape().GetDimNum();
    int64_t lastDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 1);
    int64_t lastSecondDim = mxScaleTensor->GetViewShape().GetDim(dimNum - PENULTIMATE_DIM);
    int64_t lastThirdDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 3);    // 3: 倒数第3维
    if (mxScaleTensor->GetViewStrides()[dimNum - 3] == lastDim &&   // 3： 倒数第3维
        mxScaleTensor->GetViewStrides()[dimNum - PENULTIMATE_DIM] == lastDim * lastThirdDim) {
        int64_t tmpNxD = lastDim * lastSecondDim * lastThirdDim;
        transposeFlag = true;
        // 4：batch维度从倒数第4维起
        for (int64_t batchDim = dimNum - 4; batchDim >= 0; batchDim--) {
            if (mxScaleTensor->GetViewStrides()[batchDim] != tmpNxD) {
                transposeFlag = false;
                break;
            }
            tmpNxD *= mxScaleTensor->GetViewShape().GetDim(batchDim);
        }
        if (lastSecondDim == 1 && lastThirdDim == 1) {
            transposeFlag = false;
        }
    }
    if (transposeFlag) {
        op::Shape swapedShape = mxScaleTensor->GetViewShape();
        swapedShape.SetDim(dimNum - PENULTIMATE_DIM, lastThirdDim);
        swapedShape.SetDim(dimNum - 3, lastSecondDim);  // 3： 倒数第3维
        mxScaleTensor = executor->CreateView(mxScaleTensor, swapedShape, mxScaleTensor->GetViewOffset());
    } else {
        mxScaleTensor = l0op::Contiguous(mxScaleTensor, executor);
    }
    CHECK_RET(mxScaleTensor != nullptr, false);
    return true;
}

static aclnnStatus CheckMxfp8DtypeValid(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.x1, x1_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.x2, x2_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.x1ScaleOptional, X1_SCALE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.x2Scale, X2_SCALE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.yRef, YREF_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("QuantBatchMatmulInplaceAdd CheckMxfp8DtypeVaild success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    auto x1Dtype = params.x1->GetDataType();
    auto x2Dtype = params.x2->GetDataType();
    auto x1ScaleDtype = params.x1ScaleOptional->GetDataType();
    auto x2ScaleDtype = params.x2Scale->GetDataType();
    if ((x1Dtype == DataType::DT_FLOAT8_E4M3FN || x1Dtype == DataType::DT_FLOAT8_E5M2) &&
        (x2Dtype == DataType::DT_FLOAT8_E4M3FN || x2Dtype == DataType::DT_FLOAT8_E5M2)) {
        CHECK_COND(IsMxQuantDim(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Check IsMxQuantDim failed.");
        return CheckMxfp8DtypeValid(params);
    } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the dtypes of x1 and x2 are %s and %s, \
and the dtypes of x1Scale and x2Scale are %s and %s is not supported.",
                    op::ToString(x1Dtype).GetString(), op::ToString(x2Dtype).GetString(),
                    op::ToString(x1ScaleDtype).GetString(), op::ToString(x2ScaleDtype).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("QuantBatchMatmulInplaceAdd check params success.");

    return ACLNN_SUCCESS;
}

bool ReCalcGroupSize(uint64_t inputSize, uint64_t scaleSize, uint64_t& groupSize, const char* dimensionName)
{
    if (scaleSize == 0UL) {
        std::string scaleName = strcmp(dimensionName, "n") == 0 ? "x2Scale(scale)" : "x1Scale(pertokenScale)";
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The %s dimension of %s is 0, invalid shape!", dimensionName, scaleName.c_str());
        return false;
    }
    if (groupSize == 0L) {
        if (inputSize % scaleSize != 0UL) {
            std::string scaleName = "x1Scale(pertokenScale)";
            std::string inputName = "x1";
            if (strcmp(dimensionName, "n") == 0) {
                scaleName = "x2Scale(scale)";
                inputName = "x2";
            }
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "The groupSize in %s dimension is 0 and the %s dimension of %s [%lu] is not divisible by \
the %s dimension of %s [%lu], the real groupSize in %s dimension can not be inferred.",
                dimensionName, dimensionName, inputName.c_str(), inputSize, dimensionName, scaleName.c_str(), scaleSize,
                dimensionName);
            return false;
        }
        groupSize = inputSize / scaleSize;
    }
    return true;
}

static inline bool InferGroupSize(QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    auto x1 = params.x1;
    auto x2 = params.x2;
    auto x1Scale = params.x1ScaleOptional;
    auto x2Scale = params.x2Scale;
    if (x1Scale == nullptr || x1Scale->GetViewShape().GetDimNum() < 2 || x2Scale->GetViewShape().GetDimNum() < 2) {
        return true;
    }
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    auto x1ScaleDimNum = x1Scale->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = x2Scale->GetViewShape().GetDimNum();
    auto transX1 = params.transposeX1;
    auto transX2 = params.transposeX2;
    uint64_t groupSizeK = static_cast<uint64_t>(params.groupSize) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(params.groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(params.groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    auto inputSizeM =
        transX1 ? x1->GetViewShape().GetDim(x1DimNum - 1) : x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM);
    auto scaleSizeM = 0;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        scaleSizeM = x1Scale->GetViewShape().GetDim(transX1 ? 1 : 0);
    } else {
        scaleSizeM = transX1 ? x1Scale->GetViewShape().GetDim(x1ScaleDimNum - 1) :
                               x1Scale->GetViewShape().GetDim(x1ScaleDimNum - PENULTIMATE_DIM);
    }
    CHECK_RET(ReCalcGroupSize(inputSizeM, scaleSizeM, groupSizeM, "m"), false);
    auto inputSizeK =
        transX1 ? x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM) : x1->GetViewShape().GetDim(x1DimNum - 1);
    auto scaleSizeK = 0;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        scaleSizeK = x1Scale->GetViewShape().GetDim(transX1 ? 0 : 1);
    } else {
        scaleSizeK = transX1 ? x1Scale->GetViewShape().GetDim(x1ScaleDimNum - 1) :
                               x1Scale->GetViewShape().GetDim(x1ScaleDimNum - PENULTIMATE_DIM);
    }
    CHECK_RET(ReCalcGroupSize(inputSizeK, scaleSizeK, groupSizeK, "k"), false);
    auto inputSizeN =
        transX2 ? x2->GetViewShape().GetDim(x2DimNum - PENULTIMATE_DIM) : x2->GetViewShape().GetDim(x2DimNum - 1);
    auto scaleSizeN = 0;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        scaleSizeN = x2Scale->GetViewShape().GetDim(transX2 ? 0 : 1);
    } else {
        scaleSizeN = transX2 ? x2Scale->GetViewShape().GetDim(x2ScaleDimNum - PENULTIMATE_DIM) :
                               x2Scale->GetViewShape().GetDim(x2ScaleDimNum - 1);
    }
    CHECK_RET(ReCalcGroupSize(inputSizeN, scaleSizeN, groupSizeN, "n"), false);
    OP_LOGD(
        "Infered groupSize: groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.", groupSizeM, groupSizeN, groupSizeK);
    params.groupSize =
        static_cast<int64_t>((groupSizeM << GROUP_M_OFFSET) | (groupSizeN << GROUP_N_OFFSET) | groupSizeK);
    return true;
}

static bool QBMMIAGetTransposeAttrValue(const aclTensor* tensor, bool transpose, bool isSpecialCase = true)
{
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - PENULTIMATE_DIM;
    // 对于torch的场景，NZ情况两维某一维度为1的场景无法正确判断是否转置，资料呈现不支持非连续，代码默认连续
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())) == op::Format::FORMAT_FRACTAL_NZ &&
        (tensor->GetViewShape().GetDim(dim2) == 1 || tensor->GetViewShape().GetDim(dim1) == 1)) {
        return transpose;
    }
    // check if tensor is contiguous layout
    if (tensor->GetViewStrides()[dim2] == 1 &&
        (tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2))) {
        OP_LOGD("QuantBatchMatmulInplaceAdd GetTransposeAttrValue, find tensor is not contiguous.");
        const_cast<aclTensor*>(tensor)->SetViewShape(SwapLastTwoDimValue(tensor->GetViewShape()));
        // 如果不需要校验特殊case，则直接返回
        if (!isSpecialCase) {
            return !transpose;
        }
        if (!CheckSpecialCase(tensor, dim1, dim2)) {
            return !transpose;
        }
    }
    return transpose;
}

static aclnnStatus aclnnQuantBatchMatmulInplaceAddGetWorkspaceSizeCommon(
    QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params, uint64_t* workspaceSize, aclOpExecutor* executor)
{
    // torch非连续转连续处理
    params.x2Scale = SetTensorToNDFormat(params.x2Scale);
    params.x1ScaleOptional = SetTensorToNDFormat(params.x1ScaleOptional);
    TensorContiguousProcess(params.x1, params.transposeX1, executor);
    TensorContiguousProcess(params.x2, params.transposeX2, executor);
    MxScaleContiguousProcess(params.x1ScaleOptional, executor);
    MxScaleContiguousProcess(params.x2Scale, executor);
    // 固定写法，参数检查
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(InferGroupSize(params), ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("Infer groupSize success. groupSize: %ld.", params.groupSize);
    CHECK_COND(CheckGroupSize(params), ACLNN_ERR_PARAM_INVALID, "CheckGroupSize failed.");
    bool transposeX1 = QBMMIAGetTransposeAttrValue(params.x1, params.transposeX1, true);
    bool transposeX2 = QBMMIAGetTransposeAttrValue(params.x2, params.transposeX2, true);
    CHECK_COND(
        transposeX1 == true && transposeX2 == false, ACLNN_ERR_PARAM_INVALID,
        "Only support when the transposition of x1 is true and transposition of x2 is false, but actually is %s and %s.",
        transposeX1 ? "true" : "false", transposeX2 ? "true" : "false");
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        auto x1DimNum = params.x1->GetViewShape().GetDimNum();
        auto inputSizeM = transposeX1 ? params.x1->GetViewShape().GetDim(x1DimNum - 1) :
                                        params.x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM);
        auto x2DimNum = params.x2->GetViewShape().GetDimNum();
        auto inputSizeN = transposeX2 ? params.x2->GetViewShape().GetDim(x2DimNum - PENULTIMATE_DIM) :
                                        params.x2->GetViewShape().GetDim(x2DimNum - 1);
        if (static_cast<ge::Format>(ge::GetPrimaryFormat(params.x2->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
            if (inputSizeM == 0) {
                OP_LOGD("aclnnQuantBatchMatmulInplaceAdd nz m=0");
                return ACLNN_SUCCESS;
            }
        } else {
            if (inputSizeM == 0 || inputSizeN == 0) {
                OP_LOGD("aclnnQuantBatchMatmulInplaceAdd nd m/n=0");
                return ACLNN_SUCCESS;
            }
        }
    }
    
    // Invoke l0 operator QuantBatchMatmulInplaceAdd for calculation.
    auto result = l0op::QuantBatchMatmulInplaceAdd(
        params.x1, params.x2, params.x2Scale, params.yRef, params.x1ScaleOptional, transposeX1, transposeX2, params.groupSize,
        executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(result, params.yRef, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1ScaleOptional, const aclTensor* x2Scale, aclTensor* yRef,
    bool transposeX1, bool transposeX2, int64_t groupSize, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams params{x1,   x2,          x1ScaleOptional, x2Scale,
                                                            yRef, transposeX1, transposeX2,     groupSize};

    L2_DFX_PHASE_1(
        aclnnQuantBatchMatmulInplaceAdd, DFX_IN(x1, x2, x1ScaleOptional, x2Scale, yRef, transposeX1, transposeX2, groupSize),
        DFX_OUT(yRef));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = aclnnQuantBatchMatmulInplaceAddGetWorkspaceSizeCommon(params, workspaceSize, executorPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantBatchMatmulInplaceAdd(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantBatchMatmulInplaceAdd);
    CHECK_COND(
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
        "This is an error in QuantBatchMatmulInplaceAdd launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif