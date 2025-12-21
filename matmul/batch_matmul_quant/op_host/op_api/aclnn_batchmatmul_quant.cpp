/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_batchmatmul_quant.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/dot.h"
#include "matmul/mat_mul_v3/op_host/op_api/matmul.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "level0/unsqueeze.h"
#include "batchmatmulquant.h"

using namespace std;
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
static const std::initializer_list<op::DataType> x1_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> x2_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> QUANTPARAM_DTYPE_SUPPORT_LIST = {DataType::DT_UINT64};
static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};

inline static FVector<int64_t> GetShape(const aclTensor* tensor)
{
    FVector<int64_t> shape;
    if (tensor->GetViewShape().GetDimNum() == 0) {
        shape.push_back(1);
    } else {
        size_t dimNum = tensor->GetViewShape().GetDimNum();
        for (size_t idx = 0; idx < dimNum; idx++) {
            int64_t tmpVal = tensor->GetViewShape().GetDim(idx);
            shape.push_back(tmpVal);
        }
    }
    return shape;
}

inline static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* quantParam, const aclTensor* out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(quantParam, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* quantParam, const aclTensor* bias, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, x1_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, x2_SUPPORT_LIST, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(quantParam, QUANTPARAM_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

inline static op::Shape GetBroadcastShape(const aclTensor* tensor)
{
    op::Shape shape;
    size_t dimNum = tensor->GetViewShape().GetDimNum();
    size_t loopDims = dimNum - 2;
    for (size_t idx = 0; idx < loopDims; idx++) {
        int64_t tmpVal = tensor->GetViewShape().GetDim(idx);
        shape.AppendDim(tmpVal);
    }
    if (shape.GetDimNum() == 0) {
        shape.AppendDim(1);
    }
    return shape;
}

static bool CheckShapeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* quantParam, const aclTensor* bias, const aclTensor* out,
    const bool adjX1, const bool adjX2)
{
    op::Shape x1Shape = x1->GetViewShape();
    op::Shape x2Shape = x2->GetViewShape();
    auto dimTensor1 = x1Shape.GetDimNum();
    auto dimTensor2 = x2Shape.GetDimNum();
    OP_CHECK_MIN_DIM(x1, 2, return false); // min dim num: 2
    OP_CHECK_MIN_DIM(x2, 2, return false); // min dim num: 2
    int64_t x1KDim = adjX1 ? x1->GetViewShape().GetDim(dimTensor1 - 2) : x1->GetViewShape().GetDim(dimTensor1 - 1);
    int64_t x2KDim = adjX2 ? x2->GetViewShape().GetDim(dimTensor2 - 1) : x2->GetViewShape().GetDim(dimTensor2 - 2);
    int64_t M = adjX1 ? x1->GetViewShape().GetDim(dimTensor2 - 1) : x1->GetViewShape().GetDim(dimTensor2 - 2);
    int64_t N = adjX2 ? x2->GetViewShape().GetDim(dimTensor2 - 2) : x2->GetViewShape().GetDim(dimTensor2 - 1);
    size_t maxDim = 6;
    int64_t nAlign16 = (N + 16 - 1) / 16 * 16;

    if ((dimTensor1 != dimTensor2) || (dimTensor1 > maxDim)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The dims of the two inputs should be same and be less than 7, now they are %s, %s",
            op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
        return false;
    }

    auto dimTensorquantParam = quantParam->GetViewShape().GetDimNum();
    int64_t quantParamDim0 = quantParam->GetViewShape().GetDim(0);
    if ((dimTensorquantParam != 1) || (quantParamDim0 != 1 && quantParamDim0 != nAlign16)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "quantParam[%ld] != 1 or quantParam dim == 1, but quantParamDim0[%ld] not equal to 1 or n "
            "align to multiple of 16",
            dimTensorquantParam, quantParamDim0);
        return false;
    }

    if (bias != nullptr) {
        auto dimTensorBias = bias->GetViewShape().GetDimNum();
        int64_t biasDim0 = bias->GetViewShape().GetDim(0);
        if ((dimTensorBias != 1) || (dimTensorBias == 1 && biasDim0 != N)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "dimTensorBias[%ld] != 1 or Bias dim == 1, but biasDim0[%ld] not equal to n",
                dimTensorBias, biasDim0);
            return false;
        }
    }

    if (x1KDim != x2KDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different %s, %s",
            op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
        return false;
    }
    op::Shape outShape = out->GetViewShape();
    OP_CHECK(
        (M == outShape.GetDim(dimTensor1 - 2) && N == out->GetViewShape().GetDim(dimTensor1 - 1)),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The last two dim of out shape %s need equal [%ld, %ld]",
            op::ToString(outShape).GetString(), M, N),
        return false);

    return true;
}

inline static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* quantParam, const aclTensor* bias, const aclTensor* out,
    const bool adjX1, const bool adjX2)
{
    CHECK_RET(CheckNotNull(x1, x2, quantParam, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x1, x2, quantParam, bias, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(x1, x2, quantParam, bias, out, adjX1, adjX2), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus PreInputProcess(const aclTensor* inputTensor, aclOpExecutor* executor)
{
    if (inputTensor != nullptr) {
        inputTensor = l0op::Contiguous(inputTensor, executor);
        OP_CHECK(
            inputTensor != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input perprocess failed, contiguouse return nullptr."),
            return ACLNN_ERR_INNER_NULLPTR);
        inputTensor = l0op::ReFormat(inputTensor, op::Format::FORMAT_ND);
        OP_CHECK(
            inputTensor != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "THe input perprocess failed, reformat return nullptr."),
            return ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnBatchMatmulQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* quantParam, const aclTensor* bias, bool transposeX1,
    bool transposeX2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBatchMatmulQuant, DFX_IN(x1, x2, quantParam, bias, transposeX1, transposeX2), DFX_OUT(out));

    // 固定写法, 创建OpExecutor
    auto unique_executor = CREATE_EXECUTOR();
    CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 入参检查
    aclnnStatus ret = CheckParams(x1, x2, quantParam, bias, out, transposeX1, transposeX2);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    const aclTensor* matmulOut;
    // 空tensor 处理
    if (x1->IsEmpty() || x2->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnBatchMatmulQuant do not support empty tensor!");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 连续性转换
    CHECK_RET(PreInputProcess(x1, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(PreInputProcess(x2, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(PreInputProcess(quantParam, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(PreInputProcess(bias, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // 构建matmul计算图
    matmulOut = l0op::BatchMatmulQuant(x1, x2, quantParam, bias, transposeX1, transposeX2, unique_executor.get());

    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (matmulOut->IsEmpty()) {
        *workspaceSize = 0;
        unique_executor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    matmulOut = l0op::Cast(matmulOut, out->GetDataType(), unique_executor.get());
    auto viewCopyResult = l0op::ViewCopy(matmulOut, out, unique_executor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = unique_executor->GetWorkspaceSize();
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchMatmulQuant(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchMatmulQuant);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
