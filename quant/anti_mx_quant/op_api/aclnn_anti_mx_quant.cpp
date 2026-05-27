/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_anti_mx_quant.h"
#include "anti_mx_quant.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_util.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

static constexpr size_t MIN_DIM_LEN_MXSCALE = 2;
static constexpr size_t MAX_DIM_LEN = 7;
static constexpr size_t MAX_DIM_LEN_MXSCALE= 8;
static constexpr int64_t FP4_NUMS_IN_UINT8 = 2;
static constexpr int64_t NUM_FLOAT32 = 0;
static constexpr int64_t NUM_FLOAT16 = 1;
static constexpr int64_t NUM_BFLOAT16 = 27;

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT4_E2M1, op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT8_E5M2,
    op::DataType::DT_FLOAT8_E4M3FN};

static const std::initializer_list<DataType> MXSCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E8M0};

static const std::initializer_list<DataType> Y_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const aclTensor* x, const aclTensor* mxscale, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(mxscale, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* mxscale, const aclTensor* y, int64_t dstType)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mxscale, MXSCALE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, Y_DTYPE_SUPPORT_LIST, return false);

    if (static_cast<int64_t>(y->GetDataType()) != dstType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out(y) data type must be the same as dstType.");
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y, const aclTensor* mxscale, int64_t axis)
{
    OP_LOGD("CheckShape begin");

    auto xShape = x->GetViewShape();
    auto yShape = y->GetViewShape();
    size_t xDimNum = static_cast<size_t>(xShape.GetDimNum());
    size_t yDimNum = static_cast<size_t>(yShape.GetDimNum());
    if (xDimNum != yDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim num of x and y should be same. x:%zu, y:%zu.", xDimNum, yDimNum);
        return false;
    }

    // x dim num check
    OP_CHECK_MAX_DIM(x, MAX_DIM_LEN, return false);

    // mxscale dim num check
    auto mxscaleShape = mxscale->GetViewShape();
    size_t mxscaleDimNum = static_cast<size_t>(mxscaleShape.GetDimNum());
    if (mxscaleDimNum < MIN_DIM_LEN_MXSCALE || mxscaleDimNum > MAX_DIM_LEN_MXSCALE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mxscale dim num must be in [2, 8], but got %zu.", mxscaleDimNum);
        return false;
    }

    if (mxscaleDimNum != xDimNum + 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mxscale dim num must be x dim num + 1. mxscale:%zu, x:%zu.", mxscaleDimNum,
                xDimNum);
        return false;
    }

    // axis check
    int64_t axisChange = axis >= 0 ? axis : axis + static_cast<int64_t>(xDimNum);
    if (axisChange < 0 || axisChange >= static_cast<int64_t>(xDimNum)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "axis is out of range. axis:%ld, x dim num:%zu.", axis, xDimNum);
        return false;
    }

    OP_LOGD("CheckShape end");
    return true;
}

static bool CheckAttrValid(int64_t axis, int64_t dstType)
{
    if (dstType != NUM_FLOAT32 && dstType != NUM_FLOAT16 && dstType != NUM_BFLOAT16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dstType must be 0(FLOAT32), 1(FLOAT16) or 27(BF16), but got %ld.", dstType);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* mxscale, int64_t axis, int64_t dstType,
                               const aclTensor* y)
{
    // 1. Check null ptr
    CHECK_RET(CheckNotNull(x, mxscale, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. Check dtype
    CHECK_RET(CheckDtypeValid(x, mxscale, y, dstType), ACLNN_ERR_PARAM_INVALID);

    // 3. Check attr
    CHECK_RET(CheckAttrValid(axis, dstType), ACLNN_ERR_PARAM_INVALID);

    // 4. Check shape
    CHECK_RET(CheckShape(x, y, mxscale, axis), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAntiMxQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* mxscale, int64_t axis,
                                             int64_t dstType, const aclTensor* y, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAntiMxQuant, DFX_IN(x, mxscale, axis, dstType), DFX_OUT(y));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, mxscale, axis, dstType, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (x->IsEmpty() || mxscale->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // input参数如果非连续，需要转连续
    if ((x->GetDataType() == op::DataType::DT_FLOAT4_E2M1 || x->GetDataType() == op::DataType::DT_FLOAT4_E1M2) && !IsContiguous(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the data type of x is float4, it must be contiguous");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto mxscaleContiguous = l0op::Contiguous(mxscale, uniqueExecutor.get());
    CHECK_RET(mxscaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子AntiMxQuant进行计算
    auto antiMxQuantResult = l0op::AntiMxQuant(xContiguous, mxscaleContiguous, axis, dstType, uniqueExecutor.get());
    CHECK_RET(antiMxQuantResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参y是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(antiMxQuantResult, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAntiMxQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAntiMxQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
