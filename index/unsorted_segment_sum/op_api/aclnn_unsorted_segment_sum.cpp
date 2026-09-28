/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_unsorted_segment_sum.cpp
 * \brief
 */

#include "aclnn_unsorted_segment_sum.h"
#include "unsorted_segment_sum.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "level0/fill.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 数据支持的维度大小
static constexpr size_t MIN_INPUT_DIM_NUM = 0;
static constexpr size_t MAX_INPUT_DIM_NUM = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DATA = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,  op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_UINT32,  op::DataType::DT_UINT64};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_IDS = {op::DataType::DT_INT32,
                                                                           op::DataType::DT_INT64};

static bool CheckNotNull(const aclTensor* x, const aclTensor* segmentIds, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(segmentIds, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* segmentIds, const aclTensor* out)
{
    // 检查数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST_DATA, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(segmentIds, DTYPE_SUPPORT_LIST_IDS, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, x->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* segmentIds, const aclTensor* out)
{
    OP_CHECK_MIN_DIM(x, MIN_INPUT_DIM_NUM, return false);
    OP_CHECK_MAX_DIM(x, MAX_INPUT_DIM_NUM, return false);
    OP_CHECK_MIN_DIM(segmentIds, MIN_INPUT_DIM_NUM, return false);
    OP_CHECK_MAX_DIM(segmentIds, MAX_INPUT_DIM_NUM, return false);

    auto xShape = x->GetViewShape();
    auto idsShape = segmentIds->GetViewShape();
    auto xRank = xShape.GetDimNum();
    auto idsRank = idsShape.GetDimNum();

    // x的维度个数需 >= segmentIds的维度个数
    if (xRank < idsRank) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the rank of x[%zu] should not be less than the rank of segment_ids[%zu].",
                static_cast<size_t>(xRank), static_cast<size_t>(idsRank));
        return false;
    }

    // 以segmentIds的维度个数为基准，x与segmentIds存在不一致的维度
    for (size_t i = 0; i < idsRank; i++) {
        auto xDim = xShape.GetDim(i);
        auto idsDim = idsShape.GetDim(i);
        if (xDim >= 0 && idsDim >= 0 && xDim != idsDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the dim %zu of segment_ids[%ld] does not match the dim %zu of x[%ld].", i,
                    idsDim, i, xDim);
            return false;
        }
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* segmentIds, int64_t numSegments,
                               const aclTensor* out)
{
    CHECK_RET(CheckNotNull(x, segmentIds, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, segmentIds, out), ACLNN_ERR_PARAM_INVALID);
    if (numSegments <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numSegments should be greater than 0, but got %ld.", numSegments);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckShape(x, segmentIds, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus FillOutWithZeros(const aclTensor* out, aclOpExecutor* executor)
{
    OP_LOGI("x is an empty tensor, fill output with zeros.");
    auto outShape = out->GetViewShape();
    op::ShapeVector fillDims = op::ToShapeVector(outShape);
    auto fillDimsArray = executor->AllocIntArray(fillDims.data(), outShape.GetDimNum());
    CHECK_RET(fillDimsArray != nullptr, ACLNN_ERR_RUNTIME_ERROR);
    const aclTensor* fillDimTensor = executor->ConvertToTensor(fillDimsArray, op::DataType::DT_INT64);
    CHECK_RET(fillDimTensor != nullptr, ACLNN_ERR_RUNTIME_ERROR);
    const aclScalar* zeroScalar = executor->AllocScalar(0);
    CHECK_RET(zeroScalar != nullptr, ACLNN_ERR_RUNTIME_ERROR);
    const aclTensor* zeroTensor = executor->ConvertToTensor(zeroScalar, out->GetDataType());
    CHECK_RET(zeroTensor != nullptr, ACLNN_ERR_RUNTIME_ERROR);
    const aclTensor* zeroFillTensor = l0op::Fill(fillDimTensor, zeroTensor, fillDimsArray, executor);
    CHECK_RET(zeroFillTensor != nullptr, ACLNN_ERR_RUNTIME_ERROR);
    CHECK_RET(l0op::ViewCopy(zeroFillTensor, out, executor) != nullptr, ACLNN_ERR_RUNTIME_ERROR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUnsortedSegmentSumGetWorkspaceSize(const aclTensor* x, const aclTensor* segmentIds,
                                                    int64_t numSegments, aclTensor* out, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnUnsortedSegmentSum, DFX_IN(x, segmentIds, numSegments), DFX_OUT(out));

    // 固定写法，参数检查
    auto ret = CheckParams(x, segmentIds, numSegments, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空tensor处理：x为空时所有segment均为空段，out需填充0；out为空时无需计算
    if (out->IsEmpty() || x->IsEmpty()) {
        if (!out->IsEmpty()) {
            auto fillRet = FillOutWithZeros(out, uniqueExecutor.get());
            CHECK_RET(fillRet == ACLNN_SUCCESS, fillRet);
            *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        } else {
            *workspaceSize = 0;
        }
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入 x 转换成连续的 tensor
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_RUNTIME_ERROR);

    // 将输入 segmentIds 转换成连续的 tensor
    auto segmentIdsContiguous = l0op::Contiguous(segmentIds, uniqueExecutor.get());
    CHECK_RET(segmentIdsContiguous != nullptr, ACLNN_ERR_RUNTIME_ERROR);

    auto numSegmentsTensor = uniqueExecutor.get()->ConvertToTensor(&numSegments, 1, op::DataType::DT_INT64);
    CHECK_RET(numSegmentsTensor != nullptr, ACLNN_ERR_RUNTIME_ERROR);

    // 执行 L0 算子
    auto ussOut = l0op::UnsortedSegmentSum(xContiguous, segmentIdsContiguous, numSegmentsTensor, out->GetViewShape(),
                                           uniqueExecutor.get());
    CHECK_RET(ussOut != nullptr, ACLNN_ERR_RUNTIME_ERROR);

    // 将计算结果拷贝到输出 data 上
    auto viewCopyResult = l0op::ViewCopy(ussOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_RUNTIME_ERROR);

    // 获取计算过程中需要使用的 workspace 大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUnsortedSegmentSum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUnsortedSegmentSum);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
