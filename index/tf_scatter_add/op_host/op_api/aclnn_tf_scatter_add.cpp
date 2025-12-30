/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_scatter_add_tf.cpp
 * \brief
 */

#include "aclnn_tf_scatter_add.h"
#include "tf_scatter_add.h"
#include "index/scatter_nd_add/op_api/scatter_nd_add.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MIN_INPUT_DIM_NUM = 1;
static constexpr size_t MAX_INPUT_DIM_MUM = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_INT32, op::DataType::DT_INT8,    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> NULL_SUPPORT_LIST = {};

static bool CheckNotNull(aclTensor* varRef, const aclTensor* indices, const aclTensor* updates)
{
    OP_CHECK_NULL(varRef, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(updates, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910 ||
        (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E)) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    }
    return NULL_SUPPORT_LIST;
}

static bool CheckDtypeValid(aclTensor* varRef, const aclTensor* indices, const aclTensor* updates)
{
    // 检查self的数据类型是否在算子的支持列表内
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(varRef, supportList, return false);
    // 检查index的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDEX_DTYPE_SUPPORT_LIST, return false);
    // varRef和updates的数据类型要一致
    if (varRef->GetDataType() != updates->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "updates dtype %s should be in same with varRef dtype %s.",
            op::ToString(updates->GetDataType()).GetString(), op::ToString(varRef->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool CheckShapeForScatterAdd(aclTensor* varRef, const aclTensor* indices, const aclTensor* updates)
{
    OP_CHECK_MIN_DIM(varRef, MIN_INPUT_DIM_NUM, return false);
    OP_CHECK_MIN_DIM(indices, MIN_INPUT_DIM_NUM, return false);
    OP_CHECK_MIN_DIM(updates, MIN_INPUT_DIM_NUM, return false);
    OP_CHECK_MAX_DIM(varRef, MAX_INPUT_DIM_MUM, return false);
    OP_CHECK_MAX_DIM(indices, MAX_INPUT_DIM_MUM, return false);
    OP_CHECK_MAX_DIM(updates, MAX_INPUT_DIM_MUM, return false);
    op::Shape varRefShape = varRef->GetViewShape();
    op::Shape indicesShape = indices->GetViewShape();
    op::Shape updatesShape = updates->GetViewShape();
    size_t varRefDimNum = varRefShape.GetDimNum();
    size_t indicesDimNum = indicesShape.GetDimNum();
    size_t updatesDimNum = updatesShape.GetDimNum();
    // 校验varRef的第一维为0, 且indices不为空的情况, 其他空tensor情况直接返回，与竞品保持一致
    if (varRefShape.GetDim(0) == 0 && !indices->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when varRef.shape[0] = 0, indices tensor must be empty.");
        return false;
    }
    // updates.shape = indices.shape + var.shape[1:]
    if (updatesDimNum != indicesDimNum + varRefDimNum - 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "updatesDimNum must have the same number of indicesDimNum add varDimNum - 1.");
        return false;
    }
    for (size_t i = 0; i < indicesDimNum; ++i) {
        if (updatesShape.GetDim(i) != indicesShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "updatesShape must have the same number of indicesShape.");
            return false;
        }
    }
    for (size_t i = 1; i < varRefDimNum; ++i) {
        if (updatesShape.GetDim(i + indicesDimNum - 1) != varRefShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "updatesShape must have the same number of varShape.");
            return false;
        }
    }
    return true;
}

static bool CheckShapeForScatterNdAdd(aclTensor* varRef, const aclTensor* indices, const aclTensor* updates)
{
    op::Shape varRefShape = varRef->GetViewShape();
    op::Shape indicesShape = indices->GetViewShape();
    op::Shape updatesShape = updates->GetViewShape();
    size_t varRefDimNum = varRefShape.GetDimNum();
    size_t indicesDimNum = indicesShape.GetDimNum();
    size_t updatesDimNum = updatesShape.GetDimNum();

    // indices.shape[-1] <= len(varRefShape)
    size_t rankSize = indicesShape.GetDim(indicesDimNum - 1);
    if (rankSize > varRefDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indices.shape[-1] must less than or equal to varRefDimNum.");
        return false;
    }
    // updates.shape = indices.shape[:-1] + var.shape[indices.shape[-1]:]
    if (updatesDimNum != indicesDimNum - 1 + varRefDimNum - rankSize) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "updatesDimNum must have the same number of indicesDimNum - 1 add varDimNum - rankSize.");
        return false;
    }
    for (size_t i = 0; i < indicesDimNum - 1; ++i) {
        if (updatesShape.GetDim(i) != indicesShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "updatesShape must have the same number of indicesShape.");
            return false;
        }
    }
    for (size_t i = 0; i < varRefDimNum - rankSize; ++i) {
        if (updatesShape.GetDim(i + indicesDimNum - 1) != varRefShape.GetDim(i + rankSize)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "updatesShape must have the same number of varShape.");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(aclTensor* varRef, const aclTensor* indices, const aclTensor* updates)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(varRef, indices, updates), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(varRef, indices, updates), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查updates的shape是否为目标形状
    CHECK_RET(
        CheckShapeForScatterAdd(varRef, indices, updates) || CheckShapeForScatterNdAdd(varRef, indices, updates),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTfScatterAddGetWorkspaceSize(
    aclTensor* varRef, const aclTensor* indices, const aclTensor* updates, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnTfScatterAdd, DFX_IN(varRef, indices, updates), DFX_OUT(varRef));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(varRef, indices, updates);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (varRef->IsEmpty() || indices->IsEmpty() || updates->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入varRef转换成连续的tensor
    auto varRefContiguous = l0op::Contiguous(varRef, uniqueExecutor.get());
    CHECK_RET(varRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入indices转换成连续的tensor
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入updates转换成连续的tensor
    auto updatesContiguous = l0op::Contiguous(updates, uniqueExecutor.get());
    CHECK_RET(updatesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* scatterAddRes = nullptr;
    bool useScatterNd = CheckShapeForScatterNdAdd(varRef, indices, updates);
    // bf16时转换成fp32计算
    if (varRef->GetDataType() == op::DataType::DT_BF16) {
        auto varRefContiguousFloat = l0op::Cast(varRefContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(varRefContiguousFloat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto updatesContiguousFloat = l0op::Cast(updatesContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(updatesContiguousFloat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto scatterAddResFloat =
            useScatterNd ?
                l0op::ScatterNdAdd(
                    varRefContiguousFloat, indicesContiguous, updatesContiguousFloat, false, uniqueExecutor.get()) :
                l0op::ScatterAdd(
                    varRefContiguousFloat, indicesContiguous, updatesContiguousFloat, false, uniqueExecutor.get());
        CHECK_RET(scatterAddResFloat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        scatterAddRes = l0op::Cast(scatterAddResFloat, op::DataType::DT_BF16, uniqueExecutor.get());
        CHECK_RET(scatterAddRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 执行L0算子
        scatterAddRes =
            useScatterNd ?
                l0op::ScatterNdAdd(
                    varRefContiguous, indicesContiguous, updatesContiguous, false, uniqueExecutor.get()) :
                l0op::ScatterAdd(varRefContiguous, indicesContiguous, updatesContiguous, false, uniqueExecutor.get());
        CHECK_RET(scatterAddRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将计算结果拷贝到输出data上
    auto viewCopyResult = l0op::ViewCopy(scatterAddRes, varRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTfScatterAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTfScatterAdd);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
