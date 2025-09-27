/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_heaviside.h"
#include "heaviside.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "level0/broadcast_to.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

inline static bool CheckNotNull(const aclTensor* input, const aclTensor* values, const aclTensor* out)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(values, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckDtypeValid(const aclTensor* input, const aclTensor* values, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(input, AICORE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(values, AICORE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, AICORE_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckShape(const aclTensor* input, const aclTensor* values, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(input, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(values, MAX_DIM_LEN, return false);

    Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(input, values, broadcastShape, return false);

    if (out->GetViewShape().GetShapeSize() != broadcastShape.GetShapeSize()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The out shape size is not same with broadcastShapeSize.");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* input, const aclTensor* values, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(input, values, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(input, values, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入形状是否满足
    CHECK_RET(CheckShape(input, values, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHeavisideGetWorkspaceSize(
    const aclTensor* input, const aclTensor* values, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnHeaviside, DFX_IN(input, values), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(input, values, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Heaviside算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (input->IsEmpty() || values->IsEmpty() || out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入input转换成连续的tensor
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入values转换成连续的tensor
    auto valuesContiguous = l0op::Contiguous(values, uniqueExecutor.get());
    CHECK_RET(valuesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 申请heaviside的输出tensor
    const aclTensor* result = nullptr;

    // 判断输入shape不相等需要调用BroadcastTo
    if (valuesContiguous->GetViewShape().GetShapeSize() != 1 &&
        inputContiguous->GetViewShape() != valuesContiguous->GetViewShape()) {
        const aclTensor* inputBroadcast = inputContiguous;
        const aclTensor* valuesBroadcast = valuesContiguous;
        op::Shape broadcastShape;

        if (BroadcastInferShape(inputContiguous->GetViewShape(), valuesContiguous->GetViewShape(), broadcastShape)) {
            op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
            auto broadcastShapeArray = uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
            CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
            inputBroadcast = l0op::BroadcastTo(inputContiguous, broadcastShapeArray, uniqueExecutor.get());
            CHECK_RET(inputBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            valuesBroadcast = l0op::BroadcastTo(valuesContiguous, broadcastShapeArray, uniqueExecutor.get());
            CHECK_RET(valuesBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        result = l0op::Heaviside(inputBroadcast, valuesBroadcast, uniqueExecutor.get());
    } else {
        result = l0op::Heaviside(inputContiguous, valuesContiguous, uniqueExecutor.get());
    }
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHeaviside(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHeaviside);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
} // namespace

#ifdef __cplusplus
}
#endif