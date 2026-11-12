/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_assert_async.h
 * \brief aclnn two-stage interface for AssertAsync.
 */

#ifndef OP_API_INC_ACLNN_ASSERT_ASYNC_H_
#define OP_API_INC_ACLNN_ASSERT_ASYNC_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAssertAsync 的第一段接口，根据具体的计算流程，计算 workspace 大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：断言 inputCondition（要求 numel == 1）的数值非 0；若为 0（含 bool False），
 * 设备侧打印 assertMsg 并触发算子失败；非 0 直接通过。本算子无输出张量。
 *
 * @param [in] inputCondition: 输入 Tensor，npu device 侧 aclTensor，numel == 1（rank-0 或任意收缩为
 *             单元素的 shape），数据类型支持 BOOL/INT8/INT16/INT32/INT64/UINT8/FLOAT16/FLOAT/DOUBLE/
 *             BF16/COMPLEX64/COMPLEX128，数据格式支持 ND。
 * @param [in] assertMsg: Host 侧字符串，断言失败时打印的消息，可为 nullptr（视为 ""）。
 * @param [out] workspaceSize: 返回用户需要在 npu device 侧申请的 workspace 大小。
 * @param [out] executor: 返回 op 执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAssertAsyncGetWorkspaceSize(const aclTensor* inputCondition, const char* assertMsg,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAssertAsync 的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在 npu device 侧申请的 workspace 内存起址。
 * @param [in] workspaceSize: 在 npu device 侧申请的 workspace 大小，由第一段接口获取。
 * @param [in] executor: op 执行器，包含了算子计算流程。
 * @param [in] stream: acl stream 流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAssertAsync(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_ASSERT_ASYNC_H_
