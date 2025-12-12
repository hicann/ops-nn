/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OP_API_INC_LEVEL2_ACLNN_GEGLU_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_GEGLU_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGeGluBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：完成GeGlu的反向。
 *
 * 实现说明
 * api计算基本路径：
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous])
 *     C[(self)] --> D([l0op::Contiguous])
 *     E[(gelu)] --> F([l0op::Contiguous])
 *     B --> G([l0op::GeGluV2Grad])
 *     D --> G
 *     F --> G
 *     H((dim)) --> G
 *     I((approximate)) --> G
 *     G --> J([l0op::ViewCopy])
 *     J --> K[(gradInput)]
 * ```
 */

/**
 * @param [in] gradOutput：计算输入，npu
 * device侧的aclTensor，数据类型支持FLOAT16，shape中除dim维外，其它维的大小跟self一样，
 * dim维的大小是self的一半，支持非连续的Tensor，数据格式支持ND。
 * @param [in] self：计算输入，npu
 * device侧的aclTensor，数据类型支持FLOAT16，shape中除dim维外，其它维的大小跟gradOutput一样，
 * dim维的大小是gradOutput的两倍，支持非连续的Tensor，数据格式支持ND。
 * @param [in] gelu：计算输入，npu
 * device侧的aclTensor，数据类型支持FLOAT16，shape需要与gradOutput一样，支持非连续的Tensor， 数据格式支持ND。
 * @param [in] dim: 计算属性，host侧的整数，数据类型支持INT64，当前取值只支持-1。
 * @param [in] approximate: 计算属性，host侧的整数，数据类型支持INT64，取值范围是0('none')、1('tanh') ，当前取值只支持
 * 1('tanh') 。
 * @param [in] activateLeft:
 * 计算属性，host侧的布尔值，表示激活函数操作数据块的方向，默认值为false，表示对右边做activate。
 * @param [out] gradInput：计算输出，npu
 * device侧的aclTensor，数据类型支持FLOAT16，shape需要与self一样，支持非连续的Tensor， 数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeGluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, int64_t dim, int64_t approximate,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGeGluV3Backward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * @param [in] gradOutput：计算输入，npu
 * device侧的aclTensor，数据类型支持FLOAT16，shape中除dim维外，其它维的大小跟self一样，
 * dim维的大小是self的一半，支持非连续的Tensor，数据格式支持ND。
 * @param [in] self：计算输入，npu
 * device侧的aclTensor，数据类型支持FLOAT16，shape中除dim维外，其它维的大小跟gradOutput一样，
 * dim维的大小是gradOutput的两倍，支持非连续的Tensor，数据格式支持ND。
 * @param [in] gelu：计算输入，npu
 * device侧的aclTensor，数据类型支持FLOAT16，shape需要与gradOutput一样，支持非连续的Tensor， 数据格式支持ND。
 * @param [in] dim: 计算属性，host侧的整数，数据类型支持INT64，当前取值只支持-1。
 * @param [in] approximate: 计算属性，host侧的整数，数据类型支持INT64，取值范围是0('none')、1('tanh') ，当前取值只支持
 * 1('tanh') 。
 * @param [in] activateLeft:
 * 计算属性，host侧的布尔值，表示激活函数操作数据块的方向，默认值为false，表示对右边做activate。
 * @param [out] gradInput：计算输出，npu
 * device侧的aclTensor，数据类型支持FLOAT16，shape需要与self一样，支持非连续的Tensor， 数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeGluV3BackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, int64_t dim, int64_t approximate,
    bool activateLeft, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGeGluBackward的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGeGluBackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnGeGluBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnGeGluBackward的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
   @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGeGluV3BackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnGeGluV3Backward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_GEGLU_BACKWARD_H_
