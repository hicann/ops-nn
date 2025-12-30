/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_SMOOTH_L1_LOSS_BACKWARD_H_
#define OP_API_INC_SMOOTH_L1_LOSS_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSmoothL1LossBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能： 对输入Tensor完成SmoothL1Loss backward操作
 * 对于smoothL1Loss的第一种情况, 即$|x-y|<1$,其导数为：$$\frac{\partial SmoothL1Loss(x,y)}{\partial x} = x - y $$
 * 对于smoothL1Loss的第一种情况, 即$|x-y|\geq 1$,其导数为：$$\frac{\partial SmoothL1Loss(x,y)}{\partial x} = sign(x-y)$$
 * @param [in] gradOut: npu device侧的aclTensor, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [in] self: npu device侧的aclTensor, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [in] target: npu device侧的aclTensor, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [in] reduction: host侧的参数，数据类型为int64_t，指定要应用到输出的缩减，支持0("none"),1("mean"),2"sum")。
 * @param [in] beta:指定在L1和L2损失之间更改的阈值，数据类型为double，该值必须是非负的。
 * @param [in] gradInput:npu device侧的aclTensor, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * 当reduction为"none"时，shape与self相同，否则shape为[1]
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码， 成功返回ACLNN_SUCCESS, 失败返回对应错误码。
 */
ACLNN_API aclnnStatus aclnnSmoothL1LossBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclTensor* self,
                                                                const aclTensor* target, int64_t reduction, float beta,
                                                                aclTensor* gradInput, uint64_t* workspaceSize,
                                                                aclOpExecutor** executor);

/**
 * @brief: aclnnSmoothL1LossBackward的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成SmoothL1Loss backward操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSigmoidBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码,成功返回ACLNN_SUCCESS, 失败返回对应错误码。
 */
ACLNN_API aclnnStatus aclnnSmoothL1LossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SMOOTH_L1_LOSS_BACKWARD_H_