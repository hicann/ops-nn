/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OP_API_INC_PRELU_BACKWARD_H_
#define OP_API_INC_PRELU_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnPreluBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：完成PRelu的反向。
 *
 * @param [in] gradOutput：反向传播的梯度值，即上一层的输出梯度。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32类型。
 * @param [in] self：PRelu的输出值。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32类型。
 * @param [in] weight：PRelu的权重。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32类型。
 * @param [out] gradInput：backward的输出，为输入的梯度值，即对输入进行求导后的结果。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32类型。
 * @param [out] gradWeight：backward的输出，为weight的梯度值，即对weight进行求导后的结果。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32类型。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnPreluBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                         const aclTensor* weight, aclTensor* gradInput,
                                                         aclTensor* gradWeight, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

/**
 * @brief aclnnPreluBackward的第二段接口，用于执行计算
 */
ACLNN_API aclnnStatus aclnnPreluBackward(void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                                         aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_PRELU_BACKWARD_H_
