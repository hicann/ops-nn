/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GELU_BACKWARD_V2_H_
#define OP_API_INC_GELU_BACKWARD_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGeluBackwardV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 算子功能：完成Gelu的反向。
 * @param [in] gradOutput：反向传播的梯度值，即上一层的输出梯度，和正向输出的shape一致。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持FRACTAL_NZ，NC1HWC0,ND，支持非连续的Tensor。
 * @param [in] self：gelu的输出值。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持FRACTAL_NZ，NC1HWC0,ND，支持非连续的Tensor。
 * @param [in] approximate: gelu_backward计算的入参，指定高斯近似算法，可配置为"none"或"tanh"。
 * @param [out] gradInput：gelu_backward的输出，为输入的梯度值，即对输入进行求导后的结果。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持FRACTAL_NZ，NC1HWC0,ND，支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeluBackwardV2GetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, char* approximate, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGeluBackwardV2的第二段接口，用于执行计算
 */
ACLNN_API aclnnStatus
aclnnGeluBackwardV2(void* workspace, uint64_t workspace_size, aclOpExecutor* executor, const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_GELU_BACKWARD_V2_H_
