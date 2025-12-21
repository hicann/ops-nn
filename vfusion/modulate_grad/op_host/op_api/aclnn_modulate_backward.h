/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MODULATE_GRAD_H_
#define OP_API_INC_MODULATE_GRAD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C"{
#endif

/**
 * @brief aclnnModulateBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnnop_ops_train
 * 算子功能： out = grad_output * scale + shift
 * @param [in] grad_output: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, BFLOAT16
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] input: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, BFLOAT16
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] scale: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT32, BFLOAT16
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] shift: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT32, BFLOAT16
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] grad_input: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT32, BFLOAT16
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] grad_scale: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT32, BFLOAT16
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] grad_shift: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT32, BFLOAT16
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnModulateBackwardGetWorkspaceSize(
    const aclTensor* grad_output, const aclTensor* input,const aclTensor* scale,const aclTensor* shift,
    const aclTensor* grad_input,const aclTensor* grad_scale,const aclTensor* grad_shift,uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief: aclnnModulateBackward的第二段接口，用于执行计算
 * @domain aclnnop_ops_train
 * 算子功能： out = self * scaleOptional + shiftOptional
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnModulateBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnModulateBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif //OP_API_INC_MODULATE_GRAD_H_