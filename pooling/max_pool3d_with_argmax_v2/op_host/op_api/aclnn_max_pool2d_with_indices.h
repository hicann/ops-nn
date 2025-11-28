/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OP_API_INC_LEVEL2_ACLNN_MAX_POOL2D_WITH_INDICES_H_
#define OP_API_INC_LEVEL2_ACLNN_MAX_POOL2D_WITH_INDICES_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMaxPool2dWithMask的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 对于输入信号的输入通道，提供2维池化操作：
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持float16、float32、bfloat16（仅昇腾910B AI处理器、
 * 昇腾910_93 AI处理器支持），数据格式支持NCHW(ND)， 支持非连续的Tensor。
 * @param [in] kernelSize: aclIntArray类型， 表示最大池化的窗口大小。
 * @param [in] stride: aclIntArray类型， 窗口移动的步长。
 * @param [in] padding: aclIntArray类型， 每一条边补充的层数，补充的位置填写负无穷。
 * @param [in] dilation: aclIntArray类型， 控制窗口中元素的步幅。
 * @param [in] ceilMode: aclIntArray类型， 为true时用上取整的方法计算输出形状，默认向下取整。
 * @param [in] out: npu device侧的aclTensor，数据类型支持float16、float32、bfloat16（仅昇腾910B
 * AI处理器、昇腾910_93 AI处理器支持）， 数据格式支持NCHW(ND)， 支持非连续的Tensor。
 * @param [in] indices: npu
 * device侧的aclTensor，最大值在求mask的kernel位置的bit值组成的Tensor，数据类型支持int8，数据格式支持NCHW(ND)，
 * 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnMaxPool2dWithMask的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnMaxPool2dWithMaskGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnMaxPool2dWithMask(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnMaxPool2dWithIndices的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 对于输入信号的输入通道，提供2维池化操作：
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持float32（仅昇腾910B AI处理器支持、昇腾910_93 AI处理器支持），
 * 数据格式支持NCHW(ND)，支持非连续的Tensor。
 * @param [in] kernelSize: aclIntArray类型， 表示最大池化的窗口大小。
 * @param [in] stride: aclIntArray类型， 窗口移动的步长。
 * @param [in] padding: aclIntArray类型， 每一条边补充的层数，补充的位置填写负无穷。
 * @param [in] dilation: aclIntArray类型， 控制窗口中元素的步幅。
 * @param [in] ceilMode: aclIntArray类型， 为true时用上取整的方法计算输出形状，默认向下取整。
 * @param [in] out: npu device侧的aclTensor，数据类型支持float32（仅昇腾910B AI处理器支持），数据格式支持NCHW(ND)，
 * 支持非连续的Tensor。
 * @param [in] indices: npu device侧的aclTensor，最大值索引位置组成的tensor，数据类型支持int32，数据格式支持NCHW(ND)，
 * 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMaxPool2dWithIndicesGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnMaxPool2dWithIndices的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnMaxPool2dWithIndicesGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnMaxPool2dWithIndices(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_MAX_POOL2D_WITH_INDICES_H_
