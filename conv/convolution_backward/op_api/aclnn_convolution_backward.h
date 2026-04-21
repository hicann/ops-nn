/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_CONVOLUTION_BACKWARD_H_
#define OP_API_INC_CONVOLUTION_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnConvolutionBackward的第一段接口，计算并获取workspace大小
 * @domain aclnn_ops_train
 *
 * @param [in] gradOutput: npu，卷积输出梯度。
 * device侧的aclTensor, 支持非连续的Tensor，要求和input、weight满足卷积输入输出shape的推导关系。
 * @param [in] input: npu，卷积输入。
 * device侧的aclTensor, 支持非连续的Tensor，要求和gradOutput、weight满足卷积输入输出shape的推导关系。
 * @param [in] weight: npu, 卷积权重。
 * device侧的aclTensor，支持非连续的Tensor，要求和gradOutput、input满足卷积输入输出shape的推导关系。
 * @param [in] biasSizes: npu，卷积正向过程中偏差(bias)的shape。
 * aclIntArray。
 * @param [in] stride: 反向传播过程中卷积核在输入上移动的步长。
 * aclIntArray，数组长度为weight维度减2，数值必须大于0。
 * @param [in] padding: 反向传播过程中对于输入填充。
 * aclIntArray，数组长度可以为weight维度减2，在2d场景下数组长度可以为4。数值必须大于等于0。
 * @param [in] dilation: 反向传播过程中的膨胀参数。
 * aclIntArray，数组长度可以为weight维度减2。数值必须大于0。
 * @param [in] transposed: 转置卷积使能标志位。
 * bool，当其值为True时使能转置卷积。
 * @param [in] outputPadding：反向传播过程中对于输出填充。
 * aclIntArray，在transposed为True时，数组长度可以为weight维度减2，数值必须大于等于0且小于stride。在transposed为False时，要求每个元素值为0。
 * @param [in] groups：反向传播过程中输入通道的分组数。
 * int，数值必须大于0, groups*weight的C维度=input的C维度。
 * @param [in] outputMask：输出掩码参数, 指定输出中是否包含输入、权重、偏差的梯度。
 * aclBoolArray, 反向传播过程输出掩码参数为True对应位置的梯度。
 * @param [in] cubeMathType：用于判断Cube单元应该使用哪种计算逻辑进行运算。
 * int8_t, Cube单元计算逻辑判断参数。
 * @param [out] gradInput: 卷积输入梯度在npu device侧的aclTensor。
 * @param [out] gradWeight: 卷积权重梯度在npu device侧的aclTensor。
 * @param [out] gradBias: 卷积偏置梯度在npu device侧的aclTensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* input, const aclTensor* weight, const aclIntArray* biasSizes,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool transposed,
    const aclIntArray* outputPadding, int groups, const aclBoolArray* outputMask, int8_t cubeMathType,
    aclTensor* gradInput, aclTensor* gradWeight, aclTensor* gradBias, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnConvTbcBackward的第一段接口，计算并获取workspace大小
 * @domain aclnn_ops_train
 *
 * @param [in] self: npu，卷积输出梯度
 * device侧的aclTensor，数据类型浮点类型FLOAT16，FLOAT32，BFLOAT16
 * 支持非连续的Tensor，数据格式支持ND、NCL
 * @param [in] input: npu，卷积输入
 * device侧的aclTensor，数据类型浮点类型FLOAT16，FLOAT32，BFLOAT16
 * 支持非连续的Tensor，数据格式支持ND、NCL
 * @param [in] weight: npu, 卷积权重
 * device侧的aclTensor，数据类型浮点类型FLOAT16，FLOAT32，BFLOAT16
 * 支持非连续的Tensor，数据格式支持ND、NCL
 * @param [in] bias: npu，卷积偏置
 * device侧的aclTensor，数据类型浮点类型FLOAT16，FLOAT32，BFLOAT16
 * @param [in] pad: 补边
 * int64_t，数值必须大于等于0
 * @param [out] gradInput: 卷积输入梯度在npu device侧的aclTensor
 * @param [out] gradWeight: 卷积权重梯度在npu device侧的aclTensor
 * @param [out] gradBias: 卷积偏置梯度在npu device侧的aclTensor
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvTbcBackwardGetWorkspaceSize(const aclTensor* self, const aclTensor* input,
                                                           const aclTensor* weight, const aclTensor* bias,
                                                           int64_t pad, int8_t cubeMathType,
                                                           aclTensor* gradInput, aclTensor* gradWeight,
                                                           aclTensor* gradBias, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);


/**
 * @brief aclnnDeformableConv2dBackward的第一段接口，计算并获取workspace大小
 * @domain aclnn_ops_train
 *
 * @param [in] input: npu，可变形卷积输入。
 * device侧的aclTensor，要求和gradOutput、weight满足卷积输入输出shape的推导关系。
 * @param [in] gradOutput: npu，可变形卷积输出梯度。
 * device侧的aclTensor，要求和input、weight满足卷积输入输出shape的推导关系。
 * @param [in] offsetOut: npu，可变形卷积offset输出。
 * device侧的aclTensor，要求满足可变形卷积输入输出shape的推导关系。
 * @param [in] weight: npu，可变形卷积权重。
 * device侧的aclTensor，要求和input、gradOutput满足卷积输入输出shape的推导关系。
 * @param [in] offset: npu，可变形卷积offset偏移量。
 * device侧的aclTensor，要求满足可变形卷积输入输出shape的推导关系。
 * @param [in] kernelSize: 卷积核大小。
 * aclIntArray，数组长度为2。
 * @param [in] stride: 反向传播过程中卷积核在输入上移动的步长。
 * aclIntArray，数组长度为4，数值必须大于0。
 * @param [in] padding: 反向传播过程中对于输入填充。
 * aclIntArray，数组长度为4，数值必须大于等于0。
 * @param [in] dilation: 反向传播过程中的膨胀参数。
 * aclIntArray，数组长度为4，数值必须大于0。
 * @param [in] groups: 反向传播过程中输入通道的分组数。
 * int64_t，数值必须大于0。
 * @param [in] deformableGroups: 可变形卷积的分组数。
 * int64_t，数值必须大于0。
 * @param [in] modulated: 是否使用调制可变形卷积。
 * bool，当其值为True时使用调制可变形卷积，当前只支持True。
 * @param [out] gradInput: 可变形卷积输入梯度在npu device侧的aclTensor。
 * @param [out] gradWeight: 可变形卷积权重梯度在npu device侧的aclTensor。
 * @param [out] gradOffset: 可变形卷积offset梯度在npu device侧的aclTensor。
 * @param [out] gradBias: 可变形卷积偏置梯度在npu device侧的aclTensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDeformableConv2dBackwardGetWorkspaceSize(
    const aclTensor* input, const aclTensor* gradOutput, const aclTensor* offsetOut, const aclTensor* weight,
    const aclTensor* offset,  const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, 
    const aclIntArray* dilation, int64_t groups, int64_t deformableGroups, bool modulated, aclTensor* gradInput,
    aclTensor* gradWeight, aclTensor* gradOffset, aclTensor* gradBias, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnConvolutionBackward的第二段接口，用于执行计算。
 *
 * 算子功能：完成卷积反向计算
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnConvolutionBackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvolutionBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               const aclrtStream stream);

/**
 * @brief aclnnConvTbcBackward的第二段接口，用于执行计算。
 *
 * 算子功能：完成卷积反向计算
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnConvTbcBackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvTbcBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           const aclrtStream stream);

/**
 * @brief aclnnDeformableConv2dBackward的第二段接口，用于执行计算。
 *
 * 算子功能：完成可变形卷积反向计算
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnDeformableConv2dBackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDeformableConv2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, 
 										    		const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CONVOLUTION_BACKWARD_H_
