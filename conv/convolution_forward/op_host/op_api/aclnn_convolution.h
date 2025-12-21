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
 * \file aclnn_convolution.h
 * \brief
 */

#ifndef OP_API_INC_CONVOLUTION_H
#define OP_API_INC_CONVOLUTION_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnConvolution的第一段接口，计算并获取workspace大小
 * @domain aclnn_ops_infer
 *
 * @param [in] input: npu，卷积输入。
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN。
 * 支持非连续的Tensor，数据格式支持NCL、NCHW、NCDHW。
 * @param [in] weight: npu, 卷积权重。
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN。
 * 支持非连续的Tensor，数据格式支持NCL、NCHW、NCDHW。
 * @param [in] bias: npu，偏差。
 * device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT。
 * 支持非连续的Tensor，数据格式支持NCL、NCHW、NCDHW、ND。
 * @param [in] stride: 卷积扫描步长。
 * aclIntArray，数组长度需等于input的维度减2。其值应该大于0。
 * @param [in] padding: 对input的填充。
 * aclIntArray，对于conv1d非转置卷积的数组长度可以为1或者2；对于conv2d数组长度可以为2或者4；conv3d数组长度可以为3。其值应该大于等于0。
 * @param [in] dilation: 卷积核中元素的间隔。
 * aclIntArray，数组长度需等于input的维度减2。其值应该大于0。
 * @param [in] transposed: 是否为转置卷积。
 * bool，True代表转置卷积。
 * @param [in] outputPadding：转置卷积时生效，对输出的补边。
 * aclIntArray，数组长度需等于input的维度减2。其值应大于等于0，且小于stride或dilation对应维度的值。非转置卷积情况下，忽略该属性配置。
 * @param [in] groups：表示从输入通道到输出通道的块链接个数。
 * int64_t，数值必须大于0，且满足groups*weight的C维度=input的C维度。
 * @param [out] output: 卷积输出。
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN，数据格式支持NCL、NCHW、NCDHW。
 * @param [in] cubeMathType：用于判断Cube单元应该使用哪种计算逻辑进行运算。
 * int8_t, Cube单元计算逻辑判断参数。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvolutionGetWorkspaceSize(const aclTensor* input, const aclTensor* weight,
                                                       const aclTensor* bias, const aclIntArray* stride,
                                                       const aclIntArray* padding, const aclIntArray* dilation,
                                                       bool transposed, const aclIntArray* outputPadding,
                                                       const int64_t groups, aclTensor* output, int8_t cubeMathType,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnConvTbc的第一段接口，计算并获取workspace大小
 * @domain aclnn_ops_infer
 *
 * @param [in] self: npu，Tbc卷积输入。
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、HIFLOAT8。
 * 支持非连续的Tensor，数据格式支持ND、NCL。
 * @param [in] weight: npu, Tbc卷积权重。
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、HIFLOAT8。
 * 支持非连续的Tensor，数据格式支持ND、NCL。
 * @param [in] bias: npu，偏差。
 * device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] pad: 对self的填充。
 * int64_t，其值应该大于等于0。
 * @param [out] output: Tbc卷积输出。
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、HIFLOAT8，数据格式支持NCL。
 * @param [in] cubeMathType：用于判断Cube单元应该使用哪种计算逻辑进行运算。
 * int8_t, Cube单元计算逻辑判断参数。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvTbcGetWorkspaceSize(const aclTensor* self, const aclTensor* weight,
                                                   const aclTensor* bias, const int64_t pad, aclTensor* output,
                                                   int8_t cubeMathType, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief aclnnConvDepthwise2d的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] self: npu，depthwise2d卷积输入。
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、HIFLOAT8。
 * 支持非连续的Tensor，数据格式支持NCHW。
 * @param [in] weight: npu, depthwise2d卷积权重。
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、HIFLOAT8。
 * 支持非连续的Tensor，数据格式支持NCHW。
 * @param [in] kernelSize: npu，depthwise2d卷积核尺寸。
 * device侧的aclTensor，支持非连续的Tensor。
 * @param [in] bias: npu，偏差。
 * device侧的aclTensor，支持非连续的Tensor，支持一维且数值和weight的第一维相等。
 * @param [in] stride: depthwise2d卷积扫描步长。
 * aclIntArray，数组长度需等于self的维度减2。其值应该大于0。
 * @param [in] padding: 对self的填充。
 * aclIntArray，数组长度需等于self的维度减2。其值应该大于等于0。
 * @param [in] dilation: 卷积核中元素的间隔。
 * aclIntArray，数组长度需等于self的维度减2。其值应该大于0。
 * @param [out] out: depthwise2d卷积输出。
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、HIFLOAT8。数据格式支持NCHW。
 * @param [in] cubeMathType：用于判断Cube单元应该使用哪种计算逻辑进行运算。
 * int8_t, Cube单元计算逻辑判断参数。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvDepthwise2dGetWorkspaceSize(const aclTensor* self, const aclTensor* weight,
                                                           const aclIntArray* kernelSize, const aclTensor* bias,
                                                           const aclIntArray* stride, const aclIntArray* padding,
                                                           const aclIntArray* dilation, aclTensor* out,
                                                           int8_t cubeMathType, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);

/**
 * @brief aclnnConvolution的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由aclnnConvolutionGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。调用该接口后，executor不再可用
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvolution(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

/**
 * @brief aclnnConvTbc的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由aclnnConvTbcGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。调用该接口后，executor不再可用
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvTbc(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

/**
 * @brief aclnnConvDepthwise2d的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由aclnnConvDepthwise2dGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。调用该接口后，executor不再可用
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvDepthwise2d(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CONVOLUTION_H
