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
 * \file aclnn_quant_convolution.h
 * \brief
 */

#ifndef OP_API_INC_QUANT_CONVOLUTION_H
#define OP_API_INC_QUANT_CONVOLUTION_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief quant convolution接口，计算并获取workspace大小
 * @domain aclnn_ops_infer
 *
 * @param [in] input: npu，feature map
 * device侧的aclTensor，数据类型支持INT8、HIFLOAT8、FLOAT8_E4M3FN
 * 支持非连续的Tensor，数据格式支持NCHW、NCDHW
 * @param [in] weight: npu, kernels
 * device侧的aclTensor，数据类型与input一致
 * 支持非连续的Tensor，数据格式与input一致
 * @param [in] bias: npu，偏置
 * device侧的aclTensor
 * input数据类型为INT8时，数据类型支持BFLOAT16、FLOAT、INT32
 * input数据类型为HIFLOAT8、FLOAT8_E4M3FN时，数据类型支持FLOAT
 * 支持非连续的Tensor，数据格式为ND
 * @param [in] scale: npu, scale
 * device侧的aclTensor，数据类型支持INT64、UINT64、FLOAT。数据格式支持ND
 * @param [in] offset: npu, offset
 * device侧的aclTensor，暂不支持
 * @param [in] stride: 步长
 * int64的数组，数组长度需等于input_dim - 2，例：quant conv2d的步长数组的有效长度是2维
 * @param [in] padding: 补边
 * int64的数组，数组长度支持input_dim - 2 或 （input_dim - 2）* 2，例：quant conv2d的padding数组的有效长度支持2维或4维
 * @param [in] dilation: kernel中元素的间隔，>1代表空洞卷积
 * int64的数组，数组长度需等于input_dim - 2，例：quant conv2d的dilation数组的有效长度是2维
 * @param [in] groups：分组数，表示从输入通道到输出通道的块链接个数
 * int64，大于0，且能被input和output的通道数整除，且input通道数 = weight通道数*groups
 * @param [in] offsetx：量化因子，表示量化缩放时对值域的偏移量
 * int32，最大向右偏移127，向左偏移-128
 * @param [in] roundMode：取整模式
 * output数据类型为INT8、FLOAT8_E4M3FN时，支持rint
 * output数据类型为HIFLOAT8时，支持round；其他场景不做要求
 * @param [out] output: npu
 * device侧的aclTensor
 * input数据类型为INT8时，数据类型为FLOAT16；
 * input数据类型为HIFLOAT8时，数据类型支持FLOAT16、FLOAT32、BFLOAT16、HIFLOAT8
 * input数据类型为FLOAT8_E4M3FN时，数据类型支持FLOAT16、FLOAT32、BFLOAT16、FLOAT8_E4M3FN
 * broadcast之后的shape，数据格式与input一致
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus aclnnQuantConvolutionGetWorkspaceSize(const aclTensor* input, const aclTensor* weight,
                                                            const aclTensor* bias, const aclTensor *scale,
                                                            const aclTensor *offset, const aclIntArray* stride,
                                                            const aclIntArray* padding, const aclIntArray* dilation,
                                                            bool transposed, const aclIntArray* outputPadding,
                                                            int64_t groups, int32_t offsetx,
                                                            const char* roundMode, aclTensor* output,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief quant convolution接口，进行kernellaunch
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由aclnnQuantConvolutionGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。调用该接口后，executor不再可用
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnQuantConvolution(void *workspace, const uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_CONVOLUTION_H