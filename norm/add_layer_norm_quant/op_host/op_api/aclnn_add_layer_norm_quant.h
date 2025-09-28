/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ADD_LAYER_NORM_QUANT_H_
#define OP_API_INC_LEVEL2_ADD_LAYER_NORM_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAddLayerNormQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：Add + LayerNorm + 量化计算的融合算子，将加法的计算结果做层归一化计算后进行量化，
 * 并将加法的计算结果，被量化的归一化计算结果和量化参数返回。
 * 计算公式：
 *     x = x1 + x2 + bias
 *     y = LayerNorm(x, gamma, beta)
 *     y1, out_scales1 = quant(y, scales1, zero_points1)
 *     y2, out_scales2 = quant(y, scales2, zero_points2)
 *
 * @param [in] x1:
 * 公式中的输入`x1`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与x2一致，shape需要与x2一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] x2:
 * 公式中的输入`x2`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与x1一致，shape需要与x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] gamma:
 * 公式中的输入`gamma`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与x1一致，shape需要与beta一致, 且可以广播到x1。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] beta:
 * 公式中的输入`beta`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与x1一致，shape需要与gamma一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] biasOptional:
 * 公式中的输入`bias`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与x1一致，shape需要与gamma或x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] scales1Optional:
 * 公式中的输入`scales1`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与scales2Optional一致，shape需要与gamma一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] scales2Optional:
 * 公式中的输入`scales2`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与scales1Optional一致，shape需要与gamma一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] zeroPoints1Optional:
 * 公式中的输入`zero_points1`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与scales1Optional一致，shape需要与gamma一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] zeroPoints2Optional:
 * 公式中的输入`zero_points2`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与scales1Optional一致，shape需要与gamma一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] quantMode: const char* 类型，指定要应用的量化模式，
 * 支持 "dynamic" 或 "static", 分别表示动态量化/静态量化。
 * @param [in] epsilon: double 类型，层归一化中用到的防止除0的参数。
 * @param [in] additionalOutput: bool
类型，用于指定Add的计算结果x是否输出。当为Fasle时，算子不会给xOut输出，xOut此时为脏数据。
 * @param [in] divMode: bool
类型，用于指定静态量化计算的算法。当为False时，静态量化计算计算使用乘法，反之亦然。仅仅在quantMode为"dynamic"时生效。
 * @param [in] y1Out:
 * 公式中的输出`y1`，数据类型支持INT8，shape需要与x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] y2Out:
 * 公式中的输出`y2`，数据类型支持INT8，shape需要与x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] xOut:
 * 公式中的输出`x`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与x1一致，shape需要与x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] outScales1Out:
 * 公式中的输入`out_scales1`，数据类型支持FLOAT，shape需要和x1剔除最后一维一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] outScales2Out:
 * 公式中的输入`out_scales2`，数据类型支持FLOAT，shape需要和x1剔除最后一维一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor: 返回op执行器，包含算子计算流程。
* @return aclnnStatus: 返回状态码。

 */
ACLNN_API aclnnStatus aclnnAddLayerNormQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta,
    const aclTensor* biasOptional, const aclTensor* scales1Optional, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const char* quantMode, double epsilon,
    bool additionalOutput, bool divMode, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* outScales1Out,
    aclTensor* outScales2Out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAddLayerNormQuant的第二段接口，用于执行计算。
 *
 * 算子功能：将输入tensor转换为指定的dtype类型。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAddLayerNormQuantGetWorkspaceSize。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnAddLayerNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ADD_LAYER_NORM_QUANT_H_
