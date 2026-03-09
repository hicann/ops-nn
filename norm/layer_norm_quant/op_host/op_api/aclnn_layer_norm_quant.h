/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_LAYER_NORM_QUANT_H_
#define OP_API_INC_LEVEL2_LAYER_NORM_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnLayerNormQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 
 * 算子功能：LayerNorm + 量化计算的融合算子，将输入做层归一化计算后进行量化。
 * 计算公式：
 *     y = LayerNorm(x, gamma, beta)
 *     res, scaleOut = quant(y, scale, zeroPointsOptional)
 *
 * @param [in] x:
 * 公式中的输入`x`，数据类型支持BFLOAT16、FLOAT、FLOAT16。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] gamma:
 * 公式中的输入`gamma`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与x一致，shape需要与beta一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] beta:
 * 公式中的输入`beta`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与x一致，shape需要与gamma一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] scale:
 * 公式中的输入`scale`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与x一致，shape需要与zeroPointsOptional一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] zeroPointsOptional:
 * 公式中的输入`zeroPointsOptional`，数据类型支持INT8，shape需要与scale一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] quantMode: int类型，指定要应用的量化模式，
 * 支持 1 或 0, 分别表示动态量化/静态量化。
 * @param [in] epsilon: double 类型，层归一化中用到的防止除0的参数。
 * @param [in] res:
 * 公式中的输出`res`，数据类型支持INT8，shape需要与x一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] scaleOut:
 * 公式中的输入`scaleOut`，数据类型支持FLOAT，shape需要和x剔除最后一维一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor: 返回op执行器，包含算子计算流程。
* @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLayerNormQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* gammma, const aclTensor* beta, const aclTensor* scale,
    const aclTensor* zeroPointsOptional, int quantMode, double epsilon, aclTensor* res, aclTensor* scaleOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLayerNormQuant的第二段接口，用于执行计算。
 *
 * 算子功能：将输入tensor转换为指定的dtype类型。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLayerNormQuantGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnLayerNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_LAYER_NORM_QUANT_H_
