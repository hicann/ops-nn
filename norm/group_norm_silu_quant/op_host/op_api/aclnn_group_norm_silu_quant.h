/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUP_NORM_SILU_QUANT_H_
#define OP_API_INC_GROUP_NORM_SILU_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupNormSiluQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：完成GroupNorm和Silu以及Quant融合算子功能。
 * GroupNorm计算公式为：
 * y=((x-Ex) / (sqrt(Var(x)+ϵ​x))​)∗ γ + β
 * 将channel方向分group，然后每个group内做归一化，算(C//G)HW的均值
 *
 * Silu计算公式为：
 * f(x) = x * sigmoid(x)
 * 当x大于0时,Silu激活函数将会放大x,而当x小于0时,Silu激活函数将会降低x,可以抑制过拟合
 * 
 * Quant计算公式为：
 * out = round((x / quantScale))
 *
 * @param [in] self：计算输入，shape大于等于2维
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] gamma：计算输入，shape为1维，且等于c维度。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] beta：shape为1维，且等于c维度。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] quantScale：shape为1维，且等于1或c维度。
 * npu device侧的aclTensor，数据类型支持FLOAT32类型，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] group：计算属性，host侧的整数，数据类型支持INT64,分组信息
 * @param [in] esp：计算属性，host侧的浮点数，数据类型支持double,默认le-5
 * @param [out] out：y的输出。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND。
 * @param [out] meanOut：均值的输出。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND。
 * @param [out] rstdOut：1/sqrt(Var(x)+ϵ​x)结果输出
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupNormSiluQuantGetWorkspaceSize(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, const aclTensor* quantScale, int64_t group, double eps, bool activateSilu,
    aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupNormSiluQuant的第二段接口，用于执行计算
*/
ACLNN_API aclnnStatus
aclnnGroupNormSiluQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_GROUP_NORM_SILU_QUANT_H_
