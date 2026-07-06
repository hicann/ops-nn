/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_RMS_NORM_QUANT_V3_H_
#define OP_API_INC_LEVEL2_RMS_NORM_QUANT_V3_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRmsNormQuantV3的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：RmsNorm + 量化计算的融合算子，将输入数据做RMS层归一化计算后进行量化，
 * 并将量化结果和表示归一化后的标准差的倒数返回。
 * 计算公式：
 *     rstd = rsqrt(mean(x^2, reduce_axis, keepdims=True) + epsilon)
 *     RmsNorm(x) = x * rstd * gamma + beta
 *     divMode为true时: y = round(RmsNorm(x) / scale + offset)
 *     divMode为false时: y = round(RmsNorm(x) * scale + offset)
 *
 * @param [in] x:
 * 公式中的输入x，数据类型支持FLOAT16、BFLOAT16、FLOAT32，shape维度支持1-8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] gamma:
 * 公式中的输入gamma，RmsNorm的权重参数，数据类型支持FLOAT16、BFLOAT16、FLOAT32，shape维度支持1-2维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] scale:
 * 公式中的输入scale，量化操作的缩放因子，数据类型支持FLOAT16、BFLOAT16、FLOAT32，shape维度支持1维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] offset:
 * 公式中的输入offset，量化操作的偏移量，可选参数。
 * 数据类型支持INT32、INT8、FLOAT16、BFLOAT16、FLOAT32，shape维度支持1维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] beta:
 * 公式中的输入beta，RmsNorm的偏置参数，可选参数。数据类型和shape维度与gamma保持一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] epsilon: double 类型，RMS层归一化中用到的防止除0的参数，默认值为1e-6。
 * @param [in] divMode: bool 类型，量化模式选择。true表示使用除法量化，false表示使用乘法量化。默认值为true。
 * @param [in] outputRstd: bool 类型，表示指定是否输出有效的rstd，当为false时，rstd为无效输出。默认值为false。
 * @param [out] y:
 * 公式中的输出y，量化后的输出张量。
 * 数据类型支持INT8、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN，shape需要与x一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] rstd:
 * 公式中的输出rstd，RMS归一化的标准差倒数。数据类型支持FLOAT32。
 * 支持空Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRmsNormQuantV3GetWorkspaceSize(const aclTensor* x, const aclTensor* gamma,
                                                          const aclTensor* scale, const aclTensor* offset,
                                                          const aclTensor* beta, double epsilon, bool divMode,
                                                          bool outputRstd, aclTensor* y, aclTensor* rstd,
                                                          uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRmsNormQuantV3的第二段接口，用于执行计算。
 *
 * 算子功能：执行RmsNorm+量化计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnRmsNormQuantV3GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRmsNormQuantV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_RMS_NORM_QUANT_V3_H_
