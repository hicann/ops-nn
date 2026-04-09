/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_RMS_NORM_DYNAMIC_MX_QUANT_H_
#define OP_API_INC_LEVEL2_RMS_NORM_DYNAMIC_MX_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRmsNormDynamicMxQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：RmsNorm + 量化计算的融合算子，将加法的计算结果做层归一化计算后进行量化，
 * 并将归一化计算结果，加法的计算结果，量化尺度和表示归一化后的标准差的倒数返回。
 * 计算公式：
 *     RmsNorm(x) = (x/RMS(x))*gamma) + beta
 *     y, mxscale_out = dynamicMxQuant(RmsNorm(x), round_mode, scale_alg, dst_type)
 *     rstd = 1/RMS(x)
 *
 * @param [in] x:
 * 公式中的输入x1，数据类型支持FLOAT16、BFLOAT16，shape维度支持1-7维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] gamma:
 * 公式中的输入gamma，数据类型默认与输入x1一致，若不一致，则显示设为FLOAT32，shape维度支持1维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] beta:
 * 公式中的输入beta，数据类型和shape维度与gamma保持一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] epsilon: double 类型，层归一化中用到的防止除0的参数。
 * @param [in] scaleAlg: 
 * 公式中的scale_alg，int 类型，表示mxscale_out的计算方法。
 * 支持取值0:OCP计算方法,和取值1:cuBLAS计算方法，当dst_type为FLOAT4_E2M1/FLOAT4_E1M2时仅支持取值为0。
 * @param [in] roundMode: 
 * 公式中的round_mode，string 类型，数据转换的模式。
 * 对应yOut数据类型为FLOAT4_E2M1/FLOAT4_E1M2时，支持{"rint", "floor", "round"}，
 * 对应yOut数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2时，仅支持{"rint"}。
 * @param [in] dstType: 
 * 公式中的dst_type，int 类型，表示指定数据转换后yOut的类型。
 * 输入范围为{35, 36, 40, 41}，分别对应{FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2}。
 * @param [in] outputRstd: bool 类型，表示指定是否输出有效的rstdOut，当为False时，rstdOut为无效输出。
 * @param [out] yOut:
 * 公式中的输出y，数据类型支持FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2，shape需要与x一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] mxscaleOut:
 * 公式中的输出mxscale_out，每个分组对应的量化尺度。数据类型支持FLOAT8_E8M0，shape维度支持2-8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] rstdOut:
 * 公式中的输出rstd，数据类型支持FLOAT，shape维度与输入x保持一致。
 * 支持空Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRmsNormDynamicMxQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, double epsilon,
    int64_t scaleAlg, char* roundMode, int64_t dstType, bool outputRstd, aclTensor* yOut,
    aclTensor* mxscaleOut, aclTensor* rstdOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRmsNormDynamicMxQuant的第二段接口，用于执行计算。
 *
 * 算子功能：执行RmsNorm+DynamicMxQuant计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnRmsNormDynamicMxQuantGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnRmsNormDynamicMxQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_RMS_NORM_DYNAMIC_MX_QUANT_H_