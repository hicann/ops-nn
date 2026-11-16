/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_DEQUANT_BIAS_H_
#define OP_API_INC_ACLNN_DEQUANT_BIAS_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDequantBias的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：反量化偏置叠加：
 * $$
 * y = A \times weightScale \times activateScale
 * $$
 * （无 bias）
 * $$
 * y = (A + bias) \times weightScale \times activateScale
 * $$
 * （bias 为 INT32 时，先加 bias 再缩放）
 * $$
 * y = A \times weightScale \times activateScale + bias
 * $$
 * （bias 为 FLOAT、FLOAT16、BFLOAT16 时，先缩放再加 bias；
 * activateScale 缺省时按 1 处理）
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(x, weightScale, activateScale, bias)] -->B([L0::Contiguous])
 *     B --> C([L0::DequantBias])
 *     C --> D([L0::ViewCopy])
 *     D --> E[(out)]
 * ```
 *
 * @param [in] x: device侧的aclTensor，shape为[M, N]，数据类型支持INT32，支持非连续的Tensor。
 * @param [in] weightScale: device侧的aclTensor，shape为[N]，长度与x的N维长度一致，
 * 数据类型支持FLOAT、BFLOAT16，支持非连续的Tensor。
 * @param [in] activateScaleOptional: device侧的aclTensor，shape为[M]，长度与x的M维长度一致，
 * 数据类型支持FLOAT，支持非连续的Tensor。可选输入，传入nullptr时视为不参与计算。
 * @param [in] biasOptional: device侧的aclTensor，shape为[N]，长度与x的N维长度一致，
 * 数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32，支持非连续的Tensor。可选输入，传入nullptr时视为不参与计算。
 * @param [in] outputDtype: 表示输出out的数据类型，值为1表示FLOAT16，值为27表示BFLOAT16。
 * 当weightScale数据类型为FLOAT时该参数配置为1，为BFLOAT16时配置为27。
 * @param [in] out: device侧的aclTensor，shape为[M, N]，数据类型支持FLOAT16、BFLOAT16，支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDequantBiasGetWorkspaceSize(const aclTensor* x, const aclTensor* weightScale,
                                                       const aclTensor* activateScaleOptional,
                                                       const aclTensor* biasOptional, int64_t outputDtype,
                                                       const aclTensor* out, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnDequantBias的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDequantBias(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_DEQUANT_BIAS_H_
