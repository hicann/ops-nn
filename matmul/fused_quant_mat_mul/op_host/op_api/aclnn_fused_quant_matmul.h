/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_FUSED_QUANT_MATMUL_
#define OP_API_INC_FUSED_QUANT_MATMUL_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFusedQuantMatmul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：实现量化矩阵乘和Gelu计算融合。
 * @param [in] x1: matmul左矩阵，数据类型支持：int4, int8, int32。
 * @param [in] x2: matmul右矩阵，数据类型支持：int4, int8, int32。
 * @param [in] x1Scale: x1量化参数，数据类型支持：float32。
 * @param [in] x2Scale: x2量化参数，数据类型支持：bfloat16, float32。
 * @param [in] yScaleOptional: y量化参数，当前接口暂不支持该参数。
 * @param [in] x1OffsetOptional: 预留参数，当前接口暂不支持该参数。
 * @param [in] x2OffsetOptional: 量化参数，当前接口暂不支持该参数。
 * @param [in] yOffsetOptional: 预留参数，当前接口暂不支持该参数。
 * @param [in] biasOptional: 偏置，数据类型支持：int32, bfloat16, float16, float32。
 * @param [in] x3Optional: 后融合矩阵，当前接口暂不支持该参数。
 * @param [in] fusedOpType: 融合类型，支持："gelu_tanh", "gelu_erf"。
 * @param [in] groupSizeOptional: 量化参数，数据类型支持：int64。
 * @param [out] out: 计算结果，数据类型：float16, bfloat16。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnFusedQuantMatmulGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                         const aclTensor *x1Scale, const aclTensor *x2Scale,
                                                         const aclTensor *yScaleOptional, const aclTensor *x1OffsetOptional,
                                                         const aclTensor *x2OffsetOptional, const aclTensor *yOffsetOptional,
                                                         const aclTensor *biasOptional, const aclTensor *x3Optional, 
                                                         const char *fusedOpType, int64_t groupSizeOptional,
                                                         aclTensor *out, uint64_t *workspaceSize,
                                                         aclOpExecutor **executor);

/**
 * @brief aclnnFusedQuantMatmul的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnFusedQuantMatmulGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnFusedQuantMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FUSED_QUANT_MATMUL_
