/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_ROTATE_QUANT_H
#define ACLNN_ROTATE_QUANT_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRotateQuant的第一段接口，计算并获取workspace大小
 * @domain aclnn_ops_infer
 *
 * @param [in] x: npu，待旋转量化的输入张量，必选参数。
 * device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] rotation: npu, 旋转矩阵，必选参数，数据类型与x保持一致。
 * device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] alpha: clamp需要限制的范围的缩放系数，可选参数。
 * device侧的aclTensor，shape为(1,)，数据类型为BFLOAT16，取值范围[0.0, 1.0]。
 * 如果为空或值为0.0，不做clamp处理。
 * @param [in] axis: 表示量化发生的轴（reduce轴）。
 * 目前只支持-1。
 * @param [in] roundMode: 表示数据转换的模式。
 * 支持"rint"、"round"、"floor"，默认值为"rint"。
 * @param [in] scaleAlg: 表示scale的计算算法。
 * 默认值为0，支持取值0、1、2。在mxfp4场景中只支持0和2。
 * @param [in] dstTypeMax: 表示量化目标类型的最大值。
 * 在MX场景（mxfp4/float8_xx）中，当scaleAlg=0时dstTypeMax必须为0.0；
 * 当scaleAlg=2时dstTypeMax必须在[6.0, 12.0]范围内。
 * 在INT场景中不校验此参数。
 * @param [in] trans: 表示输出y是否转置。
 * 目前只支持false。
 * @param [out] yOut: 量化后的输出张量。
 * device侧的aclTensor，MX场景数据类型支持FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2；
 * INT场景数据类型支持INT8、INT32。数据格式支持ND。
 * 当trans为false时，shape为(M, N)；当trans为true时，shape为(N, M)。
 * @param [out] scaleOut: 动态量化计算出的缩放系数。
 * device侧的aclTensor，MX场景数据类型支持FLOAT8_E8M0；INT场景数据类型支持FLOAT。数据格式支持ND。
 * MX场景shape为(*, CeilDiv(N,64), 2)；INT场景shape为(M,)。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnRotateQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* rotation, const aclTensor* alpha, int64_t axis, char* roundMode,
    int64_t scaleAlg, double dstTypeMax, bool trans, aclTensor* yOut, aclTensor* scaleOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnRotateQuant的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由aclnnRotateQuantGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。调用该接口后，executor不再可用
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnRotateQuant(void* workspace, uint64_t workspaceSize,
                                                                    aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ROTATE_QUANT_H_
