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
 * @param [in] rot: npu, 旋转矩阵，必选参数，数据类型与x保持一致。
 * device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] alpha: clamp需要限制的范围的缩放系数。
 * float，默认为0.0，不做clamp处理。
 * @param [out] yOut: 量化后的输出张量。
 * device侧的aclTensor，数据类型支持INT8、INT32，数据格式支持ND。
 * @param [out] scaleOut: 动态量化计算出的缩放系数。
 * device侧的aclTensor，数据类型支持FLOAT32，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnRotateQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* rot, float alpha, aclTensor* yOut, aclTensor* scaleOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRotateQuant的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由aclnnRotateQuantGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。调用该接口后，executor不再可用
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnRotateQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ROTATE_QUANT_H_
