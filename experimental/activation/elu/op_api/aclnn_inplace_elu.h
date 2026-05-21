/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_INPLACE_ELU_H_
#define OP_API_INC_LEVEL2_ACLNN_INPLACE_ELU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInplaceElu 第一段接口：计算 workspace 大小并生成执行器。
 * @domain aclnn_ops_infer
 * @param [in] selfRef: 输入输出张量（原地），npu device 侧 aclTensor；dtype 支持 FLOAT、FLOAT16、BFLOAT16；
 * 支持非连续，格式 ND，维度不超过 8。
 * @param [in] alpha: 激活系数，host 侧 aclScalar，需可 cast 为 FLOAT。
 * @param [in] scale: 缩放系数，host 侧 aclScalar，需可 cast 为 FLOAT。
 * @param [in] inputScale: 输入缩放系数，host 侧 aclScalar，需可 cast 为 FLOAT。
 * @param [out] workspaceSize: 需在 device 侧申请的 workspace 字节数。
 * @param [out] executor: 执行器。
 * @return aclnnStatus。
 */
ACLNN_API aclnnStatus aclnnInplaceEluGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceElu 第二段接口：在指定 stream 上执行计算。
 * @param [in] workspace: device 侧 workspace 指针。
 * @param [in] workspaceSize: workspace 大小，由 aclnnInplaceEluGetWorkspaceSize 返回。
 * @param [in] executor: 执行器。
 * @param [in] stream: aclrtStream。
 * @return aclnnStatus。
 */
ACLNN_API aclnnStatus
aclnnInplaceElu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_INPLACE_ELU_H_
