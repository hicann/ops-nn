/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_flip.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_FLIP_H_
#define OP_API_INC_LEVEL2_ACLNN_FLIP_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFlip的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 对n维张量的指定维度进行反转（倒序），dims中指定的每个轴的计算公式：
 *
 * $$
 * \operatorname{out}(i_0, i_1,
 * \ldots, i_{n-1}) = \operatorname{input}(j_0, j_1, \ldots, j_{n-1})
 * $$
 *
 * 其中，$n$是输入张量的维度，$ j_k$ = $\operatorname{dimSize}(k)$ -1 - $i_k$，$\operatorname{dimSize}(k)$
 * 表示第$k$个轴的长度。
 *
 * 计算图：
 * ```mermaid
 * graph LR
 * 	A[(Self)] -->B([l0op::Contiguous])
 *     B -->D([l0op::Flip])
 *     G((dims)) --> D([l0op::ReverseV2])
 *     D -->R([l0op::ViewCopy])
 *     R --> J[(out)]
 * ```
 * @param [in] self: 待进行flip计算的入参。npu device侧的aclTensor，
 * 数据类型支持float16,float,int32,int16,int64,bool, int8, uint8,float64，complex64, complex128, 数据格式支持ND，
 * 支持非连续的Tensor。
 * @param [in] dims: 表示需要翻转的轴。
 * @param [in] out: flip计算的出参。npu device侧的aclTensor，
 * 数据类型支持float16,float,int32,int16,int64,bool, int8, uint8,float64, complex64, complex128，数据格式支持ND，
 * 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFlipGetWorkspaceSize(const aclTensor* self, const aclIntArray* dims, aclTensor* out,
                                                uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFlip的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnFlipGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFlip(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_FLIP_H_