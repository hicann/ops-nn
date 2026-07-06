/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_ELU_H_
#define OP_API_INC_LEVEL2_ACLNN_ELU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：对输入张量self中的每个元素x调用指数线性单元激活函数ELU，并将得到的结果存入输出张量out中。
 * 计算公式：如下
 * $$
 * ELU(x) =
 * \begin{cases}
 * scale \ast x, \quad x > 0\\
 * \alpha \ast scale \ast (exp(x \ast inputScale)-1), \quad x \leq 0
 * \end{cases}
 * $$
 *
 * 实现说明：如下
 *
 * 计算图：如下
 *
 * ```mermaid
 * graph LR
 *     A[(Self)] --> B([l0op::Contiguous])
 *     B --> C([l0op::Elu])
 *     C --> D([l0op::Cast])
 *     D --> E([l0op::ViewCopy])
 *     E --> F[(out)]
 *     G((alpha)) --> C
 *     H((scale)) --> C
 *     I((inputScale)) --> C
 * ```
 */

/**
 * @brief aclnnElu 第一段接口：计算 workspace 大小并生成执行器。
 * @domain aclnn_ops_infer
 * @param [in] self: 输入张量 x，npu device 侧 aclTensor；dtype 支持 FLOAT、FLOAT16、BFLOAT16；
 * 支持非连续，格式 ND，维度不超过 8。
 * @param [in] alpha: 激活系数 α，host 侧 aclScalar，需可 cast 为 FLOAT。
 * @param [in] scale: 缩放系数 scale，host 侧 aclScalar，需可 cast 为 FLOAT。
 * @param [in] inputScale: 输入缩放系数，host 侧 aclScalar，需可 cast 为 FLOAT。
 * @param [in] out: 输出张量，npu device 侧 aclTensor；dtype 需可由 self cast，shape 与 self 一致；
 * 支持非连续，格式 ND，维度不超过 8。
 * @param [out] workspaceSize: 需在 device 侧申请的 workspace 字节数。
 * @param [out] executor: 执行器。
 * @return aclnnStatus。
 */
ACLNN_API aclnnStatus aclnnEluGetWorkspaceSize(const aclTensor* self, const aclScalar* alpha, const aclScalar* scale,
                                               const aclScalar* inputScale, aclTensor* out, uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * @brief aclnnElu 第二段接口：在指定 stream 上执行计算。
 * @param [in] workspace: device 侧 workspace 指针。
 * @param [in] workspaceSize: workspace 大小，由 aclnnEluGetWorkspaceSize 返回。
 * @param [in] executor: 执行器。
 * @param [in] stream: aclrtStream。
 * @return aclnnStatus。
 */
ACLNN_API aclnnStatus aclnnElu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_ELU_H_
