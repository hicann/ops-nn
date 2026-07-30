/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_SWIGLU_GROUP_GRAD_H_
#define OP_API_INC_LEVEL2_ACLNN_SWIGLU_GROUP_GRAD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSwigluGroupGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：完成ClampedSwiglu的反向梯度计算。从上游梯度gradY和前向输入x重算clamp mask与sigmoid，
 *           输出gradX与可选gradWeightOutOptional。
 *
 * 梯度公式：
 *   silu'(g̃) = s + f − f·s （数值改写，避免∞·0）
 *   m_g = I(g < c)（开区间约定）
 *   m_u = I(-c < u < c)（开区间约定）
 *   dg = gradY · silu'(g̃) · ũ · w_t · m_g · m_r
 *   du = gradY · f · w_t · m_u · m_r
 *   dw = Σ(gradY · yOrigin) along hidden（不含 clamp/group_index mask）
 *
 * @param [in] gradY: 上游梯度，shape (T, H)或(B, S, H)，dtype BF16/FP16/FP32。
 * @param [in] x: 前向输入，shape (T, 2H)或(B, S, 2H)，dtype 同gradY。
 * @param [in] weightOptional: 可选输入，MoE top-k路由权重，shape (T, 1)或(B, S, 1)，dtype FP32。
 *           必须与yOriginOptional同时提供或同时为空。
 * @param [in] yOriginOptional: 可选输入，前向输出y，shape (T, H)或(B, S, H)，dtype 同gradY；
 *           weight存在时，y应为已乘weight的前向输出。
 *           必须与weightOptional同时提供或同时为空。
 * @param [in] groupIndexOptional: 可选输入，各分组token数量，shape (G,)，G > 0，dtype INT64。支持空指针。
 * @param [in] clampLimit: 可选属性，截断门限值，dtype FP32。0.0 表示不截断。
 * @param [out] gradXOut: 梯度输出，shape (T, 2H)或(B, S, 2H)，dtype 同gradY/x。不支持空指针。
 * @param [out] gradWeightOutOptional: 可选输出，weight的梯度，shape (T, 1)或(B, S, 1)，dtype FP32。
 *           weightOptional非空时此参数必传非空指针；weightOptional为空时可传空指针。
 * @param [out] workspaceSize: 返回需要在Device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSwigluGroupGradGetWorkspaceSize(const aclTensor* gradY, const aclTensor* x,
                                                           const aclTensor* weightOptional,
                                                           const aclTensor* yOriginOptional,
                                                           const aclTensor* groupIndexOptional, float clampLimit,
                                                           aclTensor* gradXOut, aclTensor* gradWeightOutOptional,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnSwigluGroupGrad的第二段接口，用于执行计算。
 * @param [in] workspace: 在Device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在Device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSwigluGroupGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_SWIGLU_GROUP_GRAD_H_
