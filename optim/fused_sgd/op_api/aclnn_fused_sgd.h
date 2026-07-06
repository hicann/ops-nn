/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_FUSED_SGD_H_
#define ACLNN_FUSED_SGD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFusedSgd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：融合SGD优化器，支持momentum、weight_decay、nesterov、maximize，
 *          将参数更新、动量更新、梯度缩放等操作融合为单个kernel。
 *
 * @param [in] paramsRef: device侧的aclTensorList，需要更新的参数列表。
 *   数据类型支持FLOAT、FLOAT16、BFLOAT16。数据格式支持ND。
 * @param [in] gradsRef: device侧的aclTensorList，梯度列表。数据类型、shape需要与params一致。
 * @param [in] momentumBufferListOptionalRef: device侧的aclTensorList，动量缓冲列表。数据类型、shape需要与params一致。
 * @param [in] gradScaleOptional: device侧的aclTensor（可选），梯度缩放因子。数据类型支持FLOAT。
 * @param [in] weightDecay: 权重衰减系数，数据类型FLOAT。
 * @param [in] momentum: 动量因子，数据类型FLOAT。
 * @param [in] lr: 学习率，数据类型FLOAT。
 * @param [in] dampening: 动量阻尼系数，数据类型FLOAT。
 * @param [in] nesterov: 是否启用Nesterov动量，数据类型BOOL。
 * @param [in] maximize: 是否最大化目标函数，数据类型BOOL。
 * @param [in] isFirstStep: 是否为第一个优化步，数据类型BOOL。
 * @param [out] workspaceSize: 返回用户在device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFusedSgdGetWorkspaceSize(const aclTensorList* paramsRef, const aclTensorList* gradsRef,
                                                    const aclTensorList* momentumBufferListOptionalRef,
                                                    const aclTensor* gradScaleOptional, float weightDecay,
                                                    float momentum, float lr, float dampening, bool nesterov,
                                                    bool maximize, bool isFirstStep, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief aclnnFusedSgd的第二段接口，用于执行计算。
 *
 * 算子功能：执行融合SGD优化器。
 * @param [in] workspace: 在device侧申请的workspace内存起址。
 * @param [in] workspaceSize: workspace大小，由aclnnFusedSgdGetWorkspaceSize获取。
 * @param [in] executor: op执行器。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFusedSgd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
