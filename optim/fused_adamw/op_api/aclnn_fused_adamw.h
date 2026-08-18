/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_fused_adamw.h
 * \brief
 */

#ifndef _ACLNN_FUSED_ADAMW_H_
#define _ACLNN_FUSED_ADAMW_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFusedAdamw的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：融合applyadamw优化器，支持params、grads、expAvgs、expAvgSqs、maxExpAvgSqs、stateSteps
 *          将参数更新、动量更新、梯度缩放等操作融合为单个kernel。
 *
 * @param [in] paramsRef: device侧的aclTensorList，需要更新的参数列表。
 *   数据类型支持FLOAT、FLOAT16、BFLOAT16。数据格式支持ND。
 * @param [in] grads: device侧的aclTensorList，梯度列表。数据类型、shape需要与params一致。
 * @param [in] expAvgsRef: device侧的aclTensorList，优化器一阶矩。数据类型、shape需要与params一致。
 * @param [in] expAvgSqsRef: device侧的aclTensorList，优化器二阶矩。数据类型、shape需要与params一致。
 * @param [in] maxExpAvgSqsRef: device侧的aclTensorList，保存最大二阶矩。数据类型、shape需要与params一致。
 * @param [in] stateSteps: device侧的aclTensorList，迭代次数。数据类型、shape需要与params一致。
 * @param [in] gradScaleOptional: device侧的aclTensor，可选输入，梯度因数。数据类型、shape需要与params一致。
 * @param [in] foundInfOptional: device侧的aclTensor，可选输入，等于1时停止更新。数据类型、shape需要与params一致。
 * @param [in] lr: 学习率，数据类型DOUBLE。
 * @param [in] beta1: 一阶矩衰减系数，数据类型DOUBLE。
 * @param [in] beta2: 二阶矩衰减系数，数据类型DOUBLE。
 * @param [in] weightDecay: 权重衰减系数，数据类型DOUBLE。
 * @param [in] eps: 防止除数为0，数据类型DOUBLE。
 * @param [in] amsgrad: 是否使用算法的AMSGrad变量，数据类型BOOL。
 * @param [in] maximize: 是否最大化参数，数据类型BOOL。
 * @param [out] workspaceSize: 返回用户在device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFusedAdamwGetWorkspaceSize(
    const aclTensorList* paramsRef, const aclTensorList* grads, const aclTensorList* expAvgsRef,
    const aclTensorList* expAvgSqsRef, const aclTensorList* maxExpAvgSqsRef, const aclTensorList* stateSteps,
    const aclTensor* gradScaleOptional, const aclTensor* foundInfOptional, double lr, double beta1, double beta2,
    double weightDecay, double eps, bool amsgrad, bool maximize, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFusedAdamw的第二段接口，用于执行计算。
 *
 * 算子功能：执行融合SGD优化器。
 * @param [in] workspace: 在device侧申请的workspace内存起址。
 * @param [in] workspaceSize: workspace大小，由aclnnFusedSgdGetWorkspaceSize获取。
 * @param [in] executor: op执行器。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFusedAdamw(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // _ACLNN_FUSED_ADAMW_H_
