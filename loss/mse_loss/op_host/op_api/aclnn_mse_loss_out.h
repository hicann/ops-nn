/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MSE_LOSS_OUT_H_
#define OP_API_INC_MSE_LOSS_OUT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMseLossOut的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：计算输入x和目标y中每个元素之间的均方误差。
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16，shape需要与target满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] target: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16，shape需要与self满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] reduction: host侧的int64，指定要应用到输出的缩减，支持 0('none') | 1('mean') | 2('sum')。
 * 'none' 表示不应用减少，'mean' 表示输出将被reduce 0轴求均值，'sum' 表示输出将被reduce 0轴求和。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16。支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMseLossOutGetWorkspaceSize(const aclTensor* self, const aclTensor* target, int64_t reduction,
                                                      aclTensor* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief aclnnMseLossOut的第二段接口，用于执行计算。
 *
 * 算子功能：计算输入x和目标y中每个元素之间的均方误差。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnMseLossOutGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMseLossOut(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MSE_LOSS_OUT_H_
