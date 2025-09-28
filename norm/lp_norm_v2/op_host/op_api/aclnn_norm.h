/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_NORM_H_
#define OP_API_INC_NORM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 根据具体的计算流程，计算workspace大小
 * @domain aclnn_ops_infer
 * @param [in] self: npu device侧的aclTensor, 数据类型支持浮点数据类型
 * @param [in] p: host侧aclScalar, 数据类型支持浮点数据类型
 * @param [in] dim: npu device侧的aclIntArray, 代表对应的降维轴Axis的信息
 * @param [in] keepdim: host侧的bool, 代表是否保留对应dim维度的信息
 * @param [in] workspaceSize: 返回用户需要的npu device侧申请的workspace大小
 * @param [in] executor: 返回op执行器，包含算子计算流程
 * @return aclnnStatus: 返回状态码
 * */
ACLNN_API aclnnStatus aclnnNormGetWorkspaceSize(
    const aclTensor* self, const aclScalar* pScalar, const aclIntArray* dim, bool keepdim, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * aclnnNorm的第二段接口，用于执行计算
 * @param [in] workspace: 在npu device侧申请的workspace内存地址
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnNormGetWorkspaceSize获取
 * @param [in] executor: 返回op执行器，包含算子计算流程
 * @param [in] stream: acl stream流
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif