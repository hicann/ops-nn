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
 * \file aclnn_foreach_addcmul_scalar_v2.h
 * \brief
 */

#ifndef OP_API_INC_ACLNN_FOREACH_ADDCMUL_SCALAR_V2_H_
#define OP_API_INC_ACLNN_FOREACH_ADDCMUL_SCALAR_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnForeachAddcmulScalarV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：对输入对张量列表x2和张量列表x3执行逐元素乘法，将结果乘以标量值scalar后将结果与张量列表x1执行逐元素加法。
 * 计算公式：
 * {\rm out}_i = x1_i + {\rm scalar} × x2_i × x3_i
 * @domain aclnnop_math
 * 参数描述：
 * @param [in]   input
 * 输入Tensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。数据格式支持ND。
  * @param [in]   input
 * 输入Tensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。数据格式支持ND。
  * @param [in]   input
 * 输入Tensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。数据格式支持ND。
  * @param [in]  input
 * 输入Scalar，数据类型支持FLOAT、FLOAT16、INT32。数据格式支持ND。
 * @param [in]   out
 * 输出Tensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnForeachAddcmulScalarV2GetWorkspaceSize(
    const aclTensorList *x1,
    const aclTensorList *x2,
    const aclTensorList *x3,
    const aclScalar *scalar,
    aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnForeachAddcmulScalarV2的第二段接口，用于执行计算。
 * 功能描述：对输入对张量列表x2和张量列表x3执行逐元素乘法，将结果乘以标量值scalar后将结果与张量列表x1执行逐元素加法。
 * 计算公式：
 * {\rm out}_i = x1_i + {\rm scalar} × x2_i × x3_i
 * @domain aclnnop_math
 * 参数描述：
 * param [in] workspace: 在npu device侧申请的workspace内存起址。
 * param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnForeachAddcmulScalarV2GetWorkspaceSize获取。
 * param [in] stream: acl stream流。
 * param [in] executor: op执行器，包含了算子计算流程。
 * return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnForeachAddcmulScalarV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
