/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_RENORM_H_
#define OP_API_INC_RENORM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRenorm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：返回一个张量，其中输入张量self沿维度dim的每个子张量都经过归一化，使得子张量的p范数低于maxNorm值。
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT, shape不超过8维。支持非连续Tensor，
 * 支持空Tensor传入，数据格式支持ND。
 * @param [in] p: host侧的aclScalar，数据类型支持FLOAT，表示范数，要求数值大于0。
 * @param [in] dim:
 * host侧的INT64值，表示指定求norm的维度方向，若不指定默认为-1，范围在[-self的维度数量，self的维度数量-1]。
 * @param [in] maxNorm: host侧的aclScalar，数据类型支持FLOAT，表示最大允许的归一化值，要求数值大于等于0。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16，数据类型、shape与self一致。
 * 支持非连续Tensor，支持空Tensor传入，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRenormGetWorkspaceSize(
    const aclTensor* self, const aclScalar* p, int64_t dim, const aclScalar* maxNorm, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnRenorm的第二段接口，用于执行计算
 *
 * 算子功能：返回一个张量，其中输入张量self沿维度dim的每个子张量都经过归一化，使得子张量的p范数低于maxNorm值。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnRenormGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRenorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceRenorm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：返回一个张量，其中输入张量selfRef沿维度dim的每个子张量都经过归一化，使得子张量的p范数低于maxNorm值。
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT, shape不超过8维。支持非连续Tensor，
 * 支持空Tensor传入，数据格式支持ND。
 * @param [in] p: host侧的aclScalar，数据类型支持FLOAT，表示范数，要求数值大于0。
 * @param [in] dim:
 * host侧的INT64值，表示指定求norm的维度方向，若不指定默认为-1，范围在[-self的维度数量，self的维度数量-1]。
 * @param [in] maxNorm: host侧的aclScalar，数据类型支持FLOAT，表示最大允许的归一化值，要求数值大于等于0。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceRenormGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* p, int64_t dim, const aclScalar* maxNorm, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceRenorm的第二段接口，用于执行计算
 *
 * 算子功能：返回一个张量，其中输入张量selfRef沿维度dim的每个子张量都经过归一化，使得子张量的p范数低于maxNorm值。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceRenormGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceRenorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_RENORM_H_