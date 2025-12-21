/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ADD_RELU_H_
#define OP_API_INC_ADD_RELU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAddRelu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成加法计算后，对张量结果进行处理：大于0则不变，小于0则修改为0。
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与other构成互相推导关系，shape需要与other满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与other一致。
 * @param [in] other: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与self构成互相推导关系，shape需要与self满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与self一致。
 * @param [in] alpha: host侧的aclScalar，数据类型需要可转换成self与other推导后的数据类型。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要是self与other推导之后可转换的数据类型，shape需要是self与other
 * broadcast之后的shape，数据格式支持ND，且数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAddReluGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclScalar* alpha,
                                               aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAdd的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAddRelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceAddRelu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成加法计算后，对张量结果进行处理：大于0则不变，小于0则修改为0。
 *
 * @param [in] selfRef: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与other构成互相推导关系，shape需要与other满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与other一致。
 * @param [in] other: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与self构成互相推导关系，shape需要与self满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与self一致。
 * @param [in] alpha: host侧的aclScalar，数据类型需要可转换成self与other推导后的数据类型。
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要是self与other推导之后可转换的数据类型，shape需要是self与other
 * broadcast之后的shape，数据格式支持ND，且数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceAddReluGetWorkspaceSize(aclTensor* selfRef, const aclTensor* other, aclScalar* alpha,
                                               uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAdd的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceAddRelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ADD_RELU_H_
