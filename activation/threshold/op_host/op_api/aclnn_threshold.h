/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OP_API_INC_THRESHOLD_H_
#define OP_API_INC_THRESHOLD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnThreshold的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能: 对输入self进行阈值操作。当self中的元素大于threshold时，返回元素本身；否则，返回value。
 * @param [in] self: npu device侧的aclTensor，支持数据类型FLOAT、FLOAT16、INT32、INT8、UINT8、INT16、INT64。
 * 支持[非连续的Tensor](),数据格式支持ND[参考](#)
 * @param [in] threshold: npu device侧的aclScalar，支持数据类型FLOAT、FLOAT16、INT32、INT8、UINT8、INT16、INT64。
 * 表示对输入self的元素进行大小判断的阈值。
 * @param [in] value: npu device侧的aclScalar，支持数据类型FLOAT、FLOAT16、INT32、INT8、UINT8、INT16、INT64。
 * 表示输入self的元素小于阈值时的返回值。
 * @param [out] out: npu device侧的aclTensor，支持数据类型FLOAT、FLOAT16、INT32、INT8、UINT8、INT16、INT64。
 * 支持非连续的Tensor。且数据类型、数据格式和shape需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnThresholdGetWorkspaceSize(
    const aclTensor* self, const aclScalar* threshold, const aclScalar* value, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief aclnnThreshold的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnThresholdGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnThreshold(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceThreshold的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能: 对输入self进行阈值操作。当self中的元素大于threshold时，返回元素本身；否则，返回value。
 * @param [in] selfRef: npu device侧的aclTensor，支持数据类型FLOAT、FLOAT16、INT32、INT8、UINT8、INT16、INT64。
 * 支持[非连续的Tensor](),数据格式支持ND[参考](#)
 * @param [in] threshold: npu device侧的aclScalar，支持数据类型FLOAT、FLOAT16、INT32、INT8、UINT8、INT16、INT64。
 * 表示对输入self的元素进行大小判断的阈值。
 * @param [in] value: npu device侧的aclScalar，支持数据类型FLOAT、FLOAT16、INT32、INT8、UINT8、INT16、INT64。
 * 表示输入self的元素小于阈值时的返回值。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceThresholdGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* threshold, const aclScalar* value, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief aclnnInplaceThreshold的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceThresholdGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceThreshold(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_THRESHOLD_H_
