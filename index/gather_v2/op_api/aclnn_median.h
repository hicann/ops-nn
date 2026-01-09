/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MEDIAN_H_
#define OP_API_INC_MEDIAN_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMedian的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64，
 * 且数据类型与valuesOut相同。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND（[参考](#参考)）。
 * @param [in] valuesOut: device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64，
 * 且数据类型与self相同，valuesOut为一维Tensor，shape为(1,)。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND（[参考](#参考)）。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMedianGetWorkspaceSize(const aclTensor* self, aclTensor* valuesOut, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

/**
 * @brief aclnnMedian的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnMedianGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMedian(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnMedianDim的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: self：npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [in] dim：指定的维度，数据类型支持INT64
 * @param [in] keepDim：reduce轴的维度是否保留，数据类型支持BOOL
 * @param [in] valuesOut: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [in] indicesOut：npu
 * device侧的aclTensor，数据类型支持INT64。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMedianDimGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepDim,
                                                     aclTensor* valuesOut, aclTensor* indicesOut,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMedianDim的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnMedianDimGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMedianDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream);

/**
 * @brief aclnnNanMedian的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64，
 * 且数据类型与valuesOut相同。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND（[参考](#参考)）。
 * @param [in] out: device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64，
 * 且数据类型与self相同，out为一维Tensor，shape为(1,)。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND（[参考](#参考)）。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNanMedianGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor);

/**
 * @brief aclnnNanMedian的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnMedianGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNanMedian(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream);

/**
 * @brief aclnnNanMedianDim的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: self：npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [in] dim：指定的维度，数据类型支持INT64
 * @param [in] keepDim：reduce轴的维度是否保留，数据类型支持BOOL
 * @param [in] valuesOut: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [in] indicesOut：npu
 * device侧的aclTensor，数据类型支持INT64。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNanMedianDimGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepDim,
                                                        aclTensor* valuesOut, aclTensor* indicesOut,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnNanMedianDim的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnNanMedianDimGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNanMedianDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MEDIAN_H_