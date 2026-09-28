/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_unsorted_segment_sum.h
 * \brief
 */

#ifndef OP_API_INC_ACLNN_UNSORTED_SEGMENT_SUM_H_
#define OP_API_INC_ACLNN_UNSORTED_SEGMENT_SUM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUnsortedSegmentSum的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能： 对一个张量分段求和，即对满足segmentIds[j...] == i的所有位置j...，将x[j...]累加得到y[i]，
 * 若某个段i没有对应的元素，则y[i] = 0。
 * @param [in] x: npu device侧的aclTensor，数据类型支持FLOAT32, FLOAT16, BFLOAT16, INT32, INT64,
 * UINT32, UINT64，支持非连续的Tensor，数据格式支持ND。
 * @param [in] segmentIds: npu device侧的aclTensor，数据类型支持INT32, INT64，支持非连续的Tensor，数据格式支持ND。
 * @param [in] numSegments: int64_t类型，分段个数，取值应大于0。
 * @param [in] out: npu device侧的aclTensor，数据类型与x一致，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnUnsortedSegmentSumGetWorkspaceSize(const aclTensor* x, const aclTensor* segmentIds,
                                                              int64_t numSegments, aclTensor* out,
                                                              uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnUnsortedSegmentSum的第二段接口，用于执行计算。
 * @domain aclnn_ops_infer
 * 算子功能： 对一个张量分段求和。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnUnsortedSegmentSumGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUnsortedSegmentSum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_UNSORTED_SEGMENT_SUM_H_
