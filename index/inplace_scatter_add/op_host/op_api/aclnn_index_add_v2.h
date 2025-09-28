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
 * \file aclnn_index_add_v2.h
 * \brief
 */
#ifndef OP_API_INC_INDEX_ADD_V2_H_
#define OP_API_INC_INDEX_ADD_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnIndexAddV2的第一段接口，根据具体的计算流程，计算workspace大小。该接口暂不对外开放，仅供PTA调用。
 * @domain aclnn_ops_infer
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、INT32、FLOAT16、DOUBLE、INT16、INT8、UINT8、INT64、BOOL、BFLOAT16，
 * 支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] dim: host侧的整数，数据类型支持INT64。
 * @param [in] index: npu device侧的aclTensor，数据类型支持INT64、INT32，支持非连续的Tensor，数据格式支持ND，数据维度仅支持1维。
 * @param [in] source: npu device侧的aclTensor，数据类型支持与self一致，支持非连续的Tensor，数据格式支持ND，shape除了dim轴其他轴需要与self相同。
 * @param [in] alpha: host侧的aclScalar，数据类型与self一致。
 * @param [in] mode: host侧的整数，数据类型支持INT64。
 * @param [in] out: npu device侧的aclTensor，shape需要与self的shape一致，数据类型需要和self一致。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIndexAddV2GetWorkspaceSize(const aclTensor* self, const int64_t dim, const aclTensor* index,
                                                      const aclTensor* source, const aclScalar* alpha, int64_t mode, aclTensor* out,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnIndexAddV2的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnIndexAddV2GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIndexAddV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif