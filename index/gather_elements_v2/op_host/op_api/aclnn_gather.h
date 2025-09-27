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
 * \file aclnn_gather.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_GATHER_H_
#define OP_API_INC_LEVEL2_ACLNN_GATHER_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGather的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：对输入tensor中指定的维度进行数据聚集
 * 计算公式：$$ gather(X,index,d)_{i_0,i_1,\cdots,i_{d-1},i_{d+1},\cdots,i_{n-1}} =
 * X_{i_0,i_1,\cdots,i_{d-1},index_{i_d},i_{d+1},\cdots,i_{n-1}} $$
 *
 * @param [in] self: 待进行gather计算的入参,npu device侧的aclTensor，
 * 数据类型支持FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8、DOUBLE、UINT16、UINT32、UINT64、BOOL，数据格式支持ND,
 * 支持非连续的Tensor。
 * @param [in] dim: host侧的int64。
 * @param [in] index: 待进行gather计算的入参,npu device侧的aclTensor，
 * 数据类型支持INT32、INT64数据格式支持ND,
 * 支持非连续的Tensor。
 * @param [in] out: gather计算的出参。npu device侧的aclTensor，
 * 数据类型与self相同，数据格式支持ND，
 * 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGatherGetWorkspaceSize(
    const aclTensor* self, const int64_t dim, const aclTensor* index, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnGather的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGatherGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnGather(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_GATHER_H_