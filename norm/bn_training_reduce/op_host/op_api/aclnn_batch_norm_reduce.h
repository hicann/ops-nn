/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_BATCH_NORM_REDUCE_H_
#define OP_API_INC_BATCH_NORM_REDUCE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBatchNormReduce的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：对输入C轴以外的轴（N，H，W）进行求和及平方和。
 * 计算公式：如下所示（以4D输入为例）
 * 1. 输入x由（N1,N2,N3,N4）升维成（N1,N2,N3,N4,1）
 * $$
 * sum_i = \sum_{n=0}^{N-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{(n,i,h,w)}
 * $$
 * 
 * $$
 * squareSum_i = \sum_{n=0}^{N-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{(n,i,h,w)}^2
 * $$
 *
 * @param [in] x: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16。
 * 支持非连续的Tensor。
 * @param [out] sum: npu device侧的aclTensor，数据类型支持FLOAT。
 * @param [out] squareSum: npu device侧的aclTensor，数据类型支持FLOAT。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBatchNormReduceGetWorkspaceSize(
    const aclTensor* x, aclTensor* sum, aclTensor* squareSum, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnBatchNormReduce的第二段接口，用于执行计算。
 *
 * 算子功能：对输入C轴以外的轴（N，H，W）进行求和及平方和。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnBatchNormReduceGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的AscendCL Stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnBatchNormReduce(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_BATCH_NORM_REDUCE_H_