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
 * \file aclnn_index_copy.h
 * \brief
 */
#ifndef OP_API_INC_INDEX_COPY_H_
#define OP_API_INC_INDEX_COPY_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnIndexCopy第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：将index张量中元素值作为索引，针对指定轴dim，把source中元素复制到outRef的对应位置上。outRef中的其他元素从selfRef对应位置复制
 *
 * @param [in] selfRef: npu
 * device侧的aclTensor，输入数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、
 * DOUBLE、BOOL、COMPLEX128、COMPLEX64，支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [in] dim: host侧的int64类型。
 * @param [in] index: npu device侧的aclTensor，数据类型支持INT32、INT64,维度不能大于1，元素个数要求和张量source中
 * dim轴的大小相同。支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [in] source: npu
 * device侧的aclTensor，数据类型和selfRef一致,。支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [in] outRef: npu device侧的aclTensor, 数据类型、数据格式与selfRef相同。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIndexCopyGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index,
                                                     const aclTensor* source, aclTensor* outRef,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnIndexCopy的第二段接口，用于执行计算。
 *
 * 算子功能：将index张量中元素值作为索引，针对指定轴dim，把source中元素复制到outRef的对应位置上。并把其余元素从selfRef复制到outRef
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceIndexCopyGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIndexCopy(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream);

/**
 * @brief aclnnInplaceIndexCopy第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：将index张量中元素值作为索引，针对指定轴dim，把source中元素复制到selfRef的对应位置上。
 *
 * @param [in] selfRef: npu
 device侧的aclTensor，输入数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、
 * DOUBLE、BOOL、COMPLEX128、COMPLEX64，支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [in] dim: host侧的int64类型。
 * @param [in] index: npu device侧的aclTensor，数据类型支持INT32、INT64,维度不能大于1，元素个数要求和张量source中
 * dim轴的大小相同。支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [in] source: npu
 device侧的aclTensor，数据类型和selfRef一致,。支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。

 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceIndexCopyGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index,
                                                            const aclTensor* source, uint64_t* workspaceSize,
                                                            aclOpExecutor** executor);

/**
 * @brief aclnnInplacePut的第二段接口，用于执行计算。
 *
 * 算子功能：将index张量中元素值作为索引，针对指定轴dim，把source中元素复制到selfRef的对应位置上。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceIndexCopyGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceIndexCopy(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_INDEX_COPY_H_