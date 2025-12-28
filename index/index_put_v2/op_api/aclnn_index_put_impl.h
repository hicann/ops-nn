/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_INDEX_PUT_H_
#define OP_API_INC_INDEX_PUT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnIndexPutImpl的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] selfRef: npu device侧的aclTensor，
 * 数据类型支持FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT8、BOOL、DOUBLE，支持非连续的Tensor，数据格式支持ND。
 * @param [in] indices: npu device侧的aclTensorList，表示索引切片用。
 * @param [in] values: npu device侧的aclTensor，索引处更新的值。
 * @param [in] accumulate: bool 类型属性， True表示在索引处叠加value值, False表示在索引处替换value值。
 * @param [in] unsafe: bool 类型属性。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIndexPutImplGetWorkspaceSize(aclTensor* selfRef, const aclTensorList* indices,
                                                        const aclTensor* values, const bool accumulate,
                                                        const bool unsafe, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor);

/**
 * @brief aclnnIndexPutImpl的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aaclnnIndexPutImplGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIndexPutImpl(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ADD_H_
