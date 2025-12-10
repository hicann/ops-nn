/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef OP_API_INC_QUANTIZE_H_
#define OP_API_INC_QUANTIZE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：对输入向量进行量化
 *
 * @param [in] x: 输入Tensor，npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，数据格式支持ND。
 * @param [in] scales: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，数据格式支持ND。
 * @param [in] zeroPoints: npu device侧的aclTensor，数据类型支持INT32、INT8、UINT8、BFLOAT16，数据格式支持ND。
 * @param [in] axis: Host侧的整型，需要进行量化的轴，指定的轴不能超过输入x的维度数。
 * @param [in] dtype: 量化输出的数据类型，aclDataType类型，数据类型支持INT8、UINT8、INT32，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnQuantizeGetWorkspaceSize(
    const aclTensor* x, const aclTensor* scales, const aclTensor* zeroPoints, aclDataType dtype, int32_t axis,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnQuantize的第二段接口，用于执行计算。
 *
 * 算子功能：对输入Tensor进行量化。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnQuantizeGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnQuantize(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_QUANTIZE_H_
