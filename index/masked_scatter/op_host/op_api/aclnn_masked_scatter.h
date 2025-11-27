/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OP_API_INC_MASKED_SCATTER_H_
#define OP_API_INC_MASKED_SCATTER_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMaskedScatter的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：根据掩码mask张量中元素为True的位置，复制source中的元素到selfRef对应的位置上。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *   A[(selfRef)] -->B([l0op::Contiguous])
 *   B  -->D([l0op::MaskedScatter])
 *   D  -->J([l0op::ViewCopy])
 *   J   --> K[(selfRef)]
 *   A2[(mask)] -->B2([l0op::Contiguous])
 *   B2 --> C2([l0op::Cast])
 *   C2  -->D
 *   A1[(source)]-->B1([l0op::Contiguous])
 *   B1-->D
 * ```
 *
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] mask: npu device侧的aclTensor，数据类型支持BOOL、UINT8。shape不能大于selfRef，且需要和selfRef满足
 * broadcast关系。数据格式支持ND。
 * @param [in] source:npu device侧的aclTensor，数据类型需要与selfRef相同。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceMaskedScatterGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* mask, const aclTensor* source, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief aclnnMaskedScatter的第二段接口，用于执行计算。
 *
 * 算子功能：根据掩码mask张量中元素为True的位置，复制source中的元素到self对应的位置上。
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *   A[(selfRef)] -->B([l0op::Contiguous])
 *   B  -->D([l0op::MaskedScatter])
 *   D  -->J([l0op::ViewCopy])
 *   J   --> K[(selfRef)]
 *   A2[(mask)] -->B2([l0op::Contiguous])
 *   B2 --> C2([l0op::Cast])
 *   C2  -->D
 *   A1[(source)]-->B1([l0op::Contiguous])
 *   B1-->D
 * ```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnMaskedScatterGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceMaskedScatter(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MASKED_SCATTER_H_
