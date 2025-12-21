/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_TRANS_QUANT_PARAM_V2_H
#define OP_API_INC_TRANS_QUANT_PARAM_V2_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现transQuantParamV2计算
 * @brief aclnnTransQuantParamV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] scale: 量化参数，数据类型支持： float32。
 * @param [in] offset: 量化参数，数据类型支持：float32。
 * @param [out] out: 计算结果，数据类型：uint64_t
 * @return aclnnStatus: 返回状态码
 */

ACLNN_API aclnnStatus aclnnTransQuantParamV2GetWorkspaceSize(
    const aclTensor* scale, const aclTensor* offset, const aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnTransQuantParamV2的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnTransQuantParamV2GetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus
aclnnTransQuantParamV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_TRANS_QUANT_PARAM_V2_H