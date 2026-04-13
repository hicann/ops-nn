/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file aclnn_softsign_backward.h
 * @brief SoftsignGrad ACLNN L2 API 接口声明
 *
 * 算子: SoftsignGrad
 * 公式: output = gradients / (1 + |features|)^2
 *
 * 接口名称: aclnnSoftsignBackward
 * (用户要求使用 SoftsignBackward 而非 SoftsignGrad 作为 ACLNN 接口名)
 */

#ifndef ACLNN_SOFTSIGN_BACKWARD_H_
#define ACLNN_SOFTSIGN_BACKWARD_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnSoftsignBackward 所需的 workspace 大小
 * @param gradients [in] 上游梯度张量
 * @param features [in] Softsign 前向输入特征张量
 * @param out [in] 输出梯度张量
 * @param workspaceSize [out] 返回所需 workspace 大小
 * @param executor [out] 返回执行器
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnSoftsignBackwardGetWorkspaceSize(
    const aclTensor *gradients,
    const aclTensor *features,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief 执行 SoftsignBackward 算子计算
 * @param workspace [in] workspace 内存地址
 * @param workspaceSize [in] workspace 大小
 * @param executor [in] 执行器
 * @param stream [in] ACL 流
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnSoftsignBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SOFTSIGN_BACKWARD_H_
