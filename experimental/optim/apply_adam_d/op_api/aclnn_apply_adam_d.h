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
 * @file aclnn_apply_adam_d.h
 * @brief ACLNN L2 API declaration for ApplyAdamD.
 */

#ifndef ACLNN_APPLY_ADAM_D_H_
#define ACLNN_APPLY_ADAM_D_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnApplyAdamDGetWorkspaceSize(const aclTensor* var, const aclTensor* m, const aclTensor* v,
                                                      const aclTensor* beta1Power, const aclTensor* beta2Power,
                                                      const aclTensor* lr, const aclTensor* beta1,
                                                      const aclTensor* beta2, const aclTensor* epsilon,
                                                      const aclTensor* grad, bool useLocking, bool useNesterov,
                                                      aclTensor* varOut, aclTensor* mOut, aclTensor* vOut,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnApplyAdamD(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_APPLY_ADAM_D_H_
