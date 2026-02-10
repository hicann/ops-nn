/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_MAX_POOL3D_WITH_ARGMAX_H_
#define OP_API_INC_LEVEL2_ACLNN_MAX_POOL3D_WITH_ARGMAX_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMaxPool3dWithArgmax First segment interface. Calculate the workspace size based on the specific
 * calculation process. Function description: Calculates the aclnnMaxPool3dWithArgmax from the input tensor, at the
 * output contains 2 tensors: out and indices
 */

ACLNN_API aclnnStatus aclnnMaxPool3dWithArgmaxGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief A second interface of aclnnMaxPool3dWithArgmax, used to perform calculation.
 */

ACLNN_API aclnnStatus
aclnnMaxPool3dWithArgmax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_MAX_POOL3D_WITH_ARGMAX_H_