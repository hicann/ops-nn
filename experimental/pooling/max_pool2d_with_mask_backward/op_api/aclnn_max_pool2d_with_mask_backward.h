// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// clang-format on

/*!
 * \file aclnn_max_pool2d_with_mask_backward.h
 * \brief MaxPool2dWithMaskBackward L2 interface.
 *
 * The public indices tensor is an INT8 storage container. Its payload must
 * match the encoding expected by the target platform and pooling path.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_MAX_POOL2D_WITH_MASK_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_MAX_POOL2D_WITH_MASK_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Build the execution graph and return workspace size.
 * @param gradOutput Device tensor matching the forward output shape. Supports
 *        FP16/FP32/BF16 according to target SoC, NCHW/CHW, and non-contiguous views.
 * @param self Forward input tensor; same dtype as gradOutput and shape-matched
 *        with gradInput.
 * @param indices INT8 storage container returned by the compatible paired forward op.
 * @param kernelSize Pooling kernel, length 1 or 2.
 * @param stride Pooling stride, length 0, 1 or 2; empty means kernelSize.
 * @param padding Symmetric padding, length 1 or 2.
 * @param dilation Pooling dilation, length 1 or 2; currently each value is 1.
 * @param ceilMode Whether output shape uses ceil mode.
 * @param gradInput Output tensor, same shape as self and same dtype as gradOutput.
 * @param workspaceSize Returned device workspace bytes.
 * @param executor Returned executor.
 */
ACLNN_API aclnnStatus aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);

/** @brief Execute a graph built by aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize. */
ACLNN_API aclnnStatus aclnnMaxPool2dWithMaskBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_MAX_POOL2D_WITH_MASK_BACKWARD_H_
