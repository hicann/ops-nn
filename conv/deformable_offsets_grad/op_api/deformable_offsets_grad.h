/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_IMAGE_DEFORMABLE_OFFSETS_GRAD_OP_HOST_OP_API_DEFORMABLE_OFFSETS_GRAD_H
#define OPS_IMAGE_DEFORMABLE_OFFSETS_GRAD_OP_HOST_OP_API_DEFORMABLE_OFFSETS_GRAD_H

#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<aclTensor*, aclTensor*> DeformableOffsetsGrad(
    const aclTensor* grad_output, const aclTensor* input, const aclTensor* offsets, const aclIntArray* stride,
    const aclIntArray* pads, const aclIntArray* kernel_size, const aclIntArray* dilations, bool modulated,
    int64_t deformable_groups, aclOpExecutor* executor);
} // namespace l0op

#endif // OPS_IMAGE_DEFORMABLE_OFFSETS_GRAD_OP_HOST_OP_API_DEFORMABLE_OFFSETS_GRAD_H
