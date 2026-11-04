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
 * \file max_pool2d_with_mask_backward_proto.h
 * \brief Graph-mode registration of MaxPool2dWithMaskBackward.
 */
#ifndef MAX_POOL2D_WITH_MASK_BACKWARD_PROTO_H_
#define MAX_POOL2D_WITH_MASK_BACKWARD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Backward of MaxPool2dWithMask: scatter-add gradOutput to gradInput
 * according to the argmax mask recorded by the forward operator.
 *
 * @par Inputs:
 * @li x: An NCDHW tensor. Must be one of the following types: float16, float32, bfloat16.
 * @li grad: An NCDHW tensor with the forward output shape. Same type as x.
 * @li argmax: An NCDHW int32 tensor carrying the argmax mask of the forward pass.
 *
 * @par Attributes:
 * @li ksize: Required list of ints, pooling window size.
 * @li strides: Required list of ints, pooling stride.
 * @li pads: Required list of ints, pooling padding.
 * @li dilation: Optional list of ints, defaults to {1, 1, 1} (only 1 is supported).
 * @li ceil_mode: Optional bool, defaults to false.
 *
 * @par Outputs:
 * y: Gradient w.r.t. the forward input, same shape/type as x.
 */
REG_OP(MaxPool2dWithMaskBackward)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(argmax, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilation, ListInt, {1, 1, 1})
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPool2dWithMaskBackward)
} // namespace ge

#endif // MAX_POOL2D_WITH_MASK_BACKWARD_PROTO_H_
