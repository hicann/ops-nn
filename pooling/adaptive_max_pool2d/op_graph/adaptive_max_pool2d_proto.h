/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_POOLING_ADAPTIVE_MAX_POOL2D_PROTO_H_
#define OPS_POOLING_ADAPTIVE_MAX_POOL2D_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
 * @brief Applies a 2D adaptive max pooling over an input signal composed of several input planes.
 * The output is of size H x W, for any input size.
 * @par Inputs:
 * One input, including:
 * @li x: A Tensor. Must be one of the following data types:
 *     float16, float32, float64. \n
 * @par Attributes:
 * @li output_size: A required list of 2 ints
 *    specifying the size (H,W) of the output tensor. \n
 * @par Outputs:
 * @li y: A Tensor. Has the same data type as "x".
 * @li argmax: A Tensor. Describing the index of outputs.
 * @par Third-party framework compatibility
 * Compatible with the Pytorch operator AdaptiveMaxPool2d.
 */
REG_OP(AdaptiveMaxPool2d)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(argmax, TensorType::IndexNumberType())
    .REQUIRED_ATTR(output_size, ListInt)
    .OP_END_FACTORY_REG(AdaptiveMaxPool2d)

} // namespace ge
#endif // OPS_POOLING_ADAPTIVE_MAX_POOL2D_PROTO_H_
