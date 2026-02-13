/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file binary_cross_entropy_proto.h
 * \brief
 */

#ifndef BINARY_CROSS_ENTROPY_PROTO_H
#define BINARY_CROSS_ENTROPY_PROTO_H
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Creates a criterion that measures the Binary Cross Entropy between the target and the output. \n

* @par Inputs:
* Three inputs, including:
* @li x: A multi-dimensional tensor of type bfloat16, float16 or float32, specifying a predictive value. The value of "x" must range from 0 to 1.
* @li y: A multi-dimensional tensor of type bfloat16, float16 or float32, indicating a tag. The value of "y" must range from 0 to 1. Shape, dtype and format are the same as "x".
* @li weight: An optional multi-dimensional tensor, specifying the weight. If not null, shape, dtype and format are the same as "x" \n

* @par Attributes:
* reduction: A string specifying the reduction type to apply to the output, which must be one of: "none", "sum", or "mean". Defaults to "mean". \n

* @par Outputs:
* output: Output loss. Has the same dimension with the inputs. When "reduction" is set to "none", a tensor with the same size as "x" is output. Otherwise, a scalar is output. \n

* @par Third-party framework compatibility
* Compatible with PyTorch operator BCELoss.
*/
REG_OP(BinaryCrossEntropy)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(BinaryCrossEntropy)

} // namespace ge
#endif