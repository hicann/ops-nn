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
 * \file binary_cross_entropy_grad_proto.h
 * \brief
 */

#ifndef BINARY_CROSS_ENTROPY_GRAD_PROTO_H
#define BINARY_CROSS_ENTROPY_GRAD_PROTO_H
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs the backpropagation of BinaryCrossEntropy for training scenarios .

* @par Inputs:
* Four inputs, including:
* @li x: A 1D or 2D Tensor of type bfloat16, float16 or float32, specifying a predictive value.
* @li y: A 1D or 2D Tensor of type bfloat16, float16 or float32, indicating a tag.
* @li grad_output: A 1D or 2D Tensor of type bfloat16, float16 or float32, specifying the backpropagation gradient.
* @li weight: An optional 1D or 2D Tensor of type bfloat16, float16 or float32, specifying the weight . \n

* @par Attributes:
* reduction: A character string from "none", "mean", and "sum", specifying the gradient output mode. Defaults to "mean" . \n

* @par Outputs:
* output: A 1D or 2D Tensor. When "reduction" is set to "none", a Tensor with the same size as "x" is output. Otherwise, a Scalar is output . \n

* @attention Constraints:
* @li The value of "x" must range from 0 to 1.
* @li The value of "y" must be "0" or "1" . \n

* @par Third-party framework compatibility
* Compatible with PyTorch operator BCELossGrad.
*/
REG_OP(BinaryCrossEntropyGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(grad_output, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(BinaryCrossEntropyGrad)(BinaryCrossEntropy)

} // namespace ge
#endif