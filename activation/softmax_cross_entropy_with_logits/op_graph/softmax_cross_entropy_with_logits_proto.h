/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file softmax_v2_proto.h
 * \brief
 */
#ifndef OPS_SOFTMAX_CORSS_ENTROPY_WITH_LOGITS_H_
#define OPS_SOFTMAX_CORSS_ENTROPY_WITH_LOGITS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes softmax cross entropy cost and gradients to backpropagate. Broadcasting is supported.

*@par Inputs:
* Two inputs, including:
* @li features: A Tensor. Unnormalized scores from model output. Must be one of the following types: float16, bfloat16,
float32, double.
* A "batch_size * num_classes" matrix. The format must be ND or NHWC. Support 2D, 4D.
* When the shape of features is 4D, the data type supports bfloat16 in dynamic shape scenarios, and float32 in static
shape scenarios.
* @li labels: A Tensor. of the same type and format as "features". The true labels of the samples. A "batch_size *
num_classes" matrix.
* Has the type, format and shape as "features". \n

*@par Outputs:
* @li loss: A Tensor for per example loss (a "batch_size" vector). Has the same type and format as "features". Support
1D, 3D.
* If the shape of features and labels is not 4D, the shape of the loss is 1D.
* If the shape of features and labels is 4D, the shape of the loss is 3D.
* @li backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix).
* Has the same type, format and shape as "features". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SoftmaxCrossEntropyWithLogits.
*/
REG_OP(SoftmaxCrossEntropyWithLogits)
    .INPUT(features, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(labels, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(backprop, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SoftmaxCrossEntropyWithLogits)

} // namespace ge
#endif
