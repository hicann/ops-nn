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
 * \file sparse_softmax_cross_entropy_with_logits.h
 * \brief
 */
#ifndef OPS_LOSS_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
#define OPS_LOSS_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Computes sparse softmax cross entropy cost and gradients to backpropagate.

*@par Inputs:
*Two inputs, including:
* @li features: A Tensor. Must be one of the following types: float16, float32, double, bfloat16.
*A "batch_size * num_classes" matrix.
* @li labels: A Tensor. Must be one of the following types: 'int32', 'int64'.
*batch_size vector with values in [0, num_classes).
*This is the label for the given minibatch entry. \n


*@par Outputs:
*@li loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "features".
*@li backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix).
Has the same type as "features" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator SparseSoftmaxCrossEntropyWithLogits.
*/
REG_OP(SparseSoftmaxCrossEntropyWithLogits)
    .INPUT(features, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(loss, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .OUTPUT(backprop, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .OP_END_FACTORY_REG(SparseSoftmaxCrossEntropyWithLogits)

} // namespace ge

#endif // OPS_LOSS_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_