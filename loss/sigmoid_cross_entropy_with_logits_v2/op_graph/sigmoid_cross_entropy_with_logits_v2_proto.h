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
 * \file sigmoid_cross_entropy_with_logits_v2_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the sigmoid cross entropy loss of "predict" and "target".  Broadcasting is supported.

* @par Inputs:
* four inputs, including:
* @li predict: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the predictive value.
* @li target: A multi-dimensional Tensor, has the same dtype as "predict", specifying the target value.
* @li weight: An optional multi-dimensional Tensor, has the same dtype as "predict", specifying the weight value.
* @li pos_weight: An optional multi-dimensional Tensor, has the same dtype as "predict", specifying the pos weight value. \n

* @par Attributes:
* reduction: A optional string attr from ["none", "mean", "sum"],
*  specifying the reduction type to be applied to the output. Defaults to "mean". \n

* @par Outputs:
* loss: An ND Tensor, Sigmoid cross entropy between the predictive value and target value. Has the same dimensions as "predict". \n

* @par Third-party framework compatibility
* Compatible with PyTorch operator BCEWithLogitsLoss.
*/
REG_OP(SigmoidCrossEntropyWithLogitsV2)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(pos_weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsV2)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_H_