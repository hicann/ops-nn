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
 * \file sigmoid_cross_entropy_with_logits_proto.h
 * \brief 定义 SigmoidCrossEntropyWithLogits 算子的原型注册接口
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the sigmoid cross entropy loss of "predict" and "target".

* @par Inputs:
* two inputs, including:
* @li predict: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the predictive value, format is ND.
* @li target: A multi-dimensional Tensor, has the same dtype, format and shape as "predict", specifying the target value. \n

* @par Outputs:
* loss: A multi-dimensional Tensor, Sigmoid cross entropy between the predictive value and target value.
*       Has the same dtype, format and shape as "predict". \n

* @par Third-party framework compatibility
* Compatible with TensorFlow operator SigmoidCrossEntropyWithLogits.
*/
REG_OP(SigmoidCrossEntropyWithLogits)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogits)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_H_
