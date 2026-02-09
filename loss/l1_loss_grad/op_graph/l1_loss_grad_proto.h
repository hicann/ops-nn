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
 * \file l1_loss_grad_proto.h
 * \brief
 */
#ifndef L1_LOSS_GRAD_OP_PROTO_H_
#define L1_LOSS_GRAD_OP_PROTO_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Computes l1_loss_grad or l1_loss_backward.

* @par Inputs:
* Three inputs, including:
* @li grads: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* Required.
* @li predict: A Tensor. Has the same type as "grads". Required.
* @li label: A Tensor. Has the same type as "grads". Required. \n

* @par Attributes:
* reduction: An optional attribute of type String. Defaults to "mean". \n

* @par Outputs:
* y: A Tensor. Has the same type as "grads". \n

* @attention Constraints:
* @li In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* broadcasting is not allowed between input grads, input predict and input label, the three inputs must have the same shape. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator L1LossGrad.
*/
REG_OP(L1LossGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(L1LossGrad)
}
#endif // L1_LOSS_GRAD_OP_PROTO_H_