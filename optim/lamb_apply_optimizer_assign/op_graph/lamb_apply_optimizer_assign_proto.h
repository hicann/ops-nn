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
 * \file lamb_apply_optimizer_assign_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LAMB_APPLY_OPTIMIZER_ASSIGN_OPS_H_
#define OPS_OP_PROTO_INC_LAMB_APPLY_OPTIMIZER_ASSIGN_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief A fusion operator for bert lamb: Adam moment update + bias-corrected update.

*@par Inputs:
* Twelve inputs, including:
*@li grad: A ND Tensor specifying gradient. Support float16, float32.
*@li inputv: A ND Tensor specifying second moment v. Support float16, float32.
*@li inputm: A ND Tensor specifying first moment m. Support float16, float32.
*@li input3: A ND Tensor specifying parameters. Support float16, float32.
*@li mul0_x: A ND Tensor specifying beta1. Support float16, float32.
*@li mul1_x: A ND Tensor specifying 1 - beta1. Support float16, float32.
*@li mul2_x: A ND Tensor specifying beta2. Support float16, float32.
*@li mul3_x: A ND Tensor specifying 1 - beta2. Support float16, float32.
*@li add2_y: A ND Tensor for numerical stability (epsilon). Support float16, float32.
*@li steps: A ND Tensor specifying the step count. Support float16, float32.
*@li do_use_weight: A ND Tensor specifying whether to apply weight decay. Support float16, float32.
*@li weight_decay_rate: A ND Tensor specifying the weight decay rate. Support float16, float32. \n

*@par Outputs:
* Three outputs, including:
*@li output0: A ND Tensor specifying the update. Support float16, float32.
*@li inputv: A ND Tensor specifying the updated second moment (in-place). Support float16, float32.
*@li inputm: A ND Tensor specifying the updated first moment (in-place). Support float16, float32. \n
*\n
* next_v = inputv * mul2_x + grad^2 * mul3_x
* next_m = inputm * mul0_x + grad * mul1_x
* update = (next_m / (1 - mul0_x^steps)) / (sqrt(next_v / (1 - mul2_x^steps)) + add2_y)
*          + input3 * weight_decay_rate * do_use_weight
*\n
* @par Restrictions:
* inputv and inputm are updated in-place (their input buffers are overwritten). The kernel computes and
* writes them at the full broadcast output shape, so inputv and inputm MUST have the same shape and that
* shape MUST equal the broadcast output shape (grad and input3 may be smaller and broadcast into it).
* Passing an inputv/inputm smaller than the broadcast output shape is invalid and rejected by tiling.
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambApplyOptimizerAssign)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(inputv, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(inputm, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input3, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(steps, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(do_use_weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_decay_rate, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output0, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(inputv, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(inputm, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(LambApplyOptimizerAssign)
} // namespace ge

#endif // OPS_OP_PROTO_INC_LAMB_APPLY_OPTIMIZER_ASSIGN_OPS_H_
