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
 * \file adam_apply_one_assign_proto.h
 * \brief 
 */
#ifndef OPS_OP_PROTO_INC_ADAM_APPLY_ONE_ASSIGN_H_
#define OPS_OP_PROTO_INC_ADAM_APPLY_ONE_ASSIGN_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Performs one step of Adam optimizer parameter update, with 10 inputs and 3 outputs. Supports broadcast.

* @par Inputs:
* Ten inputs, including:
* @li input0: A ND Tensor . Must be of type float16, float32.
* @li input1: A ND Tensor . Must be of type float16, float32.
* @li input2: A ND Tensor . Must be of type float16, float32.
* @li input3: A ND Tensor . Must be of type float16, float32.
* @li input4: A ND Tensor . Must be of type float16, float32.
* @li mul0_x: A ND Tensor . Must be of type float16, float32.
* @li mul1_x: A ND Tensor . Must be of type float16, float32.
* @li mul2_x: A ND Tensor . Must be of type float16, float32.
* @li mul3_x: A ND Tensor . Must be of type float16, float32.
* @li add2_y: A ND Tensor . Must be of type float16, float32.

* @par Outputs:
* input1: A ND Tensor. Intermediate result: input1 x mul2_x + input0^2 x mul3_x. Type float16, float32.
* input2: A ND Tensor. Intermediate result: mul0_x x input2 + mul1_x x input0. Type float16, float32.
* input3: A ND Tensor. Parameter update result: input3 - out_input2 / (sqrt(out_input1) + add2_y) x input4. Type float16, float32.

* @attention Constraints:
* All input and output data types must be consistent.
* The maximum input dimension is 8.
*/
REG_OP(AdamApplyOneAssign)
    .INPUT(input0, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input3, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input4, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(input2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(input3, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(AdamApplyOneAssign)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ADAM_APPLY_ONE_ASSIGN_H_
