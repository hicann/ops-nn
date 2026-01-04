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
 * \file adam_apply_one_with_decay_assign_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ADAM_APPLY_ONE_WITH_DECAY_ASSIGN_OPS_H_
#define OPS_OP_PROTO_INC_ADAM_APPLY_ONE_WITH_DECAY_ASSIGN_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief A fusion operator for bert lamb.

*@par Inputs:
*Eleven inputs, including:
* @li input0: A ND tensor specifying gradient of the input. Support float16, float32, bfloat16.
* @li input1: A ND tensor specifying second moment. Support float16, float32, bfloat16.
* @li input2: A ND tensor specifying first moment. Support float16, float32, bfloat16.
* @li input3: A ND tensor specifying parameters to be updated. Support float16, float32, bfloat16.
* @li input4: A ND tensor specifying the learning rate. Values must be greater than 0. Shape size must be equal to 1.
Support float16, float32, bfloat16.
* @li mul0_x: A ND tensor specifying beta1. Values must be greater than 0. Shape size must be equal to 1. Support
float16, float32, bfloat16.
* @li mul1_x: A ND tensor specifying 1 - beta1. Values must range from 0 to 1. Shape size must be equal to 1. Support
float16, float32, bfloat16.
* @li mul2_x: A ND tensor specifying beta2. Values must be greater than 0. Shape size must be equal to 1. Support
float16, float32, bfloat16.
* @li mul3_x: A ND tensor specifying 1 - beta2. Values must range from 0 to 1. Shape size must be equal to 1. Support
float16, float32, bfloat16.
* @li mul4_x: A ND tensor specifying weight. Values must be greater than or equal to 0. Support float16, float32,
bfloat16.
* @li add2_y: A ND tensor ued for improving numerical stability. Values must be greater than 0. Support float16,
float32, bfloat16. \n

*@par Outputs:
*Three outputs, including:
* @li input1: A ND tensor specifying updated second moment. Support float16, float32, bfloat16.
* @li input2: A ND tensor specifying updated first moment. Support float16, float32, bfloat16.
* @li input3: A ND tensor specifying updated parameters. Support float16, float32, bfloat16. \n
*\n
* output0 = input0^2 * mul3_x + input1 * mul2_x
*\n
* output1 = input2 * mul0_x + input0 * mul1_x
*\n
* output2 = input3 - (output1 / sqrt(output0) + add2_y + input3 * mul4_x) * input4

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(AdamApplyOneWithDecayAssign)
    .INPUT(input0, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input3, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input4, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul4_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(add2_y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(input2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(input3, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(AdamApplyOneWithDecayAssign)
} // namespace ge

#endif // OPS_OP_PROTO_INC_ADAM_APPLY_ONE_WITH_DECAY_ASSIGN_OPS_H_
