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
 * \file adam_apply_one_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ADAM_APPLY_ONE_OPS_H_
#define OPS_OP_PROTO_INC_ADAM_APPLY_ONE_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief A fusion operator for bert lamb.

*@par Inputs:
* Ten inputs, including:
*@li input0: A ND Tensor specifying gradient. Support float16, float32, bfloat16.
*@li input1: A ND Tensor specifying second moment. Values must be greater than or equal to 0. Support float16, float32,
bfloat16.
*@li input2: A ND Tensor specifying first moment. Support float16, float32, bfloat16.
*@li input3: A ND Tensor specifying parameters to be updated. Support float16, float32, bfloat16.
*@li input4: A ND Tensor specifying the learning rate. Support float16, float32, bfloat16.
*@li mul0_x: A ND Tensor specifying beta1. Values must range from 0 to 1. Support float16, float32, bfloat16.
*@li mul1_x: A ND Tensor specifying 1 - beta1. Values must range from 0 to 1. Support float16, float32, bfloat16.
*@li mul2_x: A ND Tensor specifying beta2. Values must range from 0 to 1. Support float16, float32, bfloat16.
*@li mul3_x: A ND Tensor specifying 1 - beta2. Values must range from 0 to 1. Support float16, float32, bfloat16.
*@li add2_y: A ND Tensor ued for improving numerical stability. Support float16, float32, bfloat16. \n

*@par Outputs:
* Three outputs, including:
*@li output0: A ND Tensor specifying updated second moment. Support float16, float32, bfloat16.
*@li output1: A ND Tensor specifying updated first moment. Support float16, float32, bfloat16.
*@li output2: A ND Tensor specifying updated parameters. Support float16, float32, bfloat16. \n
*\n
* output0 = input0^2 * mul3_x + input1 * mul2_x
*\n
* output1 = input2 * mul0_x + input0 * mul1_x
*\n
* output2 = input3 - (output1 / sqrt(output0) + add2_y) * input4
*\n
* @par attention Constraints:
* @li All input shapes must can be broadcasted to the output shapes.
* @li All outputs must have the same shape and type as inputs.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(AdamApplyOne)
    .INPUT(input0, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input3, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input4, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(add2_y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output0, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(AdamApplyOne)
} // namespace ge

#endif // OPS_OP_PROTO_INC_ADAM_APPLY_ONE_OPS_H_
