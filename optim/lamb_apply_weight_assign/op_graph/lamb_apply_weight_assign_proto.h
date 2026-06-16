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
 * \file lamb_apply_weight_assign_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LAMB_APPLY_WEIGHT_ASSIGN_OPS_H_
#define OPS_OP_PROTO_INC_LAMB_APPLY_WEIGHT_ASSIGN_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief A fusion operator for bert lamb: trust-ratio weight update.

*@par Inputs:
* Five inputs, including:
*@li input0: A ND Tensor specifying the weight norm (scalar). Support float16, float32.
*@li input1: A ND Tensor specifying the gradient norm (scalar). Support float16, float32.
*@li input2: A ND Tensor specifying the learning rate (scalar). Support float16, float32.
*@li input3: A ND Tensor specifying the update. Support float16, float32.
*@li input_param: A ND Tensor specifying the parameters. Support float16, float32. \n

*@par Outputs:
*input_param: A ND Tensor specifying the updated parameters (in-place). Support float16, float32. \n
*\n
* ratio = where(input0 > 0, where(input1 > 0, input0 / input1, 1.0), 1.0)
* input_param = input_param - ratio * (input3 * input2)
*\n
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambApplyWeightAssign)
    .INPUT(input0, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input3, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_param, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(input_param, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(LambApplyWeightAssign)
} // namespace ge

#endif // OPS_OP_PROTO_INC_LAMB_APPLY_WEIGHT_ASSIGN_OPS_H_
