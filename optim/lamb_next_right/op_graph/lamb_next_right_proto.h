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
 * \file lamb_next_right_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LAMB_NEXT_RIGHT_OPS_H_
#define OPS_OP_PROTO_INC_LAMB_NEXT_RIGHT_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief A fusion operator for bert lamb (variance moment + denominator). \n

*@par Inputs:
* Six inputs (input_square, input_mul2 are full tensors; the rest are scalar coefficients):
*@li input_square: g. @li input_mul2: v. @li mul2_x: beta2. @li mul3_x: 1-beta2.
*@li truediv1_recip: 1/(1-beta2^t). @li add2_y: epsilon. \n

*@par Outputs:
*@li y1: next_v = v*beta2 + g^2*(1-beta2). @li y2: sqrt(next_v/(1-beta2^t)) + epsilon. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambNextRight)
    .INPUT(input_square, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_mul2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(truediv1_recip, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(LambNextRight)
} // namespace ge

#endif // OPS_OP_PROTO_INC_LAMB_NEXT_RIGHT_OPS_H_
