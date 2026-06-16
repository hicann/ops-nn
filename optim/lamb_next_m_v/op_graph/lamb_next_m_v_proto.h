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
 * \file lamb_next_m_v_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LAMB_NEXT_M_V_OPS_H_
#define OPS_OP_PROTO_INC_LAMB_NEXT_M_V_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief A fusion operator for bert lamb (moment update + bias-corrected update). \n

*@par Inputs:
* Thirteen inputs (input_mul0/1/2/3/4 are full tensors; the rest are scalar coefficients):
*@li input_mul3: g^2. @li input_mul2: v. @li input_realdiv1: 1-beta2^t.
*@li input_mul1: g. @li input_mul0: m. @li input_realdiv0: 1-beta1^t. @li input_mul4: param.
*@li mul0_x: beta1. @li mul1_sub: 1-beta1. @li mul2_x: beta2. @li mul3_sub1: 1-beta2.
*@li mul4_x: weight_decay. @li add2_y: epsilon. \n

*@par Outputs:
*@li y1: update. @li y2: next_m. @li y3: next_v. @li y4: m_unbiased/(sqrt(v_unbiased)+eps). \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambNextMV)
    .INPUT(input_mul3, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_mul2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_realdiv1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_mul1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_mul0, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_realdiv0, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_mul4, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul1_sub, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul3_sub1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mul4_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y3, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y4, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(LambNextMV)
} // namespace ge

#endif // OPS_OP_PROTO_INC_LAMB_NEXT_M_V_OPS_H_
