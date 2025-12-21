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
 * \file logit_proto.h
 * \brief
 */

#ifndef LOGIT_PROTO_H
#define LOGIT_PROTO_H
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief The logit function is a mathematical operation for converting probability to numeric variable. \n

*@par Inputs:
* x: A tensor of type float, float16 or bfloat16. Input data for the probability to logit transformation.
* Shape support 0D ~ 8D. The format must be ND.

*@par Attributes:
* eps: The epslion of "x", an optional attribute, the type is float. Defaults to -1.0. \n

*@par Outputs:
* y: A tensor with the same type, shape, format as "x". Output data from probability to logit transformation. \n
*/

REG_OP(Logit)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(eps, Float, -1.0)
    .OP_END_FACTORY_REG(Logit)

} // namespace ge
#endif