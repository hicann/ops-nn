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
 * \file gelu_v2_proto.h
 * \brief
 */
#ifndef GELU_V2_PROTO_H_
#define GELU_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The GELUV2 activation function is x*Φ(x),
* where Φ(x) the standard Gaussian cumulative distribution function.

* @par Inputs:
* One input, including:
* x: A Tensor. Must be one of the following types: bfloat16, float16, float32.

* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Attributes:
* approximate: A optional string. The gelu approximation algorithm to use: 'none' or 'tanh', default is 'none'.

* @par Third-party framework compatibility:
* Compatible with the Pytorch operator Gelu.
*/
REG_OP(GeluV2)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(approximate, String, "none")
    .OP_END_FACTORY_REG(GeluV2)

}
#endif