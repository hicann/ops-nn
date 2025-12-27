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
 * \file gelu_proto.h
 * \brief
 */
#ifndef GELU_PROTO_H_
#define GELU_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The GELU activation function is x*Φ(x),
* where Φ(x) the standard Gaussian cumulative distribution function.

* @par Inputs:
* x: A Tensor. Must be one of the following types: bfloat16, float16, float32. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator Gelu.
*/
REG_OP(Gelu)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Gelu)
}
#endif