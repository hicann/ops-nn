/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _ACTIVATION_GRAPH_SOFTPLUS_PROTO_H_
#define _ACTIVATION_GRAPH_SOFTPLUS_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Computes softplus: log(exp(x) + 1) .

*@par Inputs:
* One input:
*x: A Tensor of type bfloat16, float16 or float32. Up to 8D . \n

*@par Outputs:
*y: The activations tensor. Has the same type and format as input "x"

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Softplus.
*/
REG_OP(Softplus)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(Softplus)

} // namespace ge
#endif // _ACTIVATION_GRAPH_SOFTPLUS_PROTO_H_
