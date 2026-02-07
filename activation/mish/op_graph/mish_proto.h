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
 * \file mish_proto.h
 * \brief
 */
#ifndef MISH_PROTO_H_
#define MISH_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes hyperbolic tangent of "x" element-wise .
    
*@par Inputs:
* One input:
* x: An ND tensor. support 1D ~ 8D. Must be one of the following types:
* float16, float32, bfloat16.
*
*@par Outputs:
* y: A Tensor. Has the same type as "x" .
*
*@par Third-party framework compatibility
* Compatible with TensorFlow operator Mish.
*/
    
REG_OP(Mish)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OP_END_FACTORY_REG(Mish)
    
}
#endif