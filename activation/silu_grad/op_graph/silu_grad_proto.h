/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file silu_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SILU_GRAD_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SILU_GRAD_H_
#include "graph/operator_reg.h"

namespace ge { 

/**
* @brief Computes the gradient for the Silu of "x" .
* @par Inputs:
* Two inputs, including:
* @li dy: A tensor, which supports 1D-8D defaultly. Format support ND. Must be one of the following types: float16, bfloat16, float32.
* @li x: A tensor of the same type, shape and format as "dy".
* @par Outputs:
* dx: A tensor of the same type, shape and format as "dy".
* @par Third-party framework compatibility
* Compatible with the Torch operator SiluGrad
*/
REG_OP(SiluGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(SiluGrad)
}

#endif