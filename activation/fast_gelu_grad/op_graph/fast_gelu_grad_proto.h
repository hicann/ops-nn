/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Computes the gradient for the fast_gelu of "x" .

*@par Inputs:
*Two inputs, including:
* @li dy: A Tensor. Must be one of the following types: bfloat16, float16, float32
* @li x: A Tensor of the same type as "dy" . \n

*@par Outputs:
*z: A Tensor. Has the same type as "dy".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator FastGeluGrad
*/
REG_OP(FastGeluGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(FastGeluGrad)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_