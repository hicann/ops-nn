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
* @brief
* For Atlas 200/300/500 Inference Product, Atlas Training Series Product,
  Atlas Inference Series Product, Ascend 610 AI Processor,
  the calculation formula is x*e^(0.851*x)*(x-|x|)/(1+e^(-1.702|x|)).
* For other chips, the calculation formula is x/(1+e^(-1.702*x)).

* @par Inputs:
* One input, including:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* bfloat16, float16, float32

* @par Outputs:
* y: A Tensor. Has the same type, format and shape as "x".
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator FastGelu
*/
REG_OP(FastGelu)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(FastGelu)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_