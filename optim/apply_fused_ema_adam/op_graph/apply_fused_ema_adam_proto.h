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
 * \file apply_fused_ema_adam_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_IS_APPLY_FUSED_EMA_ADAM_H_
#define OPS_OP_PROTO_INC_IS_APPLY_FUSED_EMA_ADAM_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief A fusion operator for fused Ema-Adam.

* @par Inputs:
* Six inputs, including:
* @li grad: A Tensor with ND format specifying gradient. Support float16, float32, bfloat16.
* @li var: A Tensor with ND format specifying parameters to be updated. Support float16, float32, bfloat16.
* @li m: A Tensor with ND format specifying first moment. Support float16, float32, bfloat16.
* @li v: A Tensor with ND format specifying second moment. Support float16, float32, bfloat16.
* @li s: A Tensor with ND format specifying weight of EMA. Support float16, float32, bfloat16.
* @li step: A Tensor with ND format specifying time step. Support int64. \n

* @par Attributes:
* @li lr: A Float specifying the learning rate. Optional and defaults to "1e-3".
* @li ema_decay: A Float specifying ema decay. Must be between 0 and 1. Optional and defaults to "0.9999".
* @li beta1: A Float used for computing running averages of gradient. Optional and defaults to "0.9".
* @li beta2: A Float used for computing running averages of gradient's square. Optional and defaults to "0.999".
* @li eps: A Float ued for improving numerical stability. Optional and defaults to "1e-8".
* @li mode: An Integer must be 1 or 0. Set to "1" for AdamW and "0" for L2 regularization. Optional and defaults to "1".
* @li bias_correction: A bool. Set to "true" for bias correction and "false" for no correction. Optional and defaults to
"true".
* @li weight_decay: A Float specifying weight decay. Optional and defaults to "0".

* @par Outputs:
* Four outputs, including:
* @li var: A Tensor specifying updated parameters. Must be one of the following types: float16, float32, bfloat16.
* @li m: A Tensor specifying updated first moment. Must be one of the following types: float16, float32, bfloat16.
* @li v: A Tensor specifying updated second moment. Must be one of the following types: float16, float32, bfloat16.
* @li s: A Tensor specifying updated weight of EMA. Must be one of the following types: float16, float32, bfloat16. \n

* @attention Constraints:
* @li grad, var, m, s and v are required to be of the same datatype and shape.
*/
REG_OP(ApplyFusedEmaAdam)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(m, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(v, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(s, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(step, TensorType({DT_INT64}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(m, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(v, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(s, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(lr, Float, 1e-3f)
    .ATTR(ema_decay, Float, 0.9999)
    .ATTR(beta1, Float, 0.9)
    .ATTR(beta2, Float, 0.999)
    .ATTR(eps, Float, 1e-8f)
    .ATTR(mode, Int, 1)
    .ATTR(bias_correction, Bool, true)
    .ATTR(weight_decay, Float, 0.0)
    .OP_END_FACTORY_REG(ApplyFusedEmaAdam)

} // namespace ge

#endif // OPS_OP_PROTO_INC_IS_APPLY_FUSED_EMA_ADAM_H_
