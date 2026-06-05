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
 * \file rms_norm_grad_quant_proto.h
 * \brief
 */
#ifndef RMS_NORM_GRAD_QUANT_PROTO_H
#define RMS_NORM_GRAD_QUANT_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief RmsNormGradQuant operator interface implementation.

* @par Inputs
* @li dy: The gradient returned backward.
*         A Tensor. Support dtype: float32/float16/bfloat16, support format: ND.
* @li x: The input of the forward operator.
*        A Tensor. Support dtype: float32/float16/bfloat16, support format: ND.
* @li rstd: The intermediate computation result of the forward operator.
*           A Tensor. Support dtype: float32, support format: ND.
* @li gamma: The input of the forward operator.
*            A Tensor. Support dtype: float32/float16/bfloat16, support format: ND.
* @li scales_x: A required input tensor. Describing the weight of the quant operation.
*              Support dtype: float32/float16/bfloat16, support format: ND.
* @li offset_x: An optional input tensor. Describing the bias of the quant operation.
*               Support dtype: int32, support format: ND.
* @par Attributes
* @li quant_mode: Required string parameter. Which formula used for quantized computation.
*                 The type is String. Only support static.
* @li div_mode: A required attribute control static quant algorithm, the type is bool.
*               When true, scales will be divided by normalization output, otherwise, uses multiplication.
* @li dst_type: An optional attribute. Output y data type enum value. Support DT_HIFLOAT8, DT_INT8.
                Defaults to DT_INT8.
* @par Outputs
* @li dx: The gradient of input "x" after quantization, Has the same type and shape as "x".
*         A Tensor. Support dtype: hifloat8/int8, support format: ND.
* @li dgamma: The gradient of input "gamma". Has the same type and shape as "gamma".
*             A Tensor. Support dtype: float32, support format: ND.
*/
REG_OP(RmsNormGradQuant)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scales_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(offset_x, TensorType({DT_INT32}))
    .OUTPUT(dx, TensorType({DT_INT8, DT_HIFLOAT8}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT}))
    .ATTR(quant_mode, String, "static")
    .ATTR(div_mode, Bool, true)
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(RmsNormGradQuant)

} // namespace ge
#endif // RMS_NORM_GRAD_QUANT_PROTO_H
