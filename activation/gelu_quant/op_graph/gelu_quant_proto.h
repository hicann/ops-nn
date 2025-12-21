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
 * \file gelu_quant_proto.h
 * \brief
 */
#ifndef OPS_ACTIVATION_GELU_QUANT_GRAPH_PLUGIN_GELU_QUANT_PROTO_H_
#define OPS_ACTIVATION_GELU_QUANT_GRAPH_PLUGIN_GELU_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The fusion operator of Gelu activation function and quantum quantization.
* @par Inputs:
* @li x: A tensor. Type is DT_FLOAT32, DT_FLOAT16, DT_BF16.
      Shape supports at least 2 dimensions (M,K1), and at most 8 dimensions.
* @li input_scale: An optional tensor. When quant_mode is "static",it is a required tensor.
*     When quant_mode is "dynamic", it is an optional tensor.
*     Data type is one of DT_FLOAT32, DT_FLOAT16, DT_BF16, and is consistent with x or has higher accuracy.
*     The shape can only be one-dimensional, and the size can only be the last axis of x or 1.
* @li input_offset: An optional tensor. It is an optional input when quant_mode is "static".
*     Data type is one of DT_FLOAT32, DT_FLOAT16, DT_BF16.
*     The shape and type should be the same as input_scale. It can also be null.
* @par Outputs:
* @li y: A tensor. Type is DT_INT8, DT_FLOAT8_E4M3_FN, DT_FLOAT8_E5M2, DT_HIFLOAT8. Shape size is the same as x.
* @li out_scale: A tensor. Type is DT_FLOAT32. Represents Scale used for quantization. The value is
      output only when quant_mode is dynamic.
      The shape of out_scale matches the shape of x across all dimensions except for the last dimension.
* @par Attributes:
* @li approximate: Optional string parameter. Which formula used for activation computation.
      Type is String. The value must be none or tanh. Defaults to none.
* @li quant_mode: Optional string parameter. Which formula used for quantized computation
      Type is String. The value must be dynamic or static. Defaults to dynamic.
* @li dst_type: Optional int32 parameter, specifying the output data type.
*     The value range is [DT_INT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8]. Defaults to DT_INT8.
* @li round_mode: Optional string parameter, specifying the cast type.
*     The value range is ["rint", "round", "hybrid"]. Defaults to "rint".
*/
REG_OP(GeluQuant)
    .INPUT(x, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(input_scale, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(input_offset, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_FLOAT8_E4M3_FN, DT_FLOAT8_E5M2, DT_HIFLOAT8}))
    .OUTPUT(out_scale, TensorType({DT_FLOAT32}))
    .ATTR(approximate, String, "none")
    .ATTR(quant_mode, String, "dynamic")
    .ATTR(dst_type, Int, DT_INT8)
    .ATTR(round_mode, String, "rint")
    .OP_END_FACTORY_REG(GeluQuant)

} // namespace ge

#endif