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
 * \file quantize_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Dequantizes the input tensor into a float tensor.
* [min_range, max_range] are float32 tensors that specify the range
* for "y".
* The "mode" attribute controls exactly which calculations are used to convert
* the float values to their quantized equivalents.
* @par Inputs:
* @li x: A Tensor. Must be one of the following types: qint8, quint8, qint32, quint16, qint16.
* Shape suport 1D ~ 8D. The format support ND or NC1HWC0.
* @li min_range: A Tensor of type float32.
* Specifies the minimum scalar value possibly produced for the input. Shape suport 1D ~ 8D.
* The format support ND or NC1HWC0. Has the same format as "x".
* @li max_range: A Tensor of type float32.
* Specifies the maximum scalar value possibly produced for the input. The format support ND or NC1HWC0.
* Shape suport 1D ~ 8D. "max_range" has the same shape as "min_range". Has the same format as "x". \n

* @par Attributes:
* mode: An optional string from: "MIN_COMBINED", "MIN_FIRST", and "SCALED".
* Defaults to "MIN_COMBINED" . \n

* @par Outputs:
* y: A dictionary of type float32. The format support ND or NC1HWC0.
* "y" has the same shape and format as "x". \n

* @attention Constraints:
* @li "min_range" and "max_range" have the same shapes.
* @li "x" and "y" have the same shapes. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Dequantize.
*/
REG_OP(Dequantize)
    .INPUT(x, TensorType(DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16))
    .INPUT(min_range, TensorType{DT_FLOAT})
    .INPUT(max_range, TensorType{DT_FLOAT})
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(mode, String, "MIN_COMBINED")
    .OP_END_FACTORY_REG(Dequantize)

/**
* @brief Quantizes the input, scales/zero_points support broadcasting operations. \n
*
* @par Inputs:
* @li x: A required tensor of type float16, float32 or bfloat16, specifying the input.
* @li scales: A required tensor of type float32 or bfloat16, specifying the scaling ratio.
* @li zero_points: An optional tensor of type int8, uint8, int32, float32 or bfloat16, specifying the offset. \n
*
* @par Attributes:
* @li dtype: A required string from "torch.qint8, torch.quint8, torch.qint32, torch.hifloat8, torch.float8_e5m2, torch.float8_e4m3fn", specifying the quantified dtype.
* @li axis: An optional int, must be in the range [-rank(input x), rank(input x)), describes which dimension of the input tensor x to be processed.
* When scales's dimension number and size is 1, axis makes no effect. Default is 1. \n
*
* @attention Constraints:
* @li When scales or zero_points's dtype is bfloat16, other inputs' dtype should also be bfloat16.
* @li Scales and zero_points's shape should be the same.
* @li Scales and zero_points's dimension number should be 1 or the same with x.
* @li When scales's dimension number is 1, it should be 1 or the same with the dimension specified by attributes[axis] of input x.
* @li When scales's dimension number is the same with x, the dimension specified by attributes[axis] shoule be 1 or the same with x's, others should be 1. \n
*
* @par Outputs:
* y: A required tensor of type int8, uint8, int32, hifloat8, float8_e5m2 or float8_e4m3fn, shape should be the same with input x,
* dtype should be consistent with attributes[dtype]. \n
*/
REG_OP(Quantize)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points, TensorType({DT_INT8, DT_UINT8, DT_INT32, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .REQUIRED_ATTR(dtype, String)
    .ATTR(axis, Int, 1)
    .OP_END_FACTORY_REG(Quantize)

} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
