/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_DEQUANTIZE_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_DEQUANTIZE_PROTO_H_
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

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_DEQUANTIZE_PROTO_H_
