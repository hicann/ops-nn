/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ELEMENTS_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ELEMENTS_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Uses "updates" to update tensor "data" by "indices". \n

* @par Inputs:
* Three inputs, including:
* @li data: A ND Tensor . \n
* Must be one of the following types: complex128, complex64, double, float32, float16, int16, int32, int64, int8,
qint32, qint8, quint8, uint16, uint32, uint64, uint8, bfloat16, complex32.
* @li indices: An ND Tensor of type int32 or int64, its value shoule be less than the numbers of elements in the
* data target axis.
* @li updates: An Tensor. Same shape as indices. format:NCHW, NHWC . \n
* Must be one of the following types: complex128, complex64, double, float32, float16, int16, int32, int64, int8,
qint32, qint8, quint8, uint16, uint32, uint64, uint8, bfloat16, complex32.

* @par Attributes:
* @li axis: An optional int. Defaults to 0.
* @li reduction: An optional string. Defaults to string "none" and can be
* "add" or "mul". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "data" . \n

* @attention Constraints:
* @li In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* you are advised to replace ScatterElements with ScatterElementsV2(When there are duplicate indexes, ScatterElementsV2 provides higher precision). \n

* @par Third-party framework compatibility
* Compatible with the ONNX operator ScatterElements.
*/
REG_OP(ScatterElements)
    .INPUT(data, TensorType::NumberType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(axis, Int, 0)
    .ATTR(reduction, String, "none")
    .OP_END_FACTORY_REG(ScatterElements)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ELEMENTS_OPS_H_
