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
 * \file dynamic_quant_update_scatter_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_DYNAMIC_QUANT_UPDATE_SCATTER_H_
#define OPS_OP_PROTO_INC_DYNAMIC_QUANT_UPDATE_SCATTER_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Multiplies dynamic quantize and sparse updates into a variable reference .

* @par Inputs:
* Five inputs, including:
* @li var: An ND Tensor.
* Must be one of the following types: int8
* @li var_scale: An ND Tensor.
* Must be one of the following types: float
* @li indices: An ND Tensor.
* Must be one of the following types: int32，int64
* @li updates: An ND Tensor .
* Must be one of the following types: bfloat16，float16
* @li smooth_scales: An ND optional Tensor .
* Must be one of the following types: bfloat16，float16 \n

* @par Attributes:
* @li axis: An optional attribute. Defaults to 0, not support -1.
* @li reduce: A required attribute, can be "update". \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" .
* var_scale: A Tensor. Has the same type and format as input "var_scale" . \n

* @par Third-party framework compatibility
* Compatible with the Mindspore operator Scatter.
*/
REG_OP(DynamicQuantUpdateScatter)
    .INPUT(var, TensorType({DT_INT8}))
    .INPUT(var_scale, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(var, TensorType({DT_INT8}))
    .OUTPUT(var_scale, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(reduce, String)
    .ATTR(axis, Int, 0)
    .OP_END_FACTORY_REG(DynamicQuantUpdateScatter)

} // namespace ge

#endif // OPS_OP_PROTO_INC_DYNAMIC_QUANT_UPDATE_SCATTER_H_