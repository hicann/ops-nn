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
 * \file quant_update_scatter_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_MASKED_SCATTER_H_
#define OPS_OP_PROTO_INC_MASKED_SCATTER_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Multiplies quantize and sparse updates into a variable reference.

* @par Inputs:
* Five inputs, including:
* @li var: Src tensor. An ND tensor. Must be the following type: int8, float8_e4m3fn, float8_e5m2, hifloat8. Shape support 3-8D.
* @li indices: An ND tensor. Must be one of the following types: int32 or int64. Shape support 1D or 2D. Empty tensor not supported.
* When indices shape is 1D, indices value in range [0, var.shape(axis) - updates.shape(axis)),
* When indices shape is 2D, the first dim of indices value in range [0, var.shape(0)),
* the second dim of indices value in range [0, var.shape(axis) - updates.shape(axis)).
* @li updates: Update tensor. An ND tensor. Must be one of the following types: bfloat16, float16. The dim num should be same with var. Empty tensor not supported.
* updates.dim[0] = indices.dim[0], updates.dim[0] <= var.dim[0], updates.dim[axis] <= var.dim[axis], updates.dim[i] = var.dim[i] for i != 0 && i != axis.
* @li quant_scales: Quant scale tensor. An ND tensor. Must be one of the following types: bfloat16, float32. Shape support 1-8D. quant_scales.size() = updates.dim[quant_axis].
* @li quant_zero_points: Quant offset tensor. An ND optional tensor. Must be one of the following types: bfloat16, int32. Shape support 1-8D.
* quant_zero_points.size() = updates.dim[quant_axis].
*
* @par Attributes:
* @li reduce: A required attribute, only support value is "update", to do update operation. Dtype is string.
* @li axis: The dim to update. An optional attribute. Defaults to -2, dtype is int64, the value range is [1, len(updates.shape) -2] or [1 - len(updates.shape)，-2].
* @li quant_axis: An optional attribute. Defaults to -1, dtype is int64，currently support -1 or len(updates.shape) - 1.
* @li reciprocal_scale: False is "div", True is "mul". Defaults to False.
* @li round_mode: An optional string, specifying the cast type. The value range is ["rint", "round", "hybrid"]. Defaults to "rint".
*
* @par Outputs:
* var: An ND tensor. Has the same type and format as input "var".
*
* @attention Constraints:
* @li updates，quant_scales，quant_zero_points dtype should be one of the following combinations:[(BFLOAT16，BFLOAT16，BFLOAT16), (FLOAT16，FLOAT32，INT32)]
* @li Atlas Inference Series Product and Atlas Trainning Series Product and Atlas A2 Training Series Product/Atlas 800I A2 Inference Product and Atlas A3 Training Series Product:
* Output param var last dim should be 32B-aligned.
* @li When var is DT_INT8, DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN, round_mode only supports "rint".
* @li When var is DT_HIFLOAT8, round_mode supports "round" and "hybrid".
*
*/
REG_OP(QuantUpdateScatter)
    .INPUT(var, TensorType({DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(quant_scales, TensorType({DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_zero_points, TensorType({DT_BF16, DT_INT32}))
    .OUTPUT(var, TensorType({DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .REQUIRED_ATTR(reduce, String)
    .ATTR(axis, Int, -2)
    .ATTR(quant_axis, Int, -1)
    .ATTR(reciprocal_scale, Bool, false)
    .ATTR(round_mode, String, "rint")
    .OP_END_FACTORY_REG(QuantUpdateScatter)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MASKED_SCATTER_H_