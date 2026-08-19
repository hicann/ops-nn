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
 * \file dequant_situ_quant_proto.h
 * \brief
 */
#ifndef OPS_QUANT_DEQUANT_SITU_QUANT_PROTO_H_
#define OPS_QUANT_DEQUANT_SITU_QUANT_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Combine Dequant + Situ + Quant.

* @par Inputs:
* Seven inputs:
* @li x: Required tensor. INT8 for per-channel dequant path; INT32 for MoE grouped-matmul accumulator;
*        BF16 for pre-dequantized path; FLOAT16 for pre-dequantized path. Shape is (N..., H) for INT8
*        (dim > 1, H even) or [rows, width] for INT32/BF16/FLOAT16.
* @li weight_scale: Optional FP32. Per-channel or per-expert dequantization scale. Required for INT32 x.
* @li activation_scale: Optional FP32. Per-row dequantization scale (one value per row). Required for INT32 x.
* @li bias: Optional FP32. Dequantization bias, same shape as weight_scale.
* @li quant_scale: Optional FP32. Static quant scale or dynamic smooth scale.
* @li quant_offset: Optional FP32. Static quant offset.
* @li group_index: Optional INT64 [experts]. Per-expert consecutive routed row counts for MoE.

* @par Outputs:
* @li y: INT8 tensor. Last dim is x last dim / 2.
* @li y_scale: FP32 tensor. Per-row dynamic quant scale (meaningless for static mode).

* @par Attributes:
* @li beta: Float. The beta parameter for Situ activation. Default is 4.0.
* @li linear_beta: Float. The linear_beta parameter for Situ activation. When value <= 0, the linear_beta
* transformation is not applied. Default is 25.0.
* @li activate_left: Bool. Whether gate is the left half (true) or right half (false). Default is true.
* @li quant_type: String. The quant type to use: 'static' or 'dynamic', default is 'dynamic'.

* @attention Constraints:
* @li The last dimension of x must be even.
* @li INT8 path: weight_scale required, activation_scale/group_index must be absent, x dim > 1.
* @li INT32 path: weight_scale and activation_scale required, quant_scale/quant_offset must be absent,
*        x rank == 2, quant_type must be dynamic.
* @li BF16 path: weight_scale/activation_scale/bias/group_index must be absent, x rank == 2,
*        quant_type must be dynamic.
* @li FLOAT16 path: weight_scale/activation_scale/bias/group_index must be absent, x rank == 2,
*        quant_type must be dynamic.
* @li When quant_type is 'static', quant_scale must be provided.
* @li When quant_type is 'dynamic', quant_scale is optional (used as smoothScale).
*/
REG_OP(DequantSituQuant)
    .INPUT(x, TensorType({DT_INT8, DT_INT32, DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64, DT_INT64, DT_INT64, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(y_scale, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .ATTR(beta, Float, 4.0)
    .ATTR(linear_beta, Float, 25.0)
    .ATTR(activate_left, Bool, true)
    .ATTR(quant_type, String, "dynamic")
    .OP_END_FACTORY_REG(DequantSituQuant)
} // namespace ge

#endif // OPS_QUANT_DEQUANT_SITU_QUANT_PROTO_H_
