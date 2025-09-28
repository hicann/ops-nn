/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_quant_update_scatter_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_DYNAMIC_QUANT_UPDATE_SCATTER_V2_H_
#define OPS_OP_PROTO_INC_DYNAMIC_QUANT_UPDATE_SCATTER_V2_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Multiplies dynamic asymmetric quantize and sparse updates into a variable reference.

* @par Inputs:
* Five inputs, including:
* @li x: A tensor which layout need to be setted BSH. Shape is (B, 1, H).
* The type support float16, bfloat16, format support ND.
* @li indices: A 1D tensor with shape (B). Indices of scatter. The type support int32, format support ND.
* @li var: A tensor with shape (B, S, 1, H). Target tensor to which the quantization results are scattered.
* The type support int4, format support ND.
* @li var_scale: A tensor with shape (B, S). Target tensor to which the quantization scales are scattered.
* The type support float32, format support ND.
* @li var_offset: A tensor with shape (B, S). Target tensor to which the quantization offsets are scattered.
* The type support float32, format support ND.

* @par Outputs:
* @li var: A tensor with shape (B, S, H). Result tensor after an in-place scatter.
* The type support int4, format support ND.
* @li var_scale: A tensor with shape (1, B, S). Result tensor after an in-place scatter.
* The type support float32, format support ND.
* @li var_offset: A tensor with shape (1, B, S). Result tensor after an in-place scatter.
* The type support float32, format support ND.
*/
REG_OP(DynamicQuantUpdateScatterV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(var, TensorType({DT_INT4}))
    .INPUT(var_scale, TensorType({DT_FLOAT}))
    .INPUT(var_offset, TensorType({DT_FLOAT}))
    .OUTPUT(var, TensorType({DT_INT4}))
    .OUTPUT(var_scale, TensorType({DT_FLOAT}))
    .OUTPUT(var_offset, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(DynamicQuantUpdateScatterV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_DYNAMIC_QUANT_UPDATE_SCATTER_V2_H_