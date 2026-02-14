/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SCATTER_H_
#define OPS_OP_PROTO_INC_SCATTER_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Applies sparse updates into a variable reference.

* @par Inputs:
* @li var: The rewritten tensor. Format is ND. Support 2D ~ 8D, when axis is -1 the last dim of var should be 32B align.
* Must be one of the following types:float16, float32, int32, int8, uint8, bfloat16.
* @li indices: The index tensor. Format is ND. Support 1D ~ 2D, when discrete, 1-dim of indices should be 2.
* Must be one of the following types: int32, int64.
* Index out of bounds is not supported.
* @li updates: The source tensor. Format is ND. The number of dimensions should be equal to "var", and the dimension of
* "axis" should not be greather than "var", other dimensions should be equal to "var"
* and 0-dim of updates should be equal 0-dim of indices. Must have the same type of "var".

* @par Attributes:
* @li reduce: An required string. Can be "none" or "update".
* @li axis: An optional int. Defaults to -1, if axis < 0, it should be -1 or -2.

* @par Outputs:
* var: An ND tensor. Must have the same type, format and shape as input "var".

* @par Third-party framework compatibility
* Compatible with the Mindspore operator Scatter.
*/
REG_OP(Scatter)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT8_E8M0}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT8_E8M0}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT8_E8M0}))
    .REQUIRED_ATTR(reduce, String)
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(Scatter)

} // namespace ge

#endif // OPS_OP_PROTO_INC_RSQRT_H_