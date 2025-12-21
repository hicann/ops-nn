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
 * \file scatter_list_proto.cpp
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_SCATTER_LIST_H_
#define OPS_OP_PROTO_INC_SCATTER_LIST_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/*!
 * \file scatter_list_proto.cpp
 * \brief
 */

/**
* @brief Applies sparse updates into a variable reference.

* @par Inputs:
* @li var: The rewritten tensor list. Format is ND. Support 1D ~ 7D. Must be one of the following types:
* float16, bfloat16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
* @li indice: The index tensor. Format is ND. Support 1D ~ 2D, the first dimension should be equal to "updates", the
* second dimension should be 2. Must be one of the following types: int32, int64. Index out of bounds is not supported.
* @li updates: The source tensor. Format is ND. Support 2D ~ 8D, the first dimension should be equal to the number of
* tensors of "var", and the dimension of "axis" should not be greather than "var", other dimensions should be equal to
* "var". Must have the same type of "var".
* @li mask: The mask tensor. Format is ND. Support 1D, the first dimension should be equal to "updates". Type should be
* uint8.

* @par Attributes:
* @li reduce: An optional string. Defaults to "update".
* @li axis: An optional int. Defaults to -2.

* @par Outputs:
* var: A tensor list. Must have the same type, shape and format as input "var".

* @par Third-party framework compatibility
* Compatible with the Mindspore operator ScatterList.
*/
REG_OP(ScatterList)
    .DYNAMIC_INPUT(var, "T")
    .INPUT(indice, TensorType::IndexNumberType())
    .INPUT(updates, "T")
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .DYNAMIC_OUTPUT(var, "T")
    .ATTR(reduce, String, "update")
    .ATTR(axis, Int, -2)
    .DATATYPE(
        T, TensorType(
               {DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_UINT64}))
    .OP_END_FACTORY_REG(ScatterList)
}
#endif // OPS_OP_PROTO_INC_SCATTER_LIST_H_