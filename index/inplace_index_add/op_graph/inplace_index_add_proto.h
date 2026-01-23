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
 * \file inplace_index_add_proto.h
 * \brief
 */
#ifndef INPLACE_INDEX_ADD_PROTO_H_
#define INPLACE_INDEX_ADD_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Add updates to var according to axis and indices.

* @par Inputs:
* Three inputs, including:
* @li var: A Tensor. Must be one of the following types:
*     double, float16, float32, int16, int32, int8, uint8, int64, bool, bfloat16.
* @li indices: A Tensor. The indices of updates to select from. Its type should be int32 or int64.
    The indices should be 1-dimensional.
* @li updates: A Tensor of the same type as "var".
* @li alpha: An optional Tensor of the same type as "var". A scaling factor to updates. \n

* @par Attributes:
* axis: An required int to specify the axis to perform indices add. \n

* @par Outputs:
* var: A Tensor. Same as input "var".

* @attention Constraints:
* @li The shape values of var and updates should be the same in other dimensions except the axis dimension.
* @li The shape size of indices should be the same as updates in the axis dimension.
* @li The indices cannot contain negative values, and its value range cannot exceed the shape value range of var in the axis dimension.

* @par Third-party framework compatibility
* Compatible with the Pytorch operator index_add_.
*/
REG_OP(InplaceIndexAdd)
    .INPUT(var, TensorType({DT_INT16, DT_INT32, DT_INT8,
                            DT_UINT8, DT_FLOAT32, DT_FLOAT16, DT_DOUBLE,
                            DT_INT64, DT_BOOL, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(updates, TensorType({DT_INT16, DT_INT32, DT_INT8,
                                DT_UINT8, DT_FLOAT32, DT_FLOAT16, DT_DOUBLE,
                                DT_INT64, DT_BOOL, DT_BF16}))
    .OPTIONAL_INPUT(alpha, TensorType({DT_INT16, DT_INT32, DT_INT8,
                                       DT_UINT8, DT_FLOAT32, DT_FLOAT16, DT_DOUBLE,
                                       DT_INT64, DT_BOOL, DT_BF16}))
    .OUTPUT(var, TensorType({DT_INT16, DT_INT32, DT_INT8,
                             DT_UINT8, DT_FLOAT32, DT_FLOAT16, DT_DOUBLE,
                             DT_INT64, DT_BOOL, DT_BF16}))
    .REQUIRED_ATTR(axis, Int)
    .OP_END_FACTORY_REG(InplaceIndexAdd)

} // namespace ge
#endif // INPLACE_INDEX_ADD_PROTO_H_