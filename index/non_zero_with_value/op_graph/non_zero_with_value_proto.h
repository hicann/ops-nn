/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file non_zero_with_value_proto.h
 * \brief
 */

#ifndef NON_ZERO_WITH_VALUE_PROTO_H_
#define NON_ZERO_WITH_VALUE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns the non-zero elements of a 2D input together with their coordinates and count.
* Static max-size outputs: value/index are allocated for the worst case (all elements non-zero),
* the real valid length is given by count.

* @par Inputs:
* x: A 2D Tensor. Supported dtypes: double, float32, float16, int8, uint8, int16, uint16, int32,
*     uint32, int64, uint64, bool. \n

* @par Attributes:
* @li transpose: An optional bool. Defaults to false. On <term>Ascend 950</term> (arch35) only
*     transpose=true (coordinate-major index layout) is supported; false is not delivered.
* @li dtype: An optional attribute specifying the output index data type. Defaults to DT_INT32.
*     On <term>Ascend 950</term> (arch35) only int32 is supported. \n

* @par Outputs:
* @li value: A Tensor. Has the same type as "x". Shape is [row*col] (static max-size);
*     the first count elements are the valid non-zero values in row-major order.
* @li index: A Tensor of type int32. Shape is [2*row*col] (static max-size). Logical layout is
*     [2, row*col] coordinate-major: [0, count) hold the row indices, [row*col, row*col+count) hold
*     the corresponding column indices.
* @li count: A scalar Tensor of type int32. Shape is [1]. count[0] = number of non-zero elements. \n

* @attention Constraints:
* @li Only rank-2 input is supported.
* @li Non-zero predicate follows IEEE semantics: nan and +-inf are counted as non-zero; -0.0 is zero.

* @par Third-party framework compatibility
* Compatible with the PyTorch operator NonZeroWithValue.
*/

REG_OP(NonZeroWithValue)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                          DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(value, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                               DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(index, TensorType({DT_INT32}))
    .OUTPUT(count, TensorType({DT_INT32}))
    .ATTR(transpose, Bool, false)
    .ATTR(dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(NonZeroWithValue)

} // namespace ge

#endif // NON_ZERO_WITH_VALUE_PROTO_H_
