/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_GATHER_V2_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_GATHER_V2_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Gather slices from "x" according to "indices" by corresponding axis, produces a output tensor
* with shape(x.shape[:axis]+indices.shape[batch_dims:]+x.shape[axis+1:]). When the impl_mode is set
* as "support out of bound index", if the indices data is out of bound, the corresponding results
* will be set as 0. Otherwise, an aic_error will occur.

* @par Inputs:
* @li x: A ND (Support 1D~8D) tensor. Must be one of the following types: complex128, complex64, float64, float32, float16,
*     int16, int32, int64, int8, qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8,
*     bool, string, bfloat16.
* @li indices: A ND (Support 1D~8D) tensor of type int32 or int64.
* @li axis: A Scalar with type as int32 or int64. Must be in the range [-rank(input_tensor), rank(input_tensor)).

* @par Attributes:
* @li batch_dims: An optional int which means the number of data to be deal with. Defaults to 0.
* @li is_preprocessed: An optional bool. Whether to preprocess, wihch is true means need to be preprocess and false means not. Defaults to false.
* @li negative_index_support: An optional bool, which is true means support index is negative, and false means not. Defaults to false.

* @par Outputs:
* y: A ND Tensor which has the same type as "x".

* @attention Constraints:
* @li Value in indices must be in range [0, x.shape[axis]).
* @li Default mode is HIGH_PERCISION.
      Only HIGH_PERCISION mode support negative index, and negative index in HIGH_PERFORMANCE mode may cause precision abnormal or aicore error.
* @li Batch_dims must be in the range [max(-rank(input_tensor),-rank(indices)), min(rank(input_tensor), rank(indices))).
* @li (batch_dims + rank(input_tensor)) % rank(input_tensor) must be less than or equal to (axis + rank(input_tensor)) % rank(input_tensor).
* @li The first batch_dims dimensions of params and indices are same.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator GatherV2 .

*/
REG_OP(GatherV2)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_STRING, DT_BF16}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_STRING, DT_BF16}))
    .ATTR(batch_dims, Int, 0)
    .ATTR(is_preprocessed, Bool, false)
    .ATTR(negative_index_support, Bool, false)
    .OP_END_FACTORY_REG(GatherV2)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_GATHER_V2_OPS_H_
