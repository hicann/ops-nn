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
 * \file sparse_to_dense_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_SPARSE_TO_DENSE_H_
#define OPS_OP_PROTO_INC_SPARSE_TO_DENSE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Converts a sparse representation into a dense tensor. \n

* @par Inputs:
* Four inputs, including:
* @li indices: An ND Tensor of type int32 or int64. Dimension must be 0D, 1D or 2D.
* Data between different rows(Corresponding index of output) cannot be duplicated, and should be sorted by ascending.
* Index cannot exceed the size of each dimension of output.
* @li output_shape: A 1D Tensor has the same dtype of indices.
* @li values: A 1D Tensor, Values corresponding to each row of indices, or a scalar value to be used for all sparse indices. \n
* Must be one of the following types: float32, float16, bfloat16, int16, uint16, int32, int64, int8, uint8, bool, double.
* @li default_value: An ND Tensor of the same dtype as values .
* Size must be 1.  \n

* @par Attributes:
* @li validate_indices: An optional bool. 
* If true, indices are checked to make sure they are sorted by ascending, no repeats, and cannot exceed the size of each dimension.
* This param is currently not effective in Ascend950PR/Ascend950DT. \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "values" . \n
*/
REG_OP(SparseToDense)
    .INPUT(indices, TensorType({DT_INT32,DT_INT64}))
    .INPUT(output_shape, TensorType({DT_INT32,DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16,DT_INT16,DT_UINT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BOOL,DT_DOUBLE}))
    .INPUT(default_value, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16,DT_INT16,DT_UINT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BOOL,DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16,DT_INT16,DT_UINT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BOOL,DT_DOUBLE}))
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(SparseToDense) // namespace ge
}
#endif // OPS_OP_PROTO_INC_SPARSE_TO_DENSE_H_
