/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_INDEX_SPARSE_SLICE_PROTO_H_
#define OPS_NN_INDEX_SPARSE_SLICE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Slices a SparseTensor based on the "start" and "size". \n

*@par Inputs:
* @li indices: A 2D tensor of type int64. The indices of the SparseTensor.
* It has shape [n, rank], where n is the number of input values, and rank is the dimension of value index.
* The second dimension 'rank' supports 1 to 24(included).
* @li values: A 1D tensor. The values of the SparseTensor.
* It has shape [n] which is the number all values. The supported datatypes are:
* [DT_INT64, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
* DT_COMPLEX64, DT_COMPLEX128, DT_BOOL, DT_STRING, DT_RESOURCE, DT_BF16].
* @li shape: A 1D tensor of type int64. It has shape [rank]. It is the shape of the SparseTensor.
* @li start: A 1D tensor of type int64. It has shape [rank]. It is the  start of the slice.
* @li size: A 1D tensor of type int64. It has shape [rank]. It is the size of the slice. \n

*@par Outputs:
*@li y_indices: A tensor of type int64. The output indices of the SparseTensor after slicing.
* It has shape [m, rank] where m is unknown, it depends on compute result.
*@li y_values: A tensor. Has the same type as "values". The values of the output SparseTensor after slicing.
* It has shape [m] where m is unknown, it depends on compute result.
*@li y_shape: A tensor of type int64. It has shape [rank]. It is the shape of the output SparseTensor. \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseSlice.
*/
REG_OP(SparseSlice)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({ DT_INT64, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8,
                                DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                                DT_COMPLEX128, DT_BOOL, DT_STRING, DT_RESOURCE, DT_BF16}))
    .INPUT(shape, TensorType({DT_INT64}))
    .INPUT(start, TensorType({DT_INT64}))
    .INPUT(size, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({  DT_INT64, DT_INT32, DT_UINT16, DT_INT16,
                                    DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                                    DT_COMPLEX64, DT_COMPLEX128, DT_BOOL, DT_STRING, DT_RESOURCE, DT_BF16}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(SparseSlice)
} // namespace ge

#endif // OPS_NN_INDEX_SPARSE_SLICE_PROTO_H_