/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef OPS_NN_SPARSE_FILL_EMPTY_ROWS_PROTO_H_
#define OPS_NN_SPARSE_FILL_EMPTY_ROWS_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Fills empty rows in a sparse tensor.
 *
 * @par Inputs:
 * @li indices: A 2D tensor of type int64. Each row stores one sparse element
 * index.
 * @li values: A 1D tensor. The value of each sparse element.
 * @li dense_shape: A 1D tensor of type int64. The dense shape of the sparse
 * tensor.
 * @li default_value: A scalar tensor. The value used to fill empty rows.
 *
 * @par Outputs:
 * @li y_indices: A 2D tensor of type int64. The output sparse indices.
 * @li y_values: A 1D tensor. Has the same type as values.
 * @li empty_row_indicator: A 1D tensor of type bool. True indicates the
 * corresponding row was empty.
 * @li reverse_index_map: A 1D tensor of type int64. Maps original input values
 * to output positions.
 */
REG_OP(SparseFillEmptyRows)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_BOOL, DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                               DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}))
    .INPUT(dense_shape, TensorType({DT_INT64}))
    .INPUT(default_value, TensorType({DT_BOOL, DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                                      DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_BOOL, DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                                  DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}))
    .OUTPUT(empty_row_indicator, TensorType({DT_BOOL}))
    .OUTPUT(reverse_index_map, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(SparseFillEmptyRows)

} // namespace ge

#endif // OPS_NN_SPARSE_FILL_EMPTY_ROWS_PROTO_H_
