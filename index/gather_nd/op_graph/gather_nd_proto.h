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
 * \file gather_nd_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_GATHER_ND_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_GATHER_ND_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Gather slices from "x" into a tensor with shape specified by
* "indices". "indices" is an K-dimensional integer tensor, best thought of as a
* (K-1)-dimensional tensor of "indices" into "params", where each element
* defines a slice of "params":
*   output[\\(i_0, ..., i_{K-2}\\)] = params[indices[\\(i_0, ..., i_{K-2}\\)]]
* "indices" defines slices into the first N dimensions of
* "params", where
*           N = indices.shape[-1]
*     indices = [[0, 0], [1, 1]]
*      x = [['a', 'b'], ['c', 'd']]
*      output = ['a', 'd']
* When the impl_mode is set as "support out of bound index", if the indices
* data is out of bound, the corresponding results will be set as 0. Otherwise,
* an aic_error will occur.

* @par Inputs:
* @li x: A ND(Support 1D~8D) Tensor. Must be one of the following types:
*     complex128, complex64, float64, float32, float16, int16, int32, int64,
*     int8, qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64,
*     uint8, bool, string, bfloat16.
* @li indices: A ND(Support 1D) Tensor of type int32 or int64.

* @par Attributes:
* negative_index_support: An optional bool. Defaults to false.

* @par Outputs:
* y: A ND(Support 1D~8D) Tensor which has the same type as "x".


* @par Third-party framework compatibility
* Compatible with the TensorFlow operator GatherNd.
*/
REG_OP(GatherNd)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_STRING, DT_BF16}))
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_STRING, DT_BF16}))
    .ATTR(negative_index_support, Bool, false)
    .OP_END_FACTORY_REG(GatherNd)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_GATHER_ND_PROTO_H_
