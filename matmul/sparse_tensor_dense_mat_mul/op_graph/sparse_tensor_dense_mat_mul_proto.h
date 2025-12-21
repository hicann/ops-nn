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
 * \file sparse_tensor_dense_mat_mul_proto.h
 * \brief
 */
#ifndef OPS_SPARSE_TENSOR_DENSE_MAT_MUL_PROTO_H_
#define OPS_SPARSE_TENSOR_DENSE_MAT_MUL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Multiplies SparseTensor A (of rank 2) by dense matrix B. \n

*@par Inputs:
* @li x1_indices: A 2D tensor of type int32 or int64.
*The indices of the matrix "SparseTensor", with size [nnz, 2].
* @li x1_values: A 1D tensor. The values of the SparseTensor, with size [nnz].
* @li x1_shape: A 1D tensor of type int64. The shape of the SparseTensor, with size [2].
* @li x2: A dense matrix tensor of the same type as "x1_values". 2D. \n

*@par Outputs:
*y: A "tensor". Has the same type as "x1_values". \n

*@par Attributes:
*@li adjoint_a: An optional bool. Defaults to "False".Use the adjoint of A in the matrix multiply.
*If A is complex, this is transpose(conj(A)). Otherwise it is transpose(A).
*@li adjoint_b: An optional bool. Defaults to "False".Use the adjoint of B in the matrix multiply.
*If B is complex, this is transpose(conj(B)). Otherwise it is transpose(B). \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseTensorDenseMatMul.
*/
REG_OP(SparseTensorDenseMatMul)
    .INPUT(x1_indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x1_values, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_COMPLEX64, DT_COMPLEX128, DT_FLOAT16, DT_INT64}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_COMPLEX64, DT_COMPLEX128, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_COMPLEX64, DT_COMPLEX128, DT_FLOAT16}))
    .ATTR(adjoint_a, Bool, false)
    .ATTR(adjoint_b, Bool, false)
    .OP_END_FACTORY_REG(SparseTensorDenseMatMul)
} // namespace ge

#endif // OPS_SPARSE_TENSOR_DENSE_MAT_MUL_PROTO_H_
