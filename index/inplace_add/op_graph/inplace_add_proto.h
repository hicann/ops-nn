/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_INPLACE_ADD_H_
#define OPS_BUILT_IN_OP_PROTO_INC_INPLACE_ADD_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Adds "v" into specified rows of "x".
 * Computes y = x; y[i, :] += v; return y.
 *
 * @par Inputs:
 * @li x: A 1D to 8D Tensor. TensorType::BasicType(), Format is ND.
 * @li indices: A vector of type int32, Format is ND. Indices into the left-most dimension of x.
 * @li v: A 1D to 8D Tensor of the same type as "x", Format is ND.
 * Same dimension sizes as "x" except the first dimension, which must be the same as the size of "indices" .
 *
 * @par Outputs:
 * @li y: A 1D to 8D Tensor. Has the same type and shape as "x", Format is ND, and aliases "x".
 *
 * @attention Constraints:
 * The content of "y" is undefined if there are duplicates in indices.
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator InplaceAdd.
 */
#ifndef OPS_PROTO_DEF_INPLACEADD
#define OPS_PROTO_DEF_INPLACEADD
REG_OP(InplaceAdd)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(v, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(InplaceAdd)
#endif // OPS_PROTO_DEF_INPLACEADD

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_INPLACE_ADD_H_
