/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_INPLACE_SUB_H_
#define OPS_BUILT_IN_OP_PROTO_INC_INPLACE_SUB_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Subtracts "v" into specified rows of "x".
 * Computes y = x; y[i, :] -= v; return y.
 *
 * @par Inputs:
 * @li x: A Tensor. TensorType::BasicType(), Format is ND.
 * @li indices: A vector of type int32, Format is ND. Indices into the left-most dimension of x.
 * @li v: A Tensor of the same type as "x", Format is ND.
 * Same dimension sizes as "x" except the first dimension, which must be the same as the size of "indices" .
 *
 * @par Outputs:
 * @li y: A Tensor. Has the same type as "x", Format is ND.
 *
 * @attention Constraints:
 * The content of "y" is undefined if there are duplicates in indices.
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator InplaceSub.
 */
#ifndef OPS_PROTO_DEF_INPLACESUB
#define OPS_PROTO_DEF_INPLACESUB
REG_OP(InplaceSub)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(v, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(InplaceSub)
#endif // OPS_PROTO_DEF_INPLACESUB

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_INPLACE_SUB_H_
