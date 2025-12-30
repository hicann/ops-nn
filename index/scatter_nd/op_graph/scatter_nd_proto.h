/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ND_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ND_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Creates a new tensor by applying sparse "x" to individual values or slices within a tensor
* (initially zero for numeric, empty for string) of the given "shape" according to "indices".

* @par Inputs:
* @li indices: The index tensor. Format is ND. Support 1D ~ 8D. Must be one of the following types: int32, int64.
* @li x: The source tensor. Format is ND. Type must be the BasicType. Support 1D ~ 8D.
* @li shape: The shape of "y". Format is ND. Support 1D ~ 8D. Must be one of the following types: int32, int64.

* @par Outputs:
* y: A output tensor with same type as input "x".

* @attention Constraints:
* @li indices.shape[-1] <= shape.rank, where the range of shape.rank is [1, 7]
* @li x.shape = indices.shape[:-1] + shape[indices.shape[-1]:].

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNd.
*/
REG_OP(ScatterNd)
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(x, TensorType::BasicType())
    .INPUT(shape, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(ScatterNd)

}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ND_H_
