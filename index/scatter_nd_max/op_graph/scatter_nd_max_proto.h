/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_ND_MAX_PROTO_H_
#define SCATTER_ND_MAX_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Applies sparse "updates" to individual values or slices in a variable reference using the "max" operation.

* @par Inputs:
* @li var: The rewritten tensor. An ND tensor of type BasicType. Support 1D ~ 8D.
* @li indices: The index tensor. An ND tensor of type int32 or int64. Support 1D ~ 8D. The last dimension of "indices"
* represents that the first few dimensions of "var" are the batch dimensions.
* @li updates: The source tensor. A tensor with the same dtype as 'var'. Support 1D ~ 8D. Shape should be equal to the
* shape of "indices" except for the last dimension concats the shape of "var" except for the batch dimensions.

* @par Attributes:
* @li use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

* @par Outputs:
* var: A Tensor. Support 1D ~ 8D. Must have the same type, shape and format as input "var".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNdMax.
*/
REG_OP(ScatterNdMax)
    .INPUT(var, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::BasicType())
    .OUTPUT(var, TensorType::BasicType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdMax)
} // namespace ge

#endif // SCATTER_ND_MAX_PROTO_H_
