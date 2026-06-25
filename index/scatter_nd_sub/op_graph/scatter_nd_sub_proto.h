/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ND_SUB_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ND_SUB_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Applies sparse "updates" to individual values or slices in a variable reference using the "subtract" operation.

* @par Inputs:
* @li var: The rewritten tensor. An ND tensor. Support 1D ~ 8D. Must be one of the following types:
* float32, float16, int32, int8, uint8.
* @li indices: The index tensor. An ND tensor. Support 1D ~ 8D. Must be one of the following types: int32, int64. The
* last dimension of "indices" represents that the first few dimensions of "var" are the batch dimensions.
* @li updates: The source tensor. An ND tensor. Support 1D ~ 8D. Shape should be equal to the shape of "indices" except
* for the last dimension concats the shape of "var" except for the batch dimensions. Must have the same type of "var".

* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

* @par Outputs:
* var: An ND tensor. Support 1D ~ 8D. Must have the same type, shape and format as input "var".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNdSub.
*/
REG_OP(ScatterNdSub)
    .INPUT(var, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdSub)
}

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ND_SUB_H_
