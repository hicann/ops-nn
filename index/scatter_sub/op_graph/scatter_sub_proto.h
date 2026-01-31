/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SCATTER_SUB_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SCATTER_SUB_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {


/**
* @brief Subs sparse "updates" to a variable reference.

* @par Inputs:
* @li var: The rewritten tensor. An ND tensor. Support 1D ~ 8D. Must be one of the following types:
* float16, float32, int32, int8, uint8, bfloat16.
* @li indices: The index tensor. An ND tensor. Support 1D ~ 8D. Must be one of the following types: int32, int64.
* @li updates: The source tensor. An ND tensor. Support 1D ~ 8D. Shape should be equal to the shape of "indices" concats
* the shape of "var" except for the first dimension. Must have the same type of "var".

* @par Attributes:
* use_locking: Ignore this attribute. This attribute does not take effect even if it is set. \n

* @par Outputs:
* var: An ND tensor. Support 1D ~ 8D. Must have the same type, shape and format as input "var".

* @attention Constraints:
* updates.shape = indices.shape + var.shape[1:] or updates.shape = []. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterSub.
*/

#define SCATTER_SUB_SUPPORT_TYPES {DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8,DT_BF16}
REG_OP(ScatterSub)
    .INPUT(var, TensorType(SCATTER_SUB_SUPPORT_TYPES))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType(SCATTER_SUB_SUPPORT_TYPES))
    .OUTPUT(var, TensorType(SCATTER_SUB_SUPPORT_TYPES))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterSub)

}

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SCATTER_SUB_H_