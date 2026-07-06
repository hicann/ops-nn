/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTERMAX_PROTO_H_
#define SCATTERMAX_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Reduces sparse "updates" into a variable reference by taking the element-wise maximum. \n

* @par Inputs:
* @li var: The tensor to be updated. An ND tensor. Support 1D ~ 8D. Must be one of the following types:
* float16, float32, int32, int8, uint8.
* @li indices: The index tensor of the slices to be updated. An ND tensor. Support 1D ~ 8D.
* Must be one of the following types: int32, int64.
* @li updates: The source tensor reduced into "var". An ND tensor. Support 1D ~ 8D. The shape should be equal to the
* shape of "indices" concats the shape of "var" except for the first dimension. Must have the same type as "var". \n

* @par Attributes:
* use_locking: An optional bool. Ignore this attribute. This attribute does not take effect even if it is set.
* Defaults to "false". \n

* @par Outputs:
* var: An ND tensor. Support 1D ~ 8D. Must have the same type, shape and format as input "var". \n

* @attention Constraints:
* updates.shape = indices.shape + var.shape[1:]. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterMax.
*/
REG_OP(ScatterMax)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMax)
} // namespace ge
#endif // SCATTERMAX_PROTO_H_
