/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief According to the indices, return the value.

* @par Inputs:
* Four inputs, including:
* @li x: A ND Tensor. Must be one of the following types: int64, int32, float32, float16, bfloat16, int8, uint8, bool.
* @li indexed_sizes: A 1D Tensor of int64 with shape (N). Sizes for each one of the indexed data.
* @li indexed_strides: A 1D Tensor of int64 with shape (N). Strides for each one of the indexed data.
* @li indices: Dynamic input. A ND Tensor of int64/int32. return the value according to the indices.

* @par Outputs:
* y: The indexed output tensor. Has the same type and format as input "x".

* @attention Constraints:
* @li The value in indexed_sizes Tensor is 0 or 1, where 1 indicates that the corresponding dimension in x is indexed.
*     Must not be all zeros (at least one dimension must be indexed).
* @li Based on whether all the 1s in indexed_sizes are consective, it is categorized into a continuous axis scenario and
a non-continuous axis scenario.
*     Examples of the continuous axis scenario are as follows: x shape=(a, b, c, d), only 2 dim in the middle are
indexed, indexed_sizes=[0, 1, 1, 0] indices shape (e, f）
*     The value range of indices is [-b, b-1], [-c, c-1], then y shape is (a, e, f, d). Examples of the non-continuous
axis scenario are as follows:x shape=(a, b, c, d),
*     only the first and the third dim are indexed, indexed_sizes=[1, 0, 1, 0], indices shape=(e, f), then y shape is
(e, f, b, d)
*/
REG_OP(Index)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indexed_sizes, TensorType({DT_INT64}))
    .INPUT(indexed_strides, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64, DT_INT32}))
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Index)
} // namespace ge
#endif