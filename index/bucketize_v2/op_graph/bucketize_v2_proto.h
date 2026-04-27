/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_BUCKETIZE_V2_H_
#define OPS_BUILT_IN_OP_PROTO_INC_BUCKETIZE_V2_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Bucketize 'x' based on 'boundaries'. For example, if the inputs
are boundaries = [0, 10, 100] x = [[-5, 10000] [150, 10] [5, 100]] then
the output will be [[0, 3] [3, 2] [1, 3]].

* @par Inputs:
* Two inputs, including:
* @li x: A tensor. Must be one of the following types:
*     float16, float32, bfloat16, int8, int16, int32, int64, uint8. \n
* @li boundaries: A sorted 1-dim tensor with same type as x. It gives the boundary of the buckets. Must be one of the following types:
*     float16, float32, bfloat16, int8, int16, int32, int64, uint8. \n

* @par Attributes:
* @li out_int32: An optional true or false. If true, the dtype of y is int32. Otherwise, it is int64. Defaults to false.
* @li right: An optional true or false. If true, return upperbound index. If false return lowerbound index.

* @par Outputs:
* y: A tensor with the same shape as 'x', each value of input replaced with bucket index. 
* Must be one of the following types: int32, int64. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator bucketize. \n

* @attention Constraints:
* @li The output has the same shape as the input.
* @li ‌The boundaries must be in ascending order and non-repeating.
*/
REG_OP(BucketizeV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8}))
    .INPUT(boundaries, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_int32, Bool, false)
    .ATTR(right, Bool, false)
    .OP_END_FACTORY_REG(BucketizeV2)
} // namespace ge
#endif
