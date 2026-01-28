/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Repeat elements of input with copies of data along a specified dimension.
* @par Inputs:
* Two inputs:
* @li x: A Tensor with ND format. Support float, float16, bfloat16, int8, int16, int32, int64,
uint8, uint16, uint32, uint64, bool.
* @li repeats: A Tensor with 0-D / 1-D or a Scalar. Support int32, int64.

* @par Attributes:
* axis: An optional int32, specifying the axis to repeat. Defaults to 1000.

* @par Outputs:
* y: A Tensor, which is the same dtype as x.

* @attention Constraints:
* @li When "repeats" is 1-D tensor, the size of "repeats" must be 1 or x.size()[axis].
* @li "axis" must be within the rank of the input tensor.

* @par Third-party framework compatibility
* Compatible with the PyTorch operator RepeatInterleave.
*/
REG_OP(RepeatInterleave)
    .INPUT(x, TensorType::BasicType())
    .INPUT(repeats, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(axis, Int, 1000)
    .OP_END_FACTORY_REG(RepeatInterleave)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_