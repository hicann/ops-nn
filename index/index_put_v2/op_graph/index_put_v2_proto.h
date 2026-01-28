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
* @brief According to the index number of indexes, replace the value
* corresponding to X with the value.

* @par Inputs:
* Five inputs, including:
* @li x: A ND Tensor. Must be one of the following types: int64, int32, float32, float16, bfloat16, int8, uint8, bool.
* @li value: A Tensor of the same type as "x".
* @li indexed_sizes: A 1D Tensor of int64 with shape (N). Sizes for each one of the indexed data.
* @li indexed_strides: A 1D Tensor of int64 with shape (N). Strides for each one of the indexed data.
* @li indices: Dynamic input. A ND Tensor of int64/int32. A Tensor of the indices.

* @par Attributes:
* @li accumulate: Does it support self accumulation. Defaults to false.

* @par Outputs:
* @li x: A Tensor.

* @par Third-party framework compatibility
* Compatible with the Pytorch operator index_put.

* @par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(IndexPutV2)
    .INPUT(x, TensorType::BasicType())
    .INPUT(value, TensorType::BasicType())
    .INPUT(indexed_sizes, TensorType({DT_INT64}))
    .INPUT(indexed_strides, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64, DT_INT32}))
    .OUTPUT(x, TensorType::BasicType())
    .ATTR(accumulate, Bool, false)
    .OP_END_FACTORY_REG(IndexPutV2)
} // namespace ge
#endif