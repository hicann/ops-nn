/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file non_zero_proto.h
 * \brief
 */

#ifndef NON_ZERO_PROTO_H_
#define NON_ZERO_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns a tensor containing the indices of all non-zero
* elements of input.

* @par Inputs:
* x: A Tensor. Must be one of the following types: float16, bfloat16, float32,
*    int32, int64, double, int8, uint8, int16, uint16, uint32, uint64, bool.
* @par Attributes:
* @li transpose: An optional attribute. Type is bool. Defaults to False.
* @li dtype: An optional attribute. Specifying the output data type.
*     Either "int32" or "int64". Default to "int64". \n
* @par Outputs:
* y: A Tensor. Must be one of the following types: int32, int64. \n

* @par Third-party framework compatibility
* Compatible with the PyTorch operator NonZero.
*/

REG_OP(NonZero)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8,
                          DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                          DT_UINT64, DT_BOOL, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT64, DT_INT32}))
    .ATTR(transpose, Bool, false)
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(NonZero)

}  // namespace ge

#endif  // NON_ZERO_PROTO_H_