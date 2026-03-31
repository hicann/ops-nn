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
 * \file reverse_v2_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_REVERSE_V2_H_
#define OPS_BUILT_IN_OP_PROTO_INC_REVERSE_V2_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Reverses specific dimensions of a tensor .

* @par Inputs:
* Two inputs, including:
* @li x: An ND Tensor (up to 8D).
* Must be one of the following types: int8, uint8, int16, uint16, int32, int64, bool, bfloat16, float16, float32,
* double, complex64, complex128, string.
* @li axis: A 1D Tensor.
* Must be one of the following types: int32, int64 . \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @attention Constraints:
"axis" must be within the rank of "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ReverseV2.
*/
REG_OP(ReverseV2)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                          DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                          DT_COMPLEX64, DT_COMPLEX128, DT_STRING, DT_BF16}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                           DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128, DT_STRING, DT_BF16}))
    .OP_END_FACTORY_REG(ReverseV2)
} // namespace ge

#endif // OPS_OP_PROTO_INC_REVERSE_V2_H_