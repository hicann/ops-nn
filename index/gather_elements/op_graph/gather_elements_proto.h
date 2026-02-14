/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_elements_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_GATHER_ELEMENTS_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_GATHER_ELEMENTS_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Gather slices from "x" according to "index" by corresponding dim, produces a output tensor
* with shape(index.shape).

* @par Inputs:
* @li x: A Tensor. Must be one of the following types: float16, bfloat16, float32, uint32, int32, uint64,
* int64, uint8, int8, uint16, int16, bool, double, DT_FLOAT8_E5M2, DT_FLOAT8_E8M0, DT_FLOAT8_E4M3FN.
* @li index: The index tensor used for collecting values. A Tensor. Must be one of the following types: int32, int64.The number of dimensions 
* of index needs to be consistent with that of x, and its shape needs to be consistent with that of y. 
* Except for the dimension specified by dim, the size of other dimensions should be less than or equal 
* to the size of the corresponding dimension of x.

* @par Attributes:
* dim: the axis along which to index, int32 or int64.

* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Third-party framework compatibility
* Compatible with the PyTorch operator Gather.
*/
REG_OP(GatherElements)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32,
    DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_FLOAT8_E5M2,
    DT_FLOAT8_E8M0, DT_FLOAT8_E4M3FN}))
    .INPUT(index, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32,
    DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_FLOAT8_E5M2,
    DT_FLOAT8_E8M0, DT_FLOAT8_E4M3FN}))
    .ATTR(dim, Int, 0)
    .OP_END_FACTORY_REG(GatherElements)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_GATHER_ELEMENTS_PROTO_H_