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
 * \file embedding_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_EMBEDDING_H_
#define OPS_OP_PROTO_INC_EMBEDDING_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief According to the indices, return the embedding vectors. \n

* @par Inputs:
* @li x: A required Tensor.Must be one of the following types: complex64, complex32, double, float32,
* float16, int16, int32, int64, int8, uint16, uint32, uint64, uint8, bool, bfloat16.
* @li indices: A required Tensor. A index Tensor. Must be one of the following types: int32, int64.

* @par Outputs:
* y: The embedded output tensor. Has the same type and format as input "x".

* @attention Constraints:
* Based on whether all the 1s in indexed_sizes are consecutive, it is categorized into a continuous axis scenario and
a non-continuous axis scenario.
*/
REG_OP(Embedding)
    .INPUT(x, TensorType({DT_COMPLEX64, DT_COMPLEX32, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, DT_BOOL, DT_BF16}))
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_COMPLEX64, DT_COMPLEX32, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, DT_BOOL, DT_BF16}))
    .OP_END_FACTORY_REG(Embedding)
} // namespace ge

#endif // OPS_OP_PROTO_INC_EMBEDDING_H_
