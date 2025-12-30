/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ND_ADD_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ND_ADD_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge { 

/**
* @brief Applies sparse "updates" to individual values or slices in a variable reference using the "add" operation.

* @par Inputs:
* @li var: The rewritten tensor. An ND tensor. Support 1D ~ 8D. Must be one of the following types:
* complex128, complex64, double, float32, float16, int8, uint8, int16, uint16, int32, uint32, int64, uint64, bool.
* @li indices: The index tensor. An ND tensor. Support 1D ~ 8D. Must be one of the following types: int32, int64. The
* last dimension of "indices" represents that the first few dimensions of "var" are the batch dimensions.
* @li updates: The source tensor. An ND tensor. Support 1D ~ 8D. Shape should be equal to the shape of "indices" except
* for the last dimension concats the shape of "var" except for the batch dimensions. Must have the same type of "var".

* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

* @par Outputs:
* var: An ND tensor. Support 1D ~ 8D. Must have the same type, shape and format as input "var".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNdAdd.
*/
REG_OP(ScatterNdAdd)
    .INPUT(var, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16,
                            DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8,
                                DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(var, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16,
                             DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdAdd)    
}

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SCATTER_ND_ADD_H_