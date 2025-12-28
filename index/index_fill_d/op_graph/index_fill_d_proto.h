/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fills the elements of the input tensor with value val by selecting the indices in the order given in index.

* @par Inputs:
* Three inputs, including:
* @li x: A tensor. Must be one of the following types:
*     float16, float32, int32, bfloat16. \n
* @li assist1: A tensor. Must be one of the following types:
*     float16, float32, int32, bfloat16. \n
* @li assist2: A tensor. Must be one of the following types:
*     float16, float32, int32, bfloat16. \n

* @par Attributes:
* dim: A required int. Used to select the dimension of this tensor. \n

* @par Outputs:
* y: A tensor with the same type and shape as 'x'. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator IndexFill. \n

* @attention Constraints:
* The operator will not be enhanced in the future.
*/
REG_OP(IndexFillD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .INPUT(assist1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .INPUT(assist2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .REQUIRED_ATTR(dim, Int)
    .OP_END_FACTORY_REG(IndexFillD)
} // namespace ge
#endif