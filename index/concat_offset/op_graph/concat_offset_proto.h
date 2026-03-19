/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CONCAT_OFFSET_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CONCAT_OFFSET_H_
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Computes offsets of concat inputs within its output .

*@par Inputs:
*Two inputs, including:
* @li concat_dim: A Tensor of type int32.Supported format list ["ND"].
* @li x: A list of 1D tensor objects of type int32, each with same shape and type. 
*        It's a dynamic input.Supported format list ["ND"]. 
         The number of tensors in the x must be at least 2. 
         The shape size of each tensor in x is in range [1, 8]. \n

*@par Attributes:
*N: A required int indicating the number of tensors in the input x. \n

*@par Outputs:
*y: A Tensor list with same type as "x" . It's a dynamic output.Supported format list ["ND"]. \n

*@par Third-party framework compatibility
*@ Compatible with the TensorFlow operator ConcatOffset.
*/
REG_OP(ConcatOffset)
    .INPUT(concat_dim, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(x, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ConcatOffset)

}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_CONCAT_OFFSET_H_
