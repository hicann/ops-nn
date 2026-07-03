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
 * \file foreach_binary_op_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_foreach_binary_op_H_
#define OPS_OP_PROTO_INC_foreach_binary_op_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply an element-wise binary operation for each tensor pair (x1[i], x2[i]).
 * The operation is selected by attribute op_code: 0=add, 1=sub, 2=mul, 3=div.
 * @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensors
 * @par Attributes:
 * @li op_code: Required Int. Selects the binary op (0=add, 1=sub, 2=mul, 3=div)
 * @par Outputs:
 * @li y: A tensor list storing y[i] = x1[i] <op> x2[i]
 */
REG_OP(ForeachBinaryOp)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .REQUIRED_ATTR(op_code, Int)
    .OP_END_FACTORY_REG(ForeachBinaryOp)
} // namespace ge
#endif // OPS_OP_PROTO_INC_foreach_binary_op_H_
