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
 * \file foreach_mul_list_inplace_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_MUL_LIST_INPLACE_H_
#define OPS_OP_PROTO_INC_FOREACH_MUL_LIST_INPLACE_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply mul operation element-wise between two tensor lists. The result is written back
 *        in place to the first input "x1".
 * @par Inputs:
 * @li x1: A tensor list, the dtype can be BFloat16, Float16, Int32 or Float32, format supports ND.
 *         This value is also an output (in-place).
 * @li x2: A tensor list with the same length, dtype and format as x1.
 */
REG_OP(ForeachMulListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMulListInplace)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_MUL_LIST_INPLACE_H_
