/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_sqrt_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_SQRT_H_
#define OPS_OP_PROTO_INC_FOREACH_SQRT_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply sqrt operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * x: A tensor list containing multiple tensors. Format supports ND. The type support float16, float, bfloat16.
 * Maximum length of x is 50.
 * @par Outputs:
 * y: A tensor list which store the tensors whose value are the sqrt value of the x. Format supports ND.
 * The type support float16, float, bfloat16 and is consistent with x. Maximum length of y is 50.
 */
REG_OP(ForeachSqrt)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSqrt)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_SQRT_H_