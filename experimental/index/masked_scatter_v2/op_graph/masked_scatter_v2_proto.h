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
 * \file masked_scatter_v2_proto.h
 * \brief MaskedScatterV2 IR 原型（GE-IR 流程）
 */

#ifndef OPS_OP_PROTO_INC_MASKED_SCATTER_V2_H_
#define OPS_OP_PROTO_INC_MASKED_SCATTER_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Fills the elements of the output tensor y with values from updates in order of the true positions of mask,
* and keeps the elements of x otherwise. \n

* @par Inputs:
* Three inputs, including:
* @li x: A tensor. Must be one of the following types: float16, float32, bfloat16, int32.
* @li mask: A tensor of type bool. Must be broadcastable to the shape of x.
* @li updates: A tensor. Must be one of the following types: float16, float32, bfloat16, int32.

* @par Outputs:
* y: A tensor with the same shape and type as x. \n

* @attention Constraints:
* @li Only Ascend 950PR/Ascend 950DT support MaskedScatterV2. \n
*/
REG_OP(MaskedScatterV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT32}))
    .INPUT(mask, TensorType({DT_BOOL}))
    .INPUT(updates, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT32}))
    .OP_END_FACTORY_REG(MaskedScatterV2)
} // namespace ge
#endif // OPS_OP_PROTO_INC_MASKED_SCATTER_V2_H_
