/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_norm_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_NORM_H_
#define OPS_OP_PROTO_INC_FOREACH_NORM_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply norm operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors. Shape dimension cannot be greater than 8 dimensions.
 * Format supports ND, and data type must be float16, float32 or bfloat16. The list supports a maximum length of 256.
 * @li scalar: A scalar with one element, indicating the order of norm.
 * Format supports ND, and data type must be int64, float32.
 * @par Outputs:
 * y: A tensor list which store the tensors whose value are the norm value of the x.
 * Shape dimension cannot be greater than 8 dimensions, and shapesize should be 1.
 * Format supports ND, and data type must be float16, float32 or bfloat16. The list supports a maximum length of 256.
 */
REG_OP(ForeachNorm)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachNorm)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_NORM_H_