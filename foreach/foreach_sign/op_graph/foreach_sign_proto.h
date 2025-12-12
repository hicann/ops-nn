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
 * \file foreach_sign_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_SIGN_H_
#define OPS_OP_PROTO_INC_FOREACH_SIGN_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply sign operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * x: A tensor list containing multiple tensors. The data type can only be float16, float, int32, int8, int64, bfloat16.
 * The format support ND. Shape support 1D ~ 8D.
 * @par Outputs:
 * y: A tensor list which store the tensors whose value are the sign value of the x.
 * The format support ND. The data type can only be float16, float, int32, int8, int64, bfloat16.
 * Shape support 1D ~ 8D. The data type and shape are same as input "x".
 */
REG_OP(ForeachSign)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT64, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT64, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSign)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_SIGN_H_