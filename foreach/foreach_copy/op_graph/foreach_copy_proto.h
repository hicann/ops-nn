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
 * \file foreach_copy_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_COPY_H_
#define OPS_OP_PROTO_INC_FOREACH_COPY_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply copy operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * x: A tensor list containing multiple tensors. The data type can only be
 * float16, float, bfloat16, int8, int16, int32, uint8, uint16, uint32, int64, float64, bool.
 * The format support ND. Shape support 1D ~ 8D.
 * @par Outputs:
 * y: A tensor list which store the tensors whose value are the copy value of the x.
 * The data type can only be float16, float, bfloat16, int8, int16, int32, uint8, uint16, uint32, int64, float64, bool.
 * The format support ND. Shape support 1D ~ 8D. Has the same dtype adn shape as input "x".
 */
REG_OP(ForeachCopy)
    .DYNAMIC_INPUT(
        x, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_UINT32, DT_INT64,
                DT_DOUBLE, DT_BOOL}))
    .DYNAMIC_OUTPUT(
        y, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_UINT32, DT_INT64,
                DT_DOUBLE, DT_BOOL}))
    .OP_END_FACTORY_REG(ForeachCopy)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_COPY_H_
