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
 * \file shape_n_proto.h
 * \brief
 */
#ifndef SHAPE_N_PROTO_H_
#define SHAPE_N_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Returns shape of tensors. \n

*@par Inputs:
*x: A list of input tensors. It's a dynamic input. \n

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to "int32". \n

*@par Outputs:
*y: A list of tensors with the same length as the input list of tensors.
It's a dynamic output. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ShapeN.
*/
REG_OP(ShapeN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(ShapeN)
} // namespace ge
#endif // SHAPE_N_PROTO_H_