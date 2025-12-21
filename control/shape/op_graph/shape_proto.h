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
 * \file shape_proto.h
 * \brief
 */
#ifndef SHAPE_PROTO_H_
#define SHAPE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Returns the shape of a tensor. \n

*@par Inputs:
*x: A tensor. Must be one of the following types: float32、float16、int8、
int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string、bfloat16. \n

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to int32. \n

*@par Outputs:
*y: A tensor. The shape of the input tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Size.
*/
REG_OP(Shape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Shape)
} // namespace ge
#endif // SHAPE_PROTO_H_