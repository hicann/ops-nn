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
 * \file rank_proto.h
 * \brief
 */
#ifndef RANK_PROTO_H_
#define RANK_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Returns an integer representing the rank of input tensor. The rank of a tensor is the number of indices required to uniquely select each element of the tensor, that is, the dimension size of the tensor. \n

*@par Inputs:
*x: A Tensor of type float32, float16, int8, int16, uint16, uint8, int32, int64, uint32, uint64, bool, double, string. \n

*@par Outputs:
*y: A tensor. The rank of input tensor. Type is int32. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Rank.
*/
REG_OP(Rank)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(Rank)
} // namespace ge
#endif // RANK_PROTO_H_