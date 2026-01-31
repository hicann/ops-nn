/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_proto.h
 * \brief
 */
#ifndef OPS_POOLING_ADAPTIVE_AVG_POOL3D_PROTO_H_
#define OPS_POOLING_ADAPTIVE_AVG_POOL3D_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Applies a 3D adaptive max pooling over
* an input signal composed of several input planes.

* @par Inputs:
* One input, including:
* x: A Tensor. Must be one of the following data types:
*     float16, bfloat16, float32. \n

* @par Attributes:
* @li output_size: A required list of 0, 1 or 3 ints. \n
* 0 specifies the shape of the output tensor is the same as x. \n
* 1 specifies the size (D,D,D) of the output tensor. \n
* 3 specifies the size (D,H,W) of the output tensor. \n
* @li indices_dtype: An optional int, default value is 3.
* (3 is int32, 9 is int64) \n

* @par Outputs:
* @li y: A Tensor. Has the same data type as "x". \n
* @li indices: A Tensor. Has the same shape as "x". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveMaxPool3d.
*/
REG_OP(AdaptiveMaxPool3d)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(indices_dtype, Int, 3)
    .OP_END_FACTORY_REG(AdaptiveMaxPool3d)

} // namespace ge

#endif // OPS_POOLING_ADAPTIVE_AVG_POOL3D_PROTO_H_