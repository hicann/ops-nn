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
 * \file adaptive_avg_pool3d_proto.h
 * \brief
 */
#ifndef OPS_POOLING_ADAPTIVE_AVG_POOL3D_PROTO_H_
#define OPS_POOLING_ADAPTIVE_AVG_POOL3D_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Applies a 3D adaptive average pooling over
* an input signal composed of several input planes.

* @par Inputs:
* One input, including:
* @li x: A Tensor. Must be one of the following data types:
*     float16, bfloat16, float32. \n

* @par Attributes:
* @li output_size: A required list of 3 ints
*    specifying the size (D,H,W) of the output tensor. \n

* @par Outputs:
* @li y: A Tensor. Has the same data type as "x" \n

* @li data_format: An optional string, Specify the data format of the input and
* output data. With the default format "NDHWC" . \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveAvgPool3d.
*/
REG_OP(AdaptiveAvgPool3d)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AdaptiveAvgPool3d)

} // namespace ge

#endif // OPS_POOLING_ADAPTIVE_AVG_POOL3D_PROTO_H_