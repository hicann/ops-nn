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
 * \file adaptive_avg_pool3d_grad_proto.h
 * \brief
 */
#ifndef OPS_POOLING_ADAPTIVE_AVG_POOL3D_GRAD_PROTO_H_
#define OPS_POOLING_ADAPTIVE_AVG_POOL3D_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Applies a 3D adaptive average pooling backward over
* an input signal composed of several input planes.

* @par Inputs
* Two input, including:
* @li y_grad: A Tensor. Must be one of the following data types:
*     float16, bfloat16, float32. \n

* @li x: A Tensor. Must be one of the following data types:
*     float16, bfloat16, float32. \n

* @par Outputs
*     x_grad: A Tensor. Has the same data type as "x". \n

* @par Attributes
*     data_format: An optional string, Specify the data format of the input and
* output data. With the default format "NDHWC". \n
* For Ascend 950PR/Ascend 950DT, both "NDHWC" and "NCDHW" are supported. All other platforms support only "NDHWC".

* @par Third-party framework compatibility
* Compatible with the PyTorch operator AdaptiveAvgPool3dGrad.
*/
REG_OP(AdaptiveAvgPool3dGrad)
    .INPUT(y_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AdaptiveAvgPool3dGrad)

} // namespace ge

#endif // OPS_POOLING_ADAPTIVE_AVG_POOL3D_GRAD_PROTO_H_
