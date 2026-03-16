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
 * \file max_pool3d_grad_proto.h
 * \brief
 */
 
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MAX_POOL3D_GRAD_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MAX_POOL3D_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Performs the backpropagation of MaxPool3DGrad.

* @par Inputs:
* Three inputs, including:
* @li orig_x: An 5D Tensor. Supported type:float16, bfloat16, float32
* Must set the format, supported format list ["NCDHW, NDHWC"].
* @li orig_y: An 5D Tensor. Supported type:float16, bfloat16, float32
* Must set the format, supported format list ["NCDHW, NDHWC"].
* @li grads: An 5D Tensor. Supported type:float16, bfloat16, float32
* Must set the format, supported format list ["NCDHW, NDHWC"]. \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension (D/H/W) of the input tensor.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the strides of the sliding window for each dimension (D/H/W) of the input tensor.
* @li padding: A string specifying the padding algorithm for the input feature map.
* @li pads: A required list of int8, int16, int32, or int64 values,
* specifying the pads of the sliding window for each dimension of the input tensor.
* @li data_format: An optional string, supported values: ["NCDHW", "NDHWC"],
* default value: ["NCDHW"]. \n

* @par Outputs:
* y: A Tensor. Has the same dtype, shape and format as input "x".

* @attention Constraints:
* @li "ksize" is a list that has length 1 or 3(one value for each of D/H/W),
* every element in the list must be a numeric greater than 0.
* @li "strides" is a list that has length 0 or 1 or 3(one value for each of D/H/W),
* length 0 means use default ksize for each of D/H/W,
* every element in the list must be a numeric greater than 0.
* @li padding: a string must be either "SAME" or "VALID".
* @li "pads" is a list that has length 1 or 3(one value for each of D/H/W),
* every element in the list must be a numeric greater than or equal to 0.
* additionally, two strict size limits apply: \n
* 1. Each pads value (pD/pH/pW for D/H/W dimensions) must be less than or equal to half of the corresponding "ksize" value (kD/kH/kW / 2). \n
* 2. Each pads value (pD/pH/pW) must be less than or equal to ((corresponding ksize - 1) * corresponding dilation + 1) / 2
* (i.e., pD ≤ ((kD - 1) * dD + 1) / 2, pH ≤ ((kH - 1) * dH + 1) / 2, pW ≤ ((kW - 1) * dW + 1) / 2).

* @par Third-party framework compatibility
* Compatible with the Torch operator MaxPool3DGrad.
*/
REG_OP(MaxPool3DGrad)
    .INPUT(orig_x, TensorType::RealNumberType())
    .INPUT(orig_y, TensorType::RealNumberType())
    .INPUT(grads, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding, String, "SAME")
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(MaxPool3DGrad)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H
