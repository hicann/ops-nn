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
* Must set the format, supported format list ["NCDHW", "NDHWC"].
* @li orig_y: An 5D Tensor. Supported type:float16, bfloat16, float32
* Must set the format, supported format list ["NCDHW", "NDHWC"].
* @li grads: An 5D Tensor. Supported type:float16, bfloat16, float32
* Must set the format, supported format list ["NCDHW", "NDHWC"]. \n

* @par Attributes:
* @li ksize: A list that has length 5. The ksize of the H and W dimensions should be greater than 0.
* A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension (D/H/W) of the input tensor.
* The ksize of the N and C dimensions should be 1. e.g. For "data_format" is "NCDHW", ksize[0] = 1 and ksize[1] = 1.
* For "data_format" is "NDHWC", ksize[0] = 1 and ksize[4] = 1.  \n
* @li strides: A list that has length 5. The stride of the N and C dimensions should be 1.
* A required list of int8, int16, int32, or int64 values,
* specifying the strides of the sliding window for each dimension (D/H/W) of the input tensor.
* The stride of the N and C dimensions should be 1.  \n
* @li padding: A string specifying the padding algorithm for the input feature map. Defaults to "SAME", it support SAME, VALID and CALCULATED.
* when padding_mode is "SAME": pads 0 to ensure output shape equal to ceil(input shape / stride) ,
* (output shape equal to input shape when stride=1). \n
* when padding_mode is "VALID": no padding. The kernel slides only over valid regions, resulting in smaller output . \n
* when padding_mode is "CALCULATED": use pads to calculate output shape.
* @li pads: A list of 6 ints. Supports only padding along the D,
* H and W dimensions in sequence of head, tail, top, bottom, left and right.
* to use.
* @li data_format: An optional string, supported values: ["NCDHW", "NDHWC"],With the default format "NCDHW". \n

* @par Outputs:
* y: A Tensor. Has the same dtype, shape and format as input "orig_x".

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
#endif  // OPS_BUILT_IN_OP_PROTO_INC_MAX_POOL3D_GRAD_PROTO_H_
