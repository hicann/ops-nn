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
 * \file max_pool_grad_proto.h
 * \brief
 */
 
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MAX_POOL_GRAD_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MAX_POOL_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Computes gradients of the maxpooling function .

* @par Inputs:
* @li x1: Original forward input tensor. Support type: TensorType::RealNumberType(), Support Format:[NCHW, NHWC].
* @li x2: Original forward output tensor. Has the same type and format as input "x1".
* @li grad: Has the same type and format as input "x1". \n

* @par Attributes:
* @li ksize: A required tuple or list of int values, specifying the size of the window for
* each dimension of the input tensor.
* @li strides: A required tuple or list of int values, specifying the stride of the sliding
* window for each dimension of the input tensor.
* @li padding: A required string, specifying the type of padding algorithm
* to use.
* @li data_format: An optional string, Specify the data format of the input and
* output data. With the default format "NHWC" . \n

* @par Outputs:
* y: A mutable tensor. Has the same shape, type and format as "x1" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolGrad.
*/
REG_OP(MaxPoolGrad)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolGrad)
}
    
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H
