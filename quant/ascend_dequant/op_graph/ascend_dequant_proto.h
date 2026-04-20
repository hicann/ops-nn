/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_QUANT_OP_GRAPH_ASCEND_DEQUANT_PROTO_H
#define OPS_NN_QUANT_OP_GRAPH_ASCEND_DEQUANT_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Dequantizes the input.

* @par Inputs:
* @li x: A tensor of type int32, specifying the input. Shape support 1D ~ 8D.
* The format must be FRACTAL_NZ, NC1HWC0 or NDC1HWC0.
* @li deq_scale: A required Tensor. Must be one of the following types: float16,
* uint64. The format must be NC1HWC0 or NDC1HWC0. If deq_scale is 1D tensor,
* shape must be same as the last dimension of x. Otherwise the number of 
* dimensions should be equal to x, the last dimension of shape should be 
* the same as x, others must be 1. \n

* @par Attributes:
* @li sqrt_mode: An optional bool, specifying whether to perform square root
* on "scale", either "True" or "False". Defaults to "False".
* @li relu_flag: An optional bool, specifying whether to perform ReLU, 
* either "True" or "False". Defaults to "False".
* @li dtype: An optional int32, specifying the output data type. Defaults to "0"
* , represents dtype "DT_FLOAT". \n

* @par Outputs:
* y: The dequantized output tensor of type float16 or float32. The format must 
* be FRACTAL_NZ, NC1HWC0 or NDC1HWC0. The shape is same as x. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendDequant)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(deq_scale, TensorType({DT_FLOAT16, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(relu_flag, Bool, false)
    .ATTR(dtype, Int, DT_FLOAT)
    .OP_END_FACTORY_REG(AscendDequant)
} // namespace ge

#endif // OPS_NN_QUANT_OP_GRAPH_ASCEND_DEQUANT_PROTO_H
