/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_PROTO_AVG_POOL_GRAD_H_
#define OP_PROTO_AVG_POOL_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Computes avgpoolgrad function.

* @par Inputs:
* @li orig_input_shape: An one-dim tensor of type int32, which describes the
* original input shape [N,C,H,W] or [N,H,W,C] of forward AvgPool.
* @li input_grad: An NHWC or NCHW tensor of type float16, float32, double or bfloat16. \n

* @par Attributes:
* @li ksize: A required tuple or list of ints, 
* specifying the size of the window for each dimension of the input tensor.
* For Ascend 950PR/Ascend 950DT AI Processor: "ksize" length is 1, 2 or 4, must be greater than 0. \n
* @li strides: A required tuple or list of ints,
* specifying the stride of the sliding window for each dimension of the input tensor.
* For Ascend 950PR/Ascend 950DT AI Processor: "strides" length is 1, 2 or 4, must be greater than 0. \n
* @li padding: An optional string, specifying the type of the padding algorithm to use,
* either "VALID", "SAME".
* With "SAME" means that the outputs will have the same spatial dimensions as its inputs.
* With "VALID" means no padding.
* @li data_format: An optional string. Defaults to "NHWC". \n
* For Ascend 950PR/Ascend 950DT AI Processor: support "NCHW" or "NHWC". \n

* @par Outputs:
* @li out_grad: A mutable tensor with the same shape as "orig_input_shape" and the same type as "input_grad". \n
*\n
*     input_grad_height = (out_grad_height + pads_top + pads_bottom - ksize_height)
*                           / strides_h + 1
*\n
*     input_grad_width = (out_grad_width + pads_left + pads_right - ksize_width)
*                          / strides_w + 1
*\n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator AvgPoolGrad.
*/
REG_OP(AvgPoolGrad)
    .INPUT(orig_input_shape, TensorType({DT_INT32}))
    .INPUT(input_grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_BF16}))
    .OUTPUT(out_grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_BF16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(AvgPoolGrad)

} // namespace ge
#endif // OP_PROTO_AVG_POOL_GRAD_H_
