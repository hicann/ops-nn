/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_PROTO_AVG_POOL_V2_GRAD_H_
#define OP_PROTO_AVG_POOL_V2_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Computes avgpoolv2grad function.

* @par Inputs:
* @li orig_input_shape: An NHWC tensor of type int32.
* @li input_grad: An NHWC tensor of type float16, float32, double or bfloat16. \n

* @par Attributes:
* @li ksize: A required tuple or list of ints,
* specifying the size of the window for each dimension of the input tensor.
* @li strides: A required tuple or list of ints,
* specifying the stride of the sliding window for each dimension of the input tensor.
* @li padding_mode: An optional string, specifying the type of the padding algorithm to use,
* either "VALID", "SAME" or "CALCULATED". Default "CALCULATED".
* With "SAME" means that the outputs will have the same spatial dimensions as its inputs.
* With "VALID" means no padding.
* @li pads: An optional list of ints, specifying the pad of the input feature map.
* Default value {0,0,0,0}.
* @li global_pooling: An optional bool, whether to use the global pooling. If global_pooling =
* true, ksize and pads will be ignored. Default False.
* @li ceil_mode: An optional bool, whether to use the ceil function to calculate output height
* and width. Default False.
* @li exclusive: An optional bool, whether to exclude padding points. Default is true.
* @li data_format: An optional string. Defaults to "NCHW". \n
* @li divisor_override: An optional int, the default value is zero.
* if specified, it will be used as divisor, otherwise size of the pooling region will be used.

* @par Outputs:
* @li out_grad: A mutable tensor with the same shape and type as "orig_input_shape". \n
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
REG_OP(AvgPoolV2Grad)
    .INPUT(orig_input_shape, TensorType({DT_INT32}))
    .INPUT(input_grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_BF16}))
    .OUTPUT(out_grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_BF16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0,0,0,0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling, Bool, false)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(exclusive, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .OP_END_FACTORY_REG(AvgPoolV2Grad)

} // namespace ge
#endif // OP_PROTO_AVG_POOL_V2_GRAD_H_