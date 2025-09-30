/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 * The code snippet comes from Huawei's open-source Mindspore project.
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*!
 * \file avg_pool3_d_grad_proto.h
 * \brief
 */
#ifndef OPS_POOLING_AVG_POOL3_D_GRAD_PROTO_H_
#define OPS_POOLING_AVG_POOL3_D_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Computes average pooling3d backwards gradients.

* @par Inputs:
* @li orig_input_shape: An one-dim tensor of type int32, which describes the
* original input shape [N,C,D,H,W] of forward AvgPool3D.
* @li grads: A 5D Tensor of backwards gradient. With the format "NDHWC",
* grads supports data type float16/float32/bfloat16.

* @par Attributes:
* @li ksize: List of ints. Describes the size of the window for each dimension of the input tensor.
* Restriction: "ksize" value is in the range [1, 255] and <= "orig_input_shape" for DHW dimensions. \n
* For Atlas Inference Series Product: "ksize" length is 5. \n
* For Atlas Training Series Product: "ksize" length is 5. \n
* For Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component: "ksize"
length is 3. \n
* For Atlas A3 Training Series Product/Atlas A3 Inference Series Product: "ksize" length is 3. \n
* For Ascend 910_95 AI Processor: "ksize" length is 3, without the limit of [1, 255] and must be greater than 0. \n
* @li strides: List of ints. The stride of the sliding window for each dimension of the input tensor.
* Restriction: "strides" value is in the range [1, 63]. \n
* For Atlas Inference Series Product: "strides" length is 5. \n
* For Atlas Training Series Product: "strides" length is 5. \n
* For Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component: "strides"
length is 3. \n
* For Atlas A3 Training Series Product/Atlas A3 Inference Series Product: "strides" length is 3. \n
* For Ascend 910_95 AI Processor: "strides" length is 3, without the limit of [1, 63] and must be greater than 0. \n
* @li pads: List of ints, implicit zero paddings on both sides of the input.
* Restriction: "pads" is in the range [0, ksize/2]. \n
* For Atlas Inference Series Product: "pads" length is 6. \n
* For Atlas Training Series Product: "pads" length is 6. \n
* For Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component: "pads"
length is 3. \n
* For Atlas A3 Training Series Product/Atlas A3 Inference Series Product: "pads" length is 3. \n
* For Ascend 910_95 AI Processor: "pads" length is 3. \n
* @li ceil_mode: An optional bool. When true, will use ceil instead of floor in the formula to
* compute the output shape. Default value false.
* @li count_include_pad: An optional bool. When true, will include the zero-padding in the
* averaging calculation. Otherwise, not include paddings in averaging. Default value true.
* @li divisor_override: An optional int, if specified, it will be used as divisor, otherwise
* size of the pooling region will be used. Default value 0, which means this attribute does not take effect.
* @li data_format: An optional string, the format of the input "grads". Defaults to "NDHWC".
* For Atlas Inference Series Product: support "NDHWC" and "NCDHW". \n
* For Atlas Training Series Product: support "NDHWC" and "NCDHW". \n
* For Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component: only
support "NDHWC". \n
* For Atlas A3 Training Series Product/Atlas A3 Inference Series Product: only support "NDHWC". \n
* For Ascend 910_95 AI Processor: only support "NDHWC". \n

* @par Outputs:
* output: A mutable tensor with the same shape as "orig_input_shape" and same type as "grads".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPoolGrad.
*/

REG_OP(AvgPool3DGrad)
    .INPUT(orig_input_shape, TensorType({DT_INT32}))
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AvgPool3DGrad)

} // namespace ge

#endif // OPS_POOLING_AVG_POOL3_D_GRAD_PROTO_H_