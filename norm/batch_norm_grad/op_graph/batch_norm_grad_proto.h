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
 * \file batch_norm_grad_proto.h
 * \brief
 */

#ifndef OPS_BATCH_NORM_GRAD_PROTO_H_
#define OPS_BATCH_NORM_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Performs the backpropagation of BatchNorm.

*@par Inputs:
* Six inputs, including:
*@li y_backprop: A 4D or 5D Tensor of type bfloat16, float16 or float32, with format NCHW, NHWC, NDHWC or NCDHW, for the gradient.
*@li x: A 4D or 5D Tensor of type bfloat16, float16 or float32, with format NCHW, NHWC, NDHWC or NCDHW, the same shape with "y_backprop".
*@li scale: A 1D Tensor of type float32, with format ND, shape must be C channel.
*@li reserve_space_1: A 1D Tensor of type float32, with format ND, shape must be C channel. It is an output of BatchNorm.
* When in training mode, it represents the saved mean of "x". And in inference mode, it represents the running mean of "x".
*@li reserve_space_2: A 1D Tensor of type float32, with format ND, shape must be C channel. It is an output of BatchNorm.
*@li reserve_space_3: A 1D optional Tensor of type float32, with format ND. Not used and not involved in calculations. When in training mode,
* it represents the saved inverse standard deviation of "x" And in inference mode, it represents the running variance of "x".  \n

*@par Attributes:
*@li epsilon: An optional float32. Defaults to "1e-4". A small float number used to add with running variance of "x" in inference mode.
*@li data_format: An optional string. Defaults to "NHWC". Should be same as y_backprop/x/x_backprop's dtype.
*@li is_training: An optional bool. Defaults to "true". Specifies the operation is for training (default) or inference.
*@li output_mask: An optional ListBool. Defaults to [true, false, false]. Valid only in inference mode, it determines whether
* the outputs "x_backprop", "scale_backprop" and "offset_backprop" contain actual reseluts. \n

*@par Outputs:
*@li x_backprop: A 4D or 5D Tensor of type bfloat16, float16 or float32, with format NCHW, NHWC, NDHWC or NCDHW. For the offset of "x".
* the same shape with "x".
*@li scale_backprop: A Tensor of type float32, with format ND, for the offset of "scale", the same shape format with scale.
*@li offset_backprop: A Tensor of type float32, with format ND, for the offset of "offset", the same shape format with scale.
*@li reserve_space_4: A Tensor of type float32, with shape ND. The same shape format with scale.
*@li reserve_space_5: A Tensor of type float32, with shape ND. The same shape format with scale. \n

*@attention Constraints:
* The preceding layer of this operator must be operator BatchNorm . \n

*@see BatchNorm
*@par Third-party framework compatibility
* Compatible with the TensorFlow operators FusedBatchNormGrad, FusedBatchNormGradV2 and FusedBatchNormGradV3.
*/
REG_OP(BatchNormGrad)
    .INPUT(y_backprop, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .OUTPUT(x_backprop, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_5, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-4)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .ATTR(output_mask, ListBool, {true, false, false})
    .OP_END_FACTORY_REG(BatchNormGrad)

} // namespace ge
#endif // OPS_BATCH_NORM_GRAD_PROTO_H_
