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
 * \file batch_norm_proto.h
 * \brief
 */
#ifndef OPS_NORM_BATCH_NORM_PROTO_H_
#define OPS_NORM_BATCH_NORM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Performs batch normalization with support for 4D/5D tensors and training/inference modes.
* 
* @par Inputs
* Five inputs, with format constraints as follows:
* @li x: A 4D or 5D tensor of type float16, bfloat16, or float32. 
*        Supported data formats:
*        - 4D: NHWC (batch, height, width, channels) or NCHW (batch, channels, height, width).
*        - 5D: NDHWC (batch, depth, height, width, channels) or NCDHW (batch, channels, depth, height, width).
* @li scale: A 1D tensor of type float32, with length equal to the number of channels in "x".
*        Specifies the scaling factor (gamma) applied after normalization.
* @li offset: A 1D tensor of type float32, with length equal to the number of channels in "x".
*        Specifies the offset (beta) applied after scaling.
* @li mean: A 1D tensor of type float32, with length equal to the number of channels in "x".
*        - Inference mode (is_training=false): Must be provided as input, representing the 
*          moving mean computed during training.
*        - Training mode (is_training=true): Optional input. When provided, will be used to 
*          initialize the moving mean for updates; when None, moving mean starts from zeros.
* @li variance: A 1D tensor of type float32, with length equal to the number of channels in "x".
*        - Inference mode (is_training=false): Must be provided as input, representing the
*          moving variance computed during training.
*        - Training mode (is_training=true): Optional input. When provided, will be used to
*          initialize the moving variance for updates; when None, moving variance starts from ones.
* 
* @par Attributes
* @li epsilon: Optional float32. Small value added to variance to avoid division by zero.
*        Defaults to 0.0001f.
* @li data_format: Optional string. Specifies the data format of "x". 
*        Allowed values: "NHWC" (4D default), "NCHW" (4D), "NDHWC" (5D), "NCDHW" (5D).
* @li is_training: Optional bool. Specifies operation mode:
*        - true: Training mode (computes batch mean/variance and updates moving stats).
*        - false: Inference mode (uses provided mean/variance for normalization).
*        Defaults to true.
* @li exponential_avg_factor: Optional float32. Factor for updating moving averages during training.
*        Formula: new_mean = (1 - factor) * old_mean + factor * batch_mean.
*        Defaults to 1.0f.
* 
* @par Outputs
* Up to six outputs, with shape and format matching "x" unless specified:
* @li y: A tensor with the same rank (4D/5D), type, and format as "x", containing normalized values.
*        (Required output)
* @li batch_mean: A 1D tensor of type float32 (channel dimension).
*        - Training mode: Mean of the current batch (computed over spatial dimensions).
*        - Inference mode: Equal to input "mean" (for compatibility).
*        (Required output)
* @li batch_variance: A 1D tensor of type float32 (channel dimension).
*        - Training mode: Variance of the current batch (computed over spatial dimensions, with Bessel's correction).
*        - Inference mode: Equal to input "variance" (for compatibility).
*        (Required output)
* @li reserve_space_1: Optional 1D tensor of type float32 (channel dimension).
*        Reserved for gradient computation.
*        - Training mode: Same as batch_mean.
*        - Inference mode: Same as input "mean".
* @li reserve_space_2: Optional 1D tensor of type float32 (channel dimension).
*        Reserved for gradient computation.
*        - Training mode: saved inv_var (1/sqrt(epsilon + variance), to be reused in the backward gradient computation.
*        - Inference mode: Same as input "variance".
* @li reserve_space_3: A 1D tensor of type float32 with exactly one element.
*        Exists solely for TensorFlow compatibility and contains no meaningful data.
*/
REG_OP(BatchNorm)
    .INPUT(x, "T1")
    .INPUT(scale, "T2")
    .INPUT(offset, "T2")
    .OPTIONAL_INPUT(mean, "T2")
    .OPTIONAL_INPUT(variance, "T2")
    .OUTPUT(y, "T1")
    .OUTPUT(batch_mean, "T2")
    .OUTPUT(batch_variance, "T2")
    .OUTPUT(reserve_space_1, "T2")
    .OUTPUT(reserve_space_2, "T2")
    .OUTPUT(reserve_space_3, "T2")
    .ATTR(epsilon, Float, 1e-4f)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .ATTR(exponential_avg_factor, Float, 1.0f)
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BatchNorm)
} // namespace ge

#endif // OPS_NORM_BATCH_NORM_PROTO_H_