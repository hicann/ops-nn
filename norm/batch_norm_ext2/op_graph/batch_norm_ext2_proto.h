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
 * \file batch_norm_ext2_proto.h
 * \brief
 */
#ifndef OPS_NORM_BATCH_NORM_EXT2_PROTO_H_
#define OPS_NORM_BATCH_NORM_EXT2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Performs batch normalization on a 4D tensor, compatible with the TensorFlow operator fused_batch_norm_v2.
 *
 * @par Inputs
 * Five inputs, with format constraints as follows:
 * @li input_x: A 4D tensor of type float16 or float32, with format NHWC (batch, height, width, channels) or
 *        NCHW (batch, channels, height, width). The data to be normalized.
 * @li input_scale: A 1D tensor of type float32, with length equal to the number of channels in "input_x".
 *        Specifies the scaling factor (gamma) applied after normalization.
 * @li input_offset: A 1D tensor of type float32, with length equal to the number of channels in "input_x".
 *        Specifies the offset (beta) applied after scaling.
 * @li input_mean: An optional 1D tensor of type float32, with length equal to the number of channels in "input_x".
 *        - Inference mode (is_training=false): Must be provided, representing the population mean.
 *        - Training mode (is_training=true): Must be empty.
 * @li input_variance: An optional 1D tensor of type float32, with length equal to the number of channels in "input_x".
 *        - Inference mode (is_training=false): Must be provided, representing the population variance.
 *        - Training mode (is_training=true): Must be empty.
 *
 * @par Attributes
 * @li epsilon: Optional float32. Small value added to variance to avoid division by zero.
 *        Defaults to 0.0001f.
 * @li data_format: Optional string. Specifies the data format of "input_x".
 *        Allowed values: "NHWC" (default), "NCHW".
 * @li is_training: Optional bool. Specifies operation mode:
 *        - true: Training mode (computes batch mean/variance from the input).
 *        - false: Inference mode (uses provided mean/variance for normalization).
 *        Defaults to true.
 *
 * @par Outputs
 * Five outputs:
 * @li output_y: A tensor with the same shape, type, and format as "input_x", containing normalized values.
 *        (Required output)
 * @li output_mean: A 1D tensor of type float32 (channel dimension).
 *        - Training mode: Mean of the current batch (computed over the spatial dimensions).
 *        - Inference mode: Equal to input "input_mean" (for compatibility).
 *        (Required output)
 * @li output_variance: A 1D tensor of type float32 (channel dimension).
 *        - Training mode: Variance of the current batch computed over the spatial dimensions
 *          with Bessel's correction (unbiased).
 *        - Inference mode: Equal to input "input_variance" (for compatibility).
 *        (Required output)
 * @li output_reserve_space_1: A 1D tensor of type float32 (channel dimension). Reserved for gradient computation.
 *        - Training mode: Same as the batch mean (saved mean).
 *        - Inference mode: Same as input "input_mean".
 * @li output_reserve_space_2: A 1D tensor of type float32 (channel dimension). Reserved for gradient computation.
 *        - Training mode: The saved inverse std (1/sqrt(epsilon + variance)).
 *        - Inference mode: Same as input "input_variance".
 */
#ifndef OPS_PROTO_DEF_BATCHNORMEXT2
#define OPS_PROTO_DEF_BATCHNORMEXT2
REG_OP(BatchNormExt2)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_scale, TensorType({DT_FLOAT}))
    .INPUT(input_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(input_mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(input_variance, TensorType({DT_FLOAT}))
    .OUTPUT(output_y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_mean, TensorType({DT_FLOAT}))
    .OUTPUT(output_variance, TensorType({DT_FLOAT}))
    .OUTPUT(output_reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(output_reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNormExt2)
#endif // OPS_PROTO_DEF_BATCHNORMEXT2
} // namespace ge

#endif // OPS_NORM_BATCH_NORM_EXT2_PROTO_H_
