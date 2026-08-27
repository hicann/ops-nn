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
*@brief Performs batch normalization .

*@par Inputs:
* Five inputs, including: (NHWC or NCHW supported)
*@li input_x: A 4D Tensor of type float16 or float32.
*@li input_scale: A 1D Tensor of type float32, for the scaling factor.
*@li input_offset: A 1D Tensor of type float32, for the scaling offset.
*@li input_mean: A 1D Tensor of type float32, for the mean used for inference.
* This cannot be used if the operation is used for training.
*@li input_variance: A 1D Tensor of type float32, for the variance used for inference.
* This cannot be used if the operation is used for training . \n

*@par Attributes:
*@li epsilon: An optional float32, specifying the small value
added to variance to avoid dividing by zero. Defaults to "0.0001".
*@li data_format: An optional string, specifying the format of "x". Defaults to "NHWC".
*@li is_training: An optional bool, specifying if the operation
is used for training or inference. Defaults to "True" . \n

*@par Outputs:
* Five outputs, including: (NHWC or NCHW supported)
*@li output_y: A 4D Tensor of type float16 or float32, for the normalized "x".
*@li output_mean: A 1D Tensor of type float32, for the mean of "x".
*@li output_variance: A 1D Tensor of type float32, for the variance of "x".
*@li output_reserve_space_1: A 1D Tensor of type float32, for the mean of "x" for gradient computation.
*@li output_reserve_space_2: A 1D Tensor of type float32, for the variance of "x" for gradient computation . \n

*@attention Constraints:
*@li If the operation is used for inference, then output "reserve_space_1"
has the same value as "mean" and output "reserve_space_2" has the same value as "variance".
*@li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1‰ due to the square root instruction .

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator fused_batch_norm_v2.
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
