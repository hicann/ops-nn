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
 * \file batch_norm3d_proto.h
 * \brief
 */
#ifndef OPS_NORM_BATCH_NORM3D_PROTO_H_
#define OPS_NORM_BATCH_NORM3D_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Performs batch normalization .

*@par Inputs:
* Five inputs, including: (NDHWC, NCDHW)
*@li x: A 5D Tensor of type float16 or float32, with format NDHWC or NCDHW.
*@li scale: A Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW.
Specifies the scaling factor.
*@li offset: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
Specifies the offset.
*@li mean: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
Specifies the mean used for inference. Must be "None" if the
operation is used for training.
*@li variance: A Tensor of type float32. Must be 3D if input "x" is with format NHWC or NCHW.
Specifies the variance used for inference. Must be "None"
if the operation is used for training . \n

*@par Attributes:
*@li epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero. Defaults to
"0.0001".
*@li data_format: An optional string, specifying the format of "x". Defaults to "NCDHW".
*@li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to "True" .
\n

*@par Outputs:
* Five outputs, including: (NDHWC, NCDHW)
*@li y: A 5D Tensor of type float16 or float32 for the normalized "x", with format NDHWC or NCDHW.
*@li batch_mean: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
Specifies the mean of "x".
*@li batch_variance: A Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW.
Specifies the variance of "x".
*@li reserve_space_1: An optional Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW.
Specifies the mean of "x" for gradient computation. Pass "None" to skip this output.
*@li reserve_space_2: An optional Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the variance of "x" for gradient computation. Pass "None" to skip this output . \n

*@attention Constraints:
*@li If the operation is used for inference and outputs "reserve_space_1" and "reserve_space_2" are available,
then "reserve_space_1" has the same value as "mean" and "reserve_space_2" has the same value as "variance".
*@li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1‰ due to the square root instruction .
\n

*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator fused_batch_norm.
*@li Compatible with the TensorFlow operator fused_batch_norm_v2.
*/
#ifndef OPS_PROTO_DEF_BATCHNORM3D
#define OPS_PROTO_DEF_BATCHNORM3D
REG_OP(BatchNorm3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm3D)
#endif // OPS_PROTO_DEF_BATCHNORM3D
} // namespace ge

#endif // OPS_NORM_BATCH_NORM3D_PROTO_H_
