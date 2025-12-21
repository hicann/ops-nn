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
 * \file batch_norm_v3_proto.h
 * \brief
 */
#ifndef OPS_NORM_BATCH_NORM_V3_PROTO_H_
#define OPS_NORM_BATCH_NORM_V3_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Batch normalization (also known as batch norm) is a method used to
* make training of artificial neural networks faster and more stable
* through normalization of the layers' inputs by re-centering and re-scaling.

* @par Inputs:
* @li x: A 4D/5D tensor of type float16/bfloat16/float32, with format NCHW/NHWC/NCDHW/NDHWC. describing the feature_map.
* @li weight: A 1D tensor with the shape is same as dim C of input x, support dtypes are related to input x dtype,
* the following combinations are supported: [x: float16, weight: float16/float32], [x: bfloat16, weight:
bfloat16/float32], [x: float32, weight: float32].
* describing the weight.
* @li bias: A 1D tensor of the same dtype and shape as input weight, describing the bias.
* @li running_mean: A 1D tensor of type float16/bfloat16/float32, the shape is same as dim C of input x,
* the following combinations are supported: [x: float16, running_mean: float16/float32], [x: bfloat16, running_mean:
bfloat16/float32], [x: float32, running_mean: float32].
* describing the running mean.
* @li running_var: A 1D tensor of the same dtype and shape as input running_mean, describing the running var. \n

* @par Attributes:
* @li epsilon: An optional float32, small value added to variance to avoid dividing by zero. Defaults to "1e-5".
* @li momentum: An optional float32, the value used for the running_mean and running_var computation. Defaults to "0.1".
* @li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to "True",
now only support "True". \n

* @par Outputs:
* @li y: A 4D/5D tensor of the same dtype, shape and format as input x, describing the result.
* @li running_mean: A 1D tensor of the same dtype and shape as input running_mean, describing the running mean.
* @li running_var: A 1D tensor of the same dtype and shape as input running_var, describing the running var.
* @li save_mean: A 1D tensor of type float32, describing the mean of "x".
* @li save_rstd: A 1D tensor of type float32, describing the rstd of "x". \n
*/
REG_OP(BatchNormV3)
    .INPUT(x, "T1")
    .INPUT(weight, "T2")
    .INPUT(bias, "T2")
    .INPUT(running_mean, "T4")
    .INPUT(running_var, "T4")
    .OUTPUT(y, "T1")
    .OUTPUT(running_mean, "T4")
    .OUTPUT(running_var, "T4")
    .OUTPUT(save_mean, "T3")
    .OUTPUT(save_rstd, "T3")
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(momentum, Float, 0.1f)
    .ATTR(is_training, Bool, true)
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T3, TensorType({DT_FLOAT}))
    .DATATYPE(T4, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(BatchNormV3)
} // namespace ge

#endif // OPS_NORM_BATCH_NORM_V3_PROTO_H_