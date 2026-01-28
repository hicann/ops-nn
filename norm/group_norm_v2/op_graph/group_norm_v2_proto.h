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
 * \file group_norm_v2_proto.h
 * \brief
 */

#ifndef GROUP_NORM_V2_PROTO_H_
#define GROUP_NORM_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs group normalization . \n

* @par Inputs:
* Three inputs
* @li x: A ND tensor of type bfloat16, float16, float32. The input feature map will be processed by group normalization.
* "x" supports 2-8 dimensions (N, C, *), the calculation logic only cares about the first two dimensions (N and C),
* and the rest can all be combined into one dimension. 
* The data type and shape of x must meet the following conditions:  
* - When the data type of x is float32, C/num_groups must be a multiple of 8.  
* - When the data type of x is float16 or bfloat16, C/num_groups must be a multiple of 16.
* @li gamma: A ND tensor of type bfloat16, float16, float32. Must be 1D. Specifies the scaling factor.
* The value of "gamma" needs to be consistent with the C-axis value of "x". Has the same dype as "x".
* @li beta: A ND tensor of type bfloat16, float16, float32. Must be 1D. Specifies the offset.
* The value of "beta" needs to be consistent with the C-axis value of "x". Has the same dype as "x". \n

* @par Attributes:
* @li num_groups: An required int32, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to variance to avoid dividing by zero. Defaults to "0.00001".
* @li data_format: An optional string, specifying the format of "x". Defaults to "NHWC". This parameter is reserved and does not take effect.
* @li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to "True".

* - When set to true, it indicates training mode and uses the mean and variance of the current batch.
* - When set to false, it indicates inference mode and uses the mean and variance saved during training. \n

* @par Outputs:
* Three outputs
* @li y: A ND tensor of type bfloat16, float16, float32 for the normalized "x". Has the same type, format and shape as "x".  
* @li mean: A ND tensor of type bfloat16, float16, float32. Must be 2D (N, num_groups). Specifies the mean of "x".
* Has the same dype as "x".
* @li rstd: A ND tensor of type bfloat16, float16, float32. Must be 2D (N, num_groups). Specifies the rstd of "x".
* Has the same dype as "x". \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm.

*/
REG_OP(GroupNormV2)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NHWC")
    .ATTR(eps, Float, 0.00001f)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(GroupNormV2)

} // namespace ge

#endif // GROUP_NORM_V2_PROTO_H_