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
 * \file group_norm_swish_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_GROUP_NORM_SWISH_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_GROUP_NORM_SWISH_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs group normalization and swish. \n

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type bfloat16/float16/float32.
* @li gamma: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the offset. \n

* @par Attributes:
* @li num_groups: An required int32/int64, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to the
* denominator for numerical stability. Defaults to "0.00001".
* @li data_format: An optional String, Defaults to NCHW.
* @li activate_swish: An optional bool.  Defaults to "true".
* @li swish_scale: An optional float.  Defaults to "1". \n

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type bfloat16/float16/float32 for the normalized "x".
* @li mean: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the mean of "x".
* @li rstd: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the rstd of "x". \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm and Swish.

*/
REG_OP(GroupNormSwish)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(eps, Float, 0.00001f)
    .ATTR(activate_swish, Bool, true)
    .ATTR(swish_scale, Float, 1.0)
    .OP_END_FACTORY_REG(GroupNormSwish)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_GROUP_NORM_SWISH_OPS_H_
