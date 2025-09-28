/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_norm_swish_grad_proto.h
 * \brief
 */
#ifndef OPS_NORM_GROUP_NORM_SWISH_GRAD_H_
#define OPS_NORM_GROUP_NORM_SWISH_GRAD_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs the backward operation of group normalization and swish.

 * @par Inputs:
 * Six input, including:
 * @li dy: A Tensor. Group grad. Datatype support float32, float16, bfloat16. Format support ND.
 * @li mean: A Tensor. Mean of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * @li rstd: A Tensor. Reciprocal standard deviation of each group. Datatype support float32, float16, bfloat16. Format
support ND.
 * @li x: A Tensor. Specifies the offset. Datatype support float32, float16, bfloat16. Format support ND. Same shape as
mean.
 * @li gamma: A Tensor. Specifies the scaling factor. Datatype support float32, float16, bfloat16. Format support ND.
Same shape as dy.
 * @li beta: A Tensor. Specifies the intercept. Datatype support float32, float16, bfloat16. Format support ND. Same
shape as gamma.

* @par Attributes:
* @li num_groups: Int. Number specifying the number of group.
* @li data_format: An optional String, Defaults to NCHW.
* @li swish_scale: An optional float. Defaults to "1.0".
* @li dgamma_is_require: An optional bool, controls whether to return weight.grad. Defaults to true.
* @li dbeta_is_require: An optional bool, controls whether to return beta.grad. Defaults to true.

 * @par Outputs:
 * Three output, including:
 * @li dx: A Tensor. x factor grad. Datatype is the same as the input Datatype. Format support ND.
 * @li dgamma: A Tensor. scale factor grad. Datatype is the same as the input Datatype. Format support ND.
 * @li dbeta: A Tensor. offset factor grad. Datatype is the same as the input Datatype. Format support ND.

* @par Third-party framework compatibility
* @li Compatible with the backward of PyTorch operator GroupNorm and Swish.

*/
REG_OP(GroupNormSwishGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(swish_scale, Float, 1.0)
    .ATTR(dgamma_is_require, Bool, true)
    .ATTR(dbeta_is_require, Bool, true)
    .OP_END_FACTORY_REG(GroupNormSwishGrad)
} // namespace ge
#endif // OPS_NORM_GROUP_NORM_SWISH_GRAD_H_