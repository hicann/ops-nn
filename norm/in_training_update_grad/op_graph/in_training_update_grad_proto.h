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
 * \file in_training_update_grad_proto.h
 * \brief
 */
#ifndef OPS_NORM_IN_TRAINING_UPDATE_GRAD_PROTO_H_
#define OPS_NORM_IN_TRAINING_UPDATE_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The first (reduce-over-spatial) stage of InstanceNorm training backward.
* For each (N, C) instance it normalizes x with the given mean/variance and reduces
* dy*x_norm and dy over the spatial dimensions (keepdims). It does NOT reduce over N
* (that is the job of the downstream op INTrainingUpdateGradGammaBeta).

* @par Inputs:
* Four inputs, including:
* @li dy: A tensor with the full spatial dims. Must be one of the following types: float16, float32.
          6D with format NDC1HWC0.
* @li x: A tensor with the same dtype/format/shape as dy.
* @li variance: An NDC1HWC0 tensor of type float32, per-instance variance, spatial dims are 1.
* @li mean: An NDC1HWC0 tensor of type float32, per-instance mean, spatial dims are 1.

* @par Outputs:
* Two outputs, including:
* @li res_gamma: An NDC1HWC0 tensor of type float32, equals sum_over_spatial(dy * x_norm), spatial dims are 1.
* @li res_beta: An NDC1HWC0 tensor of type float32, equals sum_over_spatial(dy), spatial dims are 1.

* @attention The epsilon added to variance is a compile-time constant 1e-6 (not an attribute).
*/

REG_OP(INTrainingUpdateGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .OUTPUT(res_gamma, TensorType({DT_FLOAT}))
    .OUTPUT(res_beta, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingUpdateGrad)
} // namespace ge

#endif // OPS_NORM_IN_TRAINING_UPDATE_GRAD_PROTO_H_
