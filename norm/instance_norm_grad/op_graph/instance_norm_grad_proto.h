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
 * \file instance_norm_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_INSTANCE_NORM_GRAD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_INSTANCE_NORM_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Backward operator for instance normalization. \n
 * @par Inputs:
 * Five inputs, including:
 * @li dy: A tensor. Upstream gradient. Datatype support float32, float16. Format support NDHWC.
 * Same shape as "x": [N, D, H, W, C].
 * @li x: A tensor. Forward input. Datatype support float32, float16. Format support NDHWC.
 * @li variance: A tensor. Per-instance variance (raw variance, NOT rstd). Datatype support float32, float16.
 * Format support NDHWC. Shape [N, 1, 1, 1, C].
 * @li mean: A tensor. Per-instance mean. Datatype support float32, float16. Format support NDHWC.
 * Shape same as "variance".
 * @li gamma: A tensor. Per-channel scale. Datatype support float32, float16. Format support NDHWC.
 * 1-D, size equals the C-axis of "x". \n

 * @par Outputs:
 * Three outputs, including:
 * @li pd_x: A tensor. Gradient wrt x. Same datatype, format and shape as "x".
 * @li pd_gamma: A tensor. Gradient wrt gamma. Same datatype, format and shape as "gamma".
 * @li pd_beta: A tensor. Gradient wrt beta. Same datatype, format and shape as "gamma". \n

 * @attention Constraints:
 * The epsilon is hardcoded to 1e-6 inside the kernel; there is NO epsilon attribute. \n

 * @par Third-party framework compatibility
 * @li Compatible with the backward of the TensorFlow operator InstanceNormGrad.
 */
#ifndef OPS_PROTO_DEF_INSTANCENORMGRAD
#define OPS_PROTO_DEF_INSTANCENORMGRAD
REG_OP(InstanceNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(InstanceNormGrad)
#endif // OPS_PROTO_DEF_INSTANCENORMGRAD
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_INSTANCE_NORM_GRAD_OPS_H_
