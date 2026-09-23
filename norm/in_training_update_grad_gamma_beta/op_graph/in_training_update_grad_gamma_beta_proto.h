/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief Reduces the two input tensors along axis 0 and keeps the reduced dimension.
 *
 * The operator computes `pd_gamma = sum(res_gamma, axis=0, keepdims=true)` and
 * `pd_beta = sum(res_beta, axis=0, keepdims=true)`. Both inputs are required
 * float32 tensors with identical shape and logical format. Both outputs are
 * float32; their shape equals the input shape with dimension 0 set to 1.
 *
 * Ascend 950 accepts NCHW/NHWC for rank-4 inputs, NCDHW/NDHWC for rank-5
 * inputs, and ND for rank-4 or rank-5 inputs. Dynamic dimensions and dynamic
 * rank are supported. When dimension 0 is zero, the result is the zero-valued
 * empty-set sum; when another dimension is zero, the output tensor is empty.
 * The operator has no attributes and is exposed through the GE graph interface.
 *
 * Legacy products keep their existing 4D NCHW/NHWC logical GE contract and
 * may use NDC1HWC0 internally. See the operator README for product differences.
 */

#ifndef IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_PROTO_H
#define IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

#ifndef OPS_PROTO_DEF_IN_TRAINING_UPDATE_GRAD_GAMMA_BETA
#define OPS_PROTO_DEF_IN_TRAINING_UPDATE_GRAD_GAMMA_BETA
REG_OP(INTrainingUpdateGradGammaBeta)
    .INPUT(res_gamma, TensorType({DT_FLOAT}))
    .INPUT(res_beta, TensorType({DT_FLOAT}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingUpdateGradGammaBeta)
#endif // OPS_PROTO_DEF_IN_TRAINING_UPDATE_GRAD_GAMMA_BETA
} // namespace ge

#endif
