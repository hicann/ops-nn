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
 * \file in_training_reduce_v2_proto.h
 * \brief
 */
#ifndef OPS_NORM_IN_TRAINING_REDUCE_V2_PROTO_H_
#define OPS_NORM_IN_TRAINING_REDUCE_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief InstanceNorm training-forward reduce stage, paired with INTrainingUpdateV2.
* For each instance-channel (n, c), reduces over the spatial axes (H, W for 4D NCHW;
* D, H, W for 5D NCDHW) and outputs the raw sum (sum = Sigma x) and squared sum
* (square_sum = Sigma x^2). The reduction keeps N and C; outputs are raw sums
 * (no division by R, no affine). fp16 inputs are promoted to fp32 for
 * accumulation, and the outputs are always fp32.

* @par Inputs:
* One input:
* @li x: Empty tensors are supported, but only allows the reduction (spatial) axis
*        to have shape 0; the dim N and dim C must not be empty. Must be one of the
 *        following types: float16, float32. 4D with format NCHW, 5D with
*        format NCDHW, or 2D~8D with format ND (the second dim is fixed as dim C).

* @par Outputs:
* Two outputs:
* @li sum: An ND tensor of dtype float32. The number of dims is same as input x,
*          dim N and dim C are same as input x, the reduction axes are 1.
* @li square_sum: An ND tensor of the same dtype (float32) and shape as output sum.
*/

REG_OP(INTrainingReduceV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingReduceV2)
} // namespace ge

#endif // OPS_NORM_IN_TRAINING_REDUCE_V2_PROTO_H_
