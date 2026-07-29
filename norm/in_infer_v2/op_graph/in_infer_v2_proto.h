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
 * \file in_infer_v2_proto.h
 * \brief INInferV2 proto 定义（注册体与 canndev built-in 逐字一致：
 *        ops/built-in/op_proto/inc/reduce_ops.h 中的 REG_OP(INInferV2)）
 */
#ifndef OPS_NORM_IN_INFER_V2_PROTO_H_
#define OPS_NORM_IN_INFER_V2_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Performs instance normalization for inference .

*@par Inputs:
* Five inputs, including:
*@li x: A Tensor of type float16 or float32.
*@li gamma: A optional Tensor of type float32, for the scaling gamma, with shape [N, C1, 1, 1, C0].
*@li beta: A optional Tensor of type float32, for the scaling beta, with the same shape of gamma.
*@li mean: A optional ensor of type float32, for the mean, with the same shape of gamma.
*@li variance: A optional Tensor of type float32, for the variance, with the same shape of gamma. \n

*@par Attributes:
*epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero.
Defaults to "0.00001" . \n

*@par Outputs:
*@li y: A Tensor of type float16 or float32 for the normalized "x".
*@li batch_mean: A Tensor of type float32 for the result mean.
*@li batch_variance: A Tensor of type float32 for the result variance . \n

*@attention Constraints:
*For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 0.001 due to the square root instruction.
*/
REG_OP(INInferV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.00001)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INInferV2)

} // namespace ge

#endif // OPS_NORM_IN_INFER_V2_PROTO_H_
