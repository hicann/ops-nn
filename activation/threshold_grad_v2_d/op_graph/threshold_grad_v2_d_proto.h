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
 * \file threshold_grad_v2_d_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_THRESHOLD_GRAD_V2_D_H_
#define OPS_OP_PROTO_INC_THRESHOLD_GRAD_V2_D_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Thresholds each element of the input gradient.

 * @par Inputs:
 * Two inputs, including:
 * @li gradients: A tensor of type int32, float16 or bfloat16.
 * @li features: A tensor of the same type and shape as "gradients". \n

 * @par Attributes:
 * threshold: A required float used as the threshold value. \n

 * @par Outputs:
 * backprops: A tensor with the same type and shape as "gradients".
 */
#ifndef OPS_PROTO_DEF_THRESHOLDGRADV2D
#define OPS_PROTO_DEF_THRESHOLDGRADV2D
REG_OP(ThresholdGradV2D)
    .INPUT(gradients, TensorType({DT_INT32, DT_FLOAT16, DT_BF16}))
    .INPUT(features, TensorType({DT_INT32, DT_FLOAT16, DT_BF16}))
    .OUTPUT(backprops, TensorType({DT_INT32, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(threshold, Float)
    .OP_END_FACTORY_REG(ThresholdGradV2D)
#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_THRESHOLD_GRAD_V2_D_H_
