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
 * \file relu_grad_v2_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes rectified linear gradients for a ReLU operation.
 *
 * @par Inputs:
 * Two inputs, including:
 * @li gradients: A tensor of input gradient. Indicates the gradient of backpropagation. Supported type:
 * TensorType::RealNumberType(). \n In Ascend 910_95 AI Processor, support ND format. Others support 5D, format must be
 * NC1HWC0.
 * @li mask: A tensor of input mask. Indicates the positive output. Must be the following types: uint8, uint1. \n
 * In Ascend 910_95 AI Processor, support ND format. Others support 5D, format must be NC1HWC0.
 *
 * @par Outputs:
 * backprops: A tensor of output. Indicates the backpropagation result.
 * When gradient is greater than 0, backprops is the value of gradient.
 * When the value of gradient is less than or equal to 0, the value of backprops is 0.
 * Must have the same type, format and shape as "gradients". \n
 *
 * @attention Constraints:
 * The corresponding Relu operator needs to be called before using this operator on the network. \n
 */
REG_OP(ReluGradV2)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(mask, TensorType({DT_UINT8, DT_UINT1}))
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(ReluGradV2)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_