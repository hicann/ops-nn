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
 * \file soft_margin_loss_grad_proto.h
 * \brief SoftMarginLossGrad IR 注册（GE IR 图模式）
 */
#ifndef OPS_OP_PROTO_INC_SOFT_MARGIN_LOSS_GRAD_H_
#define OPS_OP_PROTO_INC_SOFT_MARGIN_LOSS_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief SoftMarginLoss backward gradient.
*  out = cof * (-target*exp(-self*target)) / (1+exp(-self*target)) * grad_output
*@par Inputs:
* @li self: predicted x, float16/float32/bfloat16, ND.
* @li target: label y, float16/float32/bfloat16, ND.
* @li grad_output: upstream gradient, float16/float32/bfloat16, ND.
*@par Attributes:
* reduction: "none"/"mean"/"sum"; mean → cof=1/N else 1.0.
*@par Outputs:
* out: broadcast(self, target, grad_output), same dtype.
*/
REG_OP(SoftMarginLossGrad)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SoftMarginLossGrad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SOFT_MARGIN_LOSS_GRAD_H_
