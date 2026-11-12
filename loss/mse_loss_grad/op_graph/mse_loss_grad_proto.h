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
 * \file mse_loss_grad_proto.h
 * \brief
 */
#ifndef MSE_LOSS_GRAD_PROTO_H_
#define MSE_LOSS_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes gradients of mse loss.

* @par Inputs:
* @li predict: An ND tensor of type float16, float32 or bfloat16.
* @li label: An ND tensor of type float16, float32 or bfloat16.
* @li dout: An ND tensor of type float16, float32 or bfloat16. \n

* @par Attributes:
* reduction: An optional string.Defaults to "mean". \n

* @par Outputs:
 * y: An ND tensor whose shape is the broadcast result of the shapes of "predict", "label" and "dout",
 * with the same type as the inputs. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator MseLossGrad.
*/
REG_OP(MseLossGrad)
    .INPUT(predict, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .INPUT(dout, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(MseLossGrad)

} // namespace ge
#endif // MSE_LOSS_GRAD_PROTO_H_
