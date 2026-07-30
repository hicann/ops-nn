/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SMOOTH_L1_LOSS
#define OPS_OP_PROTO_INC_SMOOTH_L1_LOSS

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Creates a criterion that uses a squared term if the absolute
* element-wise error falls below sigma and an L1 term otherwise.

* @par Inputs:
* @li predict: A ND multi-dimensional tensor of type float16, bfloat16 or float32,
* specifying the predictive value. \n
* @li label: A ND multi-dimensional tensor, has same type and shape as "predict",
* specifying the target value. \n

* @par Attributes:
* @li sigma: An optional float. Specifies the threshold of loss. The value must be
* non-negative. Defaults to "1.0". \n

* @par Outputs:
* loss: A multi-dimensional Tensor of type float16, bfloat16 or float32. Indicates
* the loss between the predictive value and target value.
* Has the same dimensions as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator smooth_l1_loss (reduction='none'). \n
*/
REG_OP(SmoothL1Loss)
    .INPUT(predict, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(loss, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1Loss)
} // namespace ge

#endif
