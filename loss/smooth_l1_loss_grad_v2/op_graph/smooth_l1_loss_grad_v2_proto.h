/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_grad_v2_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SMOOTH_L1_LOSS_GRAD_V2_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SMOOTH_L1_LOSS_GRAD_V2_H_

#include "graph/operator_reg.h"
#include "nn_norm.h"


namespace ge { 
/**
* @brief Calculates the reversed outputs of the function "smooth_l1_loss_v2". Broadcasting is supported.

* @par Inputs:
* Three Inputs, including:
* @li predict: A ND multi-dimensional tensor of type float16, bfloat16 or float32, specifying the predictive value.
* @li label: A ND multi-dimensional tensor, has same type and shape as "predict", specifying the target value.
* @li dout: A tensor. Has the same type as "predict". \n

* @par Attributes:
* Two Attributes, including:
* @li sigma: An optional float. Specifies the threshold of loss. The value must be non-negative. Defaults to 1.0. \n

* @li reduction: An optional string. Specifies the reduction to apply to
* the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied;
* 'mean': the sum of the output will be divided by the number of elements in
* the output;'sum': the output will be summed. Default: 'mean'. \n

* @par Outputs:
*  gradient: A tensor. Has the same type as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator SmoothL1LossBackward.
*/
REG_OP(SmoothL1LossGradV2)
    .INPUT(predict, TensorType({DT_FLOAT, DT_FLOAT16,DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(dout, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(gradient, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(sigma, Float, 1.0)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SmoothL1LossGradV2)
}

#endif