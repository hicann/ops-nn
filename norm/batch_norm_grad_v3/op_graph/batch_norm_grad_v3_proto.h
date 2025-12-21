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
 * \file batch_norm_grad_v3_proto.h
 * \brief
 */
#ifndef OPS_BATCH_NORM_GRAD_V3_PROTO_H_
#define OPS_BATCH_NORM_GRAD_V3_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
    
/**
 * @brief Computes the gradients for Batch Normalization.
 * This operation computes the gradients of the input tensor, weight, and bias during the backpropagation step
 * of batch normalization, using the saved mean and inverse standard deviation from the forward pass. It is typically
 * used in the context of training deep learning models with batch normalization layers.

* @par Inputs:
* @li dy: Gradient of the loss w.r.t the output, a tensor of type float16/bfloat16/float32. Supported formats: NCHW, NHWC, NDHWC, NCDHW.
* @li x: Input feature map, a tensor of type float16/bfloat16/float32. Supported formats: NCHW, NHWC, NDHWC, NCDHW.
* @li weight: Scale parameter, a 1D tensor of type float16/bfloat16/float32. Supported formats: ND.
* @li running_mean: Running mean, a 1D tensor of type float16/bfloat16/float32. Supported formats: ND.
* @li running_var: Running variance, a 1D tensor of type float16/bfloat16/float32. Supported formats: ND.
* @li save_mean: Saved mean from the forward pass, a 1D tensor of type float32. Supported formats: ND.
* @li save_rstd: Saved inverse standard deviation from the forward pass, a 1D tensor of type float32. Supported formats: ND.
*
* @par Attributes:
* @li is_training: (Optional) A bool value. Whether the operation is in training mode. Default: true
* @li epsilon: (Optional) A small float32 value added for numerical stability. Default: 1e-5

* @par Outputs:
* @li dx: Gradient of the loss w.r.t the input, a tensor of type float16/bfloat16/float32. Supported formats: NCHW, NHWC, NDHWC, NCDHW.
* @li dweight: Gradient of the loss w.r.t the scale parameter, a 1D tensor of type float16/bfloat16/float32. Supported formats: ND.
* When is_training is false, this output is meaningless.
* @li dbias: Gradient of the loss w.r.t the bias, a 1D tensor of type float16/bfloat16/float32. Supported formats: ND.
* When is_training is false, this output is meaningless.
*/

REG_OP(BatchNormGradV3)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(running_mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(running_var, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(save_mean, TensorType({DT_FLOAT}))
    .INPUT(save_rstd, TensorType({DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dweight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dbias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(is_training, Bool, true)
    .ATTR(epsilon, Float, 1e-5)
    .OP_END_FACTORY_REG(BatchNormGradV3)

} // namespace ge
#endif // OPS_BATCH_NORM_GRAD_V3_PROTO_H_
