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
 * \file batch_norm3_d_grad_proto.h
 * \brief
 */

#ifndef OPS_BATCH_NORM3D_GRAD_PROTO_H_
#define OPS_BATCH_NORM3D_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Performs the backpropagation of BatchNorm .

*@par Inputs:
* Five inputs, including:
*@li y_backprop: A 5D Tensor of type float16 or float32, with format NDHWC, NCDHW, for the gradient.
*@li x: A 5D Tensor of type float16 or float32, with format NDHWC, NCDHW.
*@li scale: A 5D Tensor of type float32, with format NDHWC, NCDHW.
*@li reserve_space_1: A 5D Tensor of type float32, with format NDHWC, NCDHW. It is an output of BatchNorm.
*@li reserve_space_2: A 5D Tensor of type float32, with format NDHWC, NCDHW. It is an output of BatchNorm . \n

*@par Attributes:
*@li epsilon: An optional float32. Defaults to "0.0001". A small float number added to the variance of "x".
*@li data_format: An optional string. Defaults to "NCDHW".
*@li is_training: An optional bool. Defaults to "true". Specifies the operation is for training (default) or inference .
\n

*@par Outputs:
*@li x_backprop: A Tensor of type float16 or float32, with format NDHWC, NCDHW, for the offset of "x".
*@li scale_backprop: A Tensor of type float32, with format NDHWC, NCDHW, for the offset of "scale".
*@li *offset_backprop: A Tensor of type float32, with format NDHWC, NCDHW, for the offset of "offset".
*@li *reserve_space_4: A Tensor of type float32, with shape NDHWC, NCDHW. Pass "None" to skip this output.
*@li *reserve_space_5: A Tensor of type float32, with shape NDHWC, NCDHW. Pass "None" to skip this output . \n

*@attention Constraints:
* The preceding layer of this operator must be operator BatchNorm . \n

*@see BatchNorm
*@par Third-party framework compatibility
* Compatible with the TensorFlow operators FusedBatchNormGradV2 and FusedBatchNorm3DGrad.
*/
#ifndef OPS_PROTO_DEF_BATCHNORM3DGRAD
#define OPS_PROTO_DEF_BATCHNORM3DGRAD
REG_OP(BatchNorm3DGrad)
    .INPUT(y_backprop, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_5, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm3DGrad)
#endif // OPS_PROTO_DEF_BATCHNORM3DGRAD

} // namespace ge
#endif // OPS_BATCH_NORM3D_GRAD_PROTO_H_
