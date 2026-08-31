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
 * \file batch_norm_grad_ext2_proto.h
 * \brief
 */

#ifndef OPS_BATCH_NORM_GRAD_EXT2_PROTO_H_
#define OPS_BATCH_NORM_GRAD_EXT2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Performs the backpropagation of BatchNorm .

*@par Inputs:
* Five inputs, including:
*@li y_backprop: A 4D Tensor of type float16 or float32, with format NHWC or NCHW, for the gradient.
*@li x: A 4D Tensor of type float16 or float32, with format NHWC or NCHW, the shape is same as input y_backprop.
*@li scale: A 4D Tensor of type float32, with format NHWC or NCHW, the shape is same as input y_backprop.
*@li reserve_space_1: A 4D Tensor of type float32, with format NHWC or NCHW, the shape is same as input y_backprop,
* it is an output of BatchNormExt2.
*@li reserve_space_2: A 4D Tensor of type float32, with format NHWC or NCHW, the shape is same as input y_backprop,
* it is an output of BatchNormExt2 . \n

*@par Attributes:
*@li epsilon: A required float32. A small float number added to the variance of "x".
*@li data_format: A required string for the format.
*@li is_training: A required bool for specifying the operation is for training (true) or inference (false) . \n

*@par Outputs:
*@li x_backprop: A Tensor of type float16 or float32, with format NHWC or NCHW, for the offset of "x".
*@li scale_backprop: A Tensor of type float32, with format NHWC or NCHW, for the offset of "scale".
*@li offset_backprop: A Tensor of type float32, with format NHWC or NCHW, for the offset of "offset".
*@li reserve_space_3: A Tensor of type float32, with format NHWC or NCHW.
*@li reserve_space_4: A Tensor of type float32, with format NHWC or NCHW . \n

*@attention Constraints:
* The preceding layer of this operator must be BatchNormExt2 . \n

*@see BatchNormExt2
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator FusedBatchNormGradV2.
*/
#ifndef OPS_PROTO_DEF_BATCHNORMGRADEXT2
#define OPS_PROTO_DEF_BATCHNORMGRADEXT2
REG_OP(BatchNormGradExt2)
    .INPUT(y_backprop, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BatchNormGradExt2)
#endif // OPS_PROTO_DEF_BATCHNORMGRADEXT2

} // namespace ge
#endif // OPS_BATCH_NORM_GRAD_EXT2_PROTO_H_
