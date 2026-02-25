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
 * \file instance_norm_proto.h
 * \brief
 */
#ifndef OPS_NORM_INSTANCE_NORM_PROTO_H_
#define OPS_NORM_INSTANCE_NORM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Instance normalization (also known as instance norm) is a method used to
* make training of artificial neural networks faster and more stable
* through normalization of the layers' inputs by re-centering and re-scaling.

* @par Inputs:
* Three inputs, including:
* @li x: Empty tensors are not supported. Must be one of the following types: bfloat16, float16, float32.
         4D/5D with format NCHW/NHWC/NCDHW/NDHWC.
         2D~8D with format ND, the second dim is fixed as the dim C.
* @li gamma: Empty tensors are not supported. An ND Tensor. 
             The data type is same as input x, if not, explicitly set to float32.
             The shape is same as dim C of input x.
* @li beta: Empty tensors are not supported. An ND tensor of the same dtype and shape as input gamma.

* @par Attributes:
* Two attributes, including:
* @li data_format: A optional attribute, the type is string. Defaults to "NDHWC".
* @li epsilon: A optional attribute, the type is float. Defaults to "1e-6".

* @par Outputs:
* Three outputs, including:
* @li y: Empty tensors are not supported. The shape, data type and format are the same as the input x.
* @li mean: Empty tensors are not supported. An ND tensor of the same dtype as input gamma, 
            the number of dim is same as input x, 
            the shape size of the non-reduction axis is same as input x, 
            the reduction axis is 1.
* @li variance: Empty tensors are not supported. An ND tensor of the same dtype and shape as output mean.
*/

REG_OP(InstanceNorm)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(data_format, String, "NDHWC")
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(InstanceNorm)
} // namespace ge

#endif // OPS_NORM_INSTANCE_NORM_PROTO_H_
