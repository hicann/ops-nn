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
 * \file elu_proto.h
 * \brief
 */
#ifndef ELU_PROTO_H_
#define ELU_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Activation function fused from sigmoid and ReLU, with soft saturation
* on the left and no saturation on the right .

* @par Inputs:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bfloat16, float16, float32.

* @par Attributes:
* @li alpha: An optional float32. Defines at which negative value the ELU saturates. Defaults to "1.0".
* @li scale: An optional float32. Input data scaling factor. Defaults to "1.0".
* @li input_scale: An optional float32. Negative data scaling factor. Defaults to "1.0".

* @par Outputs:
* y: A bfloat16, float16, float32, for the normalized result.
* Has the same type, shape and format as input x.

* @par Third-party framework compatibility
* @li Compatible with Tensorflow's Elu operator
* @li Compatible with Caffe's ELULayer operator
* @li Compatible with Pytorch's elu Opeartor
*
*/
REG_OP(Elu)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .ATTR(alpha, Float, 1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(input_scale, Float, 1.0)
    .OP_END_FACTORY_REG(Elu)
} // namespace ge
#endif