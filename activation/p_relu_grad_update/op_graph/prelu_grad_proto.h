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
 * \file prelu_grad_proto.h
 * \brief
 */
#ifndef OPS_NN_PRELU_GRAD_PROTO_H_
#define OPS_NN_PRELU_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Performs the backpropagation of PRelu for training scenarios .

*@par Inputs:
*@li grads: Input gradient. ND Tensors are supported.
*The data type can be float16, float32 or bfloat16.
*@li features: A ND Tensor of type float16, float32 or bfloat16. Has same dtype with grads. Support 2D~8D tensor.
*@li weights: A Scalar or 1D Tensor, has same dtype with grads.
*specifying the weight.
*The number of dimensions must be the same as the number of 
*channels(the dim1 number of features's shape) or 1.\n

*@par Outputs:
*@li dx: Reverse gradient of "features".
*Has the same shape and dtype with "features".
*@li da: Reverse gradient of "weights".
*Has the same shape and dtype with "weights". \n

*@par Third-party framework compatibility
* Compatible with PyTorch operator PReluGrad.
*/
REG_OP(PReluGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(weights, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(da, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(PReluGrad)
}
#endif //OPS_NN_PRELU_GRAD_PROTO_H_
