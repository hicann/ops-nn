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
 * \file swi_glu_grad_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SWI_GLU_GRAD_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SWI_GLU_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge{
    /**
    * @brief Compute the SwiGluGrad,
    * where the activations function in GLU is SwishGrad.

    * @par Inputs:
    * two input, including:
    * @li y_grad: A Tensor, which is the output gradient of forward operator and which \n
    * has the same shape as "x" except for the dimension specified by the "dim" parameter. \n
    * The dimension size specified by "dim" is half of the corresponding dimension of x. \n
    * Must be one of the following types: bfloat16, float16, float32.
    * @li x: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Outputs:
    * one Output, including:
    * x_grad: A Tensor, which is the gradient of x and has the same shape as "x". \n
    * Must be one of the following types: bfloat16, float16, float32.

    * @par Attributes:
    * one attribute, including:
    * @li dim: A optional int. The dimension to be split, default is -1.

    * @par Third-party framework compatibility:
    * New operator SwiGluGrad.
    */
    REG_OP(SwiGluGrad)
        .INPUT(y_grad, "T")
        .INPUT(x, "T")
        .OUTPUT(x_grad, "T")
        .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(dim, Int, -1)
        .OP_END_FACTORY_REG(SwiGluGrad)
}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_SWI_GLU_GRAD_PROTO_H_
