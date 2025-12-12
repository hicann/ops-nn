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
 * \file nn_activation.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_

#include "graph/operator_reg.h"

namespace ge{
     /**
    * @brief SquaredRelu first applies ReLU to the input tensor x, and then squares the result of the ReLU operation.
    *
    * @par Inputs:
    * x: A tensor of type float, bf16 or bfloat16. Shape support 0D ~ 8D.
    * The format must be ND.
    *
    * @par Outputs:
    * y: A Tensor with the same type, shape, format as "x". \n
    */
    REG_OP(SquaredRelu)
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .OP_END_FACTORY_REG(SquaredRelu)
}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_
