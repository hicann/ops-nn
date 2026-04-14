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
 * \file mish_grad_proto.h
 * \brief
 */
#ifndef MISH_GRAD_PROTO_H_
#define MISH_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief PyTorch mish_grad operator.
 * @par Inputs:
 * Three input, including:
 * @li grad: A tensor. Shape, datatype and format is the same as x.
 * @li x: A tensor. Support 1D ~ 8D. Must be one of the following types: float16, float32, bfloat16. Format:ND.
 * @li tanhx: A tensor. Shape, datatype and format is the same as x.
 * @par Outputs:
 * One output, including:
 * x_grad: A tensor. Shape, datatype and format is the same as x.
 */
REG_OP(MishGrad)
    .INPUT(grad, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OPTIONAL_INPUT(tanhx, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OUTPUT(x_grad, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OP_END_FACTORY_REG(MishGrad)
    
}
#endif