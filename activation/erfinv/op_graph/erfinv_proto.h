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
 * \file erfinv_proto.h
 * \brief
 */

#ifndef OP_GRAPH_ERFINV_PROTO_H_
#define OP_GRAPH_ERFINV_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes the inverse error function of each element of input.

*@par Inputs:
*One inputs, including:
* input_x: A tensor. Must be one of the following types:
*     float16, float32, bfloat16. \n

*@par Outputs:
*output_y: A ND Tensor with the same dtype and shape of input_x's. \n

*@par Third-party framework compatibility
*Compatible with the PyTorch operator Erfinv. \n
*/
REG_OP(Erfinv)
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(output_y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(Erfinv)
} // namespace ge
#endif // OP_GRAPH_ERFINV_PROTO_H_
