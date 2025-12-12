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
 * \file heaviside_proto.h
 * \brief
 */
#ifndef OPS_ACTIVATION_HEAVISIDE_GRAPH_PLUGIN_HEAVISIDE_PROTO_H_
#define OPS_ACTIVATION_HEAVISIDE_GRAPH_PLUGIN_HEAVISIDE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Apply the heaviside step function element-wise to the input. \n

* @par Inputs:
* @li input: A tensor of type float32, float16 or bfloat16. Shape support 0D ~ 8D.
* The format must be ND.
* @li values: A tensor of type float32, float16 or bfloat16. Shape support 0D ~ 8D.
* The format must be ND.

* @par Outputs:
* output: A tensor has the same type, shape and format as "input". \n
*/
REG_OP(Heaviside)
    .INPUT(input, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(values, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Heaviside)

} // namespace ge

#endif