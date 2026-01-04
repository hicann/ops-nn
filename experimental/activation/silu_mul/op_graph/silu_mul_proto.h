/*
 * Copyright (c) 2025 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file nn_activation.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_SILU_MUL_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_SILU_MUL_H_

#include "graph/operator_reg.h"

namespace ge {
/**
    * @brief Performs Silu multiplication. z = silu(x) * y. \n

    * @par Inputs:
    * x: A tensor of type float, float16 or bfloat16. Shape support 2D ~ 8D.
    * The format must be ND.
    * y: A tensor of type float, float16 or bfloat16. Shape support 2D ~ 8D.
    * The format must be ND.

    * @par Outputs:
    * z: A tensor has the same type and format as "x".
    * Other dimensions of its shape are the same as those of "x". \n
    */
REG_OP(SiluMul)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SiluMul)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_SILU_MUL_H_