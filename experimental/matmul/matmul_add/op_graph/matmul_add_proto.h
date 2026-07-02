/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
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
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_MATMUL_ADD_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_MATMUL_ADD_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(MatmulAdd)
    .INPUT(a, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MatmulAdd)
} // namespace ge

#endif
