/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
 * \file embedding_dense_grad_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_EMBEDDING_DENSE_GRAD_H_
#define OPS_OP_PROTO_INC_EMBEDDING_DENSE_GRAD_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Calculates the reversed outputs of the function "embedding". \n

* @par Inputs:
* Two inputs, including:
* @li grad: A required Tensor. A mutable Tensor of word grad. Must be one of the following types: float16, float32.
* @li indices: A required Tensor. A mutable word index Tensor. Must be one of the following types: int32, int64. \n

* @par Attributes:
* @li num_weights: A required int. The number of words in dict. \n
* @li padding_idx: An optional int judge which word to fill zeros. Defaults to "-1". \n
* @li scale_grad_by_freq: An optional bool. Defaults to "False".
*     If "True", "y" will be scaled by word frequency.
*     If "False", "y" will not be scaled by word frequency. \n

* @par Outputs:
* y: A mutable output Tensor of new word grad. Has the same type as "grad". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator EmbeddingDenseGrad.
*/
REG_OP(EmbeddingDenseGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_weights, Int)
    .ATTR(padding_idx, Int, -1)
    .ATTR(scale_grad_by_freq, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingDenseGrad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_EMBEDDING_DENSE_GRAD_H_
