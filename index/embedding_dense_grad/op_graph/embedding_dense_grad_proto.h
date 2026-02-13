/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_EMBEDDING_DENSE_GRAD_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_EMBEDDING_DENSE_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Calculates the reversed outputs of the function "embedding". \n

* @par Inputs:
* Two inputs, including:
* @li grad: A mutable Tensor of word grad. Must be one of the following types:
*     float32, bfloat16, float16.
* @li indices: A mutable word index Tensor of the int32, int64 type.\n

* @par Attributes:
* @li num_weights: An int attr which use to judge how many words in dict. \n

* @li padding_idx: An integer attribute that specifies which word index should have its gradient filled with zeros. Defaults to "-1". \n

* @li scale_grad_by_freq: An optional bool. Defaults to "False".
*     If "True", "grad_weight" will be scale by word_frequency.
*     If "False", "grad_weight" will not be scale by word_frequency. \n

* @par Outputs:
* y: A mutable output Tensor of new word grad has the same type as "grad". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator EmbeddingDenseGrad.
*/
REG_OP(EmbeddingDenseGrad)
    .INPUT(grad, TensorType({ DT_FLOAT32, DT_FLOAT16, DT_BF16 }))  /* "First operand." */
    .INPUT(indices, TensorType({ DT_INT32, DT_INT64 }))  /* "Second operand." */
    .OUTPUT(y, TensorType({ DT_FLOAT32, DT_FLOAT16, DT_BF16 }))  /* "Result, has same element type as grad input" */
    .REQUIRED_ATTR(num_weights, Int)
    .ATTR(padding_idx, Int, -1)
    .ATTR(scale_grad_by_freq, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingDenseGrad)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_EMBEDDING_DENSE_GRAD_PROTO_H_