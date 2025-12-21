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
 * \file embedding_bag_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_EMBEDDING_BAG_H_
#define OPS_OP_PROTO_INC_EMBEDDING_BAG_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Calculates the reversed outputs of the function "embedding". \n

* @par Inputs:
* Two inputs, including:
* @li grad: A mutable Tensor of word grad. Must be one of the following types:
*     float32, float16, bfloat16
* @li sort_indices: A mutable word index Tensor. Must be one of the following types:
*     int32, int64
* @li pos_idx: A mutable position of sort_indices in the grad Tensor of the int32 type.\n

* @par Attributes:
* @li num_weights: An int attr which use to judge how many words in dict. \n

* @li padding_idx: An int attr judge which word to fill zeros. Defaults to "-1". \n

* @li scale_grad_by_freq: An optional bool. Defaults to "False".
*     If "True", "grad_weight" will be scale by word_frequency.
*     If "False", "grad_weight" will not be scale by word_frequency. \n

* @par Outputs:
* y: A mutable output Tensor of new word grad has the same type as "grads". \n

* @par Restrictions:
* Warning: The grad of input last dim size should be less than 8192 bytes and the
*     sort_indices size should be larger than 32 for Ascend 910_95 AI Processors. \n
*/
REG_OP(EmbeddingDenseGradV2)
    .INPUT(grad, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16})) /* "First operand." */
    .INPUT(sort_indices, TensorType({DT_INT32, DT_INT64}))      /* "Second operand." */
    .INPUT(pos_idx, TensorType({DT_INT32}))                     /* "Thrid operand." */
    .OUTPUT(y, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))   /* "Result, has same element type as two inputs" */
    .REQUIRED_ATTR(num_weights, Int)
    .ATTR(padding_idx, Int, -1)
    .ATTR(scale_grad_by_freq, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingDenseGradV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_EMBEDDING_BAG_H_
