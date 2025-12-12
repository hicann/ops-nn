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
* Four inputs, including:
* @li weight: A required Tensor. A mutable Tensor of word grad. Must be one of the following types: float16, float32,
bfloat16.
* @li indices: A required Tensor. A mutable word index Tensor. Must be one of the following types: int32, int64.
* @li offsets: A optional Tensor. A mutable word index Tensor. Must be one of the following types: int32, int64.
* @li per_sample_weights: A optional Tensor. to indicate all weights should be taken to be 1. Must be one of the
following types: float16, float32, bfloat16.
*     If specified, per_sample_weights must have exactly the same shape as input
*     and is treated as having the same offsets, if those are not None.
*     Only supported for mode='sum'.\n

* @par Attributes:
* @li mode: An string attr which use "sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag. Defaults to
"mean".\n
* @li padding_idx: An int attr judge which word to fill zeros. Defaults to "-1". \n

* @li scale_grad_by_freq: An optional bool. Defaults to "False".
*     If "True", "grad_weight" will be scale by word_frequency.
*     If "False", "grad_weight" will not be scale by word_frequency. \n
* @li sparse: An optional bool. Defaults to "False". if True, gradient w.r.t.attr weight matrix will be a sparse tensor.
\n
* @li include_last_offset: An optional bool. Defaults to "False". if True, attr offsets has one additional element,
where the last element
*     is equivalent to the size of indices. This matches the CSR format. \n

* @par Outputs:
* four outputs: \n
* y: A mutable output Tensor of new word grad has the same type as "grads". \n
* offset2bag:A Tensor. Must be one of the following types: int32, int64. \n
* bag_size:A Tensor. Must be one of the following types: int32, int64. \n
* max_indices:A Tensor. Must be one of the following types: int32, int64. \n
* @par Third-party framework compatibility
* Compatible with the Pytorch operator EmbeddingBag.
*/
REG_OP(EmbeddingBag)
    .INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(offsets, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(per_sample_weights, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(offset2bag, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(bag_size, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(max_indices, TensorType({DT_INT32, DT_INT64}))
    .ATTR(mode, String, "mean")
    .ATTR(scale_grad_by_freq, Bool, false)
    .ATTR(sparse, Bool, false)
    .ATTR(include_last_offset, Bool, false)
    .ATTR(padding_idx, Int, -1)
    .OP_END_FACTORY_REG(EmbeddingBag)

} // namespace ge

#endif // OPS_OP_PROTO_INC_EMBEDDING_BAG_H_
