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
 * \file inplace_index_add_with_sorted_proto.h
 * \brief
 */
#ifndef OPS_NN_INDEX_INPLACE_INDEX_ADD_WITH_SORTED_PROTO_H
#define OPS_NN_INDEX_INPLACE_INDEX_ADD_WITH_SORTED_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Performs in-place addition on the "var" tensor along axis dim using
 *  "value" rows selected by "sorted_indices". "pos" gives the run-start offsets
 *  of equal-value runs inside "sorted_indices" so that multiple indices pointing
 *  to the same row can be accumulated together. "alpha" is a scalar multiplier
 *  applied to "value" before accumulation.
 *
 *  computing process (per index i, with sorted_indices[i] = r):
 *  @code{.c}
 *  var[r, :] += alpha * sum(value[j, :] for j in run(pos, i))
 *  @endcode
 *
 * @attention Constraints:
 *  - "var" and "value" must have the same dtype and rank.
 *  - "pos" must have the same length as "sorted_indices".
 *
 * @par Inputs:
 * @li var: A mutable tensor. Type: DT_FLOAT16 / DT_BF16. Format: FORMAT_ND.
 *     Shape: [M, ...], where M is the size of the indexed dimension (axis, currently only supports 0).
 * @li value: A tensor of updates. Same dtype and rank as "var". Format: FORMAT_ND.
 *     Shape: [N, ...], where N is the number of update rows and equals the length of "sorted_indices".
 * @li sorted_indices: A 1D int32 tensor of row indices into "var" along axis, sorted in
 *     non-decreasing order. Format: FORMAT_ND. Shape: [N].
 * @li pos: A 1D int32 tensor of run-start positions matching "sorted_indices".
 *     Format: FORMAT_ND. Shape: [N].
 * @li alpha: An optional scalar tensor. Type: DT_FLOAT. Format: FORMAT_ND.
 *     Shape: scalar ([] or [1]). Defaults to 1.0 when absent.
 *
 * @par Attributes:
 * axis: A required int. The dimension along which to index (currently only supports 0).
 *
 * @par Outputs:
 * var: A mutable tensor. Same dtype and shape as input "var". Format: FORMAT_ND.
 *
 * @par Third-party framework compatibility
 * Compatible with the PyTorch operator index_add_ (sorted-index variant).
 */
REG_OP(InplaceIndexAddWithSorted)
    .INPUT(var, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(sorted_indices, TensorType({DT_INT32}))
    .INPUT(pos, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(alpha, TensorType({DT_FLOAT}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(axis, Int)
    .OP_END_FACTORY_REG(InplaceIndexAddWithSorted)
} // namespace ge

#endif // OPS_NN_INDEX_INPLACE_INDEX_ADD_WITH_SORTED_PROTO_H
