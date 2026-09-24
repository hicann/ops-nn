/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_UNIQUE_WITH_COUNTS_AND_SORTING_H_
#define OPS_OP_PROTO_INC_UNIQUE_WITH_COUNTS_AND_SORTING_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Returns the unique scalar elements of the flattened input, with optional inverse indices and counts.
 * @par Inputs:
 * x: A k-dimensional tensor of BasicType or bfloat16.
 * @par Attributes:
 * @li return_inverse: Whether to return inverse indices. Defaults to false.
 * @li return_counts: Whether to return occurrence counts. Defaults to false.
 * @li sorted: Whether to sort the unique elements. Defaults to true.
 * @li out_idx: Output index/count datatype. Defaults to DT_INT64.
 * @par Outputs:
 * @li y: Unique scalar elements, with the same datatype as x.
 * @li indices: DT_INT32 or DT_INT64 indices mapping input elements to y.
 * @li counts: DT_INT32 or DT_INT64 occurrence counts for elements of y.
 * @par Third-party framework compatibility:
 * Compatible with the PyTorch operator _unique2.
 */
#ifndef OPS_PROTO_DEF_UNIQUEWITHCOUNTSANDSORTING
#define OPS_PROTO_DEF_UNIQUEWITHCOUNTSANDSORTING
REG_OP(UniqueWithCountsAndSorting)
    .INPUT(x, TensorType({BasicType(), DT_BF16}))
    .OUTPUT(y, TensorType({BasicType(), DT_BF16}))
    .OUTPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(counts, TensorType({DT_INT32, DT_INT64}))
    .ATTR(return_inverse, Bool, false)
    .ATTR(return_counts, Bool, false)
    .ATTR(sorted, Bool, true)
    .ATTR(out_idx, Type, DT_INT64)
    .OP_END_FACTORY_REG(UniqueWithCountsAndSorting)
#endif // OPS_PROTO_DEF_UNIQUEWITHCOUNTSANDSORTING
} // namespace ge

#endif // OPS_OP_PROTO_INC_UNIQUE_WITH_COUNTS_AND_SORTING_H_
