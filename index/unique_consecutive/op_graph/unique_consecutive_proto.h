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
 * \file unique_consecutive_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_UNIQUE_CONSECUTIVE_H_
#define OPS_OP_PROTO_INC_UNIQUE_CONSECUTIVE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Finds the first unique element from every consecutive group of equivalent elements.

* @par Inputs:
* x: A ND tensor of BasicType, bool or bfloat16. \n

* @par Attributes:
* @li return_idx: An optional bool. Whether to also return the indices. The default value is False. Currently only False is supported.
* @li return_count: An optional bool. Whether to also return the counts for each element. The default is False.
* @li axis: An optional int. Which one axis to apply unique. The default is 1000, which means None. Currently only 1000 is supported.
* @li out_idx: Output index/count's datatype. The default is DT_INT64.

* @par Outputs:
* @li y: "x" in the unique output "y".Has the same type as "x" .
* @li idx: The index of each value of "x".
* @li count: The counts of each value of "y".

* @attention Constraints:
* UniqueConsecutive runs on the Ascend AI CPU, which delivers poor performance.

* @par Third-party framework compatibility
* Compatible with the PyTorch operator UniqueConsecutive.
*/

REG_OP(UniqueConsecutive)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .OUTPUT(idx, TensorType::IndexNumberType())
    .OUTPUT(count, TensorType::IndexNumberType())
    .ATTR(return_idx, Bool, false)
    .ATTR(return_counts, Bool, false)
    .ATTR(axis, Int, 1000)
    .ATTR(out_idx, Type, DT_INT64)
    .OP_END_FACTORY_REG(UniqueConsecutive)

} // namespace ge

#endif // OPS_OP_PROTO_INC_UNIQUE_CONSECUTIVE_H_

