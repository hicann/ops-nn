/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SEGMENT_SUM_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SEGMENT_SUM_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Computes the sum along segments of a tensor.

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor of type NumberType. Support 1D ~ 8D.
* @li segment_ids: A 1D Tensor of type IndexNumberType, whose shape is same
* with "x.shape[0]".

* @par Outputs:
* y: A Tensor of type NumberType. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SegmentSum.

* @attention Constraints:
* @li segment_ids.dimNum = 1, and segment_ids.shape[0] = x.shape[0].
* @li segment_ids should sorted in ascending order, and segment_ids.value >= 0.
* @li y.type must be same with x.type.
* @li y.dimNum = x.dimNum, and y.shape = [max(segment_ids) + 1, x.shape[1:]]
*/

REG_OP(SegmentSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(segment_ids, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(SegmentSum)

}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_SEGMENT_SUM_H_
