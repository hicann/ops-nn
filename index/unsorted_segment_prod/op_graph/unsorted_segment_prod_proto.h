/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNSORTED_SEGMENT_PROD_PROTO_H_
#define UNSORTED_SEGMENT_PROD_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the product along segments of a tensor.
    Computes a tensor such that output[i] = prod_{j...} data[j...]
    where the product is over tuples j... such that segment_ids[j...] == i.
    If the product for a given segment ID i is empty, output[i] = 1.
    Negative segment IDs are ignored.

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor of type float16, float32, bfloat16, int32, int64, uint32, uint64, format is ND.
* @li segment_ids: A Tensor of type int32, int64, whose shape is a prefix of "x", format is ND.
* @li num_segments: A 1D Tensor contains a single element of type int32, int64, format is ND.

* @par Outputs:
* y: Have the same type and format of "x".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator UnsortedSegmentProd.
*/
REG_OP(UnsortedSegmentProd)
    .INPUT(x, TensorType::NumberType())
    .INPUT(segment_ids, TensorType::IndexNumberType())
    .INPUT(num_segments, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(UnsortedSegmentProd)
} // namespace ge

#endif  // UNSORTED_SEGMENT_PROD_PROTO_H_
