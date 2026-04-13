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
 * \file unsorted_segment_sum_proto.h
 * \brief
 */

#ifndef UNSORTED_SEGMENT_SUM_PROTO_H_
#define UNSORTED_SEGMENT_SUM_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the sum along segments of a tensor . \n
    Computes a tensor such that (output[i] = sum_{j...} x[j...] where
    the sum is over tuples j... such that segment_ids[j...] == i.If the sum
    is empty for a given segment ID i, output[i] = 0 \n
    for example:x = [[0,1,2],[3,4,5],[6,7,8]] , segment_ids = [0,0,4] num_segments = 5 \n
    output[0] = [3, 5, 7] \n
    output[1] = [0, 0, 0] \n
    output[2] = [0, 0, 0] \n
    output[3] = [0, 0, 0] \n
    output[4] = [6, 7, 8] \n
* @par Inputs:
* Three inputs, including:
* @li x: A tensor of type float32,float16,int32,int64,uint32,uint64,bfloat16. Format is ND.
* @li segment_ids: The ID of the output location.
* A tensor of type int32, int64, and the rank is less than or equal to x rank.
* Whose shape is a prefix of "x.shape". Format is ND.
* @li num_segments: A tensor of type int32, int64, format is ND.
* Indicates the output segment . \n

* @par Attributes:
* @li is_preprocessed: Deprecated attributes. Must to false. \n
* @li check_ids: Check whether the ID is verified. An optional bool. Defaults to false. \n

* @par Outputs:
* y: Type and format is the same as x . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator UnsortedSegmentSum
*/
REG_OP(UnsortedSegmentSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(segment_ids, TensorType::IndexNumberType())
    .INPUT(num_segments, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(is_preprocessed, Bool, false)
    .ATTR(check_ids, Bool, false)
    .OP_END_FACTORY_REG(UnsortedSegmentSum)
} // namespace ge

#endif  // UNSORTED_SEGMENT_SUM_PROTO_H_
