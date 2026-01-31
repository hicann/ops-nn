/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNSPRTED_SEGMENT_MIN_PROTO_H_
#define UNSPRTED_SEGMENT_MIN_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the minimum along segments of a tensor.
    Computes a tensor such that (output[i] = min_{j...} x[j...].
    the value of segment_ids must be  in [0, num_segments - 1].
    for example:x = [[0,1,2],[3,4,5],[6,7,8]], segment_ids = [0,0,1] num_segments = 5
    output[0] = [0, 1, 2]
    output[1] = [6, 7, 8]
    output[2] = [2147483648, 2147483648, 2147483648]
    output[3] = [2147483648, 2147483648, 2147483648]
    output[4] = [2147483648, 2147483648, 2147483648]

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor of type double, float32, float16, bfloat16, int8, int16, int32, int64, uint8, uint16, uint32, uint64, format is ND. bank of shape must greater zero.
* @li segment_ids: A Tensor of type int32, int64, whose shape is a prefix of "x", format is ND.
* @li num_segments: A 1D Tensor contains a single element of type int32, int64, format is ND.
* Indicates the output segment.

* @par Outputs:
* y: Have the same type and format of "x".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator UnsortedSegmentMin.
*/
REG_OP(UnsortedSegmentMin)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(segment_ids, TensorType::IndexNumberType())
    .INPUT(num_segments, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(UnsortedSegmentMin)
} // namespace ge

#endif  // UNSPRTED_SEGMENT_MIN_PROTO_H_
