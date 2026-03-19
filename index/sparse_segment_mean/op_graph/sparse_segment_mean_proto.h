/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPARSE_SEGMENT_MEAN_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPARSE_SEGMENT_MEAN_H_
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Computes the mean along sparse segments of a tensor.

*@par Inputs:
*The input indices and segment_ids must have same rank. Inputs include:
*@li x: A tensor. Must be one of the following types: float16, float32, double, bfloat16.
*@li indices: A tensor. Must be one of the following types: int32, int64.
A 1-D tensor. Has same rank as segment_ids.
*@li segment_ids: A tensor. Must be one of the following types: int32, int64. A 1-D tensor. Values should be
sorted and can be repeated. \n

*@par Outputs:
*y:A tensor. Has the same type as x. \n

*@par Third-party framework compatibility
*Compatible with TensorFlow SparseSegmentMean operator.
*/

REG_OP(SparseSegmentMean)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BFLOAT16}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(segment_ids, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(SparseSegmentMean)

}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_SPARSE_SEGMENT_MEAN_H_
