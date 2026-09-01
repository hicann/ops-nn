/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_OP_PROTO_INC_SPARSE_SEGMENT_SUM_H_
#define OPS_OP_PROTO_INC_SPARSE_SEGMENT_SUM_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Computes the sum along sparse segments of a tensor.
 *
 * @par Inputs:
 * Three inputs, including:
 * @li x: A Tensor. Must be one of the following types: int8, uint8, int16, uint16, int32, uint32, int64,
 * uint64, double, float32, float16.
 * @li indices: A 1-D Tensor. Must be one of the following types: int32, int64.
 * @li segment_ids: A 1-D Tensor. Must be one of the following types: int32, int64. Values should be sorted and can be
 * repeated.
 *
 * @par Outputs:
 * y: A Tensor. Has the same type as x.
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator SparseSegmentSum.
 */

#ifndef OPS_PROTO_DEF_SPARSESEGMENTSUM
#define OPS_PROTO_DEF_SPARSESEGMENTSUM
REG_OP(SparseSegmentSum)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_DOUBLE,
                          DT_FLOAT, DT_FLOAT16}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(segment_ids, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_DOUBLE,
                           DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(SparseSegmentSum)
#endif
} // namespace ge

#endif // OPS_OP_PROTO_INC_SPARSE_SEGMENT_SUM_H_
