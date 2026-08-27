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
 * \file sparse_segment_mean_grad_proto.h
 * \brief Proto definition for SparseSegmentMeanGrad
 */
#ifndef OPS_NN_INDEX_SPARSE_SEGMENT_MEAN_GRAD_PROTO_H_
#define OPS_NN_INDEX_SPARSE_SEGMENT_MEAN_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes gradients for SparseSegmentMean.

*@par Inputs:
*@li x: A Tensor. Must be one of the following types: float16, float, double, bfloat16.
gradient propagated to the SparseSegmentMean op.
*@li indices: A Tensor. Must be one of the following types: int32, int64.
indices passed to the corresponding SparseSegmentMean op.
*@li segment_ids: A Tensor. Must be one of the following types: int32, int64. segment_ids passed to the
corresponding SparseSegmentMean op.
*@li output_dim0: A Tensor of type int32. dimension 0 of "x" passed to
SparseSegmentMean op. \n

*@par Outputs:
*y:A Tensor. Has the same type as x. \n

*@par Third-party framework compatibility
*Compatible with tensorflow SparseSegmentMeanGrad operator.
*/
#ifndef OPS_PROTO_DEF_SPARSESEGMENTMEANGRAD
#define OPS_PROTO_DEF_SPARSESEGMENTMEANGRAD
REG_OP(SparseSegmentMeanGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BFLOAT16}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(segment_ids, TensorType({DT_INT32, DT_INT64}))
    .INPUT(output_dim0, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(SparseSegmentMeanGrad)
#endif

} // namespace ge

#endif // OPS_NN_INDEX_SPARSE_SEGMENT_MEAN_GRAD_PROTO_H_
