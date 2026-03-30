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
 * \file sorted_sparse_segment_mean_grad_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_SORTED_SPARSE_SEGMENT_MEAN_GRAD_H_
#define OPS_OP_PROTO_INC_SORTED_SPARSE_SEGMENT_MEAN_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes gradients for SortedSparseSegmentMeanGrad.

*@par Inputs:
*@li x: A Tensor. Must be one of the following types: float16, float, bfloat16.
gradient propagated to the SparseSegmentMean op.
*@li sorted_indices: A Tensor. Must be one of the following types: int32, int64.
indices passed to the corresponding SparseSegmentMean op. The input must be sorted.
*@li pre_location_indices: A Tensor. Must be one of the following types: int32, int64.
The original positions of the indices before sorting. 
*@li segment_ids: A Tensor. Must be one of the following types: int32, int64. segment_ids passed to the
corresponding SparseSegmentMean op. The input must be sorted.
*@li output_dim0: A Tensor of type int32. dimension 0 of "x" passed to
SparseSegmentMean op. \n

*@par Outputs:
*y:A Tensor. The dim of y has output_dim0 as its 0th dim, and the dim are taken from x start the 1st dim.\n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
*/

REG_OP(SortedSparseSegmentMeanGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BFLOAT16}))
    .INPUT(sorted_indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(pre_location_indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(segment_ids, TensorType({DT_INT32, DT_INT64}))
    .INPUT(output_dim0, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(SortedSparseSegmentMeanGrad)
}
#endif // OPS_OP_PROTO_INC_SORTED_SPARSE_SEGMENT_MEAN_GRAD_H_
