 /**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_REVERSE_SEQUENCE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_REVERSE_SEQUENCE_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
 *@brief Reverses variable length slices. \n

*@par Inputs:
* @li x: A ND Tensor. The input to reverse. Support 2D ~ 8D.Must be one of the following types: 
* complex64, complex128, double, float32, float16, int16, int32, int64, int8, uint16, uint32, uint8, bfloat16.
* @li seq_lengths: A 1D Tensor of type int32 or int64. \n

*@par Attributes:
*@li seq_dim: The dimension along which reversal is performed.
*@li batch_dim: An optional int. Defaults to "0". The dimension along which
reversal is performed. \n

* @par Constraints:
* @li seq_dim != batch_dim.
* @li seq_dim < rank, batch_dim < rank, rank is the dimension of x.
* @li x.shape[batch_dim] = seq_lengths.shape(0).
* @li The value range of seq_lengths is [0, x.shape[seq_dim]].

*@par Outputs:
*y: A rank ND tensor. Has the same shape as input. The extracted banded tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ReverseSequence.
*/

REG_OP(ReverseSequence)
    .INPUT(x,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(seq_lengths, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .REQUIRED_ATTR(seq_dim, Int)
    .ATTR(batch_dim, Int, 0)
    .OP_END_FACTORY_REG(ReverseSequence)
}

#endif  // OPS_BUILT_IN_OP_PROTO_INC_REVERSE_SEQUENCE_H_