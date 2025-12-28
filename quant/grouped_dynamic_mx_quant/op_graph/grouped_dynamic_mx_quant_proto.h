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
 * \file quantize_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Quantizes the input to mxfp8 group-wisely, according to group_index. \n

* @par Inputs:
* @li x: A tensor of type float16 or bfloat16, specifying the input.
* The shape only supports 2 dimensions.
* @li group_index: A tensor of type int32, specifying the index of groups.
* The shape only supports 1 dimension.

* @par Attributes:
* @li round_mode: An optional string, specifying the quantization rounding mode. 
* Defaults and only supports "rint".
* @li dst_type: An optional int, specifying the dtype of output y.
* Defaults to FLOAT8_E5M2, only supports FLOAT8_E4M3FN or FLOAT8_E5M2.
* @li blocksize: An optional int, specifying the block size of quantization.
* Defaults and only supports 32.

* @par Outputs:
* @li y: An output tensor of type FLOAT8_E4M3FN or FLOAT8_E5M2. It has the same shape and rank as input x.
* @li mxscale: An output tensor of type FLOAT8_E8M0, the shape only supports 3 dimensions. \n
* - mxscale.shape[0] = x.shape[0] / (blocksize * 2) + group_index.shape[0].
* - mxscale.shape[1] = x.shape[1].
* - mxscale.shape[2] = 2.

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe, Onnx, Tensorflow or Pytorch.
*/
REG_OP(GroupedDynamicMxQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(group_index, TensorType({DT_Int32}))
    .OUTPUT(y, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(mxscale, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(round_mode, String, "rint")
    .ATTR(dst_type, Int, DT_FLOAT8_E5M2)
    .ATTR(blocksize, Int, 32)
    .OP_END_FACTORY_REG(GroupedDynamicMxQuant)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
