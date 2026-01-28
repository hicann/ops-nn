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
 * \file dynamic_block_qunat_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Online quantizes the input tensor per block.

* @par Inputs:
- x: A tensor of type float16 or bfloat16. Shape must be 2-dimensional.

* @par Attributes:
- min_scale: (Optional) Minimum scale value for quantization. Must be a positive float.
*   Defaults to 0.0.
- round_mode: (Optional) Quantization rounding mode. Valid values:
*   - "rint": Supported for FLOAT8_E5M2/FLOAT8_E4M3FN
*   - "round": Supported for HIFLOAT8 only
*   Defaults to "rint".
- dst_type: (Optional) Target data type enum value:
*   - 34: HIFLOAT8
*   - 35: FLOAT8_E5M2
*   - 36: FLOAT8_E4M3FN
*   Defaults to 35 (FLOAT8_E5M2).
- row_block_size: (Optional) Number of elements per block in row dimension.
* Only support 1, 128, 256, 512. Defaults to 1.
- col_block_size: (Optional) Number of elements per block in column dimension.
* Only support 64, 128, 192, 256. Defaults to 128.

* @par Outputs:
- y: Quantized tensor with same shape as input x. Data type depends on dst_type.
- scale: Scale tensor of type float. Shape is [ceil(x.rows/row_block_size), ceil(x.cols/col_block_size)].

* @par Third-party framework compatibility:
* Custom operator with no direct mapping in Caffe/ONNX/TensorFlow/PyTorch.
*/
REG_OP(DynamicBlockQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(min_scale, Float, 0.0)
    .ATTR(round_mode, String, "rint")
    .ATTR(dst_type, Int, DT_FLOAT8_E5M2)
    .ATTR(row_block_size, Int, 1)
    .ATTR(col_block_size, Int, 128)
    .OP_END_FACTORY_REG(DynamicBlockQuant)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_