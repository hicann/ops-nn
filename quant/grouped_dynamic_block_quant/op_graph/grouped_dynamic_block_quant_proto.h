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
 * \file grouped_dynamic_block_quant_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Online quantizes the input tensor per block & per group.
* @par Inputs:
* @li x: A tensor of type Float16 or Bfloat16. Shape must be 2-dimensional or 3-dimensional.
* @li group_list: A tensor of type Int32. Shape must be 1-dimensional. Indicate the size or offset of each group on the -2 axis of x.

* @par Attributes:
* @li min_scale: An Optional Float. Minimum scale value for quantization. Must be a non-negative float.
*   Defaults to 0.0.
* @li round_mode: An Optional String. Quantization rounding mode. Valid values:
*   - "rint": Supported for FLOAT8_E5M2/FLOAT8_E4M3FN
*   - "round": Supported for HIFLOAT8 only
*   - "hybrid": Supported for HIFLOAT8 only
*   Defaults to "rint".
* @li dst_type: An Optional Int. Target data type enum value:
*   - 34: HIFLOAT8
*   - 35: FLOAT8_E5M2
*   - 36: FLOAT8_E4M3FN
*   Defaults to 35 (FLOAT8_E5M2).
* @li row_block_size: An Optional Int. Number of elements per block in -2 dimension.
*   Defaults to 1.
* @li col_block_size: An Optional Int. Number of elements per block in -1 dimension.
*   Defaults to 128.
* @li group_list_type: An Optional Int. Indicate the meaning of group_list.
*   Defaults to 0.

* @par Outputs:
* @li y: A tensor of type Fp8/HiF8. Quantized tensor with same shape as input x. Data type depends on dst_type.
* @li scale: A tensor of type float. Shape is [x.rows/row_block_size + group_num, ceil(x.cols/col_block_size)] or [B, x.rows/row_block_size + group_num, ceil(x.cols/col_block_size)].
*/
REG_OP(GroupedDynamicBlockQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(group_list, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(min_scale, Float, 0.0)
    .ATTR(round_mode, String, "rint")
    .ATTR(dst_type, Int, DT_FLOAT8_E5M2)
    .ATTR(row_block_size, Int, 1)
    .ATTR(col_block_size, Int, 128)
    .ATTR(group_list_type, Int, 0)
    .OP_END_FACTORY_REG(GroupedDynamicBlockQuant)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_