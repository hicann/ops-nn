/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_EXPERIMENTAL_POOLING_MAX_POOL3_D_PROTO_H_
#define OPS_EXPERIMENTAL_POOLING_MAX_POOL3_D_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Performs three-dimensional max pooling.
 *
 * @par Inputs:
 * x: A five-dimensional tensor with logical layout NCDHW or NDHWC. Supported
 * types are float16, float32 and bfloat16.
 *
 * @par Attributes:
 * @li ksize: Required pooling window sizes with one, three or five elements.
 * @li strides: Required pooling window strides with one, three or five
 * elements.
 * @li padding: Required padding mode: "VALID", "SAME" or "CALCULATED".
 * @li pads: Optional explicit paddings in front, back, top, bottom, left and
 * right order. Used by "CALCULATED" mode. Defaults to all zeros.
 * @li dilation: Optional pooling window dilations with one, three or five
 * elements. Defaults to all ones.
 * @li ceil_mode: Optional output size rounding mode for "CALCULATED" padding.
 * Zero selects floor and any nonzero value selects ceil. Defaults to zero.
 * @li data_format: Optional logical layout, "NCDHW" or "NDHWC". Defaults to
 * "NDHWC".
 *
 * @par Outputs:
 * y: The max-pooled tensor. Its data type is the same as x.
 */
REG_OP(MaxPool3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(pads, ListInt, {0, 0, 0, 0, 0, 0})
    .ATTR(dilation, ListInt, {1, 1, 1, 1, 1})
    .ATTR(ceil_mode, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(MaxPool3D)

} // namespace ge

#endif // OPS_EXPERIMENTAL_POOLING_MAX_POOL3_D_PROTO_H_
