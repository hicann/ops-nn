/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_AVG_POOL1D_AVG_MATRIX_PROTO_H_
#define OPS_NN_AVG_POOL1D_AVG_MATRIX_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Generate an auxiliary matrix .  \n

*@par Inputs:
* @li x: A tensor. Must be one of the following types:uint8, int8,int16, int32,
 int64, float16, float, double.The format must be NHWC/NCHW.

*@par Attributes:
*@li ksize: Kernel size. Input type is int.
*@li strides: Input type is int.
*@li pads: Input type is listInt .
*@li ceil_mode: Bool, default value is false.
*@li count_include_pad: Bool, default value is false.  \n

*@par Outputs:
*y_tensor: A  tensor with the same types as "x" .  \n
*@par Third-party framework compatibility

*Compatible with the TensorFlow operator Unbatch.
*/
REG_OP(AvgPool1DAvgMatrix)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, Int)
    .REQUIRED_ATTR(strides, Int)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, false)
    .OP_END_FACTORY_REG(AvgPool1DAvgMatrix)

} // namespace ge

#endif // OPS_NN_AVG_POOL1D_AVG_MATRIX_PROTO_H_
