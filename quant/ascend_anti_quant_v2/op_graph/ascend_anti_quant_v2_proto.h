/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Anti quantizes the input . 

* @par Inputs:
* @li x: A multi-dimensional tensor of type int8/int4, specifying the input.
  The maximum dimension should not exceed 8 dimensions. Format support ND. 
* @li scale: A 1-D tensor of type float32/bfloat16, specifying the scale.
  Shape is (n,), where n can be 1. If n is not 1, it must be the same as
  the size of last dimension of x. Format support ND. 
* @li offset: A optional 1-D tensor of type float32/bfloat16, specifying the offset.
  The shape and dtype of offset should be same to scale. Format support ND. 

* @par Attributes:
* @li dst_type: A optional int32, specifying the output data type. Defaults to "DT_FLOAT16".
* @li sqrt_mode: A optional bool, specifying whether to perform square root on "scale", either "True" or "False".
* Defaults to "False" . \n

* @par Outputs:
* y: The dequantized output tensor of type float16 or bfloat16. \n

*/
REG_OP(AscendAntiQuantV2)
    .INPUT(x, TensorType({DT_INT8, DT_INT4}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BFLOAT16}))
    .ATTR(dst_type, Int, DT_FLOAT16)
    .ATTR(sqrt_mode, Bool, false)
    .OP_END_FACTORY_REG(AscendAntiQuantV2)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
