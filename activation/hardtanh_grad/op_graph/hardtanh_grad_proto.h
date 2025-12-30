/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief PyTorch hardtanh_backward operator.
 *
 * @par Inputs:
 * Two inputs, including:
 * @li result: Support 1D ~ 8D. Minimum tensor of the linear region range,
 * datatype: bfloat16/float16/float32, format:ND/5HD.
 * @li grad: Maximum tensor of the linear region range,
 * datatype: bfloat16/float16/float32, format:ND/5HD. \n

 * @par Attributes:
 * Two attributes, including:
 * @li min_val: Minimum value of the linear region range, datatype:float. Defaults to "-1.0".
 * @li max_val: Maximum value of the linear region range, datatype:float. Defaults to "1.0".\n

 * @par Outputs:
 * One output, including:
 * y: Hardtanh_backward output tensor. Shape, datatype and format is the same as input result. \n

 * @attention Constraints:
 * This operator only supports datatype: bfloat16/float16/float32, format: ND/5HD. \n

 * @par Third-party framework compatibility
 * Compatible with the PyTorch operator HardtanhGrad.
 */
REG_OP(HardtanhGrad)
    .INPUT(result, TensorType({ DT_BF16, DT_FLOAT16, DT_FLOAT })) /* "First operand." */
    .INPUT(grad, TensorType({ DT_BF16, DT_FLOAT16, DT_FLOAT }))   /* "Second operand." */
    .OUTPUT(y, TensorType({ DT_BF16, DT_FLOAT16, DT_FLOAT }))     /* "Result, has same element type as two inputs" */
    .ATTR(min_val, Float, -1.0)
    .ATTR(max_val, Float, 1.0)
    .OP_END_FACTORY_REG(HardtanhGrad)
} // namespace ge
#endif