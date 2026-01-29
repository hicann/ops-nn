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
 * \file trans_quant_param_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_TRANS_QUANT_PARAMV2_OPS_H_
#define OPS_OP_PROTO_INC_TRANS_QUANT_PARAMV2_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Transfer quant param from float32 to uint64.

* @par Inputs:
* @li scale: A quantization parameter tensor. Must be one of the following types: float32.
             The format support ND. The shape support 1D and 2D.
* @li offset: An optional quantization parameter tensor. Must be one of the following types: float32.
              The format support ND. The shape support 1D and 2D. \n

* @par Attributes:
* @li round_mode: fp32 filing to fp19 mode; 0: truncation and filing; 1: r_int mode;

* @par Outputs:
* @li y: output tensor. Must be one of the following types: uint64. The format support ND.
         The shape support 1D and 2D. \n

* @attention Constraints:
* @li The passed scale, y cannot be a null pointer.

* - Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product or Atlas Inference Series Product or
* Ascend 950 AI Processor: \n
* This operator supports use with the matmul operator, such as QuantBatchMatmulV3. \n
* - Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product or Atlas Inference Series Product: \n
* This operator does not supports use with the grouped matmul operator, such as GroupedMatmul. \n
* @li When there is no offset, the y shape is consistent with the scale shape: \n
* - If y is used as matmul input(e.g., QuantBatchMatmulV3), the shape support 1D (t,), with t equal to 1 or n, or 2D(1,
n),
* where n is the same as that of the right matrix in the matmul calculation. \n
* - If y is used as grouped matmul input(e.g., GroupedMatmul), the shape support 1D (g,), or 2D(g, n), (g, 1),
* where n is the same as that of the right matrix in the grouped matmul calculation, g is the same as that of
* the number of group num(i.e., group_list parameter's shape). \n
* @li When there is an offset, use it only as matmul input (e.g., QuantBatchMatmulV3): \n
* The shape of scale, offet and y support 1D (t,), with t equal to 1 or n, or 2D(1, n), where n is the same as
* that of the right matrix in the matmul calculation.
*/
REG_OP(TransQuantParamV2)
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_UINT64}))
    .ATTR(round_mode, Int, 0)
    .OP_END_FACTORY_REG(TransQuantParamV2)

} // namespace ge

#endif