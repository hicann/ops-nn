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
 * \file wts_arq_proto.h
 * \brief WtsARQ GE IR operator registration (identical to canndev proto).
 */
#ifndef OPS_OP_PROTO_INC_WTS_ARQ_H_
#define OPS_OP_PROTO_INC_WTS_ARQ_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Weights adaptive range quantization. \n

*@par Inputs:
*w: weights need to fake quantize.
*w_min: min of weights.
*w_max: max of weights. \n

*@par Attributes:
*@li num_bits: the bits num used for quantize.
*@li offset_flag: whether using offset. \n

*@par Outputs:
*y: fake quantized weights. \n

*@par Third-party framework compatibility
*Compatible with MindSpore and TensorFlow through the GE IR graph mode
*(TensorFlow is mapped to this IR by the plugin under framework/, there is no
*aclnn API for this operator)

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
#ifndef OPS_PROTO_DEF_WTSARQ
#define OPS_PROTO_DEF_WTSARQ
REG_OP(WtsARQ)
    .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w_min, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w_max, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(num_bits, Int, 8)
    .ATTR(offset_flag, Bool, false)
    .OP_END_FACTORY_REG(WtsARQ)
#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_WTS_ARQ_H_
