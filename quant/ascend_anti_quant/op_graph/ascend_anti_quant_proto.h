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
 * \file ascend_anti_quant_proto.h
 * \brief Graph-mode IR declaration for AscendAntiQuant.
 *        This op is graph-mode only (no aclnn API). scale / offset are passed as
 *        scalar float attributes (per-tensor anti-quantization).
 */
#ifndef OPS_QUANT_ASCEND_ANTI_QUANT_GRAPH_PLUGIN_ASCEND_ANTI_QUANT_PROTO_H_
#define OPS_QUANT_ASCEND_ANTI_QUANT_GRAPH_PLUGIN_ASCEND_ANTI_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Per-tensor anti-quantization with scalar scale / offset attributes.
 *   y = cast<TOut>((x + offset) * scale)              when sqrt_mode == false
 *   y = cast<TOut>((x + offset) * scale * scale)      when sqrt_mode == true
 *
 * @par Inputs:
 * @li x: A tensor. Type is one of DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN.
 *
 * @par Outputs:
 * @li y: A tensor. Type is one of DT_FLOAT16, DT_FLOAT. Shape is the same as x.
 *
 * @par Attributes:
 * @li scale: Required float. Per-tensor scale value.
 * @li offset: Required float. Per-tensor offset value.
 * @li dtype: Optional int, output dtype. Defaults to DT_FLOAT (0). Allowed: DT_FLOAT16 (1), DT_FLOAT (0).
 * @li sqrt_mode: Optional bool. Defaults to false.
 */
REG_OP(AscendAntiQuant)
    .INPUT(x, TensorType({DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .ATTR(dtype, Int, DT_FLOAT)
    .ATTR(sqrt_mode, Bool, false)
    .OP_END_FACTORY_REG(AscendAntiQuant)

} // namespace ge

#endif // OPS_QUANT_ASCEND_ANTI_QUANT_GRAPH_PLUGIN_ASCEND_ANTI_QUANT_PROTO_H_
