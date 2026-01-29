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
 * \file relu_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes rectified linear: "max(x, 0)".
 *
 * @par Inputs:
 * x: A tensor. Must be one of the following types: float32, float16, double, int8, int32, int16, int64, uint8, uint16,
 * qint8, bfloat16.\n In Ascend 950 AI Processor, support ND format. Others support 4D, format must be one of [NHWC,
 * NCHW, HWCN].
 *
 * @par Outputs:
 * @li y: A tensor with the same dtype and shape as the "x".
 * @li mask: A tensor of value "1" (for x > 0) or "0" (for x <= 0). Type is uint8 or uint1.
 *
 * @par Third-party framework compatibility
 * Incompatible with TensorFlow or Caffe.
 *
 * @attention Constraints:
 * The last dimension of "x" must be divisible by 8.
 */
REG_OP(ReluV2)
    .INPUT(
        x, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8,
                DT_BF16}))
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8,
                DT_BF16}))
    .OUTPUT(mask, TensorType({DT_UINT8, DT_UINT1}))
    .OP_END_FACTORY_REG(ReluV2)
} // namespace ge
#endif