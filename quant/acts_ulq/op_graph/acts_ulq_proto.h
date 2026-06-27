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
 * \file acts_ulq_proto.h
 * \brief ActsULQ GE IR 算子注册
 */
#ifndef OPS_OP_PROTO_INC_ACTSULQ_H_
#define OPS_OP_PROTO_INC_ACTSULQ_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief ActsULQ fake quantization operator for QAT
 * @par Inputs:
 * @li data: A Tensor. Must be one of: float16, float32.
 * @li clamp_min: A Tensor. Same dtype as data.
 * @li clamp_max: A Tensor. Same dtype as data.
 * @par Attributes:
 * @li fixed_min: Bool. Default false.
 * @li num_bits: Int. Default 8.
 * @par Outputs:
 * @li output: A Tensor. Same dtype as data.
 * @li clamp_min_mask: A Tensor. Same dtype as data.
 * @li clamp_max_mask: A Tensor. Same dtype as data.
 * @li x_clamped_loss: A Tensor. Same dtype as data.
 */
REG_OP(ActsULQ)
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(clamp_min, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(clamp_max, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(clamp_min_mask, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(clamp_max_mask, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(x_clamped_loss, TensorType({DT_FLOAT16, DT_FLOAT}))
  .ATTR(fixed_min, Bool, false)
  .ATTR(num_bits, Int, 8)
  .OP_END_FACTORY_REG(ActsULQ)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ACTSULQ_H_
