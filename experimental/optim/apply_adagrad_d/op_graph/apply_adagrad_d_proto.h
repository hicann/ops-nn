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
 * \file apply_adagrad_d_proto.h
 * \brief REG_OP declaration for ApplyAdagradD
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_APPLY_ADAGRAD_D_H_
#define OPS_BUILT_IN_OP_PROTO_INC_APPLY_ADAGRAD_D_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Updates "var" according to the Adagrad scheme.
 *   accum += grad * grad   (when update_slots=True)
 *   var   -= lr * grad * (1 / sqrt(accum))
 *
 * @par Inputs:
 * @li var:   A mutable tensor. Supports float16, bfloat16, float32.
 * @li accum: A mutable tensor. Same type as "var".
 * @li lr:    A scalar tensor. Same type as "var".
 * @li grad:  Gradient tensor. Same type as "var".
 *
 * @par Attributes:
 * @li update_slots: Optional bool (default True). If True, accum is updated.
 * @li use_locking:  Optional bool (default False).
 *
 * @par Outputs:
 * @li var:   Updated var tensor. Same type as input "var".
 * @li accum: Updated accum tensor. Same type as input "var".
 *
 * @par Third-party framework compatibility
 * Compatible with TensorFlow operator ApplyAdagrad.
 */
REG_OP(ApplyAdagradD)
    .INPUT(var,   TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr,    TensorType::NumberType())
    .INPUT(grad,  TensorType::NumberType())
    .OUTPUT(var,  TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(update_slots, Bool, true)
    .ATTR(use_locking,  Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagradD)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_APPLY_ADAGRAD_D_H_
