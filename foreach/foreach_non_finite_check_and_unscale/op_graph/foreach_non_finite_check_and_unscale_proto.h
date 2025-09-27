/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_non_finite_check_and_unscale_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_
#define OPS_OP_PROTO_INC_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Detect whether there is Inf or Nan in scaled_grads, set found_inf to 1 if it exists,
 * and do not operate on found_inf if it does not. Finally, multiply all values of scaled_grads by inv_scale
 * @par Inputs:
 * Three inputs:
 * @li scaled_grads: A tensor list containing multiple tensors, format supports ND, can be float16, float, bfloat16,
 * maximum length of scaled_grads is 256,
 * meanwhile, this value is also an output, store the value multiplied by inv_scale.
 * @li found_inf: A single-element float tensor to which 1.0 will be written if any scaled_grad contain infs/nans,
 * with only one element, format supports ND, must be float,
 * meanwhile, this value is also an output, indicating whether there is Inf or Nan present.
 * @li inv_scale: A single-element float tensor by which scaled_grads are currently multiplied,
 * with only one element, format supports ND, must be float.
 */
REG_OP(ForeachNonFiniteCheckAndUnscale)
    .DYNAMIC_INPUT(scaled_grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(found_inf, TensorType({DT_FLOAT}))
    .INPUT(inv_scale, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ForeachNonFiniteCheckAndUnscale)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_