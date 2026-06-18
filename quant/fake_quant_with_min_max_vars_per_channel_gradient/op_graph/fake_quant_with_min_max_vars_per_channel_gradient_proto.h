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
 * \file fake_quant_with_min_max_vars_per_channel_gradient_proto.h
 * \brief
 */
#ifndef OPS_QUANT_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_PROTO_H_
#define OPS_QUANT_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Compute gradients for FakeQuantWithMinMaxVarsPerChannel. \n

*@par Inputs:
*Four inputs, including:
*@li gradients: A Tensor of type float. Backpropagated gradients above the
*   FakeQuantWithMinMaxVarsPerChannel operation, shape one of: [d], [b, d],
*   [b, h, w, d].
*@li x: A Tensor of type float. Same shape as gradients.
*@li min: A 1-D Tensor of type float, shape [d].
*@li max: A 1-D Tensor of type float, shape [d]. \n

*@par Attributes:
*@li num_bits: An optional attribute of type int. Defaults to 8.
*@li narrow_range: An optional attribute of type bool. Defaults to false. \n

*@par Outputs:
*@li backprops_wrt_x: A Tensor of type float. Same shape as x.
*@li backprops_wrt_min: A 1-D Tensor of type float, shape [d].
*@li backprops_wrt_max: A 1-D Tensor of type float, shape [d]. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator FakeQuantWithMinMaxVarsPerChannelGradient.
*/

REG_OP(FakeQuantWithMinMaxVarsPerChannelGradient)
    .INPUT(gradients, TensorType({DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(min, TensorType({DT_FLOAT}))
    .INPUT(max, TensorType({DT_FLOAT}))
    .OUTPUT(backprops_wrt_x, TensorType({DT_FLOAT}))
    .OUTPUT(backprops_wrt_min, TensorType({DT_FLOAT}))
    .OUTPUT(backprops_wrt_max, TensorType({DT_FLOAT}))
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxVarsPerChannelGradient)

} // namespace ge

#endif // OPS_QUANT_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_PROTO_H_
