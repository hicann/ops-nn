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
 * \file fake_quant_with_min_max_vars_per_channel_proto.h
 * \brief
 */
#ifndef OPS_QUANT_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_PROTO_H_
#define OPS_QUANT_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Fake-quantize x per channel using min, max as quantization range. \n

*@par Inputs:
*Three inputs, including:
*@li x: A Tensor of type float32. ND, last-axis represents channel C.
*@li min: A 1-D Tensor of type float32 with length C.
*@li max: A 1-D Tensor of type float32 with length C. \n

*@par Attributes:
*@li num_bits: An optional attribute of type int64. Range [2, 16]. Default 8.
*@li narrow_range: An optional attribute of type bool. Default false. \n

*@par Outputs:
*y: A Tensor of type float32 with the same shape as x. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator FakeQuantWithMinMaxVarsPerChannel.
*/

REG_OP(FakeQuantWithMinMaxVarsPerChannel)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(min, TensorType({DT_FLOAT}))
    .INPUT(max, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxVarsPerChannel)

}  // namespace ge

#endif  // OPS_QUANT_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_PROTO_H_
