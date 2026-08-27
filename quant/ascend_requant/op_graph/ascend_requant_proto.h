/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_REQUANT_PROTO_H
#define ASCEND_REQUANT_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Requantizes the input.

 * @par Inputs:
 * @li x: A tensor of type int32, specifying the input. The format must be
 * FRACTAL_NZ, NC1HWC0 or DNC1HWC0. Shape support 4D ~ 6D.
 * @li req_scale:A required Tensor. The type only support uint64. The format
 * must be NC1HWC0 or NDC1HWC0. If req_scale is 1D tensor, shape must be same as
 * the last dimension of x. Otherwise the number of dimensions should be equal to
 * x, the last dimension of shape should be same as x, others must be 1.
 * Shape support 5D ~ 6D. Shape must be 1 in n,d,h,w. \n

 * @par Attributes:
 * relu_flag: An optional bool, specifying whether to perform ReLU,
 * either "True" or "False". Defaults to "False" . \n

 * @par Outputs:
 * y: The dequantized output tensor of type int8. The format must be FRACTAL_NZ,
 * NC1HWC0 or NDC1HWC0. The shape is same as x. \n

 * @par Third-party framework compatibility
 * It is a custom operator. It has no corresponding operator in Caffe.
 */
#ifndef OPS_PROTO_DEF_ASCENDREQUANT
#define OPS_PROTO_DEF_ASCENDREQUANT
REG_OP(AscendRequant)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(req_scale, TensorType({DT_UINT64}))
    .OUTPUT(y, TensorType({DT_INT8}))
    .ATTR(relu_flag, Bool, false)
    .OP_END_FACTORY_REG(AscendRequant)
#endif // OPS_PROTO_DEF_ASCENDREQUANT
} // namespace ge

#endif
