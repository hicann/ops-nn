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
 * \file fake_quant_affine_cachemask_proto.h
 * \brief
 */
#ifndef OPS_QUANT_FAKE_QUANT_AFFINE_CACHEMASK_PROTO_H_
#define OPS_QUANT_FAKE_QUANT_AFFINE_CACHEMASK_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Fake-quantize the data of 'x' tensor with scale, zero_point, quant_min and quant_max. \n

*@par Inputs:
*Three inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32.
*@li scale: A Tensor of type float32 or float16. Has the same type and format as "x".
*@li zero_point: A Tensor of type int32, float16 or float32.\n

*@par Attributes:
*@li axis: An required attribute of type int64.
*@li quant_min: An required attribute of type int64.
*@li quant_max: An required attribute of type int64.\n

*@par Outputs:
*y: A Tensor of type float32 or float16.
*mask: A Tensor of type bool. \n

*@par Third-party framework compatibility
* Compatible with Pytorch operator FakeQuantAffineCachemask.
*/

REG_OP(FakeQuantAffineCachemask)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(zero_point, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(mask, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axis, Int)
    .REQUIRED_ATTR(quant_min, Int)
    .REQUIRED_ATTR(quant_max, Int)
    .OP_END_FACTORY_REG(FakeQuantAffineCachemask)

} // namespace ge

#endif // OPS_QUANT_FAKE_QUANT_AFFINE_CACHEMASK_PROTO_H_