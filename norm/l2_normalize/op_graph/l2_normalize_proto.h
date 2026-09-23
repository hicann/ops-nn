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
 * \file l2_normalize_proto.h
 * \brief L2Normalize graph mode operator prototype definition.
 */
#ifndef OPS_NORM_L2_NORMALIZE_PROTO_H_
#define OPS_NORM_L2_NORMALIZE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Normalizes elements of a specific dimension of eigenvalues (L2) .

*@par Inputs:
*x: A ND Tensor(1D-8D) of type float16 or float32, specifying the eigenvalue . \n

*@par Attributes:
*@li axis: A optional required attribute of type list, specifying the axis for normalization Defaults to {} .
*@li eps: An optional attribute of type float, specifying the lower limit of normalization. Defaults to "1e-4" . \n

*@par Outputs:
*y: A ND Tensor(1D-8D) of type float16 or float32, specifying the eigenvalue for normalization. \n

*@par Third-party framework compatibility
* Compatible with the L2 scenario of PyTorch operator Normalize.
*/
#ifndef OPS_PROTO_DEF_L2NORMALIZE
#define OPS_PROTO_DEF_L2NORMALIZE
REG_OP(L2Normalize)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, ListInt, {})
    .ATTR(eps, Float, 1e-4f)
    .OP_END_FACTORY_REG(L2Normalize)
#endif // OPS_PROTO_DEF_L2NORMALIZE
} // namespace ge

#endif // OPS_NORM_L2_NORMALIZE_PROTO_H_
