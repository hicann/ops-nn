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
 * \file where_proto.h
 * \brief
 */
#ifndef OPS_NN_INDEX_WHERE_PROTO_H_
#define OPS_NN_INDEX_WHERE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

 /**
* @brief Returns locations of nonzero / true values in a tensor. \n

* @par Inputs:
* Including:
* x: A Tensor. Must be one of the following types:
  DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_QINT8,
  DT_QUINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_QINT32,
  DT_INT64, DT_UINT64, DT_BOOL, DT_COMPLEX64, DT_COMPLEX128 \n

* @par Outputs:
* y: A Tensor of type DT_INT64. \n

* @attention Constraints:
* Where runs on the Ascend AI CPU, which delivers poor performance.\n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Where.
*/

REG_OP(Where)
    .INPUT(x, TensorType({BasicType(), DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(Where)

} // namespace ge

#endif  // OPS_NN_INDEX_WHERE_PROTO_H_
