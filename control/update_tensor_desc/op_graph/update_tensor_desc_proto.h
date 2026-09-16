/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file update_tensor_desc_proto.h
 * \brief UpdateTensorDesc 图模式 IR 注册（REG_OP），proto 内容与 canndev 保持一致。
 */

#ifndef OPS_OP_PROTO_INC_UPDATETENSORDESC_H_
#define OPS_OP_PROTO_INC_UPDATETENSORDESC_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Update the tensor_desc of the output.

* @par Inputs:
* x: A Tensor. Must be one of the following types: float16, float32, int32, int64, double,
* int8, uint8, int16, uint16, uint32, uint64, bool. Supported format "ND".

* @par attributes:
* shape: A listInt contains the data to update. \n

* @par outputs:
* y: A Tensor, has the same type as "x". Shape is same as attr "shape". \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
#ifndef OPS_PROTO_DEF_UPDATETENSORDESC
#define OPS_PROTO_DEF_UPDATETENSORDESC
REG_OP(UpdateTensorDesc)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64, DT_UINT64,
                          DT_INT16, DT_UINT16, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64, DT_UINT64,
                           DT_INT16, DT_UINT16, DT_DOUBLE}))
    .REQUIRED_ATTR(shape, ListInt)
    .OP_END_FACTORY_REG(UpdateTensorDesc)
#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_UPDATETENSORDESC_H_
