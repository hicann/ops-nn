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
 * \file tensor_scatter_update_proto.h
 * \brief
 */
#ifndef OPS_NN_INDEX_TENSOR_SCATTER_UPDATE_PROTO_H_
#define OPS_NN_INDEX_TENSOR_SCATTER_UPDATE_PROTO_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Applies sparse addition to individual values or slices in a Variable .
 *
* @par Inputs:
* Three inputs, including:
* @li x: An ND Tensor. \n
 *
* Must be one of the following types: float16, float32, double, int64, int32,
  uint8, uint16, uint32, uint64, int8, int16, bool, complex64, complex128,
  qint8, quint8, qint16, quint16, qint32, bfloat16, string. \n
* @li indices: An ND Tensor. \n
 *
* Must be one of the following types: int32
* @li updates: An ND Tensor. \n
 *
* Has the same type and format as input "x" .
*
* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator TensorScatterUpdate.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
REG_OP(TensorScatterUpdate)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OP_END_FACTORY_REG(TensorScatterUpdate)
} // namespace ge

#endif // OPS_NN_INDEX_TENSOR_SCATTER_UPDATE_PROTO_H_
