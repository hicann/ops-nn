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
 * \file math_proto_stub.cpp
 * \brief
 */
#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge{
/**
*@brief Returns x1 + x2 element-wise. Support broadcasting operations.
*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types: bool, int8, int16, int32, int64, uint8, float64,
*     float16, bfloat16, float32, complex128, complex64, complex32, string.
* @li x2: A ND Tensor. Must be one of the following types: bool, int8, int16, int32, int64, uint8, float64,
*     float16, bfloat16, float32, complex128, complex64, complex32, string. \n

*@par Outputs:
*y: A ND Tensor. Must be one of the following types: bool, int8, int16, int32, int64, uint8, float64,
*     float16, bfloat16, float32, complex128, complex64, complex32, string.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Add.
*/
REG_OP(Add)
    .INPUT(x1, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64,
        DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
        DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .INPUT(x2, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64,
        DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
        DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64,
        DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
        DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Add)

/**
*@brief Cast a tensor form src data type to dst data type.

*@par Inputs:
*One input:
* x:A ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bool, float16, float, int8, int32, uint32, uint8, bfloat16, uint1,
   int64, uint64, int16, uint16, double, complex32, complex64, complex128, qint8, quint8, qint16, quint16, qint32,
   hifloat8, float8_e5m2, float8_e4m3fn, float4_e1m2, float4_e2m1.

*@par Attributes:
*dst_type: A required attribute of type int32, specifying the dst data type.

*@par Outputs:
*y:A ND Tensor with same shape as x, and data type is specified by dst_type.

*@attention Constraints:
* @li In the scenario where the data type is converted from float16 to int16: \n
*     If the input data contains inf, inf is converted into the maximum value of int16. \n
*     If the input data contains -inf, -inf is converted into the minimum value of int16. \n
* @li In the scenarios where the data type is converted from INT32 to INT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2048, 1920).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2048, 1920).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT64 and from FLOAT32 to UINT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648, 2147483583).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from INT64 to FLOAT32: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648, 2147483647).
*/
REG_OP(Cast)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32,
        DT_UINT32, DT_UINT8, DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, 
        DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16,
        DT_QINT32, DT_BF16, DT_UINT1, DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2,
        DT_FLOAT8_E4M3FN, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32,
        DT_UINT32, DT_UINT8, DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE,
        DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16,
        DT_QINT32, DT_BF16, DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, 
        DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .REQUIRED_ATTR(dst_type, Int)
    .OP_END_FACTORY_REG(Cast)

REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_BF16, DT_UINT32, DT_UINT64, DT_STRING}))
    .OP_END_FACTORY_REG(Fill)
} // namespace ge
