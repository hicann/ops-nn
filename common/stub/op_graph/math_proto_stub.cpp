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

namespace ge {
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
    .INPUT(x1, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .INPUT(x2, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Add);

/**
 *@brief Cast a tensor form src data type to dst data type.

 *@par Inputs:
 *One input:
 * x:A ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bool, float16, float, int8, int32,
 uint32, uint8, bfloat16, uint1, int64, uint64, int16, uint16, double, complex32, complex64, complex128, qint8,
 quint8, qint16, quint16, qint32, hifloat8, float8_e5m2, float8_e4m3fn, float4_e1m2, float4_e2m1.

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
 * @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT64 and
 from FLOAT32 to UINT8: \n
 *     It can only guarantee that the input data has no precision errors within the range of (-2147483648,
 2147483583).
 * @li Atlas Inference Series Product in the scenarios where the data type is converted from INT64 to FLOAT32: \n
 *     It can only guarantee that the input data has no precision errors within the range of (-2147483648,
 2147483647).
 */
REG_OP(Cast)
    .INPUT(x, TensorType({DT_BOOL,          DT_FLOAT16,     DT_FLOAT,      DT_INT8,      DT_INT32,    DT_UINT32,
                          DT_UINT8,         DT_INT64,       DT_UINT64,     DT_INT16,     DT_UINT16,   DT_DOUBLE,
                          DT_COMPLEX64,     DT_COMPLEX128,  DT_QINT8,      DT_QUINT8,    DT_QINT16,   DT_QUINT16,
                          DT_QINT32,        DT_BF16,        DT_UINT1,      DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                          DT_FLOAT8_E4M3FN, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(y, TensorType({DT_BOOL,        DT_FLOAT16,    DT_FLOAT,     DT_INT8,     DT_INT32,       DT_UINT32,
                           DT_UINT8,       DT_INT64,      DT_UINT64,    DT_INT16,    DT_UINT16,      DT_DOUBLE,
                           DT_COMPLEX64,   DT_COMPLEX128, DT_QINT8,     DT_QUINT8,   DT_QINT16,      DT_QUINT16,
                           DT_QINT32,      DT_BF16,       DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                           DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .REQUIRED_ATTR(dst_type, Int)
    .OP_END_FACTORY_REG(Cast);

REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT,  DT_DOUBLE,     DT_INT32,   DT_UINT8,  DT_INT16,  DT_INT8,   DT_COMPLEX64,
                             DT_INT64,  DT_BOOL,       DT_QINT8,   DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16,
                             DT_UINT16, DT_COMPLEX128, DT_FLOAT16, DT_BF16,   DT_UINT32, DT_UINT64, DT_STRING}))
    .OP_END_FACTORY_REG(Fill);

REG_OP(Sort)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT8, DT_UINT8, DT_INT32, DT_INT64, DT_BF16, DT_UINT32,
                          DT_UINT16, DT_UINT64}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT8, DT_UINT8, DT_INT32, DT_INT64, DT_BF16, DT_UINT32,
                            DT_UINT16, DT_UINT64}))
    .OUTPUT(y2, TensorType({DT_INT32, DT_INT64}))
    .ATTR(axis, Int, -1)
    .ATTR(descending, Bool, false)
    .ATTR(stable, Bool, false)
    .ATTR(y2_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(Sort);

REG_OP(BroadcastTo)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(BroadcastTo);

/**
 * @brief Permutes the dimensions according to perm.
 *     The returned tensor's dimension i will correspond to the input dimension perm[i].

 * @par Inputs:
 * Two inputs, including:
 * @li x: A Tensor. Must be one of the following types:
 * bfloat16, float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
 * int16, complex32, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bool, hifloat8, float8_e5m2,
 * float8_e4m3fn, and the maximum dimension should not exceed 8 dimensions,
 * and the shape should be consistent with output.
 * @li perm: A Tensor of type int32 or int64. A permutation of the dimensions of "x", the value
 * should be within the range of [0, number of dimensions for self -1].

 * @par Outputs:
 * y: A Tensor. Has the same type as "x".

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator Transpose.
 */
REG_OP(Transpose)
    .INPUT(x, TensorType({DT_BF16,      DT_FLOAT16,   DT_FLOAT,      DT_DOUBLE,   DT_INT64,       DT_INT32,
                          DT_UINT8,     DT_UINT16,    DT_UINT32,     DT_UINT64,   DT_INT8,        DT_INT16,
                          DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,    DT_QUINT8,      DT_QINT16,
                          DT_QUINT16,   DT_QINT32,    DT_BOOL,       DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BF16,      DT_FLOAT16,   DT_FLOAT,      DT_DOUBLE,   DT_INT64,       DT_INT32,
                           DT_UINT8,     DT_UINT16,    DT_UINT32,     DT_UINT64,   DT_INT8,        DT_INT16,
                           DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,    DT_QUINT8,      DT_QINT16,
                           DT_QUINT16,   DT_QINT32,    DT_BOOL,       DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Transpose);

/**
 *@brief Returns the shape of a tensor. \n

 *@par Inputs:
 *x: A tensor. Must be one of the following types: float32、float16、int8、
 int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string、bfloat16. \n

 *@par Attributes:
 *dtype: An optional int32 or int64. The output data type. Defaults to int32. \n

 *@par Outputs:
 *y: A tensor. The shape of the input tensor. \n

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator Size.
 */
REG_OP(Shape)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Shape);

/**
 * @brief Gather slices from "params" according to "indices"."indices" must be
 *     an integer tensor of any dimension(usually 0-D or 1-D).
 *     Produces an output tensor with shape "indices.shape + params.shape[1:]" .

 * @par Inputs:
 * Two inputs, including:
 * @li x: A Tensor. Must be one of the following types: complex128, complex64, float64, float32, float16,
 *     int16, int32, int64, int8, qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8,
 *     bool, bfloat16.
 * @li indices: A Tensor of type int32 or int64 .

 * @par Attributes:
 * @li validate_indices: Whether to verify the values of indices, not enabled currently.
 * @li batch_dims: An optional int. Defaults to 0.
 * @li is_preprocessed: An optional bool. Whether to preprocess. Defaults to false.
 * @li negative_index_support: An optional bool. Defaults to false.

 * @par Outputs:
 * y: A Tensor. Has the same type as "x" .

 * @attention Constraints:
 * "indices" is in the range [0, x.shape[0]) .

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator Gather .

 */
REG_OP(Gather)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,   DT_INT32,
                          DT_INT64,      DT_INT8,      DT_QINT16, DT_QINT32, DT_QINT8,   DT_QUINT16, DT_QUINT8,
                          DT_UINT16,     DT_UINT32,    DT_UINT64, DT_UINT8,  DT_BOOL,    DT_BF16}))
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,   DT_INT32,
                           DT_INT64,      DT_INT8,      DT_QINT16, DT_QINT32, DT_QINT8,   DT_QUINT16, DT_QUINT8,
                           DT_UINT16,     DT_UINT32,    DT_UINT64, DT_UINT8,  DT_BOOL,    DT_BF16}))
    .ATTR(validate_indices, Bool, true)
    .ATTR(batch_dims, Int, 0)
    .ATTR(is_preprocessed, Bool, false)
    .ATTR(negative_index_support, Bool, false)
    .OP_END_FACTORY_REG(Gather);

/**
 *@brief Packs the list of tensors in values into a tensor with rank one higher
 * than each tensor in values, by packing them along the axis dimension.
 * Given a list of length N of tensors of shape (A, B, C); if axis == 0 then
 * the output tensor will have the shape (N, A, B, C) .

 *@par Inputs:
 * x: A list of N Tensors. Must be one of the following types: complex128,
 * complex64, double, float32, float16, int16, int32, int64, int8, qint16,
 * qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8, bfloat16,
 * complex32. It's a dynamic input.

 *@par Attributes:
 *@li axis: An optional int, default value is 0.
 *     Dimension along which to pack. The range is [-(R+1), R+1).
 *@li N: An optional int, default value is 1. Number of tensors.

 *@par Outputs:
 *y: A Tensor. Has the same type as "x".

 *@par Third-party framework compatibility
 * Compatible with the TensorFlow operator Pack.
 */
REG_OP(Pack)
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .ATTR(axis, Int, 0)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Pack);

/**
 *@brief Reshapes a tensor. Only the tensor shape is changed, without changing the data. \n

 *@par Inputs:
 *@li x: A tensor. All data types are supported. \n
 *@li shape: A tensor. Must be one of the following types: int32, int64. Defines the shape of the output tensor. \n

 *@par Attributes:
 *@li axis: An optional int32 or int64. The first dimension to reshape. Defaults to "0".
 *@li num_axes: An optional int32 or int64. The extent of the reshape. Defaults to "-1". \n

 *@par Outputs:
 *y: A tensor. The same type as input x. \n

 *@attention Constraints:
 *This operator cannot be directly called by the acllopExecute API. \n

 *@par Third-party framework compatibility
 *@li Compatible with the TensorFlow operator Reshape.
 *@li Compatible with the Caffe operator Reshape.
 */
REG_OP(Reshape)
    .INPUT(x, TensorType::ALL())
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Reshape);

} // namespace ge
