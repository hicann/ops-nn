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
 * \file nn_norm.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Performs group normalization . \n

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type float16 or float32, with format NCHW for 4D.
* @li gamma: A Tensor of type float16 or float32. Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type float16 or float32. Must be 1D. Specifies the offset.

* @par Attributes:
* @li num_groups: An required int32, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to
variance to avoid dividing by zero. Defaults to "0.0001".
* @li data_format: An optional string, specifying the format of "x".
Defaults to "NHWC".
* @li is_training: An optional bool, specifying if the operation is used for
training or inference. Defaults to "True" .

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type float16 or float32 for the normalized "x",
with format NCHW for 4D.
* @li mean: A Tensor of type float16 or float32. Must be 1D. Specifies the mean of "x".
* @li variance: A Tensor of type float16 or float32. Must be 1D. Specifies the variance of "x".

* @attention Constraints:
* @li For Atlas 200/300/500 Inference Product, only support NCHW which can be trans to 5HD.
* @li the value range of the inputs should be constrained between -10000 and 10000.

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm.

*/
REG_OP(GroupNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NHWC")
    .ATTR(eps, Float, 0.0001f)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(GroupNorm)

/**
* @brief Performs group normalization . \n

* @par Inputs:
* Three inputs
* @li x: A ND tensor of type bfloat16, float16, float32. The input feature map will be processed by group normalization.
* "x" supports 2-8 dimensions (N, C, *), the calculation logic only cares about the first two dimensions (N and C),
* and the rest can all be combined into one dimension.
* The data type and shape of x must meet the following conditions:
* - When the data type of x is float32, C/num_groups must be a multiple of 8.
* - When the data type of x is float16 or bfloat16, C/num_groups must be a multiple of 16.
* @li gamma: A ND tensor of type bfloat16, float16, float32. Must be 1D. Specifies the scaling factor.
* The value of "gamma" needs to be consistent with the C-axis value of "x". Has the same dype as "x".
* @li beta: A ND tensor of type bfloat16, float16, float32. Must be 1D. Specifies the offset.
* The value of "beta" needs to be consistent with the C-axis value of "x". Has the same dype as "x". \n

* @par Attributes:
* @li num_groups: An required int32, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to variance to avoid dividing by zero. Defaults to "0.00001".
* @li data_format: An optional string, specifying the format of "x". Defaults to "NHWC". This parameter is reserved and does not take effect.
* @li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to "True".

* - When set to true, it indicates training mode and uses the mean and variance of the current batch.
* - When set to false, it indicates inference mode and uses the mean and variance saved during training. \n

* @par Outputs:
* Three outputs
* @li y: A ND tensor of type bfloat16, float16, float32 for the normalized "x". Has the same type, format and shape as "x".
* @li mean: A ND tensor of type bfloat16, float16, float32. Must be 2D (N, num_groups). Specifies the mean of "x".
* Has the same dype as "x".
* @li rstd: A ND tensor of type bfloat16, float16, float32. Must be 2D (N, num_groups). Specifies the rstd of "x".
* Has the same dype as "x". \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm.

*/
REG_OP(GroupNormV2)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NHWC")
    .ATTR(eps, Float, 0.00001f)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(GroupNormV2)

/**
 * @brief backward operator for group normalization. \n
 * @par Inputs:
 * Five input, including:
 * @li dy: A tensor. Group grad. Datatype support float32, float16, bfloat16. Format support ND.
 * "dy" supports 2-8 dimensions (N, C, *), the calculation logic only cares about the first two dimensions (N and C),
 * and the rest can all be combined into one dimension.
 * @li mean: A tensor. Mean of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * Must be 2D (N, num_groups).
 * @li rstd: A tensor. Reciprocal standard deviation of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * Must be 2D (N, num_groups).
 * @li x: A Tensor. Specifies the offset. Datatype support float32, float16, bfloat16. Format support ND.
 * "x" supports 2-8 dimensions (N, C, *), the calculation logic only cares about the first two dimensions (N and C),
 * and the rest can all be combined into one dimension.
 * @li gamma: A tensor. Specifies the scaling factor. Datatype support float32, float16, bfloat16. Format support ND.
 * Must be 1D. The value of "gamma" needs to be consistent with the C-axis value of "x".

 * @par Attributes:
 * @li num_groups: Int. Number specifying the number of group.
 * @li data_format: An optional string. Defaults to NCHW.
 * @li dx_is_require: An optional bool, controls whether to return dx. Defaults to true.
 * @li dgamma_is_require: An optional bool, controls whether to return dgamma. Defaults to true.
 * @li dbeta_is_require: An optional bool, controls whether to return dbeta. Defaults to true.

 * @par Outputs:
 * Three output, including:
 * @li dx: A tensor. x factor grad. Datatype is the same as the input datatype. Has the same format and shape as "x".
 * @li dgamma: A tensor. Scale factor grad. Has the same datatype, format and shape as "gamma".
 * @li dbeta: A tensor. Offset factor grad. Has the same datatype, format and shape as "gamma".
 * @par Third-party framework compatibility
 * @li Compatible with the backward of PyTorch operator GroupNorm.
 */

REG_OP(GroupNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(dx_is_require, Bool, true)
    .ATTR(dgamma_is_require, Bool, true)
    .ATTR(dbeta_is_require, Bool, true)
    .OP_END_FACTORY_REG(GroupNormGrad)

/**
* @brief Performs group normalization and silu.

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type bfloat16/float16/float32.
* @li gamma: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the offset.

* @par Attributes:
* @li num_groups: An required int32/int64, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to the
* denominator for numerical stability. Defaults to "0.00001".
* @li activate_silu: An optional bool.  Defaults to "true".

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type bfloat16/float16/float32 for the normalized "x".
* @li mean: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the mean of "x".
* Ascend 910_95 AI Processor: The data type must be consistent with that of gamma and beta.
* @li rstd: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the rstd of "x".
* Ascend 910_95 AI Processor: The data type must be consistent with that of gamma and beta.

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm and Silu.

*/
REG_OP(GroupNormSilu)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(eps, Float, 0.00001f)
    .ATTR(activate_silu, Bool, true)
    .OP_END_FACTORY_REG(GroupNormSilu)

/**
* @brief Performs group normalization and swish. \n

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type bfloat16/float16/float32.
* @li gamma: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the offset. \n

* @par Attributes:
* @li num_groups: An required int32/int64, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to the
* denominator for numerical stability. Defaults to "0.00001".
* @li data_format: An optional String, Defaults to NCHW.
* @li activate_swish: An optional bool.  Defaults to "true".
* @li swish_scale: An optional float.  Defaults to "1". \n

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type bfloat16/float16/float32 for the normalized "x".
* @li mean: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the mean of "x".
* @li rstd: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the rstd of "x". \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm and Swish.

*/
REG_OP(GroupNormSwish)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(eps, Float, 0.00001f)
    .ATTR(activate_swish, Bool, true)
    .ATTR(swish_scale, Float, 1.0)
    .OP_END_FACTORY_REG(GroupNormSwish)

/**
* @brief Performs the backward operation of group normalization and swish.

 * @par Inputs:
 * Six input, including:
 * @li dy: A Tensor. Group grad. Datatype support float32, float16, bfloat16. Format support ND.
 * @li mean: A Tensor. Mean of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * @li rstd: A Tensor. Reciprocal standard deviation of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * @li x: A Tensor. Specifies the offset. Datatype support float32, float16, bfloat16. Format support ND. Same shape as mean.
 * @li gamma: A Tensor. Specifies the scaling factor. Datatype support float32, float16, bfloat16. Format support ND. Same shape as dy.
 * @li beta: A Tensor. Specifies the intercept. Datatype support float32, float16, bfloat16. Format support ND. Same shape as gamma.

* @par Attributes:
* @li num_groups: Int. Number specifying the number of group.
* @li data_format: An optional String, Defaults to NCHW.
* @li swish_scale: An optional float. Defaults to "1.0".
* @li dgamma_is_require: An optional bool, controls whether to return weight.grad. Defaults to true.
* @li dbeta_is_require: An optional bool, controls whether to return beta.grad. Defaults to true.

 * @par Outputs:
 * Three output, including:
 * @li dx: A Tensor. x factor grad. Datatype is the same as the input Datatype. Format support ND.
 * @li dgamma: A Tensor. scale factor grad. Datatype is the same as the input Datatype. Format support ND.
 * @li dbeta: A Tensor. offset factor grad. Datatype is the same as the input Datatype. Format support ND.

* @par Third-party framework compatibility
* @li Compatible with the backward of PyTorch operator GroupNorm and Swish.

*/
REG_OP(GroupNormSwishGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(swish_scale, Float, 1.0)
    .ATTR(dgamma_is_require, Bool, true)
    .ATTR(dbeta_is_require, Bool, true)
    .OP_END_FACTORY_REG(GroupNormSwishGrad)

/**
* @brief AddRmsNorm operator interface implementation. \n
*  calculating: x1, x2, gamma \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  y = gamma * (x * rstd)

* @par Inputs
* Three inputs, including:
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes
* epsilon: Input eps in the formula, which is used to prevent division-by-zero errors.
* A optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Three outputs, including:
* @li y: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(AddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(AddRmsNorm)

/**
* @brief AddRmsNormCast operator interface implementation. \n
*  calculating: x1, x2, gamma \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  y2 = gamma * (x * rstd) \n
*  y1 = cast16232(y2) \n

* @par Inputs
* Three inputs, including:
* @li x1: A tensor. Support dtype: float16/bfloat16, support format: ND.
* @li x2: A tensor. Support dtype: float16/bfloat16, support format: ND.
* @li gamma: A tensor. Support dtype: float16/bfloat16, support format: ND.

* @par Attributes
* epsilon: Input eps in the formula, which is used to prevent division-by-zero errors.
* An optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Four outputs, including:
* @li y1: A tensor. Support dtype: float32, support format: ND.
* @li y2: A tensor. Support dtype: float16/bfloat16, support format: ND.
* @li rstd: A tensor. Describing the reciprocal of (x1 + x2)'s standard deviation.
*           Support dtype: float32, support format: ND.
* @li x: A tensor. Support dtype: float16/bfloat16, support format: ND.
*/
REG_OP(AddRmsNormCast)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(AddRmsNormCast)

/**
* @brief QuantizeAddLayerNorm operator interface implementation.

* @par Inputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li bias: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li scales: A Tensor. Support dtype: [float32, bfloat16], support format: [ND].
* @li zero_points: A optional Tensor. Support dtype: [float32, bfloat16], support format: [ND].

* @par Attributes
* @li dtype: A required attribute, the type is int. No defaults value.
* @li axis: A optional attribute, the type is int. Defaults to -1.
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-5.
* @li additional_output: A optional attribute, the type is bool. Defaults to false.

* @par Outputs
* @li y: A Tensor. Support dtype: [int8], support format: [ND].
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(QuantizeAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(bias, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(QuantizeAddLayerNorm)

/**
* @brief DuaQuantizeAddLayerNorm operator interface implementation.

* @par Inputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li bias: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li scales1: A Tensor. Support dtype: [float32, bfloat16], support format: [ND].
* @li scales2: A Tensor. Support dtype: [float32, bfloat16], support format: [ND].
* @li zero_points1: A optional Tensor. Support dtype: [int8, uint8, bfloat16, int32], support format: [ND].
* @li zero_points2: A optional Tensor. Support dtype: [int8, uint8, bfloat16, int32], support format: [ND].

* @par Attributes
* @li dtype: A required attribute, the type is int. No defaults value.
* @li axis: A optional attribute, the type is float. Defaults to -1.
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-5.
* @li additional_output: A optional attribute, the type is bool. Defaults to false.

* @par Outputs
* @li y1: A Tensor. Support dtype: [int8, uint8, int32], support format: [ND].
* @li y2: A Tensor. Support dtype: [int8, uint8, int32], support format: [ND].
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(DuaQuantizeAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(scales1, ge::TensorType({DT_BF16, DT_FLOAT}))
    .INPUT(scales2, ge::TensorType({DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points1, ge::TensorType({DT_INT8, DT_UINT8, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(zero_points2, ge::TensorType({DT_INT8, DT_UINT8, DT_BF16, DT_INT32}))
    .OUTPUT(y1, ge::TensorType({DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(y2, ge::TensorType({DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(DuaQuantizeAddLayerNorm)

/**
* @brief InplaceAddRmsNorm operator interface implementation. \n
*  calculating: x1, x2, gamma \n
*  x2 = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  x1 = gamma * (x2 * rstd)

* @par Inputs
* Three inputs, including:
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes:
* epsilon: A optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Three outputs, including:
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(InplaceAddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(InplaceAddRmsNorm)

/**
* @brief AddRmsNormQuant operator interface implementation.
* Calculating input: x1, x2, gamma, scales1, scales2, zero_points1, zero_points2 \n
* Calculating process: \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  rmsnorm_out = x * rstd * gamma \n
*  if div_mode is true: \n
*    y1 = round(rmsnorm_out / scales1 + zero_points1) \n
*    y2 = round(rmsnorm_out / scales2 + zero_points2) \n
*  if div_mode is false: \n
*    y1 = round(rmsnorm_out * scales1 + zero_points1) \n
*    y2 = round(rmsnorm_out * scales2 + zero_points2) \n

* @par Inputs
* @li x1: A tensor. Input x1 for the add operation.
*         Support dtype: float32/float16/bfloat16, support format: ND.
* @li x2: A tensor. Input x2 for the add operation.
*         Support dtype: float32/float16/bfloat16, support format: ND.
* @li gamma: A tensor. Describing the weight of the rmsnorm operation.
*            Support dtype: float32/float16/bfloat16, support format: ND.
* @li scales1: A tensor. Describing the weight of the first quant operation.
*              Support dtype: float32/float16/bfloat16, support format: ND.
* @li scales2: An optional input tensor. Describing the weight of the secend quant operation.
*              Support dtype: float32/float16/bfloat16, support format: ND.
* @li zero_points1: An optional input tensor. Describing the bias of the first quant operation.
*                   Support dtype: int32/float32/float16/bfloat16, support format: ND.
* @li zero_points2: An optional input tensor. Describing the bias of the secend quant operation.
*                   Support dtype: int32/float32/float16/bfloat16, support format: ND.
* @li beta: An optional input tensor. Describing the bias of the rmsnorm operation.
*                   Support dtype: int32/float32/float16/bfloat16, support format: ND.

* @par Attributes
* @li axis: An optional attribute. Describing the axis of the quant operation, does not take effect now.
*           The type is int. Defaults to -1.
* @li epsilon: An optional attribute. Describing the epsilon of the rmsnorm operation.
*              The type is float. Defaults to 1e-6.
* @li div_mode: An optional attribute. When div_mode is true, the quant opertaion uses division, otherwise, uses multiplication.
*               The type is bool. Defaults to true.

* @par Outputs
* @li y1: A tensor. Describing the output of the first quant operation.
*                   Support dtype: int8, support format: ND.
* @li y2: A tensor. Describing the output of the second quant operation.
*                   Support dtype: int8, support format: ND.
* @li x: A tensor. Describing the output of the x1+x2 add operation.
*                  Support dtype: float32/float16/bfloat16, support format: ND.
*/
REG_OP(AddRmsNormQuant)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(scales2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points1, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points2, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8}))
    .OUTPUT(y2, TensorType({DT_INT8}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-6f)
    .ATTR(div_mode, Bool, true)
    .OP_END_FACTORY_REG(AddRmsNormQuant)

/**
* @brief Fused Operator of Add, RmsNorm and DynamicQuant.
* Calculating input: x1, x2, gamma, smooth_scale1, smooth_scale2 \n
* Calculating process: \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  rmsnorm_out = x * rstd * gamma \n
*  if smooth_scales1 exist: \n
*    scale1 = row_max(abs(rmsnorm_out * smooth_scale1)) / 127 \n
*  if smooth_scales1 not exist: \n
*    scale1 = row_max(abs(rmsnorm_out)) / 127 \n
*  y1 = round(rmsnorm_out / scale1) \n
*  if smooth_scales2 exist: \n
*    scale2 = row_max(abs(rmsnorm_out * smooth_scale2)) / 127 \n
*    y2 = round(rmsnorm_out / scale2) \n
*  if smooth_scales2 not exist:  \n
*    not calculate scale2 and y2. \n

* @par Inputs
* @li x1: A tensor. Input x1 for the add operation.
*         Support dtype: float16/bfloat16, support format: ND.
* @li x2: A tensor. Input x2 for the add operation.
*         Support dtype: float16/bfloat16, support format: ND.
* @li gamma: A tensor. Describing the weight of the rmsnorm operation.
*            Support dtype: float16/bfloat16, support format: ND.
* @li smooth_scale1: A tensor. Describing the weight of the first dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND.
* @li smooth_scale2: An optional input tensor. Describing the weight of the secend dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND.
* @li beta: An optional input tensor. Describing the offset value of dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND.
* @par Attributes
* epsilon: An optional attribute. Describing the epsilon of the rmsnorm operation.
*          The type is float. Defaults to 1e-6.

* @par Outputs
* @li y1: A tensor. Describing the output of the first dynamic quantization.
*                   Support dtype: int8, support format: ND.
* @li y2: A tensor. Describing the output of the second dynamic quantization.
*                   Support dtype: int8, support format: ND.
* @li x: A tensor. Describing the output of the x1+x2 add operation.
*                  Support dtype: float16/bfloat16, support format: ND.
* @li scale1: A tensor. Describing of the factor for the first dynamic quantization.
*                  Support dtype: float32, support format: ND.
* @li scale2: A tensor. Describing of the factor for the second dynamic quantization.
*                  Support dtype: float32, support format: ND.
*/
REG_OP(AddRmsNormDynamicQuant)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8, DT_INT8}))
    .OUTPUT(y2, TensorType({DT_INT8, DT_INT8}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(scale1, TensorType({DT_FLOAT, DT_FLOAT}))
    .OUTPUT(scale2, TensorType({DT_FLOAT, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6)
    .ATTR(outQuant1Flag, Bool, false)
    .ATTR(outQuant2Flag, Bool, false)
    .OP_END_FACTORY_REG(AddRmsNormDynamicQuant)

/**
* @brief Fused Operator of 0~4 Add, 1 RmsNorm and 1 DynamicQuant.
* Calculating input: x1, x2, gamma, smooth_scale1, smooth_scale2 \n
* Calculating process: \n
*  x = sum(x1) + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  rmsnorm_out = x * rstd * gamma \n
*  if smooth_scales1 exist: \n
*    scale1 = row_max(abs(rmsnorm_out * smooth_scale1)) / 127 \n
*  if smooth_scales1 not exist: \n
*    scale1 = row_max(abs(rmsnorm_out)) / 127 \n
*  y1 = round(rmsnorm_out / scale1) \n
*  if smooth_scales2 exist: \n
*    scale2 = row_max(abs(rmsnorm_out * smooth_scale2)) / 127 \n
*    y2 = round(rmsnorm_out / scale2) \n
*  if smooth_scales2 not exist:  \n
*    not calculate scale2 and y2. \n
*  Only for experimental use. \n

* @par Inputs
* @li x1: A tensor list. Each item of x1 is for the add operation, supports length 1 to 5.
*         Support dtype: float16/bfloat16, support format: ND.
* @li x2: A tensor. Input x2 for the add operation.
*         Support dtype: float16/bfloat16, support format: ND.
* @li gamma: A tensor. Describing the weight of the rmsnorm operation.
*            Support dtype: float16/bfloat16, support format: ND.
* @li smooth_scale1: A tensor. Describing the weight of the first dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND.
* @li smooth_scale2: An optional input tensor. Describing the weight of the secend dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND.

* @par Attributes
* epsilon: An optional attribute. Describing the epsilon of the rmsnorm operation.
*          The type is float. Defaults to 1e-6.

* @par Outputs
* @li y1: A tensor. Describing the output of the first dynamic quantization.
*                   Support dtype: int8, support format: ND.
* @li y2: A tensor. Describing the output of the second dynamic quantization.
*                   Support dtype: int8, support format: ND.
* @li x: A tensor. Describing the output of the x1+x2 add operation.
*                  Support dtype: float16/bfloat16, support format: ND.
* @li y: A tensor. Describing the output of the rmsNorm operation.
*                  Support dtype: float16/bfloat16, support format: ND.
* @li scale1: A tensor. Describing of the factor for the first dynamic quantization.
*                  Support dtype: float32, support format: ND.
* @li scale2: A tensor. Describing of the factor for the second dynamic quantization.
*                  Support dtype: float32, support format: ND.
*/
REG_OP(MultiAddRmsNormDynamicQuant)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8}))
    .OUTPUT(y2, TensorType({DT_INT8}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(scale1, TensorType({DT_FLOAT}))
    .OUTPUT(scale2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6)
    .OP_END_FACTORY_REG(MultiAddRmsNormDynamicQuant)

/**
* @brief Fused Operator of Add and LayerNorm.

* @par Inputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li bias: A optional input Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-5.
* @li additional_output: A optional attribute, the type is bool. Defaults to false.

* @par Outputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li mean: A Tensor. Support dtype: [float32], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(InplaceAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(mean, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(rstd, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(InplaceAddLayerNorm)

/**
* @brief Fused Operator of AddLayerNorm and Quantize/DynamicQuant/AscendQuantV2. \n
*  calculating: x1, x2, gamma, beta, bias, scales1, scales2, zero_points1, zero_points2 \n
*  x = x1 + x2 + bias \n
*  rstd = 1 / (sqrt(Var(x) + eps)) \n
*  y = (x - E(x)) * rstd * gamma + beta \n
*  when quant_mode = "static" and div_mode=true, output out_scales1 and out_scales2 are invalid: \n
*  y1 = round(y / scales1) + zero_points1 \n
*  y2 = round(y / scales2) + zero_points2 \n
*  when quant_mode = "static" and div_mode=false, output out_scales1 and out_scales2 are invalid: \n
*  y1 = round(y * scales1) + zero_points1 \n
*  y2 = round(y * scales2) + zero_points2 \n
*  when quant_mode = "dynamic": \n
*  tmp1 = y * scales1 \n
*  tmp2 = y * scales2 \n
*  out_scales1 = reduce_max(abs(tmp1))/127 \n
*  out_scales2 = reduce_max(abs(tmp2))/127 \n
*  y1 = round(tmp1 / out_scales1) \n
*  y2 = round(tmp2 / out_scales2) \n

* @par Inputs
* @li x1: A tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li x2: A tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li gamma: A tensor for layer norm weight params. Support dtype: float32, float16, bfloat16, support format: ND.
* @li beta: A tensor for layer norm weight params. Support dtype: float32, float16, bfloat16, support format: ND.
* @li bias: An optional input tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li scales1: An optional input tensor for one of quant scale. Support dtype: float32, float16, bfloat16, support
format: ND.
* @li scales2: An optional input tensor for another quant scale. Support dtype: float32, float16, bfloat16, support
format: ND.
* @li zero_points1: An optional input tensor for one of quant offset. Support dtype: float32, float16, bfloat16,
support format: ND.
* @li zero_points2: An optional input tensor for another quant offset. Support dtype: float32, float16, bfloat16,
support format: ND.

* @par Attributes
* @li quant_mode: An optional attribute utilized to select quant mode, can be "dynamic" or "static", the type is
string. Defaults to "dynamic".
* @li epsilon: An optional attribute for layer norm compute, the type is float. Defaults to 1e-5.
* @li additional_output: An optional attribute control whether output x valid or invalid, the type is bool. Defaults
to false, which means x output is invalid.
* @li div_mode: An optional attribute control static quant algorithm, the type is bool. Defaults
to true, which means scales while be divided by normlization output.

* @par Outputs
* @li y1: Quantize result 1.
*     A tensor. Support dtype: int8, support format: ND.
* @li y2: Quantize result 2.
*     A tensor. Support dtype: int8, support format: ND.
* @li x: Describing the result of x1 + x2 + bias.
*     A tensor. Support dtype: float32, float16, bfloat16, support format: ND.
* @li out_scales1: Describing the result of dynamic quantize scales computed via scales1.
*     A tensor. Support dtype: float32, support format: ND.
* @li out_scales2: Describing the result of dynamic quantize scales computed via scales2.
*     A tensor. Support dtype: float32, support format: ND.
*/
REG_OP(AddLayerNormQuant)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(scales1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(scales2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y1, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(y2, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(out_scales1, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(out_scales2, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .ATTR(quant_mode, String, "dynamic")
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .ATTR(div_mode, Bool, true)
    .OP_END_FACTORY_REG(AddLayerNormQuant)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_