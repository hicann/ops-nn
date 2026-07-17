/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_PROTO_H_
#define OPS_NN_PROTO_H_

namespace ge {
/**
 * @brief Applies a 2D adaptive max pooling over an input signal conposed of several input planes.
 * The output is of size H x W, for any input size.
 * @par Inputs:
 * One input, including:
 * @li x: A Tensor. Must be one of the following data types:
 *     float16, float32, float64. \n
 * @par Attributes:
 * @li output_size: A required list of 2 ints
 *    specifying the size (H,W) of the output tensor. \n
 * @par Outputs:
 * @li y: A Tensor. Has the same data type as "x".
 * @li argmax: A Tensor. Describing the index of outputs.
 * @par Third-party framework compatibility
 * Compatible with the Pytorch operator AdaptiveMaxPool2d.
 */
REG_OP(AdaptiveMaxPool2d)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(argmax, TensorType::IndexNumberType())
    .REQUIRED_ATTR(output_size, ListInt)
    .OP_END_FACTORY_REG(AdaptiveMaxPool2d)

    /**
    *@brief data conversion operator
    Convert uint16 to uint32, convert to int32, convert to float32,
    multiply by the reciprocal of pixel, convert to float16 . \n
    *@par Inputs:
    *one inputs, including:
    *@li x: A Tensor. Must be one of the following types: uint16.
    *@par Outputs:
    *y: A Tensor. Must be one of the following types: float16. \n
    *@par Third-party framework compatibility
    *only for use by corresponding operators in HDRnet networks
    */
    REG_OP(AdaCast)
    .INPUT(x, "T1")
    .OUTPUT(y, "T2")
    .ATTR(pixel, Int, 65535)
    .DATATYPE(T1, TensorType({DT_UINT16}))
    .DATATYPE(T2, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(AdaCast)

    /**
     * @brief Anti quantizes the input . \n
     * @par Inputs:
     * x: A multi-dimensional tensor of type int8, specifying the input.
     * The format support ND. Shape support 1D ~ 8D.
     * @par Attributes:
     * @li scale: A required float32, specifying the scaling ratio.
     * @li offset: A required float32, specifying the offset.
     * @li dtype: An optional int32, specifying the output data type.
     * Defaults to "DT_FLOAT".
     * @li sqrt_mode: An optional bool, specifying whether to perform square root on
     * "scale", either "True" or "False". Defaults to "False". \n
     * @par Outputs:
     * y: The dequantized output tensor of type float16 or float32.
     * The format support ND. Shape support 1D ~ 8D. The shape is same as x. \n
     * @par Third-party framework compatibility
     * It is a custom operator. It has no corresponding operator in Caffe.
     */
    REG_OP(AscendAntiQuant)
    .INPUT(x, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .ATTR(dtype, Int, DT_FLOAT)
    .ATTR(sqrt_mode, Bool, false)
    .OP_END_FACTORY_REG(AscendAntiQuant)

    /**
     * @brief Dequantizes the input.
     * @par Inputs:
     * @li x: A tensor of type int32, specifying the input. Shape support 1D ~ 8D.
     * The format must be FRACTAL_NZ, NC1HWC0 or NDC1HWC0.
     * @li deq_scale: A required Tensor. Must be one of the following types: float16,
     * uint64. The format must be NC1HWC0 or NDC1HWC0. If deq_scale is 1D tensor,
     * shape must be same as the last dimension of x. Otherwise the number of
     * dimensions should be equal to x, the last dimension of shape should be
     * the same as x, others must be 1. \n
     * @par Attributes:
     * @li sqrt_mode: An optional bool, specifying whether to perform square root
     * on "scale", either "True" or "False". Defaults to "False".
     * @li relu_flag: An optional bool, specifying whether to perform ReLU,
     * either "True" or "False". Defaults to "False".
     * @li dtype: An optional int32, specifying the output data type. Defaults to "0"
     * , represents dtype "DT_FLOAT". \n
     * @par Outputs:
     * y: The dequantized output tensor of type float16 or float32. The format must
     * be FRACTAL_NZ, NC1HWC0 or NDC1HWC0. The shape is same as x. \n
     * @par Third-party framework compatibility
     * It is a custom operator. It has no corresponding operator in Caffe.
     */
    REG_OP(AscendDequant)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(deq_scale, TensorType({DT_FLOAT16, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(relu_flag, Bool, false)
    .ATTR(dtype, Int, DT_FLOAT)
    .OP_END_FACTORY_REG(AscendDequant)

    /**
     * @brief Dequantizes the input of int16 . \n
     * @par Inputs:
     * @li x0: A tensor of type int32, specifying the input.
     * The format support NC1HWC0, FRACTAL_NZ. Shape support 4D ~ 8D.
     * @li deq_scale: A tensor of type uint64, specifying the scaling ratio.
     * The format support NC1HWC0. Shape support 5D, must be 1 in n, h, w.
     * @li x1: A tensor of type int16, specifying the input.
     * The format support NC1HWC0, ND. Shape support 1D or 5D.
     * When the format of x1 is ND, the shape length of x1 must be 1. \n
     * @par Attributes:
     * relu_flag: An optional bool, specifying whether to perform ReLU,
     * either "True" or "False". Defaults to "False" . \n
     * @par Outputs:
     * y: The dequantized output tensor of type int16.
     * The format support NC1HWC0, FRACTAL_NZ. Shape support 4D ~ 8D.
     * The shape and format are the same as input "x0". \n
     * @par Third-party framework compatibility
     * It is a custom operator. It has no corresponding operator in Caffe.
     */
    REG_OP(AscendDequantS16)
    .INPUT(x0, TensorType({DT_INT32}))
    .INPUT(deq_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(x1, TensorType({DT_INT16}))
    .OUTPUT(y, TensorType({DT_INT16}))
    .ATTR(relu_flag, Bool, false)
    .OP_END_FACTORY_REG(AscendDequantS16)

    /**
     * @brief Quantizes the input.
     * @par Inputs:
     * x: A tensor of type float16 or float32, specifying the input.
     * The format must be NC1HWC0, FRACTAL_NZ, NDC1HWC0 or ND. Shape supports 1D ~ 8D.
     * If "dst_type" is 29, the last dimension of the shape must be divisible by 2. \n
     * @par Attributes:
     * @li scale: A required float32, specifying the scaling ratio.
     * @li offset: A required float32, specifying the offset.
     * @li sqrt_mode: An optional bool, specifying whether to perform square on "scale", either "True" or "False".
     * Defaults to "False".
     * @li round_mode: An optional string, specifying the cast mode.
     * The value range is [Round, Floor, Ceil, Trunc, Hybrid]. Defaults to "Round".
     * @li dst_type: An optional int32, specifying the output data type.
     * Defaults to "2", represents dtype "DT_INT8". "29" represents dtype "DT_INT4", "34" represents dtype
     * "DT_HIFLOAT8", "35" represents dtype "DT_FLOAT8_E5M2", "36" represents dtype "DT_FLOAT8_E4M3FN". \n
     * @par Outputs:
     * y: The quantized output tensor of type int8, int4, hifloat8, float8_e5m2 or float8_e4m3fn.
     * The format must be NC1HWC0, FRACTAL_NZ, NDC1HWC0 or ND. Shape supports 1D ~ 8D.
     * Has the same format and shape as input "x". \n
     * @attention Constraints:
     * @li round_mode value range is [Round, Floor, Ceil, Trunc, Hybrid]. \n
     * Round: round to nearest, tie to even(c language rint). \n
     * Floor: round to minus infinity(c language floor). \n
     * Ceil: round to positive infinity(c language ceil). \n
     * Trunc: round to zero(c language trunc). \n
     * Hybrid: only valid when output dtype is hifloat8. \n
     * The following constraints apply to products other than Ascend 950 AI Processor: \n
     * @li When format is FRACTAL_NZ, shape supports 4D ~ 8D.
     * @li When "x" is dynamic shape, shape [-2] is not supported.
     * @li When "x" is dynamic shape, the data type of output "y" does not support int4.
     * @li When the format of "x" is ND, the data type of output "y" does not support int4. \n
     * @par Third-party framework compatibility
     * It is a custom operator. It has no corresponding operator in Caffe.
     */
    REG_OP(AscendQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(round_mode, String, "Round")
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(AscendQuant)

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
    REG_OP(AscendRequant)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(req_scale, TensorType({DT_UINT64}))
    .OUTPUT(y, TensorType({DT_INT8}))
    .ATTR(relu_flag, Bool, false)
    .OP_END_FACTORY_REG(AscendRequant)

    /**
     * @brief Requantizes the input of int16 . \n
     * @par Inputs:
     * @li x0: A tensor of type int16, specifying the input. The format must be
     * FRACTAL_NZ or NC1HWC0. Shape support 4D ~ 8D.
     * @li req_scale: A tensor of type uint64, specifying the scaling ratio.
     * The format support NC1HWC0. Shape support 5D, must be 1 in n, h, w.
     * @li x1: A tensor of type int16, specifying the input.
     * The format support NC1HWC0, FRACTAL_NZ. Shape support 4D ~ 8D.
     * Has the same format as x. \n
     * @par Attributes:
     * @li dual_output: An optional bool, specifying whether to perform dual ouput,
     * either "True" or "False". Defaults to "False".
     * @li relu_flag: An optional bool, specifying whether to perform ReLU,
     * either "True" or "False". Defaults to "False" . \n
     * @par Outputs:
     * @li y0: The dequantized output tensor of type int8.
     * The format support FRACTAL_NZ and NC1HWC0. Shape support 4D ~ 8D.
     * Has the same format and shape as input "x0".
     * @li y1: The dequantized output tensor of type int16.
     * The format support FRACTAL_NZ and NC1HWC0. Shape support 4D ~ 8D.
     * Has the same format and shape as input "x0". \n
     * @par Third-party framework compatibility
     * It is a custom operator. It has no corresponding operator in Caffe.
     */
    REG_OP(AscendRequantS16)
    .INPUT(x0, TensorType({DT_INT16}))
    .INPUT(req_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(x1, TensorType({DT_INT16}))
    .OUTPUT(y0, TensorType({DT_INT8}))
    .OUTPUT(y1, TensorType({DT_INT16}))
    .ATTR(dual_output, Bool, false)
    .ATTR(relu_flag, Bool, false)
    .OP_END_FACTORY_REG(AscendRequantS16)

    /**
     * @brief Multiplies matrix "a" by matrix "b", producing "a @ b".
     * @par Inputs:
     * Two inputs, including:
     * @li x1: A matrix Tensor. Must be one of the following types: float16,
     * float32, int32, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW].
     * @li x2: A matrix Tensor. Must be one of the following types: float16,
     * float32, int32, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW].
     * @par Attributes:
     * @li adj_x1: A bool. If True, changes the shape of "x1" from [B, M, K]
     * to [B, K, M] before multiplication.
     * @li adj_x2: A bool. If True, changes the shape of "x2" from [B, K, N]
     * to [B, N, K] before multiplication.
     * @par Outputs:
     * y: The result matrix Tensor. Must be one of the following types: float16,
     * float32, int32, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW]. BatchMatMul supports broadcasting in the
     * batch dimensions.
     * @par Third-party framework compatibility
     * Compatible with the TensorFlow operator BatchMatmul.
     */
    REG_OP(BatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMul)

    /**
     * @brief Performs batch normalization with support for 4D/5D tensors and training/inference modes.
     *
     * @par Inputs:
     * Five inputs, with format constraints as follows:
     * @li x: A 4D or 5D tensor of type float16, bfloat16, or float32.
     *        Supported data formats:
     *        - 4D: NHWC (batch, height, width, channels) or NCHW (batch, channels, height, width).
     *        - 5D: NDHWC (batch, depth, height, width, channels) or NCDHW (batch, channels, depth, height, width).
     * @li scale: A 1D tensor of type float32, with length equal to the number of channels in "x".
     *        Specifies the scaling factor (gamma) applied after normalization.
     * @li offset: A 1D tensor of type float32, with length equal to the number of channels in "x".
     *        Specifies the offset (beta) applied after scaling.
     * @li mean: A 1D tensor of type float32, with length equal to the number of channels in "x".
     *        - Inference mode (is_training=false): Must be provided as input, representing the
     *          moving mean computed during training.
     *        - Training mode (is_training=true): Optional input. When provided, will be used to
     *          initialize the moving mean for updates; when None, moving mean starts from zeros.
     * @li variance: A 1D tensor of type float32, with length equal to the number of channels in "x".
     *        - Inference mode (is_training=false): Must be provided as input, representing the
     *          moving variance computed during training.
     *        - Training mode (is_training=true): Optional input. When provided, will be used to
     *          initialize the moving variance for updates; when None, moving variance starts from ones.
     *
     * @par Attributes:
     * @li epsilon: Optional float32. Small value added to variance to avoid division by zero.
     *        Defaults to 0.0001f.
     * @li data_format: Optional string. Specifies the data format of "x".
     *        Allowed values: "NHWC" (4D default), "NCHW" (4D), "NDHWC" (5D), "NCDHW" (5D).
     * @li is_training: Optional bool. Specifies operation mode:
     *        - true: Training mode (computes batch mean/variance and updates moving stats).
     *        - false: Inference mode (uses provided mean/variance for normalization).
     *        Defaults to true.
     * @li exponential_avg_factor: Optional float32. Factor for updating moving averages during training.
     *        Formula: new_mean = (1 - factor) * old_mean + factor * batch_mean.
     *        Defaults to 1.0f.
     *
     * @par Outputs:
     * Up to six outputs, with shape and format matching "x" unless specified:
     * @li y: A tensor with the same rank (4D/5D), type, and format as "x", containing normalized values.
     *        (Required output)
     * @li batch_mean: A 1D tensor of type float32 (channel dimension).
     *        - Training mode: Mean of the current batch (computed over spatial dimensions).
     *        - Inference mode: Equal to input "mean" (for compatibility).
     *        (Required output)
     * @li batch_variance: A 1D tensor of type float32 (channel dimension).
     *        - Training mode: Variance of the current batch (computed over spatial dimensions, with Bessel's
     * correction).
     *        - Inference mode: Equal to input "variance" (for compatibility).
     *        (Required output)
     * @li reserve_space_1: Optional 1D tensor of type float32 (channel dimension).
     *        Reserved for gradient computation.
     *        - Training mode: Same as batch_mean.
     *        - Inference mode: Same as input "mean".
     * @li reserve_space_2: Optional 1D tensor of type float32 (channel dimension).
     *        Reserved for gradient computation.
     *        - Training mode: saved inv_var (1/sqrt(epsilon + variance), to be reused in the backward gradient
     * computation.
     *        - Inference mode: Same as input "variance".
     * @li reserve_space_3: A 1D tensor of type float32 with exactly one element.
     *        Exists solely for TensorFlow compatibility and contains no meaningful data.
     */
    REG_OP(BatchNorm)
    .INPUT(x, "T1")
    .INPUT(scale, "T2")
    .INPUT(offset, "T2")
    .OPTIONAL_INPUT(mean, "T2")
    .OPTIONAL_INPUT(variance, "T2")
    .OUTPUT(y, "T1")
    .OUTPUT(batch_mean, "T2")
    .OUTPUT(batch_variance, "T2")
    .OUTPUT(reserve_space_1, "T2")
    .OUTPUT(reserve_space_2, "T2")
    .OUTPUT(reserve_space_3, "T2")
    .ATTR(epsilon, Float, 1e-4f)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .ATTR(exponential_avg_factor, Float, 1.0f)
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BatchNorm)

    /**
     * @brief Continuously Differentiable Exponential Linear Uints:
     * Perform the linear uint element-wise on the input tensor X using formula:
     * max(0, x) + min(0, alpha * (exp(x/alpha) - 1)).
     * @par Inputs:
     * x: A ND tensor. Support 1D~8D. Must be one of the following types: float16, float32.
     * @par Attributes:
     * @li alpha1: An optional float32. Defines at which negative value the ELU saturates. Defaults to "1.0".
     * @li alpha2: An optional float32. Defines at which negative value the ELU saturates. Defaults to "1.0".
     * @li alpha3: An optional float32. Defines at which positive value the ELU saturates. Defaults to "1.0".
     * if x >= 0: y = alpha3 * 3 else: y = alpha1 * (exp(x/alpha2)-1)
     * @par Outputs:
     * y: A float16, float32, for the normalized result.
     * Has the same type, shape and format as input x.
     * @par Third-party framework compatibility
     * @li Compatible with ONNX's Celu operator
     */
    REG_OP(Celu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(alpha1, Float, 1.0)
    .ATTR(alpha2, Float, 1.0)
    .ATTR(alpha3, Float, 1.0)
    .OP_END_FACTORY_REG(Celu)

    /**
     * @brief Computes a 2D deformable convolution given 4D "x", "filter" and
     *  "offsets" tensors.
     * @par Inputs:
     * @li x: A 4D tensor of input image. With the format "NCHW", the data is
     * stored in the order of: [batch, in_channels, in_height, in_width].
     * @li filter: A 4D tensor of learnable filters. Must have the same type as
     * "x". With the format "NCHW" , the data is stored in the order of:
     * [out_channels, in_channels / groups, filter_height, filter_width].
     * @li offsets: A 4D tensor of x-y coordinates offset and mask. With the format
     * "NCHW", the data is stored in the order of: [batch, deformable_groups *
     * filter_height * filter_width * 3, out_height, out_width].
     * @li bias: An optional 1D tensor of additive biases to the filter outputs.
     *  The data is stored in the order of: [out_channels].
     * \n
     * \n
     *  The following are the supported data types and data formats:
     * \n
     * \n
     |  Tensor    | x       | filter  | offsets | bias    | y       |\n
        |  :-------: | :-----: | :-----: | :-----: | :-----: | :-----: |\n
        |  Data Type | float16 | float16 | float16 | float16 | float16 |\n
        |            | float32 | float32 | float32 | float32 | float32 |\n
        |  Format    | NCHW    | NCHW    | NCHW    | ND      | NCHW    |\n
        * \n
        *  For float32 type, the actual convolution calculation part on the chip is
        *  based on float16.
        * \n
        *
        * @par Attributes:
        * @li strides: Required. A list of 4 integers. The stride of the sliding
        * window for each dimension of input. The dimension order is interpreted
        * according to the data format of "x". The N and C dimensions must be
        * set to 1.
        * @li pads: Required. A list of 4 integers. The number of pixels to add to
        * each (top, bottom, left, right) side of the input.
        * @li dilations: Optional. A list of 4 integers. The dilation factor for each
        * dimension of input. The dimension order is interpreted according to the
        * data format of "x". The N and C dimensions must be set to 1. Defaults to
        * [1, 1, 1, 1].
        * @li groups: Optional. An integer of type int32. The number of blocked
        * connections from input channels to output channels. In_channels and
        * out_channels must both be divisible by "groups". Defaults to 1.
        * @li data_format: Reserved.
        * @li deformable_groups: Optional. An integer of type int32. The number of
        * deformable group partitions. In_channels must be divisible by
        * "deformable_groups". Defaults to 1.
        * @li modulated: Optional. Specify version of DeformableConv2D, true means v2,
        * false means v1, currently only support v2.
        * \n
        * \n
        *  The following value range restrictions must be met:
        * \n
        * \n
        |  Name             | Field    | Scope                       |\n
        |  :--------------: | :------: | :-------------------------: |\n
        |  Input Image Size | H        | [1, 100000 / filter_height] |\n
        |                   | W        | [1, 4096 / filter_width]    |\n
        |  Filter Size      | H        | [1, 63]                     |\n
        |                   | W        | [1, 63]                     |\n
        |  Strides          | H        | [1, 63]                     |\n
        |                   | W        | [1, 63]                     |\n
        |  Pads             | Top      | [0, 255]                    |\n
        |                   | Bottom   | [0, 255]                    |\n
        |                   | Left     | [0, 255]                    |\n
        |                   | Right    | [0, 255]                    |\n
        |  Dilations        | H        | [1, 255]                    |\n
        |                   | W        | [1, 255]                    |\n
        * \n
        *
        * @par Outputs:
        *  y:  A 4D Tensor of output feature map. Has the same type as "x". With the
        *  format "NCHW", the data is stored in the order of: [batch, out_channels,
        *  out_height, out_width].
        * \n
        *      out_height = (in_height + pad_top + pad_bottom -
        *                    (dilation_h * (filter_height - 1) + 1))
        *                   / stride_h + 1
        * \n
        *      out_width = (in_width + pad_left + pad_right -
        *                   (dilation_w * (filter_width - 1) + 1))
        *                  / stride_w + 1
        * \n
        *
        * @par Quantization supported or not
        * @li No
        *
        * @par Third-party framework compatibility
        */
    REG_OP(DeformableConv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(offsets, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(deformable_groups, Int, 1)
    .ATTR(modulated, Bool, true)
    .OP_END_FACTORY_REG(DeformableConv2D)

    /**
     * @brief Computes GlobalLpPool, GlobalLpPool consumes an input tensor X and applies lp pool pooling across the
     * values in the same channel.
     * @par Inputs:
     * x: A 4D or 5D Tensor of type float16 or float32, with format ND. \n
     * @par Attributes:
     * @li p: p value of the Lp norm used to pool over the input data. Must be one of the following types: float32.
     * Defaults to 2.0. \n
     * @par Outputs:
     * y: A 4D or 5D Tensor. Has the same type and format as "x".
     * When x is a 4D Tensor, the shape of y is [x.shape[0],x.shape[1],1,1].
     * When x is a 5D Tensor, the shape of y is [x.shape[0],x.shape[1],1,1,1].
     * @par Third-party framework compatibility
     * Compatible with the onnx operator GlobalLpPool.
     * @par Restrictions:
     * Warning: THIS FUNCTION IS DEPRECATED.
     * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
     */
    REG_OP(GlobalLpPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(p, Float, 2.0)
    .OP_END_FACTORY_REG(GlobalLpPool)

    /**
     *@brief GroupNorm and Reul operator \n
     *  calculating: x, gamma, beta \n
     *  y = relu(gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta)
     * @par Inputs:
     * Three inputs, including:
     * @li x: A Tensor. Must be one of the following types: float16, float32.
     * @li gamma: A Tensor. Must be one of the following types: float16, float32.
     * @li beta: A Tensor. Must be one of the following types: float16, float32 . \n
     * @par Attributes:
     * @li num_groups: A require attribute, the type is int32.
     * @li eps: A optional attribute, the type is float32. Defaults to 0.00001. \n
     * @par Outputs:
     * One outputs, including:
     * @li y: A Tensor. Must be one of the following types: float16, float32.
     * @par Restrictions:
     * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use/
     */
    REG_OP(GroupNormRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(eps, Float, 0.00001f)
    .OP_END_FACTORY_REG(GroupNormRelu)

    /**
     * @brief Common GRU calculation.
     * @par Inputs:
     * Eight inputs, including:
     * @li x: The input sequences packed (and pontentially padded) into on 3D Tesnor(float16).
     * @li w: The weight tensor for the gates is 3D Tensor(float16).
     * @li r: The recurrence weight tesnor is 3D Tensor(float16).
     * @li b: The bias tensor for the gates. The format must be ND
     * @li sequence_lens: Optional tensor specifying lengths of sequences(int32). The format must be ND
     * @li init_h: Optional initial value of the hidden(float16,float32).
     * @par Attributes:
     * @li activation_alpha: Optional scaling values used by some activation functions.  \n
     * @li activation_beta: Optional scaling values used by some activation functions.  \n
     * @li activations: A list of 2 (or 4 if bidirectional) activation functions for update, reset, and hidden gates. \n
     * @li clip: Cell clip threshold. \n
     * @li direction: Specify if the RNN is forward, reverse, or bidirectional. \n
     * @li hidden_size: Number of neurons in the hidden layer. \n
     * @li linear_before_reset: When computing the output of the hidden gate, apply the linear transformation before
     * multiplying by the output of the reset gate. \n
     * @par Outputs:
     * @li y: A Tensor that concats all the intermediate output values of the hidden(float16,float32).
     * @li y_h: The last output value of the hidden(float16,float32).
     */
    REG_OP(CommonGRU)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(r, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(sequence_lens, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(initial_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(activation_alpha, ListFloat, {})
    .ATTR(activation_beta, ListFloat, {})
    .ATTR(activations, ListString, {})
    .ATTR(clip, Float, -1.0)
    .ATTR(direction, String, "forward")
    .REQUIRED_ATTR(hidden_size, Int)
    .ATTR(linear_before_reset, Int, 0)
    .OP_END_FACTORY_REG(CommonGRU)

    /**
     *@brief Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0
     *otherwise The input does not need to explicitly be a 2D vector.The "axis" attribute indicates the dimension along
     *which Hardmax will be performed.The output tensor has the same shape and contains the Hardmax values of the
     *corresponding input.
     *
     *@par Inputs:
     *one input including:
     *x: input A ND Tensor.Must be one of the following types:float32,float16
     *
     *@par Attributes:
     *axis:A required int attribute that decides which dimension will be used to cal the hard_max
     *
     *@par Outputs:
     *one output including:
     *y:A ND Tensor of the same dtype as x
     *
     */
    REG_OP(HardMax)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(HardMax)

    /**
     * @brief Select one of the subgraphs to pass the input tensors and return the output tensors.
     *       If "cond" means True, the selected subgraph is "then_branch".
     *       Otherwise, the selected subgraph is "else_branch" . \n
     * @par Inputs:
     * @li cond: A Tensor. If "cond" is not a scalar of boolean type,
     *          it will be converted to a boolean according to the following rule:
     *          if "cond" is a numerical scalar, non-zero means True and zero means False;
     *          if "cond" is a string scalar, non-empty means True and empty means False;
     *          if "cond" is not a scalar, non-empty means True and empty means False.
     * @li input: The input tensors . It's a dynamic input. \n
     * @par Graphs:
     * @li then_branch: A subgraph takes 'input' and returns a list of tensors,
     *                 whose types are the same as what else_branch returns.
     * @li else_branch: A subgraph takes 'input' and returns a list of tensors,
     *                 whose types are the same as what then_branch returns . \n
     * @par Outputs:
     * output: The output tensors returned by either then_branch(input) or else_branch(input).
     *        It's a dynamic output. \n
     * @par Third-party framework compatibility
     * Compatible with the TensorFlow operator If.
     */
    REG_OP(If)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(If)

    /**
     * @brief InstanceNorm operator interface implementation.
     * @par Inputs
     * Three inputs, including:
     * @li x: A 4D or 5D Tensor. Support dtype: [float32, float16],
     *  support format: [NCHW, NHWC, NDHWC, NCDHW].
     * @li gamma: A 4D or 5D Tensor. Support dtype: [float32, float16],
     *  support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1).
     * @li beta: A 4D or 5D Tensor. Support dtype: [float32, float16],
     *  support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1).
     * @par Attributes
     * @li data_format: An optional attribute. The type is string. Default to "NDHWC".
     * @li epsilon: An optional attribute. The type is float. Default to 1e-6.
     * @par Outputs
     * Three outputs, including:
     * @li y: A 4D or 5D Tensor. Support dtype: [float32, float16],
     *  support format: [NCHW, NHWC, NDHWC, NCDHW]. Has the same type as "x".
     * @li mean: A 4D or 5D Tensor. Support dtype: [float32, float16],
     *  support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1). Has the same type as "x".
     * @li variance: A 4D or 5D Tensor. Support dtype: [float32, float16],
     *  support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1). Has the same type as "x".
     */
    REG_OP(InstanceNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(data_format, String, "NDHWC")
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(InstanceNorm)

    /**
     *@brief Computes log softmax activations .
     *@par Inputs:
     *One input:
     * logits: A ND tensor. Must be one of the following data types: double, bfloat16, float16, float32 . \n
     *@par Attributes:
     * axes: An optional list of ints. Multi-axis reduction is supported. Defaults to "{-1}" .
     * In Ascend 950 AI Processor, only single-axis reduction is supported. \n
     *@par Outputs:
     * logsoftmax: A ND tensor. Has the same data type as "logits" . \n
     *@par Third-party framework compatibility
     *Compatible with the TensorFlow operator LogSoftmax.
     */
    REG_OP(LogSoftmaxV2)
    .INPUT(logits, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(logsoftmax, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmaxV2)

    /**
     *@brief Local Response Normalization .
     *@par Inputs:
     *One input, including:
     *x: A Tensor. Must be 4-D shape, and only support the following types: float16, float32 . \n
     *@par Attributes:
     *@li depth_radius: An optional int32, specifying the half-width of the normalization window. Defaults to "5".
     * under the caffe framework, if local_size is provided and is an odd number,
     * depth_radius = (local_size - 1) / 2. local_size is the number of channels to sum over (for ACROSS_CHANNELS)
     * or the side length of the square region to sum over (for WITHIN_CHANNEL).
     *@li bias: An optional float32. An offset, usually > 0 to avoid dividing by 0.
     * Defaults to "1.0".
     *@li alpha: An optional float32. A scaling factor, usually positive.
     * Defaults to "1.0".
     *@li beta: An optional float32. An exponent. Defaults to "0.75" for the caffe framework, Defaults to "0.5" for
     * others.
     *@li norm_region: An optional string. A mode option. "ACROSS_CHANNELS":0. Defaults to "ACROSS_CHANNELS" . \n
     *@par Outputs:
     *y: A Tensor. Has the same data type and shape as "x" . \n
     * @attention Constraints:
     * This operator will be deprecated in the future. Replace it with LayerNorm operator. \n
     *@par Third-party framework compatibility:
     * Compatible with the TensorFlow operator LRN.
     */
    REG_OP(LRN)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(depth_radius, Int, 5)
    .ATTR(bias, Float, 1.0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.5)
    .ATTR(norm_region, String, "ACROSS_CHANNELS")
    .OP_END_FACTORY_REG(LRN)

    /**
     * @brief:LSTMP calculation
     * @par Inputs:
     * eight inputs:
     * @li x:A required Tensor(seq, batch, dim). Must be one of the following types: float16, float32.
     * @li real_mask:A optional Tensor(seq, batch). Must be one of the following types: float16, float32.
     * @li init_h:A optional Tensor(batch, state). Must be one of the following types: float16, float32.
     * @li init_c:A optional Tensor(batch, hidden). Must be one of the following types: float16, float32.
     * @li wx:A required Tensor(4*hidden, dim). Must be one of the following types: float16, float32.
     * @li wr:A required Tensor(4*hidden, state). Must be one of the following types: float16, float32.
     * @li bias:A optional Tensor(hidden). Must be one of the following types: float16, float32. The format must be ND.
     * @li project: A optional Tensor. Must be one of the following types: float16, float32.
     *
     * @par Outputs:
     * three outputs:
     * @li y:A Tensor. Must be one of the following types: float16, float32.
     * @li output_h:A Tensor. Must be one of the following types: float16, float32.
     * @li output_c:A Tensor. Must be one of the following types: float16, float32.
     *
     *@par Attributes:
     * time_major:An bool identifying the time major in the op. Default to false.
     * @par Restrictions:
     * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
     */
    REG_OP(LSTMP)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(wx, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(wr, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(project, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(real_mask, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(init_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(time_major, Bool, false)
    .OP_END_FACTORY_REG(LSTMP)

    /**
     * @brief CommonLSTM calculation.
     * @par Inputs:
     * eight inputs: \n
     * @li x:Each time step is a 4D Tensor. Must be one of the following types: float16, float32.
     * @li w:Each direction is a 4D Tensor. Must be one of the following types: float16, float32.
     * @li r:Each direction is a 4D Tensor. Must be one of the following types: float16, float32.
     * @li b:An optional input. Each direction is a 1D Tensor. Must be one of the following types: float16, float32. The
     * format must be ND.
     * @li sequence_lens:An optional input. A 1D Tensor.Must be one of the following types: int32. The format must be
     * ND.
     * @li initial_h:An optional input. Each direction is a 4D Tensor. Must be one of the following types: float16,
     * float32.
     * @li initial_c:An optional input. Each direction is a 4D Tensor. Must be one of the following types: float16,
     * float32.
     * @li p:An optional input. Each direction is a 1D Tensor.Must be one of the following types: float16, float32. The
     * format must be ND.
     * @par Attributes:
     * @li activation_alpha:Optional scaling values used by some activation functions. Empty is currently supported.
     * @li activation_beta:Optional scaling values used by some activation functions. Empty is currently supported.
     * @li activations:The list of activation functions. Empty is currently supported.
     * @li clip:An float identifying the cell clip in the op. Default to -1.
     * @li direction:Specify if the RNN is forward, reverse, or bidirectional. Must be one of forward(default), reverse,
     * or bidirectional.
     * @li hidden_size:Number of neurons in the hidden layer. Reserved.
     * @li input_forget:Couple the input and forget gates if 1. Reserved.
     * @par Outputs:
     * three outputs: \n
     * @li y:First dimension is time step, second dimension is direction, others is a 4D Tensor. Must be one of the
     * following types: float16, float32.
     * @li y_h:Each direction is a 4D Tensor. Must be one of the following types: float16, float32.
     * @li y_c:Each direction is a 4D Tensor. Must be one of the following types: float16, float32.
     */
    REG_OP(CommonLSTM)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(r, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(sequence_lens, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(initial_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(initial_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(p, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(activation_alpha, ListFloat, {})
    .ATTR(activation_beta, ListFloat, {})
    .ATTR(activations, ListString, {})
    .ATTR(clip, Float, -1.0)
    .ATTR(direction, String, "forward")
    .REQUIRED_ATTR(hidden_size, Int)
    .ATTR(input_forget, Int, 0)
    .OP_END_FACTORY_REG(CommonLSTM)

    /**
     * @brief Multiplies matrix "a" by matrix "b", producing "a @ b" .
     * @par Inputs:
     * Four inputs, including:
     * @li x1: A matrix Tensor. Must be one of the following types: float16,
     * float32, int32, int8, int4, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW].
     * @li x2: A matrix Tensor. Must be one of the following types: float16,
     * float32, int32, int8, int4, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW].
     * @li bias: A optional Tensor. Must be one of the following types:
     * float16, float32, int32, bfloat16. Has format [ND, NHWC, NCHW].
     * @li offset_w: A optional Tensor. Must be one of the following types:
     * int8, int4. Has format [ND, NHWC, NCHW].
     * @par Attributes:
     * @li adj_x1: A bool. If True, changes the shape of "x1" from [B, M, K] to
     * [B, K, M] before multiplication.
     * @li adj_x2: A bool. If True, changes the shape of "x2" from [B, K, N] to
     * [B, N, K] before multiplication.
     * @li offset_x: An optional integer for quantized BatchMatMulV2.
     * @par Outputs:
     * y: The result matrix Tensor. Must be one of the following types: float16,
     * float32, int32, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC]. Has the same shape
     * length as "x1" and "x2".
     * @attention Constraints:
     * if performances better in format NZ, please close
     * "MatmulTransdataFusionPass" in fusion configuration.
     * @par Third-party framework compatibility
     * Compatible with the TensorFlow operator BatchMatmul.
     */
    REG_OP(BatchMatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(BatchMatMulV2)

    /**
     *@brief Normalizes the input .
     *@par Inputs:
     * One input:
     *x: An NCHW tensor of type float16 or float32 . \n
     *@par Attributes:
     *@li eps: An optional float32 epsilon for not dividing by zero. Defaults to "1e-9" . \n
     *@li axes: A list of Intefers, along which axis to reduce. Defaults to "[0, 2, 3]" . \n
     *@par Outputs:
     *y: An NCHW tensor of type float16 or float32 . \n
     *@attention Constraints:
     * The input tensor must have the NCHW format, whose shape length must be 4.
     *@par Third-party framework compatibility
     * Compatible with the ONNX operator MeanVarianceNormalization.
     */
    REG_OP(MVNV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))  /* "First operand." */
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16})) /* "Result, has same element type as inputs" */
    .ATTR(eps, Float, 1e-9f)
    .ATTR(axes, ListInt, {0, 2, 3})
    .OP_END_FACTORY_REG(MVNV2)

    /**
    * @brief Sum the alpha according to the offset and ksize,
        and quadrature it with the sigmoid value of energy.
    * @par Inputs:
    * Three inputs, including:
    * @li alpha: A Tensor. Must be one of the following types: float32, float16.
    * @li energy: A Tensor. Must be one of the following types: float32, float16.
    * @li offset: A Tensor of type int32. \n
    * @par Outputs:
    * y: A Tensor with same type as "alpha". \n
    *
    * @par Attributes:
    * ksize: A int.
    *
    * @par Restrictions:
    * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
    */
    REG_OP(MovingSumWithSigmoid)
    .INPUT(alpha, TensorType::BasicType())
    .INPUT(energy, TensorType::BasicType())
    .INPUT(offset, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(ksize, Int)
    .OP_END_FACTORY_REG(MovingSumWithSigmoid)

    /**
     *@brief Generate the responsible flags of anchor in a single feature map.
     *@par Inputs:
     * gt_bboxes: Ground truth box, 2-D Tensor of type float32 with shape `[batch, 4]`.
     *@par Attributes:
     *@li featmap_size: The size of feature maps. It is a listint and size is 2.
     *@li strides: Stride of current level, listint.
     *@li num_base_anchors: The number of base anchors.
     *@par Outputs:
     * flags: The valid flags of each anchor in a single level, 1-D Tensor of type uint8.
     */
    REG_OP(AnchorResponseFlags)
    .INPUT(gt_bboxes, TensorType({DT_FLOAT}))
    .OUTPUT(flags, TensorType({DT_UINT8}))
    .REQUIRED_ATTR(featmap_size, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(num_base_anchors, Int)
    .OP_END_FACTORY_REG(AnchorResponseFlags)

    /**
    * @brief Anti quantizes the input .
    * @par Inputs:
    * @li x: A multi-dimensional tensor of type int8/int4, specifying the input.
        The maximum dimension should not exceed 8 dimensions. Format support ND.
    * @li scale: A 1-D tensor of type float32/bfloat16, specifying the scale.
        Shape is (n,), where n can be 1. If n is not 1, it must be the same as
        the size of last dimension of x. Format support ND.
    * @li offset: A optional 1-D tensor of type float32/bfloat16, specifying the offset.
        The shape and dtype of offset should be same to scale. Format support ND.
    * @par Attributes:
    * @li dst_type: A optional int32, specifying the output data type. Defaults to "DT_FLOAT16".
    * @li sqrt_mode: A optional bool, specifying whether to perform square root on "scale", either "True" or "False".
    * Defaults to "False" . \n
    * @par Outputs:
    * y: The dequantized output tensor of type float16 or bfloat16. \n
    */
    REG_OP(AscendAntiQuantV2)
    .INPUT(x, TensorType({DT_INT8, DT_INT4}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BFLOAT16}))
    .ATTR(dst_type, Int, DT_FLOAT16)
    .ATTR(sqrt_mode, Bool, false)
    .OP_END_FACTORY_REG(AscendAntiQuantV2)

    /**
     * @brief Compute the GeGluV2,
     * where the activations function in GLU is Gelu.
     * @par Inputs:
     * x: A Tensor. Must be one of the following types: bfloat16, float16, float32.
     * Shape supports at least 1 dimensions, and at most 8 dimensions.
     * The length of the split dimension in x must be an even number.
     * @par Outputs:
     * Two outputs, including:
     * @li y: A Tensor. Must be one of the following types: bfloat16, float16, float32.
     * The dtype of y must exactly same with input x.
     * The shape of y matches the shape of x in all dimensions except for the split dimension,
     * where its length is half of length of x's split dimension.
     * @li gelu: A Tensor. Must be one of the following types: bfloat16, float16, float32.
     * The dtype of gelu must exactly same with input x.
     * The shape of gelu matches the shape of x in all dimensions except for the split dimension,
     * where its length is half of length of x's split dimension.
     * @par Attributes:
     * Three attributes, including:
     * @li dim: An optional int. The dimension to be split, default is -1.
     * @li approximate: An optional int. Which formula used for the activation computation.
     * The gelu approximation algorithm to use: 'none'(0) or 'tanh'(1), default is 'tanh'(1).
     * Atlas Inference Series Product only support 'tanh'(1).
     * @li activate_left: An optional bool.
     * The gelu activate_left algorithm to use:
     *     'false'(activate right) or 'true'(activate left), defalut is 'false'(activate right).
     */
    REG_OP(GeGluV2)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .OUTPUT(gelu, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(dim, Int, -1)
    .ATTR(approximate, Int, 1)
    .ATTR(activate_left, Bool, false)
    .OP_END_FACTORY_REG(GeGluV2)

    /**
    *@brief Returns the size of a tensor, that is, an integer of the number of elements of the tensor. \n
    *@par Inputs:
    *x: A tensor. Must be one of the following types: float32、float16、int8、
    int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string. \n
    *@par Attributes:
    *dtype: An optional int32 or int64. The output data type. Defaults to "int32". \n
    *@par Outputs:
    *y: A tensor. The size of the input tensor. \n
    *@par Third-party framework compatibility
    *Compatible with the TensorFlow operator Size.
    */
    REG_OP(Size)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Size)

    /**
    * @brief BasicLSTMInplaceFillWindowCache calculation.
    * @par Inputs:
    * eight inputs: \n
    * @li x:Each time step is a 3D Tensor. Must be one of the following types: float16.
    * @li w:Each direction is a 3D Tensor. Must be one of the following types: int8.
    * @li r:Each direction is a 3D Tensor. Must be one of the following types: int8.
    * @li h:Each direction is a 3D Tensor. Must be one of the following types: float16.
    * @li c:Each direction is a 3D Tensor. Must be one of the following types: float16.
    * @li b:An optional input. Each direction is a 2D Tensor. Must be one of the following types: int32.
    * @li sequence_lens:An optional input. A 1D Tensor. Must be one of the following types: int32.
    * @li clean_cache:An optional input. A 1D Tensor. Must be one of the following types: int32. clean_cache=None
    behaves the same as clean_cache=2.
    * @li deq_scale:A 1D Tensor. Must be one of the following types: uint64.

    * @par Attributes:
    * @li hidden_size:Number of neurons in the hidden layer. Requied. Reserved.
    * @li activation_alpha: Optional scaling values used by some activation functions. Empty is currently supported.
    * @li activation_beta: Optional scaling values used by some activation functions. Empty is currently supported.
    * @li activations: A list of strings of activation functions. Empty is currently supported.
    * @li clip:An float identifying the cell clip in the op. Default to -1.
    * @li direction: Specify if the RNN is forward, reverse, or bidirectional. Must be forward(default).
    * @li input_forget:Couple the input and forget gates if 1. Reserved.
    * @li quant_scale_x: A float identifying the quant_scale of x_tensor. Default to -0.0.
    * @li quant_offset_x:A float identifying the quant_offset of x_tensor. Default to -0.0.
    * @li quant_sqrt_mode_x:A sqrt_mode of x_tensor. Default to False.
    * @li quant_scale_h:A float identifying the quant_scale of h_tensor. Default to -0.0.
    * @li quant_offset_h:A float identifying the quant_offset of h_tensor. Default to -0.0.
    * @li quant_sqrt_mode_h:A sqrt_mode of h_tensor. Default to False.
    * @li quant_dtype:An Int number identifying the dtype of quant. Default to 2(DT_INT8).

    * @par Outputs:
    * three outputs: \n
    * @li y:First dimension is time step, second dimension is direction, others is a 4D Tensor. Must be one of the
    following types: float16.
    * @li y_h:Each direction is a 3D Tensor. Must be one of the following types: float16.
    * @li y_c:Each direction is a 3D Tensor. Must be one of the following types: float16.
    */

    REG_OP(BasicLSTMInplaceFillWindowCache)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(w, TensorType({DT_INT8}))
    .INPUT(r, TensorType({DT_INT8}))
    .INPUT(h, TensorType({DT_FLOAT16}))
    .INPUT(c, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(b, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(sequence_lens, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(clean_cache, TensorType({DT_INT32}))
    .INPUT(deq_scale, TensorType({DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OUTPUT(h, TensorType({DT_FLOAT16}))
    .OUTPUT(c, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(hidden_size, Int)
    .ATTR(activation_alpha, ListFloat, {})
    .ATTR(activation_beta, ListFloat, {})
    .ATTR(activations, ListString, {})
    .ATTR(clip, Float, -1.0)
    .ATTR(direction, String, "forward")
    .ATTR(input_forget, Int, 0)
    .ATTR(quant_scale_x, Float, 0.0)
    .ATTR(quant_offset_x, Float, 0.0)
    .ATTR(quant_sqrt_mode_x, Bool, false)
    .ATTR(quant_scale_h, Float, 0.0)
    .ATTR(quant_offset_h, Float, 0.0)
    .ATTR(quant_sqrt_mode_h, Bool, false)
    .ATTR(quant_dtype, Int, DT_INT8)
    .OP_END_FACTORY_REG(BasicLSTMInplaceFillWindowCache)

    /*
    * @brief BasicGRUInplaceFillWindowCache calculation.
    * @par Inputs:
    * eight inputs: \n
    * @li x:Each time step is a 3D Tensor. Must be one of the following types: float16.
    * @li w:Each direction is a 3D Tensor. Must be one of the following types: int8.
    * @li r:Each direction is a 3D Tensor. Must be one of the following types: int8.
    * @li h:Each direction is a 3D Tensor. Must be one of the following types: float16.
    * @li b:An optional input. Each direction is a 2D Tensor. Must be one of the following types: int32.
    * @li sequence_lens:An optional input. A 1D Tensor. Must be one of the following types: int32.
    * @li clean_cache:An optional input. A 1D Tensor. Must be one of the following types: int32.
    * @li deq_scale:A 1D Tensor. Must be one of the following types: uint64.

    * @par Attributes:
    * @li hidden_size:Number of neurons in the hidden layer. Requied. Reserved.
    * @li activation_alpha: Optional scaling values used by some activation functions. Empty is currently supported.
    * @li activation_beta: Optional scaling values used by some activation functions. Empty is currently supported.
    * @li activations: A list of strings of activation functions. Empty is currently supported.
    * @li clip:An float identifying the cell clip in the op. Default to -1.
    * @li direction: Specify if the RNN is forward, reverse, or bidirectional. Must be forward(default).
    * @li linear_before_reset: Apply the linear transformation before multiplying by the output of the reset gate.
    Default to 1(Int).
    * @li quant_scale_x: A float identifying the quant_scale of x_tensor. Default to -0.0.
    * @li quant_offset_x:A float identifying the quant_offset of x_tensor. Default to -0.0.
    * @li quant_sqrt_mode_x:A sqrt_mode of x_tensor. Default to False.
    * @li quant_scale_h:A float identifying the quant_scale of h_tensor. Default to -0.0.
    * @li quant_offset_h:A float identifying the quant_offset of h_tensor. Default to -0.0.
    * @li quant_sqrt_mode_h:A sqrt_mode of h_tensor. Default to False.
    * @li quant_dtype:An Int number identifying the dtype of quant. Default to 2(DT_INT8).

    * @par Outputs:
    * two outputs: \n
    * @li y:First dimension is time step, second dimension is direction, others is a 4D Tensor. Must be one of the
    following types: float16.
    * @li y_h:Each direction is a 3D Tensor. Must be one of the following types: float16.
    */

    REG_OP(BasicGRUInplaceFillWindowCache)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(w, TensorType({DT_INT8}))
    .INPUT(r, TensorType({DT_INT8}))
    .INPUT(h, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(b, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(sequence_lens, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(clean_cache, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(deq_scale, TensorType({DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OUTPUT(h, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(hidden_size, Int)
    .ATTR(activation_alpha, ListFloat, {})
    .ATTR(activation_beta, ListFloat, {})
    .ATTR(activations, ListString, {})
    .ATTR(clip, Float, -1.0)
    .ATTR(direction, String, "forward")
    .ATTR(linear_before_reset, Int, 1)
    .ATTR(quant_scale_x, Float, 0.0)
    .ATTR(quant_offset_x, Float, 0.0)
    .ATTR(quant_sqrt_mode_x, Bool, false)
    .ATTR(quant_scale_h, Float, 0.0)
    .ATTR(quant_offset_h, Float, 0.0)
    .ATTR(quant_sqrt_mode_h, Bool, false)
    .ATTR(quant_dtype, Int, DT_INT8)
    .OP_END_FACTORY_REG(BasicGRUInplaceFillWindowCache)

    /**
    * @brief Sum X1 and X2 according to the offset recorded in seq_len1 and seq_len2. \n

    * @par Inputs:
    * Four inputs, including:
    * @li x1: A Tensor. Support BasicType.
    * @li x2: A Tensor. Support BasicType.
    * @li seq_len1: A Tensor. Support int32.
    * @li seq_len2: A Tensor. Support int32. \n

    * @par Outputs:
    * y: A Tensor with same type as "x1". \n

    * @par Restrictions:
    * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
    */
    REG_OP(DynSeqOuter)
    .INPUT(x1, TensorType::BasicType())
    .INPUT(x2, TensorType::BasicType())
    .INPUT(seq_len1, TensorType({DT_INT32}))
    .INPUT(seq_len2, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(DynSeqOuter)

    /**
    * @brief Finds values and indices of the "k" largest elements for the last
    * dimension . \n

    * @par Inputs:
    * Two inputs, including:
    * @li x: A 1D or higher tensor of type RealNumberType, with the last dimension
    * at least "k".
    * @li k: A 0D Tensor of type int32.
    * Number of top elements to look for along the last dimension (along each row
    * for matrices) . \n

    * @par Attributes:
    * @li sorted: An optional bool. Defaults to "True".
    * If "True", the returned "k" elements are themselves sorted.
    * If "False", the returned "k" elements are not sorted.
    * @li largest: An optional bool, controls whether to return largest or smallest elements. Defaults to true.
    * If "True", the "k" largest elements are returned in descending order.
    * If "False", the "k" smallest elements are returned in ascending order.
    * @li dim: An optional int. Default is -1. 0-D. Number of top elements to look for along the last dimension (along
    each row for matrices). \n

    * @par Outputs:
    * @li values: A Tensor, specifying the sorted data. Has the same type as
    * "x".
    * @li indices: A Tensor of type int32, specifying the indices of sorted data . \n

    * @see TopK()
    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator TopKV2.
    */
    REG_OP(TopK)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType({DT_INT32}))
    .ATTR(sorted, Bool, true)
    .ATTR(largest, Bool, true)
    .ATTR(dim, Int, -1)
    .OP_END_FACTORY_REG(TopK)

    /**
    * @brief Performs average pooling on the input.

    * @par Inputs:
    * x: A tensor of shape [N, C, H, W] or [N, H, W, C] which supports data type float16, float32, double.

    * @par Attributes:
    * @li ksize: A required ListInt, list of 4 ints, specifying the size (N, C, H, and W)
    * of the sliding window, where N = C = 1,
        * and H and W are positive integers within the range [1, 255].
    * @li strides: A required ListInt, list of 4 ints, specifying the stride of the
        * sliding window. The strides of the N and C dimensions are 1.
        * The strides of the H and W dimensions are positive integers within
        * the range [1, 63].
    * @li padding_mode: An optional String, specifying the padding algorithm,
        * either "VALID", "SAME" and "CALCULATED".
        * With "SAME" means that the outputs will have the same spatial dimensions
        * as its inputs. With "VALID" means no padding.
    * @li pads: A optional ListInt. Pad value when padding_mode is "CALCULATED".
    * @li data_format: An optional String, specifying the data format of "ksize"
        * and "strides", either "NHWC", or "NCHW" (default).
    * @li global_pooling: An optional Bool. Global or not. If true, pads will change to {0,0,0,0}
    * and ksize will change to [input_h, input_w].
    * @li ceil_mode: An optional Bool. Use ceil or floor to calculate the output size when
    * padding_mode is "CALCULATED".
    * @li exclusive: An optional Bool. Ignore padding area or not when calculating average.
    * @li divisor_override: An optional Int, its valid range is [1, 255], and the default value is zero.
    * if specified, it will be used as divisor, otherwise size of the pooling region will be used.

    * @par Outputs:
    * y: The average pooled output tensor. Has the same type and format as
    * input "x".

    * @attention Constraints:
    * @li Only single input and single output are supported.
    * @li Global pooling is supported.
    * @li "ksize_H" and "ksize_W" are positive integers within the range [1, 255].
    * ksize_H * ksize_W < 256
    * @li Due to instruction restrictions,
        * the values of "strides_h" and "strides_w" are positive integers within
        * the range [1, 63].
    * @li If the sliding window range exceeds the original width and height of the input feature map,
        * and the calculation result of count_include_pad is False, the behavior of dividing by 0 will appear.
        * This scenario does not conform to the normal logic of the operator.
        * It is recommended to modify attributes such as ceil_mode or stride to satisfy that the sliding window
        * always has an intersection with the input feature map. In this abnormal scenario,
        * different chips may return different results, and four abnormal results may appear: 0, 65504, Nan, and INF.
    * @li When the C axis is greater than 1, if points with the same H and W dimensions in x contain one INF input
        * on the C axis, the output of the INF input covered by the sliding window on this C axis is INF, and the
        * outputs of other C axis without INF input covered by the sliding window are Nan. If points with the same
        * H and W dimensions in x contain more than one INF input on the C axis, the outputs of all INF input data
        * covered by the sliding window on the C axis are Nan.
    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator AvgPoolV2.
    */
    REG_OP(AvgPoolV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling, Bool, false)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(exclusive, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .OP_END_FACTORY_REG(AvgPoolV2)

    /**
    * @brief Performs average pooling on the input.

    * @par Inputs:
    * x: A tensor of shape [N, C, H, W] or [N, H, W, C] which supports data type float16, float32, bfloat16, double. \n

    * @par Attributes:
    * @li ksize: A required list of 4 ints, specifying the size of the sliding window,
    * The ksize of the N and C dimensions are 1.
    * @li strides: A required list of 4 ints, specifying the stride of the
    * sliding window. The strides of the N and C dimensions are 1.
    * @li padding: A required string, specifying the padding algorithm,
        * either "VALID" or "SAME". With "SAME" means that the outputs will have the
        * same spatial dimensions as its inputs. With "VALID" means no padding.
    * @li data_format: An optional string, specifying the data format of "ksize"
    * and "strides", either "NCHW", or "NHWC" (default). \n

    * @par Outputs:
    * y: The average pooled output tensor. Has the same type and format
    * as input "x". \n

    * @attention Constraints:
    * @li This operator applies only to a TensorFlow network.
    * @li Only single input and single output are supported.
    * @li For Atlas Training Series Product, Atlas A2 Training Series Product/Atlas 800I A2 Inference Product,
    * Atlas A3 Training Series Product: "ksize_H" and "ksize_W" are positive integers within the range [1, 255].
    * ksize_H * ksize_W < 256. \n
    * For Ascend 950 AI Processor: The ksize of the H and W dimensions should be greater than 0.
    * @li For Atlas Training Series Product, Atlas A2 Training Series Product/Atlas 800I A2 Inference Product,
    * Atlas A3 Training Series Product: the values of "strides_h" and "strides_w" are positive integers within
    * the range [1, 63]. \n
    * For Ascend 950 AI Processor: The stride of the H and W dimensions should be greater than 0.
    * @li When the C axis is greater than 1, if points with the same H and W dimensions in x contain one INF input
    * on the C axis, the output of the INF input covered by the sliding window on this C axis is INF, and the
    * outputs of other C axis without INF input covered by the sliding window are Nan. If points with the same
    * H and W dimensions in x contain more than one INF input on the C axis, the outputs of all INF input data
    * covered by the sliding window on the C axis are Nan. this constraints not for Ascend 950 AI Processor.
    * @li The ouput "y" shape at the N and C dimensions should be equal with input "x" shape at same dimensions. The
    output
    * shape at the H and W dimensions is calculated by below formula: \n
    * @code{.c}
        when "padding_mode" is "SAME":
                    out_height = (in_height + stride_h - 1) / stride_h
                    out_width = (in_width + stride_w - 1) / stride_w
        when "padding_mode" is "VALID":
                    out_height = (in_height + stride_h - ksize_h) / stride_h
                    out_width = (in_width + stride_w - ksize_w) / stride_w
        It not support out_height < 0 or out_width < 0.
    * @endcode
        * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator AvgPool.
    */
    REG_OP(AvgPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(AvgPool)

    /**
    * @brief Computes a 2D convolution given 4D "x", "filter" and "bias" tensors.
    * Like this, output = CONV(x, filter) + bias.
    * @par Inputs:
    * @li x: A required 4D tensor of input image. With the format "NHWC" which shape is
    * [n, h, w, in_channels] or the format "NCHW" which shape is [n, in_channels, h, w].
    * @li filter: A required 4D tensor of convolution kernel.
    * With the format "HWCN" which shape is [kernel_h, kernel_w, in_channels / groups, out_channels]
    * or the format "NCHW" which shape is [out_channels, in_channels / groups, kernel_h, kernel_w].
    * @li bias: An optional 1D tensor of additive biases to the outputs.
    * The data is stored in the order of: [out_channels].
    * @li offset_w: An optional quantitative offset tensor. Reserved.
    *\n
    * The following are the supported data types and data formats (except IPV350 and Ascend 950 AI Processor):
    *\n
    | Tensor    | x        | filter   | bias     | y        |\n
    | :-------: | :------: | :------: | :------: | :------: |\n
    | Data Type | float16  | float16  | float16  | float16  |\n
    |           | float16  | float16  | float16  | float32  |\n
    |           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
    |           | bfloat16 | bfloat16 | bfloat16 | float32  |\n
    |           | float32  | float32  | float32  | float32  |\n
    |           | int8     | int8     | int32    | int32    |\n
    | Format    | NCHW     | NCHW     | ND       | NCHW     |\n
    |           | NHWC     | HWCN     | ND       | NHWC     |\n
    |           | NCHW     | HWCN     | ND       | NCHW     |\n
    *\n
    * The following are the supported data types and data formats for IPV350:
    *\n
    | Tensor    | x       | filter  | bias    | y       |\n
    | :-------: | :-----: | :-----: | :-----: | :-----: |\n
    | Data Type | int16   | int8    | int32   | int32   |\n
    |           | int8    | int8    | int32   | int32   |\n
    | Format    | NCHW    | NCHW    | ND      | NCHW    |\n
    |           | NHWC    | HWCN    | ND      | NHWC    |\n
    *\n
    * The following are the supported data types and data formats for Ascend 950 AI Processor:
    *\n
    | Tensor    | x        | filter   | bias     | y        |\n
    | :-------: | :------: | :------: | :------: | :------: |\n
    | Data Type | float16  | float16  | float16  | float16  |\n
    |           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
    |           | float32  | float32  | float32  | float32  |\n
    |           | hifloat8 | hifloat8 | float32  | hifloat8 |\n
    | Format    | NCHW     | NCHW     | ND       | NCHW     |\n
    |           | NHWC     | HWCN     | ND       | NHWC     |\n
    *\n
    * @par Attributes:
    * @li strides: Required. A list of 4 integers. The stride of the sliding window
    * for each dimension of input. The dimension order is determined by the data
    * format of "x". The n and in_channels dimensions must be set to 1.
    * When the format is "NHWC", its shape is [1, stride_h, stride_w, 1],
    * when the format is "NCHW", its shape is [1, 1, stride_h, stride_w].
    * @li pads: Required. A list of 4 integers. The number of pixels to add to each
    * (pad_top, pad_bottom, pad_left, pad_right) side of the input.
    * @li dilations: Optional. A list of 4 integers. The dilation factor for each
    * dimension of input. The dimension order is determined by the data format of
    * "x". The n and in_channels dimensions must be set to 1.
    * When the format is "NHWC", its shape is [1, dilation_h, dilation_w, 1],
    * when the format is "NCHW", its shape is [1, 1, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
    * @li groups: Optional. An integer of type int32. The number of groups
    * in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
    * @li data_format: Optional. It is a string represents input's data format.
    * Defaults to "NHWC". Reserved.
    * @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
    * and is used for filling in pad values. Ensure that the output is within the
    * effective range. Defaults to 0. Reserved.
    * @par Outputs:
    * y: A 4D tensor of output feature map.
    * With the format "NHWC" which shape is [n, out_height, out_width, out_channels]
    * or the format "NCHW" which shape is [n, out_channels, out_height, out_width].
    *\n
    *     out_height = (h + pad_top + pad_bottom -
    *                   (dilation_h * (kernel_h - 1) + 1))
    *                  / stride_h + 1
    *\n
    *     out_width = (w + pad_left + pad_right -
    *                  (dilation_w * (kernel_w - 1) + 1))
    *                 / stride_w + 1
    *\n
    * @attention Constraints:
    * @li The following value range restrictions must be met:
    *\n
    | Name             | Field      | Scope       |\n
    | :--------------: | :--------: | :---------: |\n
    | x size           | h          | [1, 100000] |\n
    |                  | w          | [1, 4096]   |\n
    | filter size      | kernel_h   | [1, 511]    |\n
    |                  | kernel_w   | [1, 511]    |\n
    | strides          | stride_h   | [1, 63]     |\n
    |                  | stride_w   | [1, 63]     |\n
    | pads             | pad_top    | [0, 255]    |\n
    |                  | pad_bottom | [0, 255]    |\n
    |                  | pad_left   | [0, 255]    |\n
    |                  | pad_right  | [0, 255]    |\n
    | dilations        | dilation_h | [1, 255]    |\n
    |                  | dilation_w | [1, 255]    |\n
    | offset_x         | -          | [-128, 127] |\n
    *\n
    * @li The w dimension of the input image supports cases exceeding 4096, but it may
    * cause compilation errors.
    *\n
    * @li If any dimension of x/filter/bias/offset_w/y shape exceeds max
    * int32(2147483647), the product of each dimension of x/filter/bias/offset_w/y
    * shape exceeds max int32(2147483647) or the value of strides/pads/dilations/offset_x
    * exceeds the range in the above table, the correctness of the operator cannot be guaranteed. \n
    * In Ascend 950 AI Processor: If any dimension of x/filter/bias/offset_w/y shape exceeds max
    * 1000000, the product of each dimension of x/filter/bias/offset_w/y
    * shape exceeds max int32(2147483647) or the value of strides/pads/dilations/offset_x
    * exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
    *\n
    * @li When the specifications of the Conv2D exceeds the constraints mentioned above,
    * a timeout AI Core error may be reported.
    *\n
    * @par Quantization supported or not
    * Yes
    *\n
    * @par Third-party framework compatibility
    * @li Compatible with the TensorFlow operator "conv2d".
    * @li Compatible with the Caffe operator 2D "Convolution".
    * @li Compatible with the ONNX operator 2D "Conv".
    * @li Compatible with the PyTorch operator "Conv2D".
    */
    REG_OP(Conv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2D)

    /**
    * @brief Computes a 3D convolution with 5D "x", "filter" and "bias" tensors.
    * Like this, output = CONV(x, filter) + bias.
    * @par Inputs:
        * @li x: A required 5D tensor of input image.
                The format of x is NCDHW or NDHWC.
                The data is stored in the order of: [n, in_channels, d, h, w] or [n, d, h, w, in_channels]. \n
                Any dimension of x shape must be in [1, 2147483646] except Ascend 950 AI Processor. \n
                In Ascend 950 AI Processor, any dimension of x shape must be in [1, 1000000].
        * @li filter: A required 5D tensor of convolution kernel.
                    Must have the same type as "x".
                    The format support NCDHW or DHWCN.
                    The data is stored in the order of:[out_channels, in_channels, kernel_d, kernel_h, kernel_w] or
                    [kernel_d, kernel_h, kernel_w, in_channels, out_channels]. \n
                    The value of kernel_h * kernel_w * kernel_k0 must be in [0, 65535],
                    kernel_k0 is determined by the data type, indicating the number of elements aligned to 32B. \n
                    The kernel_h and kernel_w dimensions must be in [1, 511]. \n
                    The other values of filter_size must be in [1, 2147483646]. \n
                    When format is DHWCN and type is float32,
                    filter should be a constants except Ascend 950 AI Processor. \n
                    In Ascend 950 AI Processor, the kernel_h and kernel_w dimensions must be in [1, 255],
                    And the other values of filter_size must be in [1, 1000000].
        * @li bias: An optional 1D tensor of additive biases to the outputs.
                    The data is stored in the order of: [out_channels].
                    "out_channels" must equals to the "out_channels" of output y. \n
                    In Ascend 950 AI Processor, the out_channels dimension must be in [1, 1000000]
        * @li offset_w: An optional quantitative offset tensor. Reserved.
    *\n
    *\n
    * The following are the supported data types and data formats for Ascend 950 AI Processor:
    *\n
    | Tensor    | x        | filter   | bias     |   y      |\n
    | :-------: | :------: | :------: | :------: | :------: |\n
    | Data Type | float16  | float16  | float16  | float16  |\n
    |           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
    |           | float32  | float32  | float32  | float32  |\n
    |           | hifloat8 | hifloat8 | float32  | hifloat8 |\n
    | Format    | NCDHW    | NCDHW    | ND       | NCDHW    |\n
    |           | NDHWC    | DHWCN    | ND       | NDHWC    |\n
    *\n
    * The following are the supported data types and data formats for other products:
    *\n
    | Tensor    | x        | filter   | bias     | y        |\n
    | :-------: | :------: | :------: | :------: | :------: |\n
    | Data Type | float16  | float16  | float16  | float16  |\n
    |           | bfloat16 | bfloat16 | float32  | bfloat16 |\n
    | Format    | NCDHW    | NCDHW    | ND       | NCDHW    |\n
    |           | NDHWC    | DHWCN    | ND       | NDHWC    |\n
    |           | NCDHW    | DHWCN    | ND       | NCDHW    |\n
    *\n
    * @par Attributes:
        * @li strides: Required. A list of 5 integers. Specifies the stride of the
                    sliding window for each dimension of "x". The dimension order is determined by the data format of
    "x". The n and in_channels dimensions must be 1. \n When the format is "NDHWC", its shape is [1, stride_d, stride_h,
    stride_w, 1], when the format is "NCDHW", its shape is [1, 1, stride_d, stride_h, stride_w]. \n The stride_h and
    stride_w dimensions must be in [1, 63]. The stride_d must be in [1, 2147483646] except Ascend 950 AI Processor.
    \n In Ascend 950 AI Processor the stride_d must be in [1, 1000000].
        * @li pads: Required. A list of 6 integers. Supports only padding along the d, h and w dimensions in sequence of
                    pad_head, pad_tail, pad_top, pad_bottom, pad_left and pad_right. \n
                    The pad_top, pad_bottom, pad_left and pad_right must be in [0, 255].
                    The pad_head and pad_tail must be in [0, 2147483646] except Ascend 950 AI Processor. \n
                    In Ascend 950 AI Processor the pad_head and pad_tail must be in [1, 1000000].
        * @li dilations: Optional. A list of 5 integers. Specifies the dilation
                        factor for each dimension of "x". The dimension order is determined by the data format of "x".
    \n When the format is "NDHWC", its shape is [1, dilation_d, dilation_h, dilation_w, 1], when the format is "NCDHW",
    its shape is [1, 1, dilation_d, dilation_h, dilation_w]. \n Default value is [1, 1, 1, 1, 1]. \n The dilation_h and
    dilation_w dimensions must be in [1, 255]. The dilation_d dimensions must be in [0, 2147483646] except Ascend 950 AI
    Processor. \n In Ascend 950 AI Processor the dilation_d dimensions must be in [1, 1000000].
        * @li groups: Optional. An integer of type int32. The number of groups
                    in group convolution. In_channels and out_channels must both be divisible by "groups".
                    The value of groups must be in [1, 65535]. Default value is 1.
        * @li data_format: Optional. It represents data format of the input x and output y, and is a string
                        dtype with "NCDHW" and "NDHWC". Defaults to "NDHWC".
        * @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
                        and is used for filling in pad values. Ensure that the output is within the
                        effective range. Defaults to 0.
    * @par Outputs:
    * y: A 5D tensor of output feature map. Has the same type as "x". With the format "NCDHW" or "NDHWC",
            the data is stored in the order of: [n, out_channels, out_depth, out_height, out_width] or
            [n, out_depth, out_height, out_width, out_channels]. \n
    *\n
    *     out_depth = (d + pad_head + pad_tail -
    *                  (dilation_d * (kernel_d - 1) + 1))
    *                 / stride_d + 1
    *\n
    *     out_height = (h + pad_top + pad_bottom -
    *                   (dilation_h * (kernel_h - 1) + 1))
    *                  / stride_h + 1
    *\n
    *     out_width = (w + pad_left + pad_right -
    *                  (dilation_w * (kernel_w - 1) + 1))
    *                 / stride_w + 1
    *\n
            Any dimension of y shape must be in [1, 2147483646] except Ascend 950 AI Processor. \n
            In Ascend 950 AI Processor, any dimension of y shape must be in [1, 1000000].
    * @attention Constraints:
        * @li The input x size after padding should be greater than the filter size.
        * @li The w dimension of the input x supports cases exceeding 4096, but it may
        * cause compilation errors.
        * @li If any dimension of x/filter/bias/y shape exceeds max int32 minus one (2147483646),
        * the product of each dimension of x/filter/bias/y shape exceeds max int32 minus one (2147483646) or
        * the value of strides/pads/dilations/offset_x exceeds the range which is described in Attributes,
        * the correctness of the operator cannot be guaranteed. \n
        * In Ascend 950 AI Processor: If any dimension of x/filter/bias/y shape exceeds max
        * 1000000, the product of each dimension of x/filter/bias/y
        * shape exceeds max int32(2147483647) or the value of stride/padding/dilation/offset_x
        * exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
        * @li If the Conv3D enters the Direct Memory Access(DMA) copy process, a timeout AI Core error may be reported.
        * You are advised to reduce the Conv3D specifications and try again.
        * You can view the warning log to check whether the DMA copy process is entered.
        * For example: 'The Conv3D has entered the DMA processing process. A timeout AI Core error may be reported.
        * If a timeout AI Core error is reported, reduce the Conv3D specifications and try again' \n
    * @par Third-party framework compatibility
        * @li Compatible with the TensorFlow operator "conv3d".
        * @li Compatible with the Caffe operator "Convolution".
        * @li Compatible with the ONNX operator 3D "Conv".
        * @li Compatible with the PyTorch operator "Conv3D".
    */
    REG_OP(Conv3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv3D)

    /**
    *@brief Computes the transpose of convolution 2d with respect to the input.
    *@par Inputs:
        * Five inputs:
        * @li input_size: A Tensor of type int32 or int64. An integer vector
        * representing the shape of input, where input is a 4-D tensor
        * [batch, height, width, channels] or [batch, channels, height, width].
        * @li x: A Tensor of type int8, float16, bfloat16. 4-D with shape [batch,
        * out_height, out_width, out_channels] or [batch, out_channels, out_height,
        * out_width].
        * @li filter: A Tensor of type int8, float16, bfloat16. Must have the same
        * type as "x".
        * 4-D with shape [filter_height, filter_width, in_channels, out_channels].
        * or [out_channels, filter_height, filter_width, in_channels].
        * or [out_channels, in_channel, filter_height, filter_width].
        * @li bias: An optional 1D tensor of type float16, float32, int32.
        *  Format is "ND".
        * @li offset_w: An optional 1D tensor of type int8 for quantized inference. Reserved.
        *\n
        *\n
        * The following are the supported data types and data formats:\n
        *\n
        *\n
        | Tensor    | x       | filter  | bias    | y      |\n
        |-----------|---------|---------|---------|--------|\n
        | Data Type | float16 | float16 | float16 | float16 |\n
        |           | bfloat16| bfloat16| float32 | bfloat16|\n
        |           | float16 | float16 | float32 | float32 |\n
        |           | float32 | float32 | float32 | float32 |\n
        |           | int8    | int8    | int32    | int32   |\n
        | Format    | NCHW    | NCHW    | ND      | NCHW    |\n
        |           | NHWC    | HWCN    | ND      | NHWC    |\n
        *\n
        * int8 for x and filter is not supported in 1971.
        * When input x and filter is int8, a dequant or requant operator must be followed.
        *
    *@par Attributes:
        * @li strides: A required tuple/list of 4 integers. The stride of the sliding
        * window for H/W dimension. The index of H/W is the same as data_format.
        * @li pads: A required tuple/list of 4 integers, [top, bottom, left, right]
        * pads on feature map.
        * @li groups: An optional integer of blocked connections from input channels to output
        * channels. Defaults to "1".
        * @li dilations: An optional tuple/list of 4 integers, The dilation factor for each
        * dimension of input. The value of N/C dimensions must be 1. Must be with shape
        * [1, 1, dilation_height, dilation_width] or [1, dilation_height, dilation_width, 1].
        * Defaults to [1, 1, 1, 1].
        * @li data_format: An optional string from: "NHWC", "NCHW".
        * Defaults to "NHWC". Specify the data format of the input and output data.
        * @li output_padding: An optional tuple/list of integers. The size will be added
        * in the output shape. The value of N/C dimensions must be 1. Defaults to [0, 0, 0, 0].
        * @li offset_x: An optional int. Input offset, used for quantized inference.
        * The negative offset added to the input image for int8 type. Ensure offset_x
        * within the effective range of int8 [-128, 127]. Defaults to "0".
        *\n
        *\n
        * The following value range restrictions must be met:\n
        *\n
        *\n
        | Name             | Field    | Scope        |\n
        |------------------|----------|--------------|\n
        | input_size       | H        | [1, 4096]    |\n
        |                  | W        | [1, 4096]    |\n
        | x (out_backprop) | H*strideH| [1, 4096]    |\n
        |                  | W*strideW| [1, 4096]    |\n
        | filter           | H        | [1, 255]     |\n
        |                  | W        | [1, 255]     |\n
        | y (fmap)         | H        | [1, 4096]    |\n
        |                  | W        | [1, 4096]    |\n
        | strides          | H        | [1, 63]      |\n
        |                  | W        | [1, 63]      |\n
        | pads             | Top      | [0, 255]     |\n
        |                  | Bottom   | [0, 255]     |\n
        |                  | Left     | [0, 255]     |\n
        |                  | Right    | [0, 255]     |\n
        | groups           |          | [1, 65535]   |\n
        | dilations        | H        | [1, 255]     |\n
        |                  | W        | [1, 255]     |\n
        | output_padding   | N        | [0, 0]       |\n
        |                  | C        | [0, 0]       |\n
        |                  | H        | [0, 4096]    |\n
        |                  | W        | [0, 4096]    |\n
        | Offset_x         |          | [-128, 127]  |\n
        *\n
        * In Atlas Training Series Product, fmap or out_backprop's H and W not support 1 when\n
        * fmap_h + pad_top + pad_bottom != (filter_height - 1) * dilation_h + 1
        * and filter_width > fmap_width.
        * If filter_h = 1 and filter_w = 1, out_backprop_w * stride_h * stride_w
        *  < 4096. \n
        *
    *@par Outputs:
        * y: A Tensor. A Tensor of type float16, bfloat16, float32, int32, and has
        *  same format as input_size.
        *\n
        *     out_backprop_height = (fmap_height + pad_top + pad_bottom -
        *                           (dilation_h * (filter_height - 1) + 1) - output_padding_height)
        *                           / stride_h + 1
        *\n
        *     out_backprop_width = (fmap_width + pad_left + pad_right -
        *                          (dilation_w * (filter_width - 1) + 1) - output_padding_width)
        *                          / stride_w + 1
        *\n
        *
    */
    REG_OP(Conv2DTranspose)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8, DT_BF16}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2DTranspose)

    /**
    *@brief Computes the transpose of convolution 3d with respect to the input.

    *@par Inputs:
        * @li input_size: A Tensor of type int32 or int64. An integer vector
        * representing the shape of input.
        * Any value of input_size must be in [1, 2147483647].
        * @li x: A Tensor of type float16 or bfloat16. The format
        * is NDHWC or NCDHW.
        * @li filter: A Tensor of type float16 or bfloat16, currently does not support int8.
        * The format is NDHWC, NCDHW or DHWCN.
        * height (H) and width (W) dimensions must be in [1, 511].
        * The other dimensions of filter shape must be in [1, 2147483647].
        * @li bias: Optional. An optional 1D tensor of type float16 and float32. When x
        * is float16, bias is float16. When x is bfloat16, bias is float32.
        * Currently bias is not supported on Atlas 200/500 A2 Inference Product and
        * Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component.
    Reserved.
        * @li offset_w: Optional. An optional 1D tensor for quantized deconvolution.
        * Currently offset_w is not supported on Atlas 200/500 A2 Inference Product and
        * Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component.
    Reserved.

    *@par Attributes:
        * @li strides: Required. A tuple/list of 5 integers. Specifies the stride of
        * the sliding window for each dimension of "x".
        * Has the same format as "x".
        * The batch(N) and channels(C) must be 1.
        * The other values of strides must be in [1, 2147483647].
        * @li pads: Required. A tuple/list of 6 integers.
        * [front, back, top, bottom, left, right] pads on feature map.
        * The top, bottom, left and right must be in [0, 255].
        * The front and back must be in [0, 2147483647].
        * @li dilations: Optional. A tuple/list of 5 integers,
        * The dilation factor for each dimension of input.
        * Has the same format as "x".
        * The batch(N) and channel(C) must be 1.
        * The depth(D), height(H) and width(W) must be in [1, 255].
        * In graph mode, only configuration to [1, 1, 1, 1, 1] is supported.
        * @li groups: Optional. Number of blocked connections from input channels to
        *  output channels. Defaults to 1.
        * The value of groups must be in [1, 65535].
        * @li data_format: Optional. A string from: "NDHWC", "NCDHW".
        * Defaults to "NDHWC". Specify the data format of the input and output data.
        * @li output_padding: Optional. The size will be added in the output shape.
        * Defaults to [0, 0, 0, 0, 0].
        * Currently output_padding is not supported on Atlas 200/500 A2 Inference Product and
        * Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component.
        * In graph mode, only configuration to [0, 0, 0, 0, 0] is supported.
        * @li offset_x: Optional. Input offset_x value. Defaults to 0.
        * Currently offset_x is not supported on Atlas 200/500 A2 Inference Product and
        * Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component.
    Reserved.

    *@par Outputs:
        * y: A Tensor. Has the same format as "x", has the type float16, float32, bfloat16.
        * Any dimension of y shape must be in [1, 2147483647].

    *@attention Constraints:\n
        * Due to hardware resource restrictions,
        * the operator fails to be executed in scenarios of some parameter value combinations.
        * Analyze and rectify the fault based on the log information.
        * If the fault persists, visit https://www.hiascend.com/support for technical support.
    */
    REG_OP(Conv3DTranspose)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv3DTranspose)

    /**
     * @brief Computes rectified linear: "max(x, 0)".
     *
     * @par Inputs:
     * x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types:
     * float32, float64, int32, uint8, int16, int8, int64, uint16, float16, qint8, bfloat16.
     *
     * @par Outputs:
     * y: A tensor. Has the same type as "x".
     *
     * @par Third-party framework compatibility
     * @li Compatible with the TensorFlow operator Relu.
     * @li Compatible with the Caffe operator ReLULayer.
     *
     */
    REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16,
                          DT_QINT8, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16,
                           DT_QINT8, DT_BF16}))
    .OP_END_FACTORY_REG(Relu)

    /**
    *@brief Computes reciprocal of square root of "x" element-wise: y = 1/sqrt{x}.

    *
    *@par Inputs:
    * x: An ND or 5HD tensor. Must be one of the following types: bfloat16, float, double, float16,
        * complex64, complex128.
    *
    *@par Outputs:
    * y: An ND or 5HD tensor. Has the same dtype as "x".
    *
    *@par Third-party framework compatibility
    *Compatible with the TensorFlow operator Rsqrt.
    *
    */
    REG_OP(Rsqrt)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Rsqrt)

    /**
    *@brief Add tensor with value.

    *@par Inputs:
    *One input, including: \n
    * x: A ND Tensor. Must be one of the following types:int32,int16, float16, float32, bfloat16,int64. \n

    *@par Attributes:
    *value: A scale. Must be float. \n

    *@par Outputs:
    *y: A ND Tensor. Has the same dtype and shape as "x1". \n

    *@par Third-party framework compatibility:
    * Compatible with the PyTorch operator adds.
    *@attention Constraints:
    * For parameters of the float32 type, there is no precision loss. For INT32 and INT64 parameters,
    * precision loss occurs when the parameter value exceeds 2^24. it is recommended to use Add.
    */
    REG_OP(Adds)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_INT64}))
    .REQUIRED_ATTR(value, Float)
    .OP_END_FACTORY_REG(Adds)

    /**
    *@brief Layernorm operator interface implementation \n
    *  calculating: x, gamma, beta \n
    *  mean  = np.mean(x, reduce_axis, keepdims=True) \n
    *  variance = np.mean(np.power((x - mean),2), reduce_axis, keepdims=True) \n
    *  y = gamma*((x - mean) / np.sqrt(variance + epsilon)) + beta

    *@par Inputs:
    *Three inputs, including:
    * @li x: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
    * The shape is [A1,...,Ai,R1,...,Rj].
    * @li gamma: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
    * Has the same dtype and shape as beta. The shape is [R1,...,Rj].
    * @li beta: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
    * Has the same dtype and shape as gamma. The shape is [R1,...,Rj]. \n

    *@par Attributes:
    * @li begin_norm_axis: An optional attribute, the dtype is int32. Defaults to 0.
    * Indicates the index of the R1 axis in the shape of x.
    * @li begin_params_axis: An optional attribute, the dtype is int32. Defaults to 0.
    * In Ascend 950 AI Processor, begin_params_axis and begin_norm_axis refer to the same axis in the shape of x.
    * @li epsilon: An optional attribute, the dtype is float32. Defaults to 1e-7 . \n

    *@par Outputs:
    *Three outputs, including:
    * @li y: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
    * Has the same dtype, shape and format as x.
    * @li mean: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
    * Has the same shape as variance, which is [A1,...,Ai,1,...,1], where there are j 1's after Ai.
    * @li variance: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
    * Has the same shape as mean, which is [A1,...,Ai,1,...,1], where there are j 1's after Ai.
    */
    REG_OP(LayerNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(begin_norm_axis, Int, 0)
    .ATTR(begin_params_axis, Int, 0)
    .ATTR(epsilon, Float, 0.0000001f)
    .OP_END_FACTORY_REG(LayerNorm)

    /**
    * @brief Performs max pooling on the input .

    * @par Inputs:
    * One input:
    * x: A 4-D Tensor. Supported type:float16, float32, double, int8, int16,
    * int32, int64, uint8, uint16, qint8. Supported format: NHWC, NCHW.

    * @par Attributes:
    * @li ksize: A required list of int8, int16, int32, or int64 values,
    * specifying the size of the window for each dimension of the input tensor.
    * No default value.
    * @li strides: A required list of int8, int16, int32, or int64 values,
    * specifying the stride of the sliding window for each dimension of
    * the input tensor. No default value.
    * @li padding: A required string. Supported modes: SAME, VALID. No default value. \n
    * when padding is "SAME": pads 0 to ensure output shape equal to ceil(input shape / stride) ,
    * (output shape equal to input shape when stride=1). \n
    * when padding is "VALID": no padding. The kernel slides only over valid regions, resulting in smaller output .
    * @li data_format: An optional string. Supported format: NHWC, NCHW. Defaults to "NHWC" . \n

    * @par Outputs:
    * y: A 4-D Tensor. Has the same type and format as input "x" . \n

    * @attention Constraints:
    * @li "ksize" is a list that has length 4. The ksize of the H and W dimensions should be greater than 0.
    * The ksize of the N and C dimensions should be 1. e.g. For "data_format" is "NCHW", ksize[0] = 1 and ksize[1] = 1.
    * For "data_format" is "NHWC", ksize[0] = 1 and ksize[3] = 1. \n
    * For Non-Ascend 950 AI Processor: The produce of the ksize in H and W dimensions
    * should be less than or equal to 255. e.g. For "data_format" is "NCHW", ksize[2] * ksize[3] <= 255. \n
    * @li "strides" is a list that has length 4. The stride of the N and C dimensions should be 1. \n
    * For Non-Ascend 950 AI Processor: The stride of the H and W dimensions should be greater than 0 and
    * smaller than 64. \n
    * For Ascend 950 AI Processor: The stride of the H and W dimensions should be greater than 0.
    * @li The ouput "y" shape at the N and C dimensions should be equal with input "x" shape at same dimensions. The
    output
    * shape at the H and W dimensions is calculated by below formula: \n
    * @code{.c}
        when "padding" is "SAME":
            out_height = (in_height + stride_h - 1) / stride_h
            out_width = (in_width + stride_w - 1) / stride_w
        when "padding" is "VALID":
            out_height = (in_height + stride_h - ksize_h) / stride_h
            out_width = (in_width + stride_w - ksize_w) / stride_w
        It not support out_height < 0 or out_width < 0.
    * @endcode
    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator MaxPool.
    */
    REG_OP(MaxPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
                          DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                           DT_UINT16, DT_QINT8}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPool)

    /**
    * @brief DynamicGRUV2 calculation.
    * @par Inputs:
    * seven inputs:
    * @li x:Must be one of the following types: float16.
    * @li weight_input:Must be one of the following types: float16.
    * @li weight_hidden:Must be one of the following types: float16.
    * @li bias_input:Must be one of the following types: float16, float32. The format must be ND.
    * @li bias_hidden:Must be one of the following types: float16, float32. The format must be ND.
    * @li seq_length:Must be one of the following types: int32, float16 in ND.
    * @li init_h:Must be one of the following types: float16, float32.

    * @par Attributes:
    * @li direction:An string identifying the direction in the op. Default to "UNIDIRECTIONAL". Support "UNIDIRECTIONAL"
    and "REDIRECTIONAL".
    * @li cell_depth:An integer identifying the cell depth in the op. Default to 1.
    * @li keep_prob:An float identifying the keep prob in the op. Default to 1.
    * @li cell_clip:An float identifying the cell clip in the op. Default to -1.
    * @li num_proj:An integer identifying the num projection in the op. Default to 0.
    * @li time_major:An bool identifying the time major in the op. Default to true.
    * @li activation:An string identifying the type of activation function in the op. Default to "tanh". Only tanh is
    currently supported.
    * @li gate_order:An string identifying the gate order in weight and bias. Default to "zrh". "rzh" is another option.
    * @li reset_after:An bool identifying whether to apply reset gate after matrix multiplication. Default to true.
    * @li is_training:An bool identifying is training in the op. Default to true.

    * @par Outputs:
    * six outputs:
    * @li y:Must be one of the following types: float16, float32.
    * @li output_h:Must be one of the following types: float16, float32.
    * @li update:Must be one of the following types: float16, float32.
    * @li reset:Must be one of the following types: float16, float32.
    * @li new:Must be one of the following types: float16, float32.
    * @li hidden_new:Must be one of the following types: float16, float32.
    */
    REG_OP(DynamicGRUV2)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(weight_input, TensorType({DT_FLOAT16}))
    .INPUT(weight_hidden, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias_hidden, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32, DT_FLOAT16}))
    .OPTIONAL_INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(update, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(reset, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(hidden_new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(direction, String, "UNIDIRECTIONAL")
    .ATTR(cell_depth, Int, 1)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(cell_clip, Float, -1.0)
    .ATTR(num_proj, Int, 0)
    .ATTR(time_major, Bool, true)
    .ATTR(activation, String, "tanh")
    .ATTR(gate_order, String, "zrh")
    .ATTR(reset_after, Bool, true)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(DynamicGRUV2)

    /**
    * @brief Function axpy with softmax and dropoutdomask . \n

    * @par Inputs:
    * Three inputs, including:
    * @li x1: A mutable tensor. The type supports float16 and float32.
    * @li x2: A mutable tensor. The type supports float16 and float32. Has the same type and shape as "x1".
    * @li mask: A mutable tensor. The type supports uint8. Has the same shape as "x1". \n

    * @par Attributes:
    * @li alpha: A attribute used to scale tensor. The type is float . \n
    * @li input_keep_prob: A attribute used to judge which units should be keep.
    *     The type is float . \n
    * @li axis: A list of int. The dimension softmax would be performed on. Defaults
    *     to "[-1]" . \n

    * @par Outputs:
    * @li y1: A mutable tensor. Has the same type as "x1". \n
    * @li y2: A mutable tensor. Has the same type as "x1". \n

    * @par Restrictions:
    * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
    */
    REG_OP(AxpyWithSoftmaxAndDropOutDoMask)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(alpha, Float)
    .REQUIRED_ATTR(input_keep_prob, Float)
    .ATTR(axis, ListInt, {-1})
    .OP_END_FACTORY_REG(AxpyWithSoftmaxAndDropOutDoMask)

    /**
    * @brief Return the unique elements of the input tensor with counts and sorted elements. \n

    * @par Inputs:
    * x: A tensor. Input "x" is a k-dimensional tensor. \n

    * @par Attributes:
    * @li return_inverse: An optional DType from: "bool". Defaults to False.
    * @li return_counts: An optional DType from: "bool". Defaults to False.
    * @li sorted: An optional DType from "bool". Defaults to True. \n
    * @li out_idx: Output index/count's datatype. Defaults to DT_INT64.

    * @par Outputs:
    * @li y: A Tensor. The output list of unique scalar elements. Has the same type as "x".
    * @li indices: A tensor of type DT_INT32, DT_INT64.
    *              Representing the indices for where elements in the original input map to in the output.
    * @li counts: A tensor of type DT_INT32, DT_INT64.
                    Representing the number of occurrences for each unique value or tensor. \n

    * @par Third-party framework compatibility
    * Compatible with Pytorch operator _unique2.
    */
    REG_OP(UniqueWithCountsAndSorting)
    .INPUT(x, TensorType({BasicType(), DT_BF16}))
    .OUTPUT(y, TensorType({BasicType(), DT_BF16}))
    .OUTPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(counts, TensorType({DT_INT32, DT_INT64}))
    .ATTR(return_inverse, Bool, false)
    .ATTR(return_counts, Bool, false)
    .ATTR(sorted, Bool, true)
    .ATTR(out_idx, Type, DT_INT64)
    .OP_END_FACTORY_REG(UniqueWithCountsAndSorting)

    /**
    *@brief Creates a tensor with the given "shape" and "dtype". \n

    *@par Inputs:
    *shape: The shape of the output tensor. \n

    *@par Attributes:
    *@li dtype: Optional. The data type of the output tensor. Defaults to "int32".
    *@li init: An optional bool. If true, initializes the returned tensor with the default value of "dtype". Defaults to
    "false". \n

    *@par Outputs:
    *y: A tensor. \n

    *@par Third-party framework compatibility
    *Compatible with the TensorFlow operator Empty.
    */
    REG_OP(Empty)
    .INPUT(shape, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE, DT_BF16, DT_STRING, DT_COMPLEX64, DT_COMPLEX128}))
    .ATTR(dtype, Int, DT_INT32)
    .ATTR(init, Bool, false)
    .OP_END_FACTORY_REG(Empty)

    /**
    * @brief Multiplies matrix "a" by matrix "b", producing "a @ b".
    * @par Inputs:
    * Four inputs, including:
    * @li x1: A matrix Tensor. 2D. Must be one of the following types: float32,
    * float16, int32, int8, int4, bfloat16, hifloat8. Has format [ND, NHWC, NCHW].
    * @li x2: A matrix Tensor. 2D. Must be one of the following types: float32,
    * float16, int32, int8, int4, bfloat16, hifloat8. Has format [ND, NHWC, NCHW].
    * @li bias: A 1D Tensor. Must be one of the following types: float32,
    * float16, int32, bfloat16. Has format [ND, NHWC, NCHW].
    * @li offset_w: A Optional 1D Tensor for quantized inference. Type is int8, int4, bfloat16.
    * Reserved.

    * @par Attributes:
    * @li transpose_x1: A bool. If True, changes the shape of "x1" from [K, M] to
    * [M, K] before multiplication.
    * @li transpose_x2: A bool. If True, changes the shape of "x2" from [N, K] to
    * [K, N] before multiplication.
    * @li offset_x: An optional integer for quantized MatMulV2.
    * The negative offset added to the input x1 for int8 type. Ensure offset_x
    * within the effective range of int8 [-128, 127]. Defaults to "0".

    * @par Outputs:
    * y: The result matrix Tensor. 2D. Must be one of the following types: float32,
    * float16, int32, bfloat16, hifloat8. Has format [ND, NHWC, NCHW].

    * @attention Constraints:
    * if performances better in format NZ, please close
    * "MatmulTransdataFusionPass" in fusion configuration.

    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator MatMul.
    */
    REG_OP(MatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(MatMulV2)

    /**
    * @brief Resize images to size using nearest neighbor interpolation. \n

    * @par Inputs:
    * Inputs include:
    * @li x: A 4-D tensor. Represents the original image. Must set the format, supported format list ["NCHW, NHWC"].
    * Must be one of the following types: int8, uint8, int16, uint16, int32, int64, float16, float32,
    * double, bfloat16.
    * @li size: A 1-D int32 tensor of 2 elements: new_height, new_width.
    * Indicates the size of the target image, which is used to determine the height and width of the output image.
    * Must be the type int32. \n

    * @par Attributes:
    * @li align_corners: An optional bool. Determines whether to align the corners of the input and output images.
    * If set to True, the corner pixels of the input and output images are aligned,
    * preserving the value of the corner pixels. When set to false,
    * the scaling process scales according to proportions and does not strictly align the corners.
    * Defaults to false.
    * @li half_pixel_centers: An optional bool. Determines the pixel center position during interpolation.
    * If this parameter is set to True, the interpolation algorithm considers the center point of the pixel
    * to estimate the pixel value more accurately. When set to false, the pixel center is on the integer coordinate
    point.
    * Defaults to false. \n

    * @li scales: An optional listfloat. Multiplier for spatial size. Defaults to {0.0f, 0.0f} .
    * @par Outputs:
    * y: A 4-D tensor. Indicates the target image. Has the same type and format as input "x".
         The N, C dimension must be the same as x. \n

    * @par Third-party framework compatibility
    * Compatible with tensorflow ResizeNearestNeighbor operator.
    */

    REG_OP(ResizeNearestNeighborV2)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                          DT_BF16}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                           DT_BF16}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .ATTR(scales, ListFloat, {0.0f, 0.0f})
    .OP_END_FACTORY_REG(ResizeNearestNeighborV2)

    /**
    * @brief Resize the input tensor. \n
    currently, only support resize image tensor using nearest neighbor and linear interpolation.

    * @par Inputs:
    * Input x must be a 4-D tensor. Inputs include: \n
    * @li x: A Tensor. Must be one of the following types: uint8, int8, int16, \n
    int32, int64, float16, float, double. 4-D with shape [batch, height, width, channels] \n
    or shape [batch, channels, height, width].
    * @li roi: A 1-D float Tensor. Only takes effect when attr coordinate_transformation_mode \n
    is "tf_crop_and_resize". Must be one of the following types: float16, float, double.
    * @li scales: A 1-D float Tensor, the scale array along each dimension, Only one of \n
    'scales' and 'sizes' can be specified. Must be float type.
    * @li sizes: A 1-D int64 Tensor, The size of the output tensor. Only one of \n
    'scales' and 'sizes' can be specified.  If 'size' is specified, then set scales \n
    to empty data (zero shape) in this operator's input list. Must be one of \n
    the following types: int32, int64.

    * @par Attributes:
    * @li coordinate_transformation_mode: An optional String. how to transform \n
    the coordinate in the resized tensor to the coordinate in the original tensor. \n
    options: pytorch_half_pixel, align_corners, asymmetric, \n
    tf_crop_and_resize.
    * @li cubic_coeff_a: An optional Float. Defaults to -0.75, only used in cubic interpolation. \n
    other optional: -0.5
    * @li exclude_outside: An optional Int. Defaults to 0, If set to 1, the weight of sampling \n
    locations outside the tensor will be set to 0 and the weight will be renormalized \n
    so that their sum is 1.0.
    * @li extrapolation_value: An optional Float. Defaults to 0.0f. When coordinate_transformation_mode \n
    is "tf_crop_and_resize" and x_original is outside the range [0, length_original - 1], \n
    this value is used as the corresponding output value.
    * @li mode: An optional String. Defaults to nearest. Three interpolation modes: nearest (default), \n
    linear and cubic.
    * @li nearest_mode: An optional String. Defaults to round_prefer_floor. Four modes: round_prefer_floor, \n
    round_prefer_ceil, floor, ceil. Only used by nearest interpolation.

    * @par Outputs:
    * y: A Tensor. Has the same type as x.

    * @attention Constraints: \n
    * Input x must be a 4-D tensor.

    * @par Third-party framework compatibility
    * Compatible with tensorflow ResizeNearestNeighborV2 operator.
    */

    REG_OP(Resize)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OPTIONAL_INPUT(roi, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(sizes, TensorType({DT_INT64, DT_INT32}))
    .OUTPUT(y,
            TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(coordinate_transformation_mode, String, "half_pixel")
    .ATTR(cubic_coeff_a, Float, -0.75)
    .ATTR(exclude_outside, Int, 0)
    .ATTR(extrapolation_value, Float, 0.0)
    .ATTR(mode, String, "nearest")
    .ATTR(nearest_mode, String, "round_prefer_floor")
    .OP_END_FACTORY_REG(Resize)

    /**
    * @brief According to the indices and indices_mask, return the value.

    * @par Inputs:
    * Four inputs, including:
    * @li x: A ND tensor. Must be one of the following types:
    *     float, float16, int64, int32, bool, uint8, int8.
    * @li indices: Dynamic input. A ND tensor of int64. return the value according to the indices.

    * @par Attributes:
    *
    * @li indices_mask: A list int. Indicates which dimensions of input needs to be indexed.

    * @par Outputs:
    * @li y: The indexed output tensor. Has the same type and format as input "x".
    */
    REG_OP(IndexByTensor)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL}))
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL}))
    .ATTR(indices_mask, ListInt, {})
    .OP_END_FACTORY_REG(IndexByTensor)

    /**
    * @brief Gather slices from "params" according to "indices"."indices" must be
        an integer tensor of any dimension(usually 0-D or 1-D).
        Produces an output tensor with shape "indices.shape + params.shape[1:]" .

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
    .OP_END_FACTORY_REG(Gather)

    /**
    * @brief Applies sparse addition to individual values or slices in a Variable .

    * @par Inputs:
    * Three inputs, including:
    * @li x: An ND Tensor. \n

    * Must be one of the following types: float16, float32, int32, int8, uint8
    * @li indices: An ND Tensor. \n

    * Must be one of the following types: int32
    * @li updates: An ND Tensor. \n

    * Must be one of the following types: float16, float32, int32, int8, uint8

    * @par Outputs:
    * y: A Tensor. Has the same type and format as input "x" . \n

    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator TensorScatterAdd.

    * @par Restrictions:
    * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
    */
    REG_OP(TensorScatterAdd)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
    .OP_END_FACTORY_REG(TensorScatterAdd)

    /**
    * @brief Choose the value of X with value according to mask.

    * @par Inputs:
    * two inputs, including:
    *  @li x: A Tensor of dtype is float16 or float32.
    *  @li mask: A Tensor of dtype is bool. \n

    * @par Outputs:
    * y: A tensor with the same type as x. \n

    * @par Third-party framework compatibility
    * Compatible with the Numpy operator select.
    * Replaces the pytorch operator masked_select in some scenarios.\n
    */
    REG_OP(MaskedSelectV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mask, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(MaskedSelectV2)

    /**
    * @brief Choose the value of X with value according to mask.

    * @par Inputs:
    * two inputs, including:
    * @li x: A tensor of type BasicType.
    * @li mask: A tensor of type bool, true for selecting related number of x out, false for no selecting. \n

    * @par Outputs:
    * y: A tensor with the same type as x. \n

    * @attention Constraints:
    * @li The input tensors of x and mask must meet the broadcast relationship.
    * @li The dimnum of y must be 1.

    * @par Third-party framework compatibility
    * Compatible with the Numpy operator select.\n
    */
    REG_OP(MaskedSelect)
    .INPUT(x, TensorType::BasicType())
    .INPUT(mask, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MaskedSelect)

    /**
    * @brief Quantizes the input of int8.

    * @par Inputs:
    * @li x: A tensor. Must be one of the following types: int8. The format support NZ.
    * @li offset: A tensor. Must be one of the following types: int8. The format support NZ. \n

    * @par Attributes:
    * @li dst_type: Declare the output dtype. Support DT_INT8, DT_INT4. Defaults to DT_INT8. \n

    * @par Outputs:
    * @li y: A output Tensor. Must be one of the following types: int8, int4. The format support NZ. \n

    * @par Third-party framework compatibility
    * It is a custom operator. It has no corresponding operator in Caffe, Onnx, Tensorflow or Pythorch.
    */
    REG_OP(AscendWeightQuant)
    .INPUT(x, TensorType({DT_INT8}))
    .INPUT(offset, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(AscendWeightQuant)

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
    REG_OP(L2Normalize)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, ListInt, {})
    .ATTR(eps, Float, 1e-4f)
    .OP_END_FACTORY_REG(L2Normalize)

    /**
    *@brief Performs the backpropagation of L2Normalize for training scenarios .

    *@par Inputs:
    * Three inputs, including:
    *@li x: A ND Tensor(1D-8D) of type float16 or float32, specifying
    * the eigenvalue of forward inputs.
    *@li y: A ND Tensor(1D-8D) of type float16 or float32, specifying
    * the normalization result of the forward output. the same shape with x.
    *@li dy: A ND Tensor(1D-8D) of type float16 or float32, specifying
    * the reverse input gradient. the same shape with x . \n

    *@par Attributes:
    *@li dim: A required attribute of type int, specifying the axis to be
    * normalized.  Defaults to {}.
    *@li eps: An optional attribute of type float, specifying the lower limit of
    * normalization. Defaults to "1e-4" . \n

    *@par Outputs:
    *dx: A ND Tensor(1D-8D), Reverse gradient of eigenvalue "x". Has the same shape as "x" . \n

    *@par Third-party framework compatibility
    * Compatible with the L2 scenario of PyTorch operator NormalizeGrad.
    */
    REG_OP(L2NormalizeGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(dim, ListInt, {})
    .ATTR(eps, Float, 0.0001f)
    .OP_END_FACTORY_REG(L2NormalizeGrad)

    /**
    *@brief Performs batch normalization .

    *@par Inputs:
    * Five inputs, including: (NDHWC, NCDHW)
    *@li x: A 5D Tensor of type float16 or float32, with format NDHWC or NCDHW.
    *@li scale: A Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW.
    Specifies the scaling factor.
    *@li offset: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
    Specifies the offset.
    *@li mean: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
    Specifies the mean used for inference. Must be "None" if the
    operation is used for training.
    *@li variance: A Tensor of type float32. Must be 3D if input "x" is with format NHWC or NCHW.
    Specifies the variance used for inference. Must be "None"
    if the operation is used for training . \n

    *@par Attributes:
    *@li epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero. Defaults
    to "0.0001".
    *@li data_format: An optional string, specifying the format of "x". Defaults to "NCDHW".
    *@li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to
    "True" . \n

    *@par Outputs:
    * Five outputs, including: (NDHWC, NCDHW)
    *@li y: A 5D Tensor of type float16 or float32 for the normalized "x", with format NDHWC or NCDHW.
    *@li batch_mean: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
    Specifies the mean of "x".
    *@li batch_variance: A Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW.
    Specifies the variance of "x".
    *@li reserve_space_1: An optional Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW.
    Specifies the mean of "x" for gradient computation. Pass "None" to skip this output.
    *@li reserve_space_2: An optional Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
    Specifies the variance of "x" for gradient computation. Pass "None" to skip this output . \n

    *@attention Constraints:
    *@li If the operation is used for inference and outputs "reserve_space_1" and "reserve_space_2" are available,
    then "reserve_space_1" has the same value as "mean" and "reserve_space_2" has the same value as "variance".
    *@li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1‰ due to the square root
    instruction . \n

    *@par Third-party framework compatibility
    *@li Compatible with the TensorFlow operator fused_batch_norm.
    *@li Compatible with the TensorFlow operator fused_batch_norm_v2.
    */
    REG_OP(BatchNorm3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm3D)

    /**
    *@brief Performs batch normalization .

    *@par Inputs:
    * Five inputs, including: (NHWC or NCHW supported)
    *@li input_x: A 4D Tensor of type float16 or float32.
    *@li input_scale: A 1D Tensor of type float32, for the scaling factor.
    *@li input_offset: A 1D Tensor of type float32, for the scaling offset.
    *@li input_mean: A 1D Tensor of type float32, for the mean used for inference.
    * This cannot be used if the operation is used for training.
    *@li input_variance: A 1D Tensor of type float32, for the variance used for inference.
    * This cannot be used if the operation is used for training . \n

    *@par Attributes:
    *@li epsilon: An optional float32, specifying the small value
    added to variance to avoid dividing by zero. Defaults to "0.0001".
    *@li data_format: An optional string, specifying the format of "x". Defaults to "NHWC".
    *@li is_training: An optional bool, specifying if the operation
    is used for training or inference. Defaults to "True" . \n

    *@par Outputs:
    * Five outputs, including: (NHWC or NCHW supported)
    *@li output_y: A 4D Tensor of type float16 or float32, for the normalized "x".
    *@li output_mean: A 1D Tensor of type float32, for the mean of "x".
    *@li output_variance: A 1D Tensor of type float32, for the variance of "x".
    *@li output_reserve_space_1: A 1D Tensor of type float32, for the mean of "x" for gradient computation.
    *@li output_reserve_space_2: A 1D Tensor of type float32, for the variance of "x" for gradient computation . \n

    *@attention Constraints:
    *@li If the operation is used for inference, then output "reserve_space_1"
    has the same value as "mean" and output "reserve_space_2" has the same value as "variance".
    *@li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1‰ due to the square root
    instruction . \n

    *@par Third-party framework compatibility
    * Compatible with the TensorFlow operator fused_batch_norm_v2.
    */
    REG_OP(BatchNormExt2)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_scale, TensorType({DT_FLOAT}))
    .INPUT(input_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(input_mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(input_variance, TensorType({DT_FLOAT}))
    .OUTPUT(output_y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_mean, TensorType({DT_FLOAT}))
    .OUTPUT(output_variance, TensorType({DT_FLOAT}))
    .OUTPUT(output_reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(output_reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNormExt2)

    /**
    *@brief Performs the backpropagation of BatchNorm .

    *@par Inputs:
    * Five inputs, including:
    *@li y_backprop: A 5D Tensor of type float16 or float32, with format NDHWC, NCDHW, for the gradient.
    *@li x: A 5D Tensor of type float16 or float32, with format NDHWC, NCDHW.
    *@li scale: A 5D Tensor of type float32, with format NDHWC, NCDHW.
    *@li reserve_space_1: A 5D Tensor of type float32, with format NDHWC, NCDHW. It is an output of BatchNorm.
    *@li reserve_space_2: A 5D Tensor of type float32, with format NDHWC, NCDHW. It is an output of BatchNorm . \n

    *@par Attributes:
    *@li epsilon: An optional float32. Defaults to "0.0001". A small float number added to the variance of "x".
    *@li data_format: An optional string. Defaults to "NCDHW".
    *@li is_training: An optional bool. Defaults to "true". Specifies the operation is for training (default) or
    inference . \n

    *@par Outputs:
    *@li x_backprop: A Tensor of type float16 or float32, with format NDHWC, NCDHW, for the offset of "x".
    *@li scale_backprop: A Tensor of type float32, with format NDHWC, NCDHW, for the offset of "scale".
    *@li *offset_backprop: A Tensor of type float32, with format NDHWC, NCDHW, for the offset of "offset".
    *@li *reserve_space_4: A Tensor of type float32, with shape NDHWC, NCDHW. Pass "None" to skip this output.
    *@li *reserve_space_5: A Tensor of type float32, with shape NDHWC, NCDHW. Pass "None" to skip this output . \n

    *@attention Constraints:
    * The preceding layer of this operator must be operator BatchNorm . \n

    *@see BatchNorm
    *@par Third-party framework compatibility
    * Compatible with the TensorFlow operators FusedBatchNormGradV2 and FusedBatchNorm3DGrad.
    */
    REG_OP(BatchNorm3DGrad)
    .INPUT(y_backprop, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_5, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm3DGrad)

    /**
    *@brief Performs the backpropagation of BatchNorm .

    *@par Inputs:
    * Five inputs, including:
    *@li y_backprop: A 4D Tensor of type float16 or float32, with format NHWC or NCHW, for the gradient.
    *@li x: A 4D Tensor of type float16 or float32, with format NHWC or NCHW, the shape is same as input y_backprop.
    *@li scale: A 4D Tensor of type float32, with format NHWC or NCHW, the shape is same as input y_backprop.
    *@li reserve_space_1: A 4D Tensor of type float32, with format NHWC or NCHW, the shape is same as input y_backprop,
    * it is an output of BatchNormExt2.
    *@li reserve_space_2: A 4D Tensor of type float32, with format NHWC or NCHW, the shape is same as input y_backprop,
    * it is an output of BatchNormExt2 . \n

    *@par Attributes:
    *@li epsilon: A required float32. A small float number added to the variance of "x".
    *@li data_format: A required string for the format.
    *@li is_training: A required bool for specifying the operation is for training (true) or inference (false) . \n

    *@par Outputs:
    *@li x_backprop: A Tensor of type float16 or float32, with format NHWC or NCHW, for the offset of "x".
    *@li scale_backprop: A Tensor of type float32, with format NHWC or NCHW, for the offset of "scale".
    *@li offset_backprop: A Tensor of type float32, with format NHWC or NCHW, for the offset of "offset".
    *@li reserve_space_3: A Tensor of type float32, with format NHWC or NCHW.
    *@li reserve_space_4: A Tensor of type float32, with format NHWC or NCHW . \n

    *@attention Constraints:
    * The preceding layer of this operator must be BatchNormExt2 . \n

    *@see BatchNormExt2
    *@par Third-party framework compatibility
    * Compatible with the TensorFlow operator FusedBatchNormGradV2.
    */
    REG_OP(BatchNormGradExt2)
    .INPUT(y_backprop, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BatchNormGradExt2)

    /**
    * @brief Calculate the hard sigmoid function.

    * @par Inputs:
    * One input, including:
    * input_x: A ND tensor. The shape should be within the range of 0D to 8D.
    * Must be one of the following types: float16, float32, int32, bfloat16.

    * @par Attributes:
    * @li alpha: An optional float. Slope of the operator, defaults to 0.16666666.
    * @li beta: An optional float. Offset of the operator, defaults to 0.5.

    * @par Outputs:
    * output_y: A ND tensor with the same dtype and shape as 'input_x'.

    * @par Third-party framework compatibility
    * Compatible with the Pytorch operator Hardsigmoid.
    */
    REG_OP(HardSigmoid)
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(output_y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(alpha, Float, 0.16666666)
    .ATTR(beta, Float, 0.5)
    .OP_END_FACTORY_REG(HardSigmoid)

    /**
    * @brief Calculate the soft shrinkage function.

    * @par Inputs:
    * One inputs, including:
    * input_x: A tensor. Must be one of the following types:
    *     float16, float32, bfloat16. \n

    * @par Attributes:
    * lambd: An optional float. Defaults to 0.5. \n

    * @par Outputs:
    * output_y: A Tensor with the same dtype and shape of input_x's. \n

    * @par Third-party framework compatibility
    * Compatible with the Pytorch operator Softshrink. \n
    */
    REG_OP(SoftShrink)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output_y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(lambd, Float, 0.5)
    .OP_END_FACTORY_REG(SoftShrink)

    /**
    * @brief Calculate the reversed outputs of the function "soft_shrink".

    * @par Inputs:
    * Two inputs, including:
    * @li input_grad: A tensor. The shape should be within the range of 0D to 8D. Must be one of the following types:
    *     float16, float32, bfloat16. \n
    * @li input_x: A tensor of the same dtype and shape as "input_grad". The shape should be within the range of 0D to
    8D. \n

    * @par Attributes:
    * lambd: An optional float. Defaults to 0.5. lambd should be greater or equal to 0. \n

    * @par Outputs:
    * output_y: A Tensor of the same dtype and shape as "input_grad". The shape should be within the range of 0D to 8D.
    \n

    * @par Third-party framework compatibility
    * Compatible with the Pytorch operator SoftShrinkGrad. \n
    */
    REG_OP(SoftShrinkGrad)
    .INPUT(input_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output_y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(lambd, Float, 0.5)
    .OP_END_FACTORY_REG(SoftShrinkGrad)

    /**
     *@brief Operators for managing cache memory.

     *@par Inputs:
     *src: A ND Tensor with TensorType::NumberType().

     *@par Attributes:
     *@li max_size: The maximum memory size required for caching operation.
     *@li type: An optional int32 or int64 which has a default value of 6, indicating a prefetch operation.
     *@li offset: An optional int32 or int64 specifies the offset of the CMO operation address, which must not exceed
     the *size of the input memory. \n
     */
    REG_OP(Cmo)
    .INPUT(src, TensorType::NumberType())
    .REQUIRED_ATTR(max_size, Int)
    .ATTR(type, Int, 6)
    .ATTR(offset, Int, 0)
    .OP_END_FACTORY_REG(Cmo)

    /**
    *@brief Creates a criterion that measures the loss given input tensors x1 x2 and a Tensor label y with values 1 or
    -1.

    *@par Inputs:
    *@li x1: A ND Tensor with one of the following types: int8, uint8, int32, float16, float32, int16, int64, double.
    *@li x2: A ND Tensor with one of the following types: int8, uint8, int32, float16, float32, int16, int64, double.
    * x1 and x2 can be broadcast.
    *@li target: A ND Tensor with one of the following types: int8, uint8, int32, float16, float32, int16, int64,
    double.
    * target and x1, x2 can be broadcast.\n

    *@par Attributes:
    *@li margin: An optional float32. Defaults to "0.0".
    *@li reduction: An optional string. Defaults to "mean". \n

    *@par Outputs:
    *@li y: A ND Tensor with Must be float32.
    *@par Third-party framework compatibility
    * Compatible with the PyTorch operator CosineEmbeddingLoss.
    */
    REG_OP(CosineEmbeddingLoss)
    .INPUT(x1, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(x2, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(target, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(margin, Float, 0)
    .ATTR(reduction, String, "mean")
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(CosineEmbeddingLoss)

    /**
    *@brief Count adam result. \n

    *@par Inputs:
    *Eleven inputs, including:
    * @li var: A ND Tensor of weight. Support float16/float32.\n
    * @li m: A ND Tensor of the 1st moment estimates. Datatype and shape are same as var.\n
    * @li v: A ND Tensor of the 2nd moment estimates. Datatype and shape are same as var.\n
    * @li lr: A ND Tensor of learning rate. Datatype is same as var. Shape (1, ).\n
    * @li beta1: A ND Tensor of the exponential decay rate for the 1st moment estimates. Datatype is same as var. Shape
    (1, ).\n
    * @li beta2: A ND Tensor of the exponential decay rate for the 2nd moment estimates. Datatype is same as var. Shape
    (1, ).\n
    * @li epsilon: A ND Tensor for numerical stability. Datatype is same as var. Shape (1, ).\n
    * @li grad: A ND Tensor. Datatype and shape are same as var.\n
    * @li max_grad_norm: An Optional Tensor. Datatype is same as var. Shape (1, ).\n
    * @li global_grad_norm: A ND Tensor. Datatype is same as var. Shape (1, ).\n
    * @li weight_decay: A ND Tensor. Datatype is same as var. Shape (1, ).\n
    * @li step_size: An Optional Tensor. Datatype is same as var. Shape (1, ).\n

    * @par Attributes:
    * @li adam_mode: An optional bool. Defaults to "adam". \n

    *@par Outputs:
    *Three inputs, including:
    * @li var: A ND Tensor of weight. Datatype and shape are same as var.\n
    * @li m: A ND Tensor of the 1st moment estimates. Datatype and shape are same as var.\n
    * @li v: A ND Tensor of the 2nd moment estimates. Datatype and shape are same as var.\n
    */
    REG_OP(ApplyAdamV2)
    .INPUT(var, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(m, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(v, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(epsilon, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(max_grad_norm, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(global_grad_norm, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(weight_decay, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(step_size, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(var, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(m, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(v, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(adam_mode, String, "adam")
    .OP_END_FACTORY_REG(ApplyAdamV2)

    /**
     *@brief Forwards the value of an available tensor from input "x" to output "y".
     *       Merge waits for at least one of the input tensors to become available.
     *       It is usually combined with Switch to implement branching.
     *       Merge forwards the first tensor to become available to output "y",
     *       and sets "value_index" the index of the tensor in inputs .

     *@par Inputs:
     *x: The input tensors, one of which will become available.
     *   Must be one of the following types: float16, float32, float64, int8,
     *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . It's a dynamic input. \n

     *@par Outputs:
     *@li y: The available tensor. Has the same type as "x".
     *@li value_index: A scalar of type int32, for the index of the chosen input
     *                 tensor . \n

     *@see Switch()

     *@par Third-party framework compatibility
     *Compatible with the TensorFlow operator Merge.
     */
    REG_OP(Merge)
    .DYNAMIC_INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OUTPUT(value_index, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(Merge)

    /**
     *@brief Forwards the value of an available tensor from input "x" to output "y".
     *       Merge waits for at least one of the input tensors to become available.
     *       It is usually combined with Switch to implement branching.
     *       Merge forwards the first tensor to become available to output "y",
     *       and sets "value_index" the index of the tensor in inputs .

     *@par Inputs:
     *x: The input tensors, one of which will become available.
     *   Must be one of the following types: float16, float32, float64, int8,
     *   int16, int32, int64, uint8, uint16, uint32, uint64, bool, string . It's a dynamic input. \n

     *@par Outputs:
     *@li y: The available tensor. Has the same type as "x".
     *@li value_index: A scalar of type int32, for the index of the chosen input
     *                 tensor . \n

     *@see Switch() | Merge()

     *@par Third-party framework compatibility
     *Compatible with the TensorFlow operator RefMerge.
     */
    REG_OP(RefMerge)
    .DYNAMIC_INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OUTPUT(value_index, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(RefMerge)

    /**
     *@brief Forwards "data" to the output port determined by "pred".
     *       If "pred" is "true", the data input is forwarded to "output_true".
     *       Otherwise, the data is forwarded to "output_false" .

     *@par Inputs:
     *@li data: The tensor to be forwarded.
     *          Must be one of the following types: float16, float32, float64,
     *          int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool, string.
     *@li pred: A boolean scalar. The output port that will receive data . \n

     *@par Outputs:
     *@li output_false: If "pred" is "false", data will be forwarded to this output.
     *                  Has the same type as "data".
     *@li output_true: If "pred" is "true", data will be forwarded to this output.
     *                 Has the same type as "data" . \n

     *@see Merge()

     *@par Third-party framework compatibility
     *Compatible with the TensorFlow operator Switch.
     */
    REG_OP(Switch)
    .INPUT(data, TensorType::ALL())
    .INPUT(pred, TensorType({DT_BOOL}))
    .OUTPUT(output_false, TensorType::ALL())
    .OUTPUT(output_true, TensorType::ALL())
    .OP_END_FACTORY_REG(Switch)

    /**
     *@brief Forwards "data" to the output port determined by "pred".
     *       If "pred" is "true", the data input is forwarded to "output_true".
     *       Otherwise, the data is forwarded to "output_false" .

     *@par Inputs:
     *@li data: The ref tensor to be forwarded.
     *          Must be one of the following types: float16, float32, float64,
     *          int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool, string.
     *@li pred: A boolean scalar. The output port that will receive data . \n

     *@par Outputs:
     *@li output_false: If "pred" is "false", data will be forwarded to this output.
     *                  Has the same type as "data".
     *@li output_true: If "pred" is "true", data will be forwarded to this output.
     *                 Has the same type as "data" . \n

     *@see Merge() | Switch()

     *@par Third-party framework compatibility
     *Compatible with the TensorFlow operator RefSwitch.
     */
    REG_OP(RefSwitch)
    .INPUT(data, TensorType::ALL())
    .INPUT(pred, TensorType({DT_BOOL}))
    .OUTPUT(output_false, TensorType::ALL())
    .OUTPUT(output_true, TensorType::ALL())
    .OP_END_FACTORY_REG(RefSwitch)

    /**
     *@brief Forwards "data" to the output port determined by "pred_value" .

     *@par Inputs:
     *@li data: The tensor to be forwarded.
     *          Must be one of the following types: float16, float32, float64,
     *          int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool.
     *@li pred_value: An int64 tensor which determines the output port that will receive data . \n

     *@par Outputs:
     *output: The output tensors, one of which will become available.
     *        Has the same type as "data". It's a dynamic output.
     */
    REG_OP(SwitchN)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                             DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))
    .INPUT(pred_value, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                                        DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(SwitchN)

    /**
     *@brief Creates or finds a child frame, and makes "x" available to the child
     *       frame. This op is used together with Exit to create loops in the graph.
     *       The Executor uses the unique "frame_name" to identify frames.
     *       If "is_constant" is "true", output "y" is a constant in the child
     *       frame; otherwise it may be changed in the child frame .

     *@par Inputs:
     *x: The tensor to be made available to the child frame.
     *   Must be one of the following types: float16, float32, float64, int8,
     *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . \n

     *@par Attributes:
     *@li frame_name: A required string. The name of the child frame.
     *@li is_constant: A required bool. If true, the output is constant in
     *                 the child frame . \n

     *@par Outputs:
     *y: A Tensor. Has the same type as "x" . \n

     *@see Exit()

     *@par Third-party framework compatibility
     *Compatible with the TensorFlow operator Enter.
     */
    REG_OP(Enter)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
                          DT_UINT32, DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
                           DT_UINT32, DT_UINT64, DT_BOOL}))
    .REQUIRED_ATTR(frame_name, String)
    .REQUIRED_ATTR(is_constant, Bool)
    .OP_END_FACTORY_REG(Enter)

    /**
     *@brief Makes the input available to the next iteration .

     *@par Inputs:
     *x: The tensor to be made available to the next iteration.
     *   Must be one of the following types: float16, float32, float64, int8,
     *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . \n

     *@par Outputs:
     *y: A Tensor. Has the same type as "x" . \n

     *@par Third-party framework compatibility
     *Compatible with the TensorFlow operator NextIteration.
     */
    REG_OP(NextIteration)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
                          DT_UINT32, DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
                           DT_UINT32, DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(NextIteration)

    /**
    * @brief Applies set operation along last dimension of 2 Tensor inputs. \n

    * @par Inputs:
    * @li xyz1: A Tensor. Must be one of the following types: float16, bfloat16, float32. Point set with shape (B, 2, N)
    * @li xyz2: A Tensor. Must have the same type and shape as x1. \n

    * @par Outputs:
    * @li dist1: A Tensor. Must be one of the following types: float16, bfloat16, float32. with shape (B, N)
    * @li dist2: A Tensor. Must have the same type and shape as dist1.
    * @li idx1: A Tensor of type int32. with shape (B, N)
    * @li idx2: A Tensor. Must have the same type and shape as idx1.
    */
    REG_OP(ChamferDistance)
    .INPUT(xyz1, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(xyz2, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(dist1, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(dist2, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(idx1, TensorType({DT_INT32}))
    .OUTPUT(idx2, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(ChamferDistance)

    /**
    * @brief Computes LpNormReduce.

    * @par Inputs:
    * x: A ND tensor of type float16, bfloat16, float32.
    *
    * @li p: An optional int, "inf" or "-inf", default value is 2, p >= 0.
    * @li axes: ListInt, an optional attribute, indicates dimensions over which to compute the norm.
    * Default is {}, meaning all axes will be computed.
    * @li keepdim: An optional bool. If set to true, the reduced dimensions are retained in the result
    * as dimensions with size one. Default is false.
    * @li epsilon: An optional float. A value added to the denominator for numerical stability. Default is 1e-12.

    * @par Outputs:
    * y: A ND tensor has the same dtype as "x". The shape of "y" is depending on "axes" and "keepdim".

    * @attention Constraints:
    * @li When the attribute "axes" is specified as the axis with a shape dimension value of 1 in the input tensor,
    * there may be precision difference in the calculation results.
    * @li When the tensor "x" is empty and "p" is infinity, we cannot reduce the whole tensor or reduce over an empty
    dimension.

    * @attention Constraints:
    * This operator will be deprecated in the future. Replace it with LpNormReduceV2 operator.

    * @par Third-party framework compatibility
    * Compatible with the Pytorch operator LpNormReduce.
    */
    REG_OP(LpNormReduce)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(p, Int, 2)
    .ATTR(axes, ListInt, {})
    .ATTR(keepdim, Bool, false)
    .ATTR(epsilon, Float, 1e-12f)
    .OP_END_FACTORY_REG(LpNormReduce)

    /**
    * @brief Computes LpNormReduce.

    * @par Inputs:
    * x: A ND tensor of dtype float16, bfloat16, float32.
    *
    * @par Attributes:
    * @li p: An optional float, "inf" or "-inf", indicates the order of norm. Default is 2.0.
    * @li axes: ListInt, an optional attribute, indicates dimensions over which to compute the norm.
    * Default is {}, meaning all axes will be computed.
    * @li keepdim: An optional bool. If set to true, the reduced dimensions are retained in the result
    * as dimensions with size one. Default is false.
    * @li epsilon: An optional float. A value added to the denominator for numerical stability. Default is 1e-12.

    * @par Outputs:
    * y: A ND tensor has the same dtype as "x". The shape of "y" is depending on "axes" and "keepdim".

    * @attention Constraints:
    * @li When the attribute "p" is negative, there may be precision difference in the calculation results.
    * @li When the attribute "axes" is specified as the axis with a shape dimension value of 1 in the input tensor,
    * there may be precision difference in the calculation results.
    * @li When the tensor "x" is empty and "p" < 0 or "p" is infinity, we cannot reduce the whole tensor or reduce
    * over an empty dimension.

    * @par Third-party framework compatibility
    * Compatible with the Pytorch operator LpNormReduce.
    */
    REG_OP(LpNormReduceV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(p, Float, 2.0)
    .ATTR(axes, ListInt, {})
    .ATTR(keepdim, Bool, false)
    .ATTR(epsilon, Float, 1e-12f)
    .OP_END_FACTORY_REG(LpNormReduceV2)

    /**
    * @brief Computes LpNormUpdate.

    * @par Inputs:
    * x: A ND tensor of type float16, bfloat16, float32.
    *
    * @par Attributes:
    * @li p: An optional int, "inf" or "-inf", default value is 2, p >= 0.
    * @li epsilon: Float, default is 1e-12.

    * @par Outputs:
    * y: A ND tensor has the same shape and dtype as "x".

    * @attention Constraints:
    * This operator will be deprecated in the future. Replace it with LpNormUpdateV2 operator.

    * @par Third-party framework compatibility
    * Compatible with the Pytorch operator LpNormUpdate.
    */
    REG_OP(LpNormUpdate)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(p, Int, 2)
    .ATTR(epsilon, Float, 1e-12f)
    .OP_END_FACTORY_REG(LpNormUpdate)

    /**
    * @brief Computes LpNormUpdate.

    * @par Inputs:
    * x: A ND tensor of dtype float16, bfloat16, float32.
    *
    * @par Attributes:
    * @li p: An optional float, "inf" or "-inf", indicates the order of norm. Default is 2.0.
    * @li epsilon: An optional float. A value added to the denominator for numerical stability. Default is 1e-12.

    * @par Outputs:
    * y: A ND tensor has the same shape and dtype as "x".

    * @attention Constraints:
    * When the attribute "p" is negative, there may be precision difference in the calculation results.

    * @par Third-party framework compatibility
    * Compatible with the Pytorch operator LpNormUpdate.
    */
    REG_OP(LpNormUpdateV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(p, Float, 2.0)
    .ATTR(epsilon, Float, 1e-12f)
    .OP_END_FACTORY_REG(LpNormUpdateV2)

    /**
    *@brief Performs Region of Interest (ROI) Pooling . \n

    *@par Inputs:
    * Three inputs, including:
    *@li x: A tensor of type float16 or float32, describing the feature
    * map. The data of x must be greater than or equal to "0.0".
    *@li rois: A tensor of type float16 or float32, with 3D shape
    * [batch, 5, roi_max_num], describing the RIOs. Each ROI consists of five
    * elements: "batch_id", "x1", "y1", "x2", and "y2", which "batch_id" indicates
    * the index of the input feature map, "x1", "y1", "x2", or "y2" must be
    * greater than or equal to "0.0".
    * roi_max_num must be less than or equal to 6000 and must be divided by 16.
    * The input data of the rois cannot exceed the width and height range of the x,
    * otherwise, the accuracy of the output result may not be as expected.
    *@li roi_actual_num: A  optional tensor of type int32, with shape [batch, 8], specifying
    * the number of ROIs per batch . \n

    *@par Attributes:
    *@li pooled_h: A required int32, specifying the pooled H. Must be greater
    * than 0.
    *@li pooled_w: A required int32, specifying the pooled W. Must be greater
    * than 0.
    *@li spatial_scale_h: An required scaling factor for mapping the input
    * coordinates of height to the ROI coordinates.
    *@li spatial_scale_w: An required scaling factor for mapping the input
    * coordinates of width to the ROI coordinates . \n

    *@par Outputs:
    *y: A tensor of type float16 or float32, describing the result
    * feature map . \n

    *@attention Constraints:
    * For the feature map input:
    *@li If pooled_h = pooled_w = 2, the feature map size must not exceed 50.
    *@li If pooled_h = pooled_w = 3, the feature map size must not exceed 60.
    *@li If pooled_h = pooled_w = 4, the feature map size must not exceed 70.
    *@li If pooled_h = pooled_w = 5, the feature map size must not exceed 70.
    *@li If pooled_h = pooled_w = 6, the feature map size must not exceed 80.
    *@li If pooled_h = pooled_w = 7, the feature map size must not exceed 80.
    *@li If pooled_h = pooled_w = 8, the feature map size must not exceed 80.
    *@li If pooled_h = pooled_w = 9, the feature map size must not exceed 70.
    *@li If pooled_h = pooled_w = 10, the feature map size must not exceed 70.
    *@li If pooled_h = pooled_w = 11, the feature map size must not exceed 70.
    *@li If pooled_h = pooled_w = 12, the feature map size must not exceed 70.
    *@li If pooled_h = pooled_w = 13, the feature map size must not exceed 70.
    *@li If pooled_h = pooled_w = 14, the feature map size must not exceed 70.
    *@li If pooled_h = pooled_w = 15, the feature map size must not exceed 70.
    *@li If pooled_h = pooled_w = 16, the feature map size must not exceed 70.
    *@li If pooled_h = pooled_w = 17, the feature map size must not exceed 50.
    *@li If pooled_h = pooled_w = 18, the feature map size must not exceed 40.
    *@li If pooled_h = pooled_w = 19, the feature map size must not exceed 40.
    *@li If pooled_h = pooled_w = 20, the feature map size must not exceed 40.
    *@par Third-party framework compatibility
    * It is a custom operator. It has no corresponding operator in Caffe.
    */
    REG_OP(ROIPooling)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(rois, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(roi_actual_num, TensorType({DT_INT32}))
    .REQUIRED_ATTR(pooled_h, Int)
    .REQUIRED_ATTR(pooled_w, Int)
    .REQUIRED_ATTR(spatial_scale_h, Float)
    .REQUIRED_ATTR(spatial_scale_w, Float)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ROIPooling)

    /**
    *@brief Performs Position Sensitive PS ROI Pooling . \n

    *@par Inputs:
    * Two inputs, including:
    *@li x: A tensor of type float16 or float32, describing the feature
    * map, dimension C1 must be equal to
    * (int(output_dim+15)/C0))*group_size*group_size.
    *@li rois: A tensor of type float16 or float32, with shape
    * [batch, 5, rois_num], describing the ROIs, each ROI consists of five
    * elements: "batch_id", "x1", "y1", "x2", and "y2", which "batch_id" indicates
    * the index of the input feature map, "x1", "y1", "x2", or "y2" must be
    * greater than or equal to "0.0" . \n

    *@par Attributes:
    *@li output_dim: A required int32, specifying the number of output channels,
    * must be greater than 0.
    *@li group_size: A required int32, specifying the number of groups to encode
    * position-sensitive score maps, must be within the range (0, 128).
    *@li spatial_scale: A required float32, scaling factor for mapping the input
    * coordinates to the ROI coordinates . \n

    *@par Outputs:
    *y: A tensor of type float16 or float32, describing the result
    * feature map . \n

    *@attention Constraints:
    * HC1HWC0: channel must be Group_size squared, rois_num is a multiple of 16
    */
    REG_OP(PSROIPoolingV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(spatial_scale, Float)
    .REQUIRED_ATTR(output_dim, Int)
    .REQUIRED_ATTR(group_size, Int)
    .OP_END_FACTORY_REG(PSROIPoolingV2)

    /**
    * @brief Performs max_pool_ext2 on the input .

    * @par Inputs:
    * One input:
    * x: A Tensor of type: float16, float32, float64, int8, int16, int32, int64, uint8, uint16, qint8.


    * @par Attributes:
    * @li ksize: A required list of int8, int16, int32, or int64 values,
    * specifying the size of the window for each dimension of the input tensor. No default value.
    * @li strides: A required list of int8, int16, int32, or int64 values,
    * specifying the stride of the sliding window for each dimension of the input tensor. No default value.
    * @li padding: A required string. No default value.
    * @li data_format: An optional string . \n

    * @par Outputs:
    * y: A Tensor. Has the same type and format as input "x" . \n

    * @attention Constraints:
    * @li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1, ksize[1] * ksize[2] <= 255.
    * @li "stride" is a list that has length 4: strides[0] = 1 or strides[3] = 1,
    * strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
    * @li "padding" is either "SAME" or "VALID" . \n

    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator MaxPoolV2.
    */
    REG_OP(MaxPoolExt2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
                          DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                           DT_UINT16, DT_QINT8}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolExt2)

    /**
    * @brief Computes second-order gradients of the maxpooling function .

    * @par Inputs:
    * @li x1: Original forward input tensor. Supported type:float, double, int32,
     * uint8, int16, int8, int64, uint16, float16, uint32, uint64.
    * @li x2: Has the same type and format as input "x1".
    * @li grad:Has the same type and format as input "x1" . \n

    * @par Attributes:
    * @li ksize: A required list or tuple,
    * specifying the size of the sliding window.
    * @li strides: A required list or tuple,
    * specifying the stride of the sliding window.
    * @li padding: A required string, window sliding mode. Either SAME or VALID.
    * @li data_format: An optional string.
    * Format of the original input, either NCHW or NHWC. Defaults to NHWC . \n

    * @attention Constraints:
    * @li Only Atlas Training Series Product is supported.
    * @li "x1" and "grads" must have the same shape.
    * @li "x2" and "y" must have the same shape. Otherwise, an error is reported.
    * @li "x1", "x2", "grads", and "y" must be 5D tensors.
    * @li ksize[H] and ksize[W] is in the range [1, 255].
    * @li strides[H] and strides[W] is in the range [1, 63].
    * @li Other dimensions of ksize and strides is 1 . \n

    * @par Outputs:
    * y: Has the same type and format as input "x1" . \n

    * @par Third-party framework compatibility
    * @li Compatible with the TensorFlow operator MaxPoolGradGrad.
    */
    REG_OP(MaxPoolGradGrad)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolGradGrad)

    /**
    * @brief Computes second-order gradients of the maxpooling function .

    * @par Inputs:
    * @li x: Original forward input tensor. Supported type: float16, Support format: NC1HWC0.
    * @li grad: Gradient tensor. Supported type: float16, Support format: NC1HWC0.
    * @li argmax: An tensor of type uint16 or int64, Support format: NC1HWC0.
    * @par Attributes:
    * @li ksize: A required list, specifying the size of the sliding window.
    * @li strides: A required list, specifying the stride of the sliding window.
    * @li padding: A required string, window sliding mode. Either SAME or VALID.
    * @par Outputs:
    * y:Result tensor. Supported type: float16, Support format: NC1HWC0.

    * @attention Constraints:
    * @li Only the cloud platform is supported.
    * @li "x1" and "grads" must have the same shape.
    * @li length of the shape of x, grads, argmax, y must be 5.
    * @li shape of argmax must be (fmap_n, fmap_c1, kernel_h * kernel_w,
    * (shape_max_pool[2] * shape_max_pool[3] + 15) // 16 * 16, 1),
    * or (fmap_n, fmap_c1, kernel_h * kernel_w,
    * (shape_max_pool[2] * shape_max_pool[3] + 31) // 16, 16), else failed . \n

    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator MaxPoolGradGradWithArgmax.
    */
    REG_OP(MaxPoolGradGradWithArgmax)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .INPUT(argmax, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .OP_END_FACTORY_REG(MaxPoolGradGradWithArgmax)

    /**
    * @brief Performs max pooling on the input and outputs both max values and indices .

    * @par Inputs:
    * One input:
    * x: An 5hd Tensor of type float16.
    * Must set the format, supported format list ["NC1HWC0"].
    * @par Attributes:
    * @li ksize: A required list of int8, int16, int32, or int64 values,
    * specifying the size of the window for each dimension of the input tensor. No default value.
    * @li strides: A required list of int8, int16, int32, or int64 values,
    * specifying the stride of the sliding window for each dimension of the input tensor. No default value.
    * @li pads: A required list of int8, int16, int32, or int64 values,
    * specifying the pad of the input feature map. No default value. \n
    * @li dtype: A optional int. default value is 3.
    * @li dilation: A optional list of int8, int16, int32, or int64 values.
    * @li ceil_mode: A optional bool. default value is false . \n

    * @par Outputs:
    * y: A Tensor. Has the same type and format as input "x".
    * argmax:  A Tensor. type:uint16.
    * @attention Constraints:
    * @li ksize: a list that has length 4:
    * ksize[0] = 1, ksize[1] = 1, ksize[2] * ksize[3] <= (ub_size-8)*1024//6//2//16.
    * @li strides: a list that has length 4:
    * strides[0] = 1, strides[1] = 1, 1 <= strides[2] <= 2048, 1 <= strides[3] <= 2048.
    * @li pads: a list that has length 4:
    * pads[0] = 1, pads[1] = 1, 1 <= pads[2] <= (ksize[2]//2), 1 <= pads[3] <= (ksize[3]//2).
    * @li dilation: a list that has length 4.
    * @li ceil_mode: is a bool, default is false . \n

    * @par Third-party framework compatibility
    * Compatible with the PyTorch operator max_pool2d_with_indices.
    */
    REG_OP(MaxPoolWithArgmaxV2)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OUTPUT(argmax, TensorType({DT_UINT16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dtype, Int, 3)
    .ATTR(dilation, ListInt, {1, 1, 1, 1})
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolWithArgmaxV2)

    /**
    * @brief Performs the backpropagation of MaxPoolWithArgmaxV2.

    * @par Inputs:
    * Three inputs, including:
    * @li x: An 5hd tensor of type float16.
    * Must set the format, supported format list ["NC1HWC0"]
    * @li grad: An 5hd tensor of type float16.
    * Must set the format, supported format list ["NC1HWC0"]
    * @li argmax: An 5hd tensor of type uint16 or int64.
    * Must set the format, supported format list ["NC1HWC0"] \n
    * For Ascend 950 AI Processor: The uint16 data type is not supported.

    * @par Attributes:
    * @li ksize: A required list of int8, int16, int32, or int64 values,
    * specifying the size of the window for each dimension of the input tensor. No default value.
    * @li strides: A required list of int8, int16, int32, or int64 values,
    * specifying the stride of the sliding window for each dimension of the input tensor. No default value.
    * @li pads: A required list of int8, int16, int32, or int64 values,
    * specifying the pad of the input feature map. No default value. \n
    * @li dtype: A optional int. default value is 3.
    * @li dilation: A optional list of int8, int16, int32, or int64 values.
    * @li ceil_mode: A optional bool. default value is false. \n

    * @par Outputs:
    * y: A Tensor. Has the same type and format as input "x". \n

    * @attention Constraints:
    * @li ksize: a list that has length 4:
    * ksize[0] = 1, ksize[1] = 1, ksize[2] * ksize[3] <= (ub_size-8)*1024//7//2//16.
    * @li strides: a list that has length 4:
    * strides[0] = 1, strides[1] = 1, 1 <= strides[2] <= 2048, 1 <= strides[3] <= 2048.
    * @li pads: a list that has length 4:
    * pads[0] = 1, pads[1] = 1, 1 <= pads[2] <= (ksize[2]//2), 1 <= pads[3] <= (ksize[3]//2).
    * @li dilation: a list that has length 4.
    * @li ceil_mode: is a bool, default is false. \n

    * @see max_pool_grad_with_argmaxv2
    * @par Third-party framework compatibility
    * Compatible with the PyTorch backward operator of max_pool2d_with_indices.
    */
    REG_OP(MaxPoolGradWithArgmaxV2)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT16}))
    .INPUT(argmax, TensorType({DT_UINT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dtype, Int, 3)
    .ATTR(dilation, ListInt, {1, 1, 1, 1})
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolGradWithArgmaxV2)

    /**
    * @brief Computes gradients of the maxpooling function .

    * @par Inputs:
    * @li orig_input: Original forward input tensor. Support type: float16, float32, Support format:[NCHW, NHWC].
    * @li orig_output: Has the same shape and type as "x1", Support format:[NCHW, NHWC].
    * @li grad: Has the same shape and type as "x1", Support format:[NCHW, NHWC]. \n

    * @par Attributes:
    * @li ksize: A required list of int8, int16, int32, or int64 values,
    * specifying the size of the window for each dimension of the input tensor.
    * No default value.
    * @li strides: A required list of int8, int16, int32, or int64 values,
    * specifying the stride of the sliding window for each dimension of
    * the input tensor. No default value.
    * @li padding_mode: A required string. Defaults to "CALCULATED".
    * @li pads:A required list of int8, int16, int32, or int64 values,
    * a data to caculate when padding_mode is "CALCULATED".
    * @li data_format: An optional string. Defaults to "NHWC" .
    * @li global_pooling: An optional bool. Whether to use the global pooling.
    * If global_pooling = true, kernel size and paddings will be ignored.
    * Default False. When the input parameters are set to float16, global_pooling does not support being set to true.
    * @li ceil_mode: An optional bool. Whether to use the ceil function to calculate output
    * height and width. If it is set to False, the floor function will be used. Default False \n

    * @par Outputs:
    * out_grad: A mutable tensor. Has the same shape, type and format as "x1" . \n

    * @attention Constraints:
    * @li Computing gradients of global pooling is not supported, which means
    * "ksize < x1".
    * @li "ksize" is in the range [1, 255]. "strides" is in the range [1, 63]
    * @li in static situation, orig_input, orig_output, grad and y cannot support float32.

    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator MaxPoolGrad.
    */
    REG_OP(MaxPoolV3Grad)
    .INPUT(orig_input, TensorType::RealNumberType())
    .INPUT(orig_output, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .OUTPUT(out_grad, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling, Bool, false)
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolV3Grad)

    /**
     *@brief Updates '*var' according to the Adam algorithm..
     *   lr_t := {learning_rate} * sqrt{1 - beta_2^t} / (1 - beta_1^t)
     *   m_t := beta_1 * m_{t-1} + (1 - beta_1) * g
     *   v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g
     *   vhat_t := max{vhat_{t-1}, v_t}
     *   variable := variable - lr_t * m_t / (sqrt{vhat_t} + epsilon)
     *
     *@par Inputs:
     *@li var: A mutable tensor. Must be one of the data types defined in
     *    TensorType::NumberType(). Should be from a Variable().
     *@li m: A mutable tensor. Has the same type as "var". Should be from a
     *    Variable().
     *@li v: A mutable tensor. Has the same type as "var". Should be from a
     *    Variable().
     *@li vhat: A mutable tensor. Has the same type as "var". Should be from a
     *    Variable().
     *@li beta1_power: A mutable tensor. Has the same type as "var". Should be from a
     *    Variable().
     *@li beta2_power: A mutable tensor. Has the same type as "var". Should be from a
     *    Variable().
     *@li lr: A tensor for the learning rate. Has the same type as "var". Should be
     *    from a Variable().
     *@li grad: A tensor for the gradient. Has the same type as "var". Should be
     *    from a Variable().
     *
     *@par Attributes:
     *@li beta1: A scalar. Has the same type as "var".
     *@li beta2: A scalar. Has the same type as "var".
     *@li epsilon: A scalar. Has the same type as "var".
     *@li use_locking: An optional bool. Defaults to "False".
     *    If "True", updating of the "var" tensor is protected by a lock;
     *    otherwise the behavior is undefined, but may exhibit less contention.
     *
     *@par Outputs:
     *@li var: A mutable tensor. Has the same type as input "var".
     *@li m: A mutable tensor. Has the same type as input "var"
     *@li v: A mutable tensor. Has the same type as input "var"
     *@li vhat: A mutable tensor. Has the same type as input "var"
     *
     *@attention Constraints:
     * The input tensors must have the same shape.
     *
     *@par Third-party framework compatibility
     * Compatible with the TensorFlow operator ResourceApplyKerasMomentum.
     *
     */
    REG_OP(ApplyAdamWithAmsgrad)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(vhat, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamWithAmsgrad)

    /**
     *@brief Updates "var" according to the AddSign update.
     *  t-1 mean previous period.
     *  m_t <- beta1 * m_{t-1} + (1 - beta1) * grad
     *  update <- exp(logbase * sign_decay * sign(grad) * sign(m_t)) * grad
     *  var <- var - lr * update
     *
     *@attention Constraints:
     *  the input tensors must have the same shape.
     *
     *@par Inputs:
     *@li var: A mutable tensor. Should be from a Variable().
     *@li m: A mutable tensor. Has the same type as "var".
     *     Should be from a Variable().
     *@li lr: A scalar. Has the same type as "var".
     *@li logbase: A scalar. Has the same type as "var".
     *@li sign_decay: A scalar. Has the same type as "var".
     *@li beta: A scalar. Has the same type as "var".
     *@li grad: A tensor for the gradient. Has the same type as "var".
     *
     *@par Attributes:
     * use_locking: An optional bool. Defaults to "False".
     *     If "True", updating of the "var", "ms", and "mom" tensors is protected
     *     by a lock; otherwise the behavior is undefined, but may exhibit less
     *     contention.
     *
     *@par Outputs:
     * var: A mutable tensor. Has the same type as input "var".
     *
     *@par Third-party framework compatibility
     *Compatible with the TensorFlow operator ApplyPowerSign.
     *
     */
    REG_OP(ApplyPowerSign)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(logbase, TensorType::NumberType())
    .INPUT(sign_decay, TensorType::NumberType())
    .INPUT(beta, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyPowerSign)

    /**
     *@brief Updates "var" according to the adagrad scheme.
     *   accum += grad * grad
     *   var -= lr * grad * (1 / sqrt(accum))
     *
     *@attention Constraints:
     *@li The input and output tensors must have the same shape.
     *
     *@par Inputs:
     *@li var: A mutable tensor. Should be from a Variable(). Support float16, bfloat16 and float32.
     *@li accum: A mutable tensor. Has the same type as "var".
     *     Should be from a Variable().
     *@li lr: A scalar. Has the same type as "var".
     *@li grad: A tensor for the gradient. Has the same type as "var".
     *
     *@par Attributes:
     *@li update_slots: An optional bool. Defaults to "True". If "True", the accum tensor will be updated.
     *@li use_locking: An optional bool. Defaults to "False".
     *     If "True", updating of the "var", "ms", and "mom" tensors is protected
     *     by a lock; otherwise the behavior is undefined, but may exhibit less
     *     contention.
     *
     *@par Outputs:
     * var: A mutable tensor. Has the same type as input "var".
     *
     *@par Third-party framework compatibility
     *Compatible with the TensorFlow operator ApplyAdagrad.
     *
     */
    REG_OP(ApplyAdagrad)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(update_slots, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagrad)

    /**
     * @brief Updates "var" according to the adagradv2 scheme.
     *   accum += grad * grad
     *   var -= lr * grad * (1 / sqrt(accum) + epsilon)
     *
     * @par Inputs:
     * @li var: A mutable tensor. Must be one of the data types defined in
     * TensorType::NumberType(). Should be from a Variable().
     * @li accum: A mutable tensor. Has the same type as "var". Should be from a
     * Variable().
     * @li lr: A tensor for the learning rate. Has the same type as "var". Should be
     * from a Variable().
     * @li grad: A tensor for the gradient. Has the same type as "var". Should be
     * from a Variable().
     * @li epsilon: A scalar. Has the same type as "var".
     *
     * @par Attributes:
     * @li update_slots: An optional bool. Defaults to "True".
     * If "True", "accum" will be updated
     * @li use_locking: An optional bool. Defaults to "False".
     * If "True", updating of the "var" tensor is protected by a lock;
     * otherwise the behavior is undefined, but may exhibit less contention.
     *
     * @par Outputs:
     * var: A mutable tensor. Has the same type as input "var".
     *
     * @attention Constraints:
     * The input tensors must have the same shape.
     *
     * @par Third-party framework compatibility
     * Compatible with the TensorFlow operator ApplyAdagrad.
     *
     */
    REG_OP(ApplyAdagradV2)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(update_slots, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagradV2)

    /**
    *@brief Updates "var" according to the proximal adagrad scheme . \n

    *@par Inputs:
    *Eight inputs, including:
    * @li var: A mutable Tensor. Must be one of the following types:
    *     TensorType::NumberType(). Should be a Variable Tensor.
    * @li gradient_accumulator: A mutable Tensor. Must have the same
    *     type as "var". Should be a Variable Tensor.
    * @li gradient_squared_accumulator: A mutable Tensor of the same type as "var".
    *     Should be a Variable Tensor.
    * @li grad: A Tensor of the same type as "var", for the gradient.
    * @li lr: A Tensor of the same type as "var".
    *     Scaling factor. Must be a scalar.
    * @li l1: A Tensor of the same type as "var".
    *     L1 regulariation. Must be a scalar.
    * @li l2: A Tensor of the same type as "var".
    *     L2 regulariation. Must be a scalar.
    * @li global_step: A Tensor of type int32 or int64.
    *     Training step number. Must be a scalar . \n

    *@par Attributes:
    *use_locking: An optional bool. Defaults to "False".
    *     If "True", updating of the var and accum tensors will be
    *     protected by a lock; otherwise the behavior is undefined,
    *     but may exhibit less contention . \n

    *@par Outputs:
    *var: A mutable Tensor. Has the same type as "var" . \n

    *@par Third-party framework compatibility
    *Compatible with the TensorFlow operator ApplyAdagradDA.
    */
    REG_OP(ApplyAdagradDA)
    .INPUT(var, TensorType::NumberType())
    .INPUT(gradient_accumulator, TensorType::NumberType())
    .INPUT(gradient_squared_accumulator, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagradDA)

    /**
    * @brief Implements stochastic gradient descent (optionally with momentum).
    * Nesterov momentum is based on the formula from
    * On the importance of initialization and momentum in deep learning.

    * @par Inputs:
    * @li parameters: A mutable tensor of type float16, float32 or bfloat16.
    * Support format: [NC1HWC0,NDC1HWC0,ND,FRACTAL_Z,FRACTAL_Z_3D].
    * Specifies the iterable of parameters to optimize or dicts defining parameter
    * groups.
    * @li gradient: A tensor of type float16, float32 or bfloat16.
    * Support format: [NC1HWC0,NDC1HWC0,ND,FRACTAL_Z,FRACTAL_Z_3D].
    * Specifies the gradient of training step.
    * @li learning_rate: A tensor of type float16, float32 or bfloat16.
    * Support format: [ND].
    * Specifies the learing_rate of training step.
    * @li accum: A tensor of type float16, float32 or bfloat16.
    * Support format: [NC1HWC0,NDC1HWC0,ND,FRACTAL_Z,FRACTAL_Z_3D].
    * Specifies the velocity of training step.
    * @li momentum: A tensor of type float16, float32 or bfloat16.
    * Support format: [ND].
    * Specifies the momentum factor.
    * @li stat: A tensor of type float16, float32 or bfloat16.
    * Support format: [NC1HWC0,NDC1HWC0,ND,FRACTAL_Z,FRACTAL_Z_3D].
    * Specifies the status representing the first step or not . \n

    * @par Attributes:
    * @li dampening: An optional float, specifying the dampening for momentum.
    * Defaults to "0.0".
    * @li weight_decay: An optional float, specifying the L2 penalty. Defaults to
    * "0.0".
    * @li nesterov: An optional bool, specifying whether to enable Nesterov
    * momentum. Defaults to "False" . \n

    * @par Outputs:
    * parameters: Tensor of the same type and format as input "parameters" . \n

    * @see ApplyMomentum()

    * @par Third-party framework compatibility
    * @li Compatible with the PyTorch operator SGD.
    */
    REG_OP(SGD)
    .INPUT(parameters, TensorType(DT_FLOAT, DT_FLOAT16, DT_BF16))
    .INPUT(gradient, TensorType(DT_FLOAT, DT_FLOAT16, DT_BF16))
    .INPUT(learning_rate, TensorType(DT_FLOAT, DT_FLOAT16, DT_BF16))
    .INPUT(accum, TensorType(DT_FLOAT, DT_FLOAT16, DT_BF16))
    .INPUT(momentum, TensorType(DT_FLOAT, DT_FLOAT16, DT_BF16))
    .INPUT(stat, TensorType(DT_FLOAT, DT_FLOAT16, DT_BF16))
    .OUTPUT(parameters, TensorType(DT_FLOAT, DT_FLOAT16, DT_BF16))
    .ATTR(dampening, Float, 0.0)
    .ATTR(weight_decay, Float, 0.0)
    .ATTR(nesterov, Bool, false)
    .OP_END_FACTORY_REG(SGD)

    /**
     *@brief Updates '*var' according to the momentum scheme.
     *   accum = accum * momentum - x1 * x2 * lr
     *   if use_nesterov is True:
     *       var += accum * momentum - x1 * x2 * lr
     *   else:
     *       var += accum
     *
     *@par Inputs:
     *@li var: A mutable tensor. Should be from a Variable(). Supported dtype: float32.
     *    Supported format: NC1HWC0, C1HWNCoC0, ND, FRACTAL_Z.
     *@li accum: A mutable tensor. Has the same shape, data type, and format as "var".
     *    Should be from a Variable(). Supported dtype: float32
     *@li x1: A mutable Tensor. Has the same shape, data type, and format as "var".
     *    Should be from a Variable(). Supported dtype: float32
     *@li momentum: A scalar. Has the same data type as "var". Supported dtype: float32
     *@li x2: A scalar has the same data type as "var". Supported dtype: float32
     *@li lr: A scalar. has the same data type as "var". Supported dtype: float32
     *
     *@par Attributes:
     *@li use_nesterov: An optional bool. Defaults to "False".
     *    If "True", var will be updated by using Nesterov momentum.
     *@li use_locking: An optional bool. Defaults to "False".
     *    If "True", updating of the "var" tensor is protected by a lock;
     *    otherwise the behavior is undefined, but may exhibit less contention.
     *
     *@par Outputs:
     * @li var: A mutable tensor. Has the same data type, shape, and format as input "var".
     * @li accum: A mutable tensor. Has the same data type, shape, and format as input "accum".
     *
     *@attention Constraints:
     * @li var: A mutable tensor. Has the same type as input "var".
     * @li accum: A mutable tensor. Has the same type as input "accum".
     *
     *@par Third-party framework compatibility
     * Compatible with the TensorFlow operator ResourceApplyKerasMomentum.
     *
     */
    REG_OP(FusedMulApplyKerasMomentum)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(x1, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .INPUT(x2, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .ATTR(use_nesterov, Bool, false)
    .OP_END_FACTORY_REG(FusedMulApplyKerasMomentum)

    /**
    *@brief Update "g" according to the LARS algorithm . \n

    *@par Inputs:
    *Six inputs, including:
    * @li w: A ND Tensor. Must be of type float32
    * @li g: A ND Tensor of the same type and shape as "w".
    * @li w_square_sum: A 1D Tensor of  square_sum(w), has the same type as "w",  Must be a scalar or 1D tensor.
    * @li g_square_sum: A 1D Tensor of  square(g), has the same type as "w", Must be a scalar or 1D tensor.
    * @li weight_decay: A 1D Tensor of the same type as "w",  Must be a scalar or 1D tensor.
    * @li learning_rate: A 1D Tensor of the same type as "w", Must be a scalar or 1D tensor. \n

    *@par Attributes:
    *Three Attributes, including:
    * @li hyperpara: An optional float. Default value is 0.001.
    * @li epsilon: An optional float. Default value is 1e-5.Avoid denominator is 0.
    * @li use_clip: An optional bool. Defaults to "False".
    *     If "True", updating learning rate . \n

    *@par Outputs:
    *g_new: a ND Tensor of the same type as "w".
    */
    REG_OP(LarsV2Update)
    .INPUT(w, TensorType(DT_FLOAT))
    .INPUT(g, TensorType(DT_FLOAT))
    .INPUT(w_square_sum, TensorType(DT_FLOAT))
    .INPUT(g_square_sum, TensorType(DT_FLOAT))
    .INPUT(weight_decay, TensorType(DT_FLOAT))
    .INPUT(learning_rate, TensorType(DT_FLOAT))
    .OUTPUT(g_new, TensorType(DT_FLOAT))
    .ATTR(hyperpara, Float, 0.001)
    .ATTR(epsilon, Float, 0.00001)
    .ATTR(use_clip, Bool, false)
    .OP_END_FACTORY_REG(LarsV2Update)

    /**
    *@brief Finds unique elements in a 1D tensor. \n

    *@par Inputs:
    *x: 1D tensor. Support all types mentioned in TensorType.
    *Input "x" is a k-dimensional tensor. \n

    *@par Attributes:
    *out_idx: A required DType from: "int32, int64". \n

    *@par Outputs:
    *@li y: A Tensor. Has the same type as "x".
    *@li idx: A Tensor of type "out_idx".
    *@li count: A Tensor of type "out_idx". \n

    *@attention Constraints:
    *@li UniqueWithCounts runs on the Ascend AI CPU, which delivers poor performance. \n
    *@li Dtype bfloat16, uint32, uint64 only support Ascend 950 AI Processor. \n

    *@par Third-party framework compatibility
    *Compatible with the TensorFlow operator UniqueWithCounts.
    */
    REG_OP(UniqueWithCounts)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                          DT_STRING, DT_BF16, DT_UINT32, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                           DT_STRING, DT_BF16, DT_UINT32, DT_UINT64}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(count, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(out_idx, Type)
    .OP_END_FACTORY_REG(UniqueWithCounts)

    /**
    *@brief Finds unique elements in a 1D tensor. \n

    *@par Inputs:
    *x: 1D tensor. Support all types mentioned in TensorType.
    *Input "x" is a 1D tensor. \n

    *@par Attributes:
    *out_idx: An optional DType from: "int32, int64". Defaults to "int32". \n

    *@par Outputs:
    *@li y: "x" in the unique output "y".
    *@li idx: A tensor the same size as "x". The index of each value of "x". \n

    *@attention Constraints:
    *@li Unique runs on the Ascend AI CPU, which delivers poor performance. \n
    *@li Dtype bfloat16, uint32, uint64 only support Ascend 950 AI Processor. \n

    *@par Third-party framework compatibility
    *Compatible with the TensorFlow operator Unique.
    */
    REG_OP(Unique)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE,
                          DT_BF16, DT_UINT32, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE,
                           DT_BF16, DT_UINT32, DT_UINT64}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_idx, Type, DT_INT32)
    .OP_END_FACTORY_REG(Unique)

    /**
    *@brief Gather selected dims of input which returns the shape of tensor shape after gathershapes.\n

    *@par Inputs:
    *x: A list of input tensors. All data types are supported. It's a dynamic input. \n

    *@par Attributes:
    *@li axes: An 2-D list of int32 or int64 required. Select some dims of input.
    *@li dtype: An optional int32, which indicates the data type of output. Defaults to DT_INT32. \n

    *@par Outputs:
    *shape: The shape of tensor shape after gathershapes. Must be one of the following types: int32、int64. \n
    */
    REG_OP(GatherShapes)
    .DYNAMIC_INPUT(x, TensorType::ALL())
    .OUTPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(axes, ListListInt)
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(GatherShapes)

    /**
    *@brief Returns a tensor containing the indices of all non-zero elements of
    *input.

    *@par Inputs:
    *x: A Tensor. Must be one of the following types: float16, float32, int32,
    *int64, double, int8, uint8, int16, uint16, uint32, uint64, bool.
    *Supported format "ND". \n

    *@par Attributes:
    *@li transpose: The output tensor will be transposed if true. Defaults to False.
    *@li dtype: Must be one of the following types: int32. Defaults to `int32`. \n

    *@par Outputs:
    *@li value: A Tensor. Has the same type as "x" .
    *@li index: A Tensor. The type is INT32, means index for input.
    *@li count: A Scalar. The type is INT32, means count for non_zero ele in input. \n

    *@par Third-party framework compatibility
    *Compatible with the PyTorch operator NonZeroWithValue.
    */
    REG_OP(NonZeroWithValue)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                          DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(value, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                               DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(index, TensorType({DT_INT32}))
    .OUTPUT(count, TensorType({DT_INT32}))
    .ATTR(transpose, Bool, false)
    .ATTR(dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(NonZeroWithValue)

    /**
    *@brief Computes the inverse of one or more square invertible matrices or
    their adjoints (conjugate transposes) . \n

    *@par Inputs:
    *The input x is a tensor of shape [..., M, M] whose inner-most 2 dimensions
    form square matrices. Inputs include:
    *x:A Tensor of input. Shape is [..., M, M] . \n

    *@par Attributes:
    *adjoint:An optional bool. Defaults to False.Boolean indicating whether to
    deal with matrix or its (block-wise) adjoint . \n

    *@par Outputs:
    *y:A Tensor. Has the same type as x . \n

    *@attention Constraints:
    *The input x is a tensor of shape [..., M, M] whose inner-most 2 dimensions
    form square matrices.  \n

    *@par Third-party framework compatibility
    *Compatible with tensorflow MatrixInverse operator.
    */
    REG_OP(MatrixInverse)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .ATTR(adjoint, Bool, false)
    .OP_END_FACTORY_REG(MatrixInverse)

    /**
    * @brief Performs reduced batch normalization .

    * @par Inputs:
    * x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
    * Indicates the input tensor, that is, the original data to be normalized.

    * @par Outputs:
    * @li sum: A 1D tensor of type float32 for SUM reduced "x". It represents the sum of the input tensor "x" on the C
    axis.
    * The shape of sum is consistent with the C axis of "x". Has the same format as "x".
    * @li square_sum: A 1D tensor of type float32 for SUMSQ reduced "x". It represents the sum of squares of the input
    tensor "x" on the C axis.
    * The shape of sum is consistent with the C axis of "x". Has the same format as "x". \n

    * @attention Constraints:
    * This operator is a BatchNorm fusion operator for updating the moving
    * averages for training.
    * This operator is used in conjunction with BNTrainingReduce.
    */
    REG_OP(BNTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingReduce)

    /**
    * @brief Performs reduced batch normalization .

    * @par Inputs:
    * x: A 5D tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW.
    * Represents the input tensor in batch normalization training.
    * When the C axis is 0, other dimensions support empty tensors; when the C axis is not 0, other dimensions do not
    support empty tensors.

    * @par Outputs:
    * @li sum: A 1D tensor of type float32 for SUM reduced "x". It represents the sum of the input tensor "x" on the C
    axis.
    * The shape of sum is consistent with the C axis of "x". Has the same format as "x".
    * @li square_sum: A 1D tensor of type float32 for SUMSQ reduced "x". It represents the sum of squares of the input
    tensor "x" on the C axis.
    * The shape of sum is consistent with the C axis of "x". Has the same format as "x". \n

    * @attention Constraints:
    * This operator is a BatchNorm fusion operator for updating the moving
    * averages for training.
    * This operator is used in conjunction with BN3DTrainingReduce.
    */
    REG_OP(BN3DTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BN3DTrainingReduce)

    /**
    * @brief Performs the backpropagation of BatchNorm .

    * @par Inputs:
    * Seven inputs, including:
    * @li grads: A 4D tensor of type float16 or float32 or bfloat16, for the gradient, with format NHWC or NCHW.
    * The gradient of the loss function with respect to the output of the batch normalization layer.
    * @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
    * It represents the data input to the batch normalization layer during the forward pass.
    * Has the same type, format and shape as "grads".
    * @li diff_scale: A 1D tensor of type float32, the shape is same as dim C of input grads.
    * Indicates the gradient of the loss function to the scaling parameter "scale".
    * Has the same format as "grads".
    * @li diff_offset: A 1D tensor of type float32, the shape is same as dim C of input grads.
    * Represents the gradient of the loss function to the offset parameter.
    * Has the same format as "grads".
    * @li scale: A 1D tensor of type float32, the shape is same as dim C of input grads.
    * The scaling parameter in batch normalization, used to adjust the normalized output.
    * Has the same format as "grads".
    * @li batch_mean: A 1D tensor of type float32, the shape is same as dim C of input grads, for the mean of "x".
    * Has the same format as "grads".
    * @li batch_variance: A 1D tensor of type float32, the shape is same as dim C of input grads, for the variance of
    "x".
    * Has the same format as "grads". \n

    * @par Attributes:
    * epsilon: An optional float32. Defaults to "0.0001".
    * Represents a small positive number added to the variance of "x" to prevent division by zero. \n

    * @par Outputs:
    * y: A Tensor of type float16, float32 or bfloat16, with format NHWC or NCHW.
    * It represents the gradient of the loss function with respect to the input data x.
    * Has the same type, format and shape as "grads". \n

    * @attention Constraints:
    * The preceding layer of this operator must be BNTrainingUpdateGrad . \n

    * @see BNTrainingUpdateGrad
    */
    REG_OP(BNTrainingReduceGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(diff_scale, TensorType({DT_FLOAT}))
    .INPUT(diff_offset, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(epsilon, Float, 0.0001)
    .OP_END_FACTORY_REG(BNTrainingReduceGrad)

    /**
    * @brief Performs the backpropagation of BatchNorm .

    * @par Inputs:
    * Seven inputs, including:
    * @li grads: A 5Dtensor of type float16 or float32 or bfloat16, for the gradient, with format NDHWC or NCDHW.
    * @li x: A 5D tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW.
    * @li diff_scale: A 1D tensor of type float32,
    * for the mean of "x". shape must be C channel.
    * @li diff_offset: A 1D tensor of type float32,
    * for the variance of "x". shape must be C channel.
    * @li scale: A 1D tensor of type float32.
    * @li batch_mean: A 1D tensor of type float32,
    * for the mean of "x". shape must be C channel.
    * @li batch_variance: A 1D tensor of type float32,
    * for the variance of "x" . shape must be C channel. \n

    * @par Attributes:
    * epsilon: An optional float32. Defaults to "0.0001". A small float number
    * added to the variance of "x" . \n

    * @par Outputs:
    * y: A 5D Tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW. \n

    * @attention Constraints:
    * The preceding layer of this operator must be BN3DTrainingReduceGrad . \n

    * @see BN3DTrainingReduceGrad
    */
    REG_OP(BN3DTrainingReduceGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(diff_scale, TensorType({DT_FLOAT}))
    .INPUT(diff_offset, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(epsilon, Float, 0.0001)
    .OP_END_FACTORY_REG(BN3DTrainingReduceGrad)

    /**
    * @brief Performs reduced batch normalization .

    * @par Inputs:
    * Seven inputs, including:
    * @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW. Empty tensors are not
    supported.
    * Input tensor, that is, the original data that needs to be normalized.
    * @li sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator
    BNTrainingReduce.
    * It represents the sum of the input tensor "x" on the C axis. Has the same format as "x".
    * @li square_sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator
    BNTrainingReduce.
    * It represents the sum of squares of the input tensor "x" on the C axis. Has the same format as "x".
    * @li scale: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling factor. Has the
    same format as "x".
    * @li offset: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling offset. Has the
    same format as "x".
    * @li mean: A 1D tensor of type float32, the shape is same as dim C of input "x", for the updated mean. Has the same
    format as "x".
    * @li variance: A 1D tensor of type float32, the shape is same as dim C of input "x", for the updated variance. Has
    the same format as "x". \n

    * @par Attributes:
    * @li epsilon: A required float32, specifying the small value added to variance
    * to avoid dividing by zero.
    * @li factor: A required float32, specifying the weight for updating the mean
    * and variance . \n

    * @par Outputs:
    * Five outputs, including:
    * @li y: A 4D tensor of type float16 or float32 or bfloat16, for normalized "x". Empty tensors are not supported.
    * Has the same dype, format and shape as "x".
    * @li mean: A 1D tensor of type float32, for the updated mean. shape must be C channel. Has the same format as "x".
    * @li variance: A 1D tensor of type float32, for the updated variance. shape must be C channel. Has the same format
    as "x".
    * @li batch_mean: A 1D tensor of type float32, for the mean of "x". shape must be C channel. Has the same format as
    "x".
    * @li batch_variance: A 1D tensor of type float32, for the variance of "x" . shape must be C channel. Has the same
    format as "x". \n

    * @attention Constraints:
    * @li This operator is a BatchNorm fusion operator for updating the moving
    * averages for training. This operator is used in conjunction with
    * BNTrainingUpdate.
    * @li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to the
    * square root instruction.
    */
    REG_OP(BNTrainingUpdate)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(factor, Float)
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdate)

    /**
    * @brief Performs reduced batch normalization .

    * @par Inputs:
    * Seven inputs, including:
    * @li x: A 5D tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW.
    * @li sum: A 1D tensor of type float32 for the output of operator BN3DTrainingUpdate. shape must be C channel.
    * @li square_sum: A 1D tensor of type float32 for the output of operator BN3DTrainingUpdate. shape must be C
    channel.
    * @li scale: A 1D tensor of type float32, for the scaling factor. shape must be C channel.
    * @li offset: A 1D tensor of type float32, for the scaling offset. shape must be C channel.
    * @li mean: A 1D tensor of type float32, for the updated mean. shape must be C channel.
    * @li variance: A 1D tensor of type float32, for the updated variance . shape must be C channel. \n

    * @par Attributes:
    * @li epsilon: A required float32, specifying the small value added to variance
    * to avoid dividing by zero.
    * @li factor: A required float32, specifying the weight for updating the mean
    * and variance . \n

    * @par Outputs:
    * Five outputs, including:
    * @li y: A 5D tensor of type float16 or float32 or bfloat16, for normalized "x", with format NDHWC or NCDHW.
    * @li mean: A 1D tensor of type float32, for the updated mean. shape must be C channel.
    * @li variance: A 1D tensor of type float32, for the updated variance. shape must be C channel.
    * @li batch_mean: A 1D tensor of type float32, for the mean of "x". shape must be C channel.
    * @li batch_variance: A 1D tensor of type float32, for the variance of "x" . shape must be C channel. \n

    * @attention Constraints:
    * @li This operator is a BatchNorm fusion operator for updating the moving
      averages for training.
    * This operator is used in conjunction with BN3DTrainingUpdate.
    * @li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to the square
    * root instruction.
    */
    REG_OP(BN3DTrainingUpdate)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(factor, Float)
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BN3DTrainingUpdate)

    /**
    * @brief Performs batch normalization for inference .

    * @par Inputs:
    * Five inputs, including:
    * @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
    * @li scale: A 1D tensor of type float32, for the scaling factor, the shape is same as dim C of input x.
    * @li offset: A 1D tensor of type float32, for the scaling offset, the shape is same as dim C of input x.
    * @li mean: A 1D tensor of type float32, for the mean, the shape is same as dim C of input x.
    * @li variance: A 1D tensor of type float32, for the variance, the shape is same as dim C of input x. \n

    * @par Attributes:
    * epsilon: An optional float32, specifying the small value added to variance to
    * avoid dividing by zero. Defaults to "0.0001" . \n

    * @par Outputs:
    * y: A 4D tensor of type float16 or float32 or bfloat16 for the normalized "x", with format NHWC or NCHW. \n

    * @attention Constraints:
    * For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to the
    * square root instruction.
    */
    REG_OP(BNInfer)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(BNInfer)

    /**
    * @brief Performs reduced batch normalization. For some scenes which don't
    * contain assign moving average .

    * @par Inputs:
    * Five inputs, including:
    * @li x: A 4D tensor of type float16/float32/bfloat16, with format NHWC or NCHW. Empty tensors are not supported.
    * Input tensor, that is, the original data that needs to be normalized.
    * @li sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator
    BNTrainingReduce.
    * It represents the sum of the input tensor "x" on the C axis. Has the same format as "x".
    * @li square_sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator
    BNTrainingReduce.
    * It represents the sum of squares of the input tensor "x" on the C axis. Has the same format as "x".
    * @li scale: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling factor. Has the
    same format as "x".
    * @li offset: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling offset. Has the
    same format as "x". \n

    * @par Attributes:
    * epsilon: A required float32, specifying the small value added to variance
    * to avoid dividing by zero. \n

    * @par Outputs:
    * Three outputs, including:
    * @li y: A 4D tensor of type float16 or float32 or bfloat16, for normalized "x". Empty tensors are not supported.
    * Has the same dype, format and shape as "x".
    * @li batch_mean: A 1D tensor of type float32, for the mean of "x". shape must be C channel. Has the same format as
    "x".
    * @li batch_variance: A 1D tensor of type float32, for the variance of "x" . shape must be C channel. Has the same
    format as "x". \n

    * @attention Constraints:
    * @li This operator is used in conjunction with BNTrainingReduce.
    * @li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to
    * the square root instruction.
    */
    REG_OP(BNTrainingUpdateV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateV2)

    /**
    * @brief Performs reduced batch normalization v3. For some scenes which
    * don't contain assign moving average .

    * @par Inputs:
    * Five inputs, including:
    * @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW. Empty tensors are not
    supported.
    * Input tensor, that is, the original data that needs to be normalized.
    * @li sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator
    BNTrainingReduce.
    * It represents the sum of the input tensor "x" on the C axis. Has the same format as "x".
    * @li square_sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator
    BNTrainingReduce.
    * It represents the sum of squares of the input tensor "x" on the C axis. Has the same format as "x".
    * @li scale: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling factor. Has the
    same format as "x".
    * @li offset: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling offset. Has the
    same format as "x". \n

    * @par Attributes:
    * epsilon: A required float32, specifying the small value added to variance
    * to avoid dividing by zero. \n

    * @par Outputs:
    * @li y: A 4D tensor of type float16 or float32 or bfloat16, for normalized "x". Empty tensors are not supported.
    * Has the same dype, format and shape as "x".
    * @li batch_mean: A 1D tensor of type float32, for the mean of "x". shape must be C channel. Has the same format as
    "x".
    * @li batch_variance: A 1D tensor of type float32, for the variance of "x" . shape must be C channel. Has the same
    format as "x".
    * @li reserve_1: A 1D tensor of type float32, for the mean of batch "x".
    * Has the same type, shape and format as input "sum".
    * @li reserve_2: A 1D tensor of type float32, for the variance of batch "x".
    * Has the same type, shape and format as input "sum". \n

    * @attention Constraints:
    * @li This operator is used in conjunction with BNTrainingReduce.
    * @li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to
    * the square root instruction.
    */
    REG_OP(BNTrainingUpdateV3)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateV3)

    /**
    * @brief Performs the backpropagation of BatchNorm .

    * @par Inputs:
    * Four inputs, including:
    * @li grads: A 4D tensor of type float16 or float32 or bfloat16,
    * for the gradient, with format NHWC or NCHW.
    * Indicates the gradient of the loss function with respect to the output of the batch normalization layer.
    * @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
    * Indicates the data input to the batch normalization layer during the forward propagation process.
    * Has the same type, format and shape as "grads".
    * @li batch_mean: A 1D tensor of type float32,
    * for the mean of "x". Shape must be C channel.
    * Has the same format as "grads".
    * @li batch_variance: A 1D tensor of type float32,
    * for the variance of "x" . Shape must be C channel.
    * Has the same format as "grads". \n

    * @par Attributes:
    * epsilon: An optional float32. Defaults to "0.0001".
    * Represents a very small positive number that is added to the variance of "x" to prevent division by zero. \n

    * @par Outputs:
    * @li diff_scale: A 1D Tensor of type float32,
    * for the offset of "scale". Shape must be C channel.
    * Has the same format as "grads".

    * @li diff_offset: A 1D Tensor of type float32,
    * for the offset of "offset". Shape must be C channel.
    * Has the same format as "grads". \n

    */
    REG_OP(BNTrainingUpdateGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .OUTPUT(diff_scale, TensorType({DT_FLOAT}))
    .OUTPUT(diff_offset, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateGrad)

    /**
    * @brief Performs the backpropagation of BatchNorm .

    * @par Inputs:
    * Four inputs, including:
    * @li grads: A 5D tensor of type float16 or float32 or bfloat16,
    * for the gradient, with format NDHWC or NCDHW.
    * @li x: A 5D tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW,
    * the shape is same as input grads.
    * @li batch_mean: A 1D tensor of type float32,
    * for the mean of "x", the shape is same as dim C of input grads.
    * @li batch_variance: A 1D tensor of type float32,
    * for the variance of "x", the shape is same as dim C of input grads. \n

    * @par Attributes:
    * epsilon: An optional float32. Defaults to "0.0001". A small float number
    * added to the variance of "x" . \n

    * @par Outputs:
    * @li diff_scale: A 1D Tensor of type float32,
    * for the offset of "scale", the shape is same as dim C of input grads.
    * @li diff_offset: A 1D Tensor of type float32,
    * for the offset of "offset", the shape is same as dim C of input grads. \n
    */
    REG_OP(BN3DTrainingUpdateGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .OUTPUT(diff_scale, TensorType({DT_FLOAT}))
    .OUTPUT(diff_offset, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BN3DTrainingUpdateGrad)

    /**
    *@brief Performs instance normalization for inference .

    *@par Inputs:
    * Five inputs, including:
    *@li x: A Tensor of type float16 or float32.
    *@li gamma: A optional Tensor of type float32, for the scaling gamma, with shape [N, C1, 1, 1, C0].
    *@li beta: A optional Tensor of type float32, for the scaling beta, with the same shape of gamma.
    *@li mean: A optional ensor of type float32, for the mean, with the same shape of gamma.
    *@li variance: A optional Tensor of type float32, for the variance, with the same shape of gamma. \n

    *@par Attributes:
    *epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero.
    Defaults to "0.00001" . \n

    *@par Outputs:
    *@li y: A Tensor of type float16 or float32 for the normalized "x".
    *@li batch_mean: A Tensor of type float32 for the result mean.
    *@li batch_variance: A Tensor of type float32 for the result variance . \n

    *@attention Constraints:
    *For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 0.001 due to the square root
    instruction.
    */
    REG_OP(INInferV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.00001)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INInferV2)

    /**
    *@brief Performs reduce instance normalization.

    *@par Inputs:
    *x: A 4D tensor of type float16 or float32, format [NCHW, NHWC]\n

    *@par Outputs:
    *@li sum: A 4D tensor of type float32 for SUM reduced "x", format [NCHW, NHWC], and HW=1.
    *@li square_sum: A 4D tensor of type float32 for SUMSQ reduced "x", format [NCHW, NHWC], and HW=1. \n

    *@attention Constraints:
    * This operator is a InstanceNorm fusion operator for updating the moving averages for training.
    * This operator is used in conjunction with INTrainingUpdateV2.
    */
    REG_OP(INTrainingReduceV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingReduceV2)

    /**
    *@brief Performs update instance normalization. \n

    *@par Inputs:
    * Seven inputs, including:
    *@li x: A 4D tensor of type float16 or float32, format [NCHW, NHWC].
    *@li sum: A 4D tensor of type float32 for the output of operator INTrainingReduceV2, format [NCHW, NHWC], and HW=1.
    *@li square_sum: A 4D tensor of type float32 for the output of operator INTrainingReduceV2, format [NCHW, NHWC], and
    HW=1.
    *@li gamma: A 4D optional tensor of type float32, for the scaling gamma, format [NCHW, NHWC], and HW=1.
    *@li beta: A 4D optional tensor of type float32, for the scaling beta, format [NCHW, NHWC], and HW=1.
    *@li mean: A 4D optional tensor of type float32, for the updated mean, format [NCHW, NHWC], and HW=1.
    *@li variance: A 4D optional tensor of type float32, for the updated variance, format [NCHW, NHWC], and HW=1.\n

    *@par Attributes:
    *@li momentum: A optional float32, specifying the momentum to update mean and var. default to 0.1.
    *@li epsilon: A optional float32, specifying the small value added to variance to avoid dividing by zero. default to
    0.00001. \n

    *@par Outputs:
    * Three outputs
    *@li y: A 4D tensor of type float16 or float32, for normalized "x", format [NCHW, NHWC].
    *@li batch_mean: A 4D tensor of type float32, for the updated mean, format [NCHW, NHWC], and HW=1.
    *@li batch_variance: A 4D tensor of type float32, for the updated variance, format [NCHW, NHWC], and HW=1. \n

    *@attention Constraints:
    * This operator is a InstanceNorm fusion operator for updating the moving averages for training.
    * This operator is used in conjunction with INTrainingReduceV2.
    */
    REG_OP(INTrainingUpdateV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(momentum, Float, 0.1)
    .ATTR(epsilon, Float, 0.00001)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingUpdateV2)

    /**
    *@brief Performs the backpropagation of InstanceNorm. \n

    *@par Inputs:
    * Seven inputs, including:
    *@li dy: A 4D tensor of type float16 or float32, format [NCHW, NHWC].
    *@li x: A 4D tensor of type float16 or float32, format [NCHW, NHWC].
    *@li variance: A 4D tensor of type float32, for the variance of "x", format [NCHW, NHWC] and HW=1.
    *@li mean: A 4D tensor of type float32, for the mean of "x", format [NCHW, NHWC] and HW=1.
    *@li res_gamma: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1.
    *@li res_beta: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1.
    *@li gamma: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1. \n

    *@par Outputs:
    *pd_x: A 4D tensor of type float16 or float32, for the offset of "x", format [NCHW, NHWC]. \n

    *@attention Constraints:
    * The preceding layer of this operator must be INTrainingUpdateGrad. \n
    */
    REG_OP(INTrainingReduceGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(res_gamma, TensorType({DT_FLOAT}))
    .INPUT(res_beta, TensorType({DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingReduceGrad)

    /**
    *@brief Performs the backpropagation of InstanceNorm. \n

    *@par Inputs:
    * Four inputs, including:
    *@li dy: A 4D tensor of type float16 or float32, for the gradient, format [NCHW, NHWC].
    *@li x: A 4Dtensor of type float16 or float32, format [NCHW, NHWC].
    *@li variance: A 4D tensor of type float32, for the variance of "x", format [NCHW, NHWC] and HW=1.
    *@li mean: A 4D tensor of type float32, for the mean of "x", format [NCHW, NHWC] and HW=1. \n

    *@par Outputs:
    *@li res_gamma: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1.
    *@li res_beta: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1. \n

    */
    REG_OP(INTrainingUpdateGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .OUTPUT(res_gamma, TensorType({DT_FLOAT}))
    .OUTPUT(res_beta, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingUpdateGrad)

    /**
    *@brief Performs the backpropagation of InstanceNorm. \n

    *@par Inputs:
    * Two inputs, including:
    *@li res_gamma: A 4D tensor of type float32,  format [NCHW, NHWC].
    *@li res_beta: A 4D tensor of type float32, format [NCHW, NHWC]. \n

    *@par Outputs:
    *@li pd_gamma: A 4D tensor of type float32, format [NCHW, NHWC].
    *@li pd_beta: A 4D tensor of type float32, format [NCHW, NHWC]. \n

    */
    REG_OP(INTrainingUpdateGradGammaBeta)
    .INPUT(res_gamma, TensorType({DT_FLOAT}))
    .INPUT(res_beta, TensorType({DT_FLOAT}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingUpdateGradGammaBeta)

    /**
    *@brief Performs reduced group normalization.

    *@par Inputs:
    *x: A Tensor of type float16 or float32, with format NCHW NHWC . \n

    *@par Outputs:
    *@li sum: A Tensor of type float32 for SUM reduced "x". shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC.
    *@li square_sum: A Tensor of type float32 for SUMSQ reduced "x".shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1]
    for NHWC.

    *@par Attributes:
    *num_groups: A optional Int, specifying the num of groups. required, same to GNTrainingUpdate, default to 2 . \n

    *@attention Constraints:
    * This operator is a GroupNorm fusion operator for updating the moving averages for training.
    * This operator is used in conjunction with GNTrainingUpdate.
    */
    REG_OP(GNTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .ATTR(num_groups, Int, 2)
    .OP_END_FACTORY_REG(GNTrainingReduce)

    /**
    *@brief Performs update group normalization .

    *@par Inputs:
    * Seven inputs, including: (NCHW NHWC supported)
    *@li x: A Tensor of type float16 or float32.
    *@li sum: A tensor of type float32,
    shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
    for the output of operator GNTrainingReduce.
    *@li square_sum: A tensor of type float32,
    shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
    for the output of operator GNTrainingReduce.
    *@li scale: A optional tensor of type float32,
    shape is [1, G, 1, 1, 1] for NCHW, [1, 1, 1, G, 1] for NHWC
    is for the scaling gamma.
    *@li offset: A optional tensor of type float32,
    shape is [1, G, 1, 1, 1] for NCHW, [1, 1, 1, G, 1] for NHWC
    for the scaling beta.
    *@li mean: A optional tensor of type float32,
    shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
    for the updated mean.
    *@li variance: A optional tensor of type float32,
    shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
    for the updated variance.

    *@par Attributes:
    *@li epsilon: A optional float32, specifying the small value added to variance to avoid dividing by zero, default to
    0.0001.
    *@li num_groups: a optional int, specifying the num of groups. required, same to GNTrainingReduce, default to 2. \n

    *@par Outputs:
    * Three outputs, including:
    *@li y: A Tensor of type float16 or float32, for normalized "x".
    *@li batch_mean: A Tensor of type float32, for the updated mean.
    *@li batch_variance: A Tensor of type float32, for the updated variance . \n

    *@attention Constraints:
    *@li This operator is a InstanceNorm fusion operator for updating the moving averages for training.
    * This operator is used in conjunction with GNTrainingUpdate.
    *@li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to the square root
    instruction.
    */
    REG_OP(GNTrainingUpdate)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(num_groups, Int, 2)
    .ATTR(epsilon, Float, 0.0001)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(GNTrainingUpdate)

    /**
    * @brief Computes the softmax focal loss of "pred" and "target".

    * @par Inputs:
    * Three inputs, including:
    * @li pred: A 2-dimensional Tensor of type float16 or float32, specifying the predicted value.
    * @li target: A 1-dimensional Tensor of type int32, specifying the target value.
    * @li weight: A 1-dimensional Tensor, specifying the weight value on class_wise. \n

    * @par Attributes:
    * @li gamma: An optional float, specifying the exponent of the modulating factor (1 - pt)
    * to balance easy/hard examples. Defaults to 2.0.
    * @li alpha: An optional float, specifying the weighting factor in range (1, 0) to balance
    * the importance of positive/negative examples or less than 0 for ignore. Defaults to 0.25.
    * @li reduction: A optional character string from "none", "mean", and "sum", specifying the
    * reduction type to be applied to the output. Defaults to "mean".  reduction only support
    * "none" currently for matching mmcv.\n

    * @par Outputs:
    * y: Softmax focal loss between the predicted value and target value. Has the same dimensions as "pred". \n

    * @par Third-party framework compatibility
    * Compatible with mmcv operator SoftmaxFocalLoss.
    */
    REG_OP(SoftmaxFocalLoss)
    .INPUT(pred, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(gamma, Float, 2.0)
    .ATTR(alpha, Float, 0.25)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SoftmaxFocalLoss)

    /**
    * @brief Computes the regression box of the RPN. It is a FasterRCNN operator.

    * @par Inputs:
    * Two inputs, including:
    * @li predict: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the predictive value.
    * The maximum dimension is 8.
    * @li label: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the target value.
    * The maximum dimension is 8, predict and label can be broadcast.

    * @par Attributes:
    * sigma: Must be a floating point number. Defaults to "1.0".

    * @par Outputs:
    * loss: Indicates the loss between the predictive value and target value.
    * Has the same dtype and dimensions as "predict".

    * @attention Constraints:
    * This operator does not perform the "reduce" operation on the loss value.
    * Call other reduce operators to perform "reduce" operation on the loss if required.

    * @par Third-party framework compatibility
    * Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SmoothL1Loss.
    */
    REG_OP(SmoothL1Loss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1Loss)

    /**
    * @brief Performs the backpropagation of SmoothL1Loss for training scenarios .

    * @par Inputs:
    * Three inputs, including:
    * @li predict: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the predictive value.
    * @li label: A multi-dimensional Tensor of float16 or float32 or bfloat16, specifying the target value.
    * @li dout: A multi-dimensional Tensor of float16 or float32 or bfloat16,
        specifying the gradient transferred from the upper layer . \n

    * @par Attributes:
    * sigma: Must be a floating point number. Defaults to "1.0" . \n

    * @par Outputs:
    * gradient: Return gradient. Has the same dimensions and type as "predict" . \n

    * @par Third-party framework compatibility
    * Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SmoothL1LossGrad.
    */
    REG_OP(SmoothL1LossGrad)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1LossGrad)

    /**
    *@brief Layernorm operator interface implementation with given sum and square sum of input tensor \n
    *  calculating: x, gamma, beta, sum, square_sum \n
    *  mean  = sum / reduce_axis \n
    *  variance = square_sum / reduce_aixs - mean * mean \n
    *  variance = np.mean(np.power((x - mean),2), reduce_axis, keepdims=True) \n
    *  y = gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta

    *@par Inputs:
    *Five inputs, including:
    * @li x: A Tensor. Must be one of the following types: float16.
    * @li gamma: A Tensor. Must be one of the following types: float16.
    * @li beta: A Tensor. Must be one of the following types: float16.
    * @li sum. A Tensor. Must be one of the following types: float.
    * @li square_sum. A Tensor. Must be one of the following types: float. \n

    *@par Attributes:
    * @li epsilon: A optional attribute, the type is float32. Defaults to 1e-5. \n

    *@par Outputs:
    *Three outputs, including:
    * @li y: A Tensor. Must be one of the following types: float16.
    */
    REG_OP(LayerNormUpdate)
    .INPUT(x1, TensorType({DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(epsilon, Float, 0.00001f)
    .OP_END_FACTORY_REG(LayerNormUpdate)

    /**
    * @brief Computes the gradient for Local Response Normalization .

    * @par Inputs:
    * @li grads: A 4D Tensor of type float16 or float32.
    * @li x: A 4D Tensor of type float16 or float32.
    * @li y: A 4D Tensor of type float16 or float32 .

    * @par Attributes:
    * @li depth_radius: An optional int, specifying the half-width of the
    * normalization window. Defaults to "5".
    * @li bias: An optional float32. An offset, usually > 0 to avoid dividing by 0.
    * Defaults to "1".
    * @li alpha: An optional float32. A scaling factor, usually positive.
    * Defaults to "1".
    * @li beta: An optional float32. An exponent. Defaults to "0.5" .

    * @par Outputs:
    * z: A Tensor. Has the same type and shape as "grads" .

    * @attention Constraints:
    * "x" and "y" must have the same shape and type as "grads" .

    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator LRNGrad.
    */
    REG_OP(LRNGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(depth_radius, Int, 5)
    .ATTR(bias, Float, 1.0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.5)
    .OP_END_FACTORY_REG(LRNGrad)

    /**
    * @brief InstanceNormGrad operator interface implementation.

    * @par Inputs:
    * Five inputs, including:
    * @li dy: Represents the input gradient tensor. Support dtype: float16, float32. Suppor shape 4D or 5D.
    * Support format: [NCHW, NHWC, NDHWC, NCDHW]. Has the same dtype, format and shape as "x".
    * @li x: Represents the input tensor. Support dtype: float16, float32. Suppor shape 4D or 5D.
    * Support format: [NCHW, NHWC, NDHWC, NCDHW].
    * @li variance: Represents the variance tensor. Support dtype: float16, float32. Suppor shape 4D or 5D.
    * Support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1). Has the same dtype and format as "x".
    * The shapes of "variance" and "mean" are consistent, and the N and C axes are consistent with those of "x", and the
    other dimensions are 1.
    * @li mean: Represents the mean tensor. Support dtype: float16, float32. Suppor shape 4D or 5D.
    * Support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1). Has the same dtype and format as "x".
    * The shapes of "variance" and "mean" are consistent, and the N and C axes are consistent with those of "x", and the
    other dimensions are 1.
    * @li gamma: Represents the optional weight parameter. Support dtype: float16, float32. Suppor shape 4D or 5D.
    * Support format: [NCHW, NHWC, NDHWC, NCDHW].  Has the same dtype as "x".
    * The C axis of "gamma" is consistent with that of "x", and the other dimensions are 1. \n

    * @par Outputs:
    * Three outputs, including:
    * @li pd_x: Represents the gradient tensor of the input tensor. Support dtype: float16, float32. Suppor shape 4D or
    5D.
    * Support format: [NCHW, NHWC, NDHWC, NCDHW].  Has the same dtype, format and shape as "x".
    * @li pd_gamma: Represents the gradient tensor of the weight parameter. Support dtype: float16, float32. Suppor
    shape 4D or 5D.
    * Support format: [NCHW, NHWC, NDHWC, NCDHW]. Has the same dtype and format as "x". Has the same shape as "gamma".
    * @li pd_beta: Represents the gradient tensor of the bias parameter. Support dtype: float16, float32. Suppor shape
    4D or 5D.
    * Support format: [NCHW, NHWC, NDHWC, NCDHW]. Has the same dtype and format as "x". Has the same shape as "gamma".
    */
    REG_OP(InstanceNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(InstanceNormGrad)

    /**
    * @brief Computes Centralization. result = x - mean(x, axes)

    * @par Inputs:
    *  x: An ND tensor of type float16, float32.
    * @par Attributes:
    * axes: The dimensions to reduce. Must be one of the following types: int, list, tuple, NoneType, default: -1.
    * Must be in the range [-rank(x), rank(x)).
    * @par Outputs:
    * y: A Tensor. Has the same type as "x". \n

    * @par Third-party framework compatibility
    * custom operator \n
    */
    REG_OP(Centralization)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(Centralization)

    /**
     * @brief Calculate the loss. Creates a criterion that optimizes a two-class classification
     * logistic loss between input_x and input_y (containing 1 or -1).

     * @par Inputs:
     * Tow inputs, including:
     * @li input_x: A tensor. Must be one of the following types:
     *     float16, float32, bfloat16. \n
     * @li input_y: A tensor. Must be one of the following types:
     *     float16, float32, bfloat16. \n

     * @par Attributes:
     * reduction: An optional string. Defaults to "mean". \n

     * @par Outputs:
     * output_z: while reduction == "none", A Tensor with the same type and shape of input_x's. \n
     *          while reduction == "sum" or "mean", A Tensor with the same type of input_x , shape of which is (1,)

     * @par Third-party framework compatibility
     * Compatible with the Pytorch operator SoftMarginLoss. \n
     */
    REG_OP(SoftMarginLoss)
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(input_y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OUTPUT(output_z, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(SoftMarginLoss)

    /**
     * @brief Calculate the PoissonNllLoss function.  \n
     *        target follow distribution of Poisson(input)loss(input,target) = input - target * log(input) +
     log(target!)

     * @par Inputs:
     * Two inputs, including:
     * @li input_x: A tensor. Must be one of the following types: float16, float32.
     * @li target: A tensor. Must be one of the following types: float16, float32. \n

     * @par Attributes:
     * four Attributes, including:
     * @li log_input: An optional bool. Defaults to "True"
     * @li full: An optional bool. Defaults to "False"
     * @li eps: An optional float. Defaults to "1e-8"
     * @li reduction: An optional string from "none", "mean",and "sum". Defaults to "mean" \n

     * @par Outputs:
     * loss: A Tensor has same element type as two inputs. \n

     * @par Third-party framework compatibility
     * Compatible with the Pytorch operator PoissonNllLoss. \n
     */
    REG_OP(PoissonNllLoss)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(log_input, Bool, true)
    .ATTR(full, Bool, false)
    .ATTR(eps, Float, 1e-8f)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(PoissonNllLoss)

    /**
    * @brief Creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss)
    *        between input x (a 2D mini-batch Tensor) and output y (which is a 2D Tensor of target class indices) \n

    * @par Inputs:
    * Two inputs, including:
    * @li x: A tensor. Must be one of the following types:
    *     float16, float32, bfloat16.
    * @li target: A tensor. Must be the following types:
    *     int32. \n

    * @par Attributes:
    * reduction: An optional string. Defaults to "mean" \n

    * @par Outputs:
    * @li y: A Tensor has same element type as input x. \n
    * @li is_target: A Tensor has same element type as input target. \n

    * @par Third-party framework compatibility
    * Compatible with the Pytorch operator MultiLabelMarginLoss. \n
    */
    REG_OP(MultilabelMarginLoss)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(target, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(is_target, TensorType({DT_INT32}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(MultilabelMarginLoss)

    /**
     * @brief Performs batch normalization . \n
     * @par Inputs:
     * Two inputs
     * @li input_x: A Tensor. Support float32. shape (n, c, d).
     * @li seq_len: A Tensor. Each batch normalize data num. Support Int32. Shape (n, ). \n
     * @par Attributes:
     * @li normalize_type: Str. Support "per_feature" or "all_features".
     * @li epsilon: An optional float32, specifying the small value added to
     * variance to avoid dividing by zero. Defaults to "0.00001" . \n
     * @par Outputs:
     * One outputs
     * @li output_y: A Tensor for the normalized "x".Support float32. shape (n, c, d).\n
     */
    REG_OP(NormalizeBatch)
    .INPUT(input_x, TensorType({DT_FLOAT}))
    .INPUT(seq_len, TensorType({DT_INT32}))
    .OUTPUT(output_y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(normalize_type, String)
    .ATTR(epsilon, Float, 0.00001f)
    .OP_END_FACTORY_REG(NormalizeBatch)

    /**
    * @brief Loss function that measures the softmax cross entropy.

    * @par Inputs:
    * Three inputs, including:
    * @li scores: A Tensor. Must be one of the following types: float16, bfloat16, float32, double.
    * A "batch_size * num_classes" matrix.
    * @li labels: A Tensor. Must be one of the following types: "int32", "int64".
    * @li weights: A manual rescaling weight given to each class, the same dtype with scores.
    * If given, it has to be a 1D Tensor assigning weight to each of the classes.
    * Otherwise, it is treated as if having all ones. \n

    * @par Attributes:
    * ignore_index:Specifies a target value that is ignored and does not contribute to the input gradient.
    * It's an optional value, Defaults to 0. \n
    * reduction: A character string from "none", "mean", and "sum", specifying the gradient output mode. Defaults to
    "mean" . \n

    * @par Outputs:
    * @li loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "scores".
    * @li log_prop: A Tensor. Has the same type as "scores" . \n

    * @par Third-party framework compatibility
    * Compatible with the ONNX operator SoftmaxCrossEntropyLoss.
    */
    REG_OP(SoftmaxCrossEntropyLoss)
    .INPUT(scores, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(weights, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(ignore_index, Int, 0)
    .ATTR(reduction, String, "mean")
    .OUTPUT(loss, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(log_prop, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(SoftmaxCrossEntropyLoss)

    /**
    * @brief MMCV Function: softmax_focal_loss_grad.

    * @par Inputs:
    * Three inputs, including:
    * @li pred: the predicted tensor. The type support float16 and float32.
    * @li target: the target label Tensor. The type support Int32.
    * @li dout: the grad of previous op grad, which has the sampe shape wth pred. The type support float16 and float32.
    * @li weight: A optional input Tensor, default is None, which helps to calculate the loss by supplying sample
    weights. The type support float16 and float32.
    *     shape of pred should be (B, D), B means batch size, D means the number of labels.
    *     shape of target should be (B, D).
    *     shape of weight should be (D, ) \n

    * @par Attributes:
    * @li alpha: A attribute is used to reweight the sample. The type is float . Default is 0.25\n
    * @li gamma: A attribute is used to calculate the power of the probability.
    *     The type is float . Default is 2.0\n
    * @li reduction: a type of the reduce method. default is 'mean', which means computing the average loss.
                    'sum' means computing the sum of the loss, 'none' means no reducing. Default is 'mean' \n

    * @par Outputs:
    * grad: A mutable Tensor. Has the same type and shape as "pred". \n

    * @par Third-party framework compatibility
    * Compatible with the MMCV operator SoftmaxFocalLossGrad.
    */
    REG_OP(SoftmaxFocalLossGrad)
    .INPUT(pred, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(alpha, Float, 0.25)
    .ATTR(gamma, Float, 2.0)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SoftmaxFocalLossGrad)

    /**
    * @brief Performs Matrix-to-matrix Multiply,
    * producing y = alpha[0] * a @ b + beta[0] * c.
    * @attention Constraints:
    * For better performance, The k-axis must be aligned to 16 (input type
    * is float16) or 32 (input type is int8).

    * @par Inputs:
    * Five inputs, including:
    * @li a: A matrix Tensor. Must be one of the following types:float32, float16,
    * int8, int32. Has format ND.
    * @li b: A matrix Tensor. Must be one of the following types:float32, float16,
    * int8, int32. Has format ND.
    * @li c: A matrix Tensor. Must be one of the following types:float32, float16,
    * int8, int32. Has format ND.
    * @li alpha: A 1D Tensor. The shape of alpha is [1].Must be one of the
    * following types: float32, float16, int8, int32. Has format ND.
    * @li beta: A 1D Tensor. The shape of beta is [1]. Must be one of the following
    * types: float32, float16, int8, int32. Has format ND.

    * @par Attributes:
    * Two attributes, including:
    * @li transpose_a: Optional. A bool. If True, changes the shape of "a" from
    * [M, K] to [K, M] before multiplication.
    * @li transpose_b: Optional. A bool. If True, changes the shape of "b" from
    * [K, N] to [N, K] before multiplication.

    * @par Outputs:
    * y: The result matrix Tensor. Must be one of the following types: float32,
    * float16, int8, int32. Has format [ND], the format should be equal to a.
    */
    REG_OP(GEMM)
    .INPUT(a, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .INPUT(b, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .INPUT(c, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .ATTR(transpose_a, Bool, false)
    .ATTR(transpose_b, Bool, false)
    .OP_END_FACTORY_REG(GEMM)

    /**
    * @par Inputs:
    * Three inputs, including:
    * @li x: An ND Tensor .

    * Must be one of the following types: float
    * @li indices: An ND Tensor . \n

    * Must be one of the following types: int32
    * @li updates: An ND Tensor .

    * Must be one of the following types: float

    * @par Outputs:
    * y: A Tensor. Has the same type and format as input "x" . \n

    * @par Outputs:
    * argmax: A Tensor. Has the same type and format as input "indices" . \n
    */
    REG_OP(ScatterMaxWithArgmax)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(argmax, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(ScatterMaxWithArgmax)

    /**
    * @brief Concatenates a list of N tensors along the first dimension.
    * @par Inputs:
    * @li x: A list of Tensors. Must be one of the following types:  int32,
    * float16, float32. Tensors to be concatenated. All must have size 1 in
    *  the first dimension and same shape. It's a dynamic input. \n

    * @par Attributes:
    * @li equation: The subscripts for the Einstein summation. \n
    * @li N: tensor size of input. \n

    * @par Outputs:
    * @li y: Sums the product of the elements of the input operands along
    * dimensions specified
    * using a notation based on the Einstein summation convention. \n

    * @attention Constraints:
    * Input N must be Int. \n

    * @par Third-party framework compatibility
    * Compatible with Tensorflow 2.x einsum operator.
    */
    REG_OP(Einsum)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(equation, String)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(Einsum)
} // namespace ge
#endif
