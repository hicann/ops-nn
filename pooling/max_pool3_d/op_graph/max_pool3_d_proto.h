/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Performs max pooling on the input . \n

* @par Inputs:
* One input:
* x: A 5D tensor. Supported type:float16, float32. Additional support for bfloat16 in Ascend 950 AI Processor. 
* The double type is reserved but currently unsupported. Support format: NHDWC, NCDHW. Additional support for ND in Ascend 950 AI Processor.

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of
* the input tensor. No default value.
* @li padding: An required string, specifying the type of
* the padding algorithm to use. It support "SAME", "VALID" or "CALCULATED".
* @li pads: An optional list of int8, int16, int32, or int64 values.
* It must have 6 elements, specifying the front, backend, top, bottom, left, and right padding for input tensor.
* The pads should be greater than or equal to 0 and smaller than the corresponding kernel size.
* It only takes effect when padding is "CALCULATED". Default value is {0,0,0,0,0,0}.
* @li dilation: Dilation of kernel. default value is {1,1,1,1,1}.
* @li ceil_mode: Use the floor or ceil function to calculate output depth, height and width. 
* It support 0(floor) or 1(ceil). Default value is 0. 
* It can be set to 1 only when padding is "CALCULATED".
* @li data_format: An optional string, specify the data format of the input and
* output data. It support "NDHWC"(default), "NCDHW"". \n

* @par Outputs:
* y: A 5D tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* @li "ksize" is a list that has length 1, 3, or 5. The ksize of the H and W dimensions should be greater than 0.
* The ksize of the N and C dimensions should be 1. e.g. For "data_format" is "NCDHW", ksize[0] = 1 and ksize[1] = 1.
* For "data_format" is "NDHWC", ksize[0] = 1 and ksize[4] = 1.  \n
* For Atlas Training Series Product, Atlas A2 Training Series Product/Atlas 800I A2 Inference Product,
* Atlas A3 Training Series Product: The produce of the ksize in D, H and W dimensions
* should be less than or equal to 255. e.g. For "data_format" is "NCDHW", ksize[2] * ksize[3] * ksize[4] <= 255. \n
* @li "strides" is a list that has length 1, 3, or 5. The stride of the N and C dimensions should be 1.  \n
* For Atlas Training Series Product, Atlas A2 Training Series Product/Atlas 800I A2 Inference Product,
* Atlas A3 Training Series Product: The stride of the D, H and W dimensions should be greater than 0 and
* smaller than 64.  \n
* The stride of the D, H and W dimensions should be greater than 0.
* @li "data_format" only support "NCDHW" and "NDHWC". \n
* @li The ouput "y" shape at the N and C dimensions should be equal with input "x" shape at same dimensions. The output
* shape at the D, H and W dimensions is calculated by below formula: \n
* @code{.c}
  when "padding" is "SAME":
          out_depth = (in_depth + stride_d - 1) / stride_d
          out_height = (in_height + stride_h - 1) / stride_h
          out_width = (in_width + stride_w - 1) / stride_w
  when "padding" is "VALID":
          out_depth = (in_depth + stride_d - ((ksize_d - 1) * dilation_d + 1)) / stride_d
          out_height = (in_height + stride_h - ((ksize_h - 1) * dilation_h + 1)) / stride_h
          out_width = (in_width + stride_w - ((ksize_w - 1) * dilation_w + 1)) / stride_w
  when "padding" is "CALCULATED":
          if "ceil_mode" is 0:
                  out_depth = (in_depth + pad_front + pad_backend - ((ksize_d - 1) * dilation_d + 1)) / stride_d + 1
                  out_height = (in_height + pad_top + pad_bottom - ((ksize_h - 1) * dilation_h + 1)) / stride_h + 1
                  out_width = (in_width + pad_left + pad_right - ((ksize_w - 1) * dilation_w + 1)) / stride_w + 1
          else :
                  out_depth = (in_depth + pad_front + pad_backend - ((ksize_d - 1) * dilation_d + 1) + stride_d - 1) / stride_d + 1
                  out_height = (in_height + pad_top + pad_bottom - ((ksize_h - 1) * dilation_h + 1) + stride_h - 1) / stride_h + 1
                  out_width = (in_width + in_width + pad_right - ((ksize_w - 1) * dilation_w + 1) + stride_w - 1) / stride_w + 1
                  if  (out_depth - 1) * stride_d >= in_depth + pad_front :
                          out_depth = out_depth - 1
                  if  (out_height - 1) * stride_h >= in_height + pad_top :
                          out_height = out_height - 1
                  if  (out_width - 1) * stride_w >= in_width + in_width :
                          out_width = out_width - 1
  It not support out_height < 0 or out_width < 0.
* @endcode
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPool3D.
*/
REG_OP(MaxPool3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_BF16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(pads, ListInt, {0,0,0,0,0,0})
    .ATTR(dilation, ListInt, {1, 1, 1, 1, 1})
    .ATTR(ceil_mode, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(MaxPool3D)

}  // namespace ge

#endif