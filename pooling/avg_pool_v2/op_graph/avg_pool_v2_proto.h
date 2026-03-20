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
 * \file avg_pool_v2_proto.h
 * \brief
 */
#ifndef OPS_POOLING_AVG_POOL_V2_PROTO_H_
#define OPS_POOLING_AVG_POOL_V2_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"



namespace ge {
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
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_DOUBLE}))
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

} // namespace ge

#endif // OPS_POOLING_AVG_POOL_PROTO_H_