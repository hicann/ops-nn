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
 * \file nn_pooling_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Performs max pooling on the input and outputs both max values and
* indices .

* @par Inputs:
* One input:
* x: An 4D Tensor. Supported type:float16, bfloat16, float32
* Must set the format, supported format list ["NCHW, NHWC"]. \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor.
* No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of
* the input tensor. No default value.
* @li padding: A required string. Must be ["SAME", "VALID"]. \n
* when padding is "SAME": pads 0 to ensure output shape equal to ceil(input shape / stride) ,
* (output shape equal to input shape when stride=1). \n
* when padding is "VALID": no padding. The kernel slides only over valid regions, resulting in smaller output .
* @li Targmax: A required int32 or int64. Default value: 9(int64).
* Specify the dtype for argmax output, currently not used(only int32 or int64).
* @li include_batch_in_index: A required bool. Must be false. \n
* when include_batch_in_index is true: Indicates whether to include the batch dimension,
* when calculating the index of the maximum output value argmax.
* when include_batch_in_index is false: Indicates is not included the batch dimension.
* @li data_format: A required string. Supported format: NHWC, NCHW. Default value: [NHWC].
* @li nan_prop: A required bool. Default value: false. \n
* when nan_prop is true: NAN values are used for comparison. The maximum value can be NAN.
* when nan_prop is false: NAN values are ignored.

* @par Outputs:
* @li y: A Tensor. Has the same type and format as input "x".
* @li argmax: A Tensor. Has the same format as input "x", Supported format list ["NCHW, NHWC"].

* @attention Constraints:
* @li "ksize" is a list that has length 4: [NHWC] required ksize[0] = 1 and ksize[3] = 1,
* [NCHW] required ksize[0] = 1 and ksize[1] = 1.
* @li "stride" is a list that has length 4: [NHWC] required strides[0] = 1 and strides[3] = 1,
* [NCHW] required strides[0] = 1 and strides[1] = 1.
* @li "padding" is either "SAME" or "VALID" .
* @li Targmax: only is 3 and 9, int32 corresponds to 3, int64 corresponds to 9.
* @li include_batch_in_index: only support include_batch_in_index is false.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolWithArgmax.
*/
REG_OP(MaxPoolWithArgmax)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(argmax, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(Targmax, Int, 9)
    .ATTR(include_batch_in_index, Bool, false)
    .ATTR(data_format, String, "NHWC")
    .ATTR(nan_prop, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolWithArgmax)


}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H
