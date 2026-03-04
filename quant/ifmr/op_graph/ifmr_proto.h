/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IFMR_PROTO_H
#define IFMR_PROTO_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief IFMR(Input Feature Map Reconstruction).

* @par Inputs:
* @li data: A Tensor of feature map.
* @li data_min: A Tensor of min value of feature map.
* @li data_max: A Tensor of max value of feature map.
* @li cumsum: A Tensor of cumsum bin of data . \n

* @par Attributes:
* @li min_percentile: Min init percentile.
* @li max_percentile: Max init percentile.
* @li search_range: Search range.
* @li search_step: Step size of searching.
* @li with_offset: Whether using offset.
* @li quant_bits: Bits of quant, an optional attr, default value is 8. \n

* @par Outputs:
* @li scale: Optimal scale.
* @li offset: Optimal offset. \n

* @par Third-party framework compatibility
* Compatible with mindspore.
*/

REG_OP(IFMR)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(data_min, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(data_max, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(cumsum, TensorType({DT_INT32}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(min_percentile, Float)
    .REQUIRED_ATTR(max_percentile, Float)
    .REQUIRED_ATTR(search_range, ListFloat)
    .REQUIRED_ATTR(search_step, Float)
    .REQUIRED_ATTR(with_offset, Bool)
    .ATTR(quant_bits, Int, 8)
    .OP_END_FACTORY_REG(IFMR)
} // namespace ge

#endif // IFMR_PROTO_H