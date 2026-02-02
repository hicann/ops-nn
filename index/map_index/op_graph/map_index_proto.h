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
 * \file map_index_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_MAP_INDEX_H_
#define OPS_OP_PROTO_INC_MAP_INDEX_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns index of shape in the map.

* @par Inputs:
* Three inputs, including:
* @li x: One dimensional tensor of type int32, specifying queried shape,
* Format support ND, the max dim is 400 (128 on Lhisi(Hi3796CV300CS, SD3403), 24000 on Ascend 910_95 AI Processor).
* @li data_seq: One dimensional tensor of type int32, 
* Format support ND, specifying the mapped table is queried.
* The length of data_seq must be multiple of the length of x, and the length of data_seq / x <= 100 (256 on Ascend 910_95 AI Processor).
* @li level_index: One dimensional tensor of type int32, the length of level_index must be equal to the length of data_seq divided by the length of x.
* Format support ND, specifying secondary index. \n

* @par Attributes:
 *@li transpose: An optional bool. specifying the input is transposed on A3 or A5, A3 is true, A5 is false.

* @par Outputs:
* y: A scalar of type int32, specifying index of shape in the map.
* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(MapIndex)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(data_seq, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(level_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(transpose, Bool, false)
    .OP_END_FACTORY_REG(MapIndex)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MAP_INDEX_H_