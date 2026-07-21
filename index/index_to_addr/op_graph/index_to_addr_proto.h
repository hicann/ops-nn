/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_INDEX_INDEX_TO_ADDR_PROTO_H_
#define OPS_NN_INDEX_INDEX_TO_ADDR_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Converts block index to address table.
 *
 * @par Inputs:
 * @li base_addr: Base address tensor, supports DT_INT64 and DT_UINT64.
 * @li x: Index tensor, supports DT_INT64 and DT_UINT64.
 *
 * @par Attributes:
 * @li ori_shape: Original matrix shape.
 * @li block_size: Block matrix shape.
 * @li ori_storage_mode: Storage mode of original tensor, default "Matrix".
 * @li block_storage_mode: Storage mode of block tensor, default "Matrix".
 * @li rank_id: Rank id, default 0.
 * @li dtype: Base tensor dtype, default DT_FLOAT.
 *
 * @par Outputs:
 * @li addrs_table: Address table tensor, supports DT_INT64 and DT_UINT64.
 */
REG_OP(IndexToAddr)
    .INPUT(base_addr, TensorType({DT_INT64, DT_UINT64}))
    .INPUT(x, TensorType({DT_INT64, DT_UINT64}))
    .OUTPUT(addrs_table, TensorType({DT_INT64, DT_UINT64}))
    .REQUIRED_ATTR(ori_shape, ListInt)
    .REQUIRED_ATTR(block_size, ListInt)
    .ATTR(ori_storage_mode, String, "Matrix")
    .ATTR(block_storage_mode, String, "Matrix")
    .ATTR(rank_id, Int, 0)
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(IndexToAddr)

} // namespace ge

#endif // OPS_NN_INDEX_INDEX_TO_ADDR_PROTO_H_
