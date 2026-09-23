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
 * \file embedding_hash_table_evict_proto.h
 * \brief embedding_hash_table_evict
 */

#ifndef EMBEDDING_HASH_TABLE_EVICT_PROTO_H_
#define EMBEDDING_HASH_TABLE_EVICT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief embedding hashtable evict.
 *
 * @par Inputs:
 * @li table_handle: A tensor. Must be int64. Contains addr of table's infos.
 * @li keys: A tensor. Must be int64. Keys to evict.
 * @li sampled_values: An optional tensor. Must be float32. Random sampled values used by random init mode.
 *
 * @par Attributes:
 * @li table_cap: Required, int, table capacity.
 * @li embedding_dim: Required, int, embedding value dimension.
 * @li init_mode: Optional, string, set to "random" or "constant", default is "constant".
 * @li const_val: Optional, float, constant value used by constant init mode, default is 0.0.
 */
#ifndef OPS_PROTO_DEF_EMBEDDINGHASHTABLEEVICT
#define OPS_PROTO_DEF_EMBEDDINGHASHTABLEEVICT
REG_OP(EmbeddingHashTableEvict)
    .INPUT(table_handle, TensorType({DT_INT64}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(sampled_values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(table_cap, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(init_mode, String, "constant")
    .ATTR(const_val, Float, 0.0)
    .OP_END_FACTORY_REG(EmbeddingHashTableEvict)
#endif // OPS_PROTO_DEF_EMBEDDINGHASHTABLEEVICT

} // namespace ge
#endif // EMBEDDING_HASH_TABLE_EVICT_PROTO_H_
