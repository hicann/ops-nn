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
 * \file init_embedding_hash_table_proto.h
 * \brief
 */

#ifndef INIT_EMBEDDING_HASH_TABLE_PROTO_H_
#define INIT_EMBEDDING_HASH_TABLE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Init EmbeddingHashTable.
*
*@par Inputs:
*Inputs include:
* table_handle: A Tensor. Dtype support: int64
* sampled_values: A Tensor. Dtype support: float32
*
*@par Attributes:
* @li bucket_size: An required attribute indicates hashtable size.
* @li embedding_dim: An required attribute indicates hashtable value size.
* @li initializer_mode: An optional attribute indicates init hashtable mode.
* @li constant_value: An optional attribute indicates hashtable constant mode const value.
*
*@attention Constraints:
* Only support value's dtype is float32.
*/
REG_OP(InitEmbeddingHashTable)
    .INPUT(table_handle, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(sampled_values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(bucket_size, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(initializer_mode, String, "random")
    .ATTR(constant_value, Float, 0.0)
    .OP_END_FACTORY_REG(InitEmbeddingHashTable)

} // namespace ge
#endif  // INIT_EMBEDDING_HASH_TABLE_PROTO_H_
