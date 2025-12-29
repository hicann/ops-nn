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
 * \file embedding_hash_table_import_proto.h
 * \brief
 */

#ifndef EMBEDDING_HASH_TABLE_IMPORT_PROTO_H_
#define EMBEDDING_HASH_TABLE_IMPORT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief embedding hashtable data import. \n

* @par Inputs:
* @li table_handles: A Tensor, dtype is DT_INT64. 1-D. Indicates addr of hashtable.
* @li embedding_dims: A Tensor, dtype is DT_INT64. 1-D. Indicates the length of values.
* @li bucket_sizes: A Tensor, dtype is DT_INT64. 1-D. Indicates the hash bucket size.
* @li keys: A list of Tensor, dtype is DT_INT64, dynamic input. Indicates the hashtable keys number.
* @li counters: A list of Tensor, dtype is DT_UINT64, dynamic input. Indicates the hashtable counters.
* @li filter_flags: A list of Tensor, dtype is DT_UINT8, dynamic input. Indicates filter_flags.
* @li values: A list of Tensor, dtype is DT_FLOAT32, dynamic input. Indicates the import values .

* @attention Constraints:
* @li table_handles, embedding_dims and bucket_sizes have same shape len.
* @li actual kernel compute filter_flags be 64 bit.
* @li single struct contains keys、counters、filter_flags and values, should be algin to 8 bytes.
*/
REG_OP(EmbeddingHashTableImport)
.INPUT(table_handles, TensorType({DT_INT64}))
.INPUT(embedding_dims, TensorType({DT_INT64}))
.INPUT(bucket_sizes, TensorType({DT_INT64}))
.DYNAMIC_INPUT(keys, TensorType({DT_INT64}))
.DYNAMIC_INPUT(counters, TensorType({DT_UINT64}))
.DYNAMIC_INPUT(filter_flags, TensorType({DT_UINT8}))
.DYNAMIC_INPUT(values, TensorType({DT_FLOAT}))
.OP_END_FACTORY_REG(EmbeddingHashTableImport)

} // namespace ge
#endif  // EMBEDDING_HASH_TABLE_IMPORT_PROTO_H_
