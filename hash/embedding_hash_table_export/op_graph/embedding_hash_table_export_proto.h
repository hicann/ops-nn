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
 * \file embedding_hash_table_export_proto.h
 * \brief
 */

#ifndef EMBEDDING_HASH_TABLE_EXPORT_PROTO_H_
#define EMBEDDING_HASH_TABLE_EXPORT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief embedding hashtable export. \n

* @par Inputs:
* @li table_handles: A Tensor, dtype is DT_INT64, 1-D. indicates the table handle address.
* @li table_sizes: A Tensor, dtype is DT_INT64,  1-D. indicates the size of hash map for each table.
* @li embedding_dims: A Tensor, dtype is DT_INT64,  1-D. indicates the dim of embedding value in hashtable.
* @li bucket_sizes: A Tensor, dtype is DT_INT64. 1-D. indicates the hash bucket size. \n

* @par Outputs:
* @li keys: A list of Tensor objects, dtype is DT_INT64. indicates the hashtable keys. It's a dynamic output.
* @li counters: A list of Tensor objects, dtype is DT_UINT64. indicates the hashtable counter. It's a dynamic output.
* @li filter_flags: A list of Tensor objects, dtype is DT_UINT8. indicates the hashtable filter flag. It's a dynamic
*     output.
* @li values: A list of Tensor objects, dtype is DT_FLOAT. indicates the hashtable value. It's a dynamic output. \n

* @par Attributes:
* @li export_mode: An optional String. Represents export mode, can be either "all" or "new". Defaults to "all".
* @li filtered_export_flag: An optional Bool. Represents filter export flag on counter filter scenario.
      Defaults to false. \n

* @attention Constraints:
* @li table_handles, table_sizes, embedding_dims and bucket_sizes have same shape len.
* @li the number of tensors in each dynamic output tensor list (keys/counters/filter_flags/values) shall be equal to
    the shape len of table_handles. \n
*/
REG_OP(EmbeddingHashTableExport)
    .INPUT(table_handles, TensorType({DT_INT64}))
    .INPUT(table_sizes, TensorType({DT_INT64}))
    .INPUT(embedding_dims, TensorType({DT_INT64}))
    .INPUT(bucket_sizes, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(keys, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(counters, TensorType({DT_UINT64}))
    .DYNAMIC_OUTPUT(filter_flags, TensorType({DT_UINT8}))
    .DYNAMIC_OUTPUT(values, TensorType({DT_FLOAT}))
    .ATTR(export_mode, String, "all")
    .ATTR(filtered_export_flag, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingHashTableExport)

} // namespace ge
#endif  // EMBEDDING_HASH_TABLE_EXPORT_PROTO_H_
