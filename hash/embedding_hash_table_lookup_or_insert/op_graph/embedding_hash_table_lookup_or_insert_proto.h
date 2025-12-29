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
 * \file embedding_hash_table_lookup_or_insert_proto.h
 * \brief
 */

#ifndef EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_PROTO_H_
#define EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief insert keys in NPU hashtable and return coresponding value.
*
*@par Inputs:
*Inputs include:
* @li table_handle: A tensor. Must be int64. Contains addr of table's infos. shape of [5].
* @li keys: A tensor. Must be int64. Keys to be insert.
*
*@par Outputs:
* values: A tensor. Must be float32. N-D with shape [N, embedding_dim].
*
*@par Attributes:
* @li bucket_size: Required, int, table capacity.
* @li embedding_dim: Required, int, value dims.
* @li filter_mode: Optional, string, set to "no_filter" or "counter", set "counter" to enable the counter-based filter mode, set "no_filter" to disable the filter function. default is "no_filter".
* @li filter_freq: Optional, int, filter threshold. default is 0.
* @li default_key_or_value: Optional, bool, set true will return the value of default_key, set false will return default_value. default is false.
* @li default_key: Optional, int, default key set by customer,  default is 0.
* @li default_value: Optional, float, default value set by customer. default is 0.
* @li filter_key_flag: Optional, bool, set true to enable filter_key, set false to disable the function of filter key. default is false.
* @li filter_key: Optional, int, filter input key and return default_value when flag is true. default is -1.
*/
REG_OP(EmbeddingHashTableLookupOrInsert)
.INPUT(table_handle, TensorType({DT_INT64}))
.INPUT(keys, TensorType({DT_INT64}))
.OUTPUT(values, TensorType({DT_FLOAT32}))
.REQUIRED_ATTR(bucket_size, Int)
.REQUIRED_ATTR(embedding_dim, Int)
.ATTR(filter_mode, String, "no_filter")
.ATTR(filter_freq, Int, 0)
.ATTR(default_key_or_value, Bool, false)
.ATTR(default_key, Int, 0)
.ATTR(default_value, Float, 0)
.ATTR(filter_key_flag, Bool, false)
.ATTR(filter_key, Int, -1)
.OP_END_FACTORY_REG(EmbeddingHashTableLookupOrInsert)

} // namespace ge
#endif  // EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_PROTO_H_
