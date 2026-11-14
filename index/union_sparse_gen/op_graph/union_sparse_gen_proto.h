/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_OP_PROTO_INC_UNION_SPARSE_GEN_H_
#define OPS_OP_PROTO_INC_UNION_SPARSE_GEN_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Merges explicit KV block lists within each query group into a sorted union,
 * and produces per-token query membership bitmasks. Groups never cross requests.
 *
 * @par Inputs:
 * @li sparse_block_idx: Required INT32 ND tensor of shape [H,T,K], containing logical
 * KV block IDs. H is the sparse head count, T the stored query row count, and K the
 * per-row capacity. Only the valid prefix of each row participates in the union.
 * @li sparse_block_count: Required INT32 ND tensor of shape [H,T], specifying the
 * valid prefix length of each row. Counts are clamped to [0,K].
 * @li cu_sqlen_q: Optional INT64 ND tensor of shape [B+1], specifying Q storage
 * prefixes. Starts at zero, is nondecreasing, and ends at T. B is the request count.
 * Without it, seq_used_q and positive max_q_seqlen are required, with T=B*max_q_seqlen.
 * @li cu_sqlen_kv: Optional INT64 ND tensor of shape [B+1], specifying nondecreasing
 * KV storage prefixes starting at zero. Without it, seq_used_kv is required.
 * Q and KV independently select their storage convention by prefix presence.
 * @li seq_used_q: Optional INT32 ND tensor of shape [B]. Specifies nonnegative
 * effective Q lengths, taking precedence over Q prefix differences. Each effective
 * length must fit its storage interval. If absent, Q prefix differences are used.
 * @li seq_used_kv: Optional INT32 ND tensor of shape [B]. Specifies nonnegative
 * effective KV lengths, taking precedence over KV prefix differences. With KV
 * prefixes, each length must fit its storage interval. If absent, differences are used.
 *
 * @par Attributes:
 * @li block_shape: Optional ListInt, default [1,128]. Q and KV block granularities;
 * only [1,128] is supported. Let Y=block_shape[1].
 * @li union_size: Optional int64, default 8. Query rows per group; only 8 is supported.
 * @li mask_type: Optional int64, default 0. Supports 0 (noncausal) and 1 (bottom-right
 * aligned causal). Both preserve query membership and clip tokens to effective KV
 * length. Mode 1 additionally requires p<KV_b-Q_b+qLocal+1 for each query.
 * @li totalq_union_num: Optional int64, default -1. Exact group count
 * G=sum_b ceil(Q_b/union_size). Zero is valid for an empty task; -1 requests automatic
 * inference from effective lengths, whose values must be available to shape inference.
 * @li max_logical_blocks: Optional int64, default -1. Exact maximum logical block
 * count L=max(1,max_b ceil(KV_b/Y)); must be positive when supplied. The default -1
 * requests automatic inference from effective lengths. Output capacity U=min(union_size*K,L).
 * @li max_q_seqlen: Optional int64, default -1; must be at least -1. With Q prefixes,
 * a positive value bounds effective Q length. Without Q prefixes, a positive value
 * is required and defines the equal-width Q storage slots.
 * @li max_kv_seqlen: Optional int64, default -1; must be at least -1. Compatibility
 * attribute; does not replace or bound effective KV lengths.
 * @li layout_sparse_pattern: Optional int64, default 4. Only layout 4 ([H,T,K]) is supported.
 * @li is_input_idx_sorted: Optional bool, default true. Promises ascending valid
 * input prefixes; only true is supported. Duplicate valid IDs are merged.
 * @li residual_block_mode: Optional int64, default 0. Only explicit-list mode 0 is
 * supported; unselected blocks are not automatically inserted.
 *
 * @par Outputs:
 * @li union_sparse_idx: INT32 ND tensor of shape [H,G,U]. Contains ascending unique
 * valid block IDs for each group, padded with -1. Invalid block IDs are excluded.
 * @li union_sparse_count: INT32 ND tensor of shape [H,G]. Actual union block counts.
 * @li exact_mask: INT32 ND tensor of shape [H,G,U*Y]. Bit i indicates that group-local
 * query i selected the block and can see this KV token. Padding and tokens outside
 * effective KV length are zero. Mask type does not change the block union or count.
 *
 * @par Third-party framework compatibility
 * Custom Ascend operator for sparse attention metadata.
 */
REG_OP(UnionSparseGen)
    .INPUT(sparse_block_idx, TensorType({DT_INT32}))
    .INPUT(sparse_block_count, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_sqlen_q, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(cu_sqlen_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(seq_used_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seq_used_kv, TensorType({DT_INT32}))
    .OUTPUT(union_sparse_idx, TensorType({DT_INT32}))
    .OUTPUT(union_sparse_count, TensorType({DT_INT32}))
    .OUTPUT(exact_mask, TensorType({DT_INT32}))
    .ATTR(block_shape, ListInt, {1, 128})
    .ATTR(union_size, Int, 8)
    .ATTR(mask_type, Int, 0)
    .ATTR(totalq_union_num, Int, -1)
    .ATTR(max_logical_blocks, Int, -1)
    .ATTR(max_q_seqlen, Int, -1)
    .ATTR(max_kv_seqlen, Int, -1)
    .ATTR(layout_sparse_pattern, Int, 4)
    .ATTR(is_input_idx_sorted, Bool, true)
    .ATTR(residual_block_mode, Int, 0)
    .OP_END_FACTORY_REG(UnionSparseGen)
} // namespace ge

#endif
