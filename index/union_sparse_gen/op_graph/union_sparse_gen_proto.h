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
 * Without it, Q uses T/B equal-width slots. If seq_used_q is also absent, max_q_seqlen
 * gives the slot width, with B=ceil(T/max_q_seqlen); the final slot uses its remaining Q rows.
 * @li cu_sqlen_kv: Optional INT64 ND tensor of shape [B+1], specifying nondecreasing
 * KV storage prefixes starting at zero. Without it or seq_used_kv, max_kv_seqlen
 * gives the fixed effective KV length. Q and KV select length sources independently.
 * @li seq_used_q: Optional INT32 ND tensor of shape [B]. Specifies nonnegative
 * effective Q lengths, taking precedence over Q prefix differences. Each effective
 * length must fit its storage interval. If absent, Q prefix differences or fixed maxQ are used.
 * @li seq_used_kv: Optional INT32 ND tensor of shape [B]. Specifies nonnegative
 * effective KV lengths, taking precedence over KV prefix differences. With KV
 * prefixes, each length must fit its storage interval. If absent, differences or fixed maxKV are used.
 *
 * @par Attributes:
 * @li block_shape: Optional ListInt, default [1,128]. Q and KV block granularities;
 * supports [1,Y], Y in {4,8,16,128}.
 * @li union_size: Optional int64, default 8. Query rows per group; only 8 is supported.
 * @li mask_type: Optional int64, default 0. Supports 0 (noncausal) and 1 (bottom-right
 * aligned causal). Both preserve query membership and clip tokens to effective KV
 * length. Mode 1 additionally requires p<KV_b-Q_b+qLocal+1 for each query.
 * @li totalq_union_num: Optional int64, default -1. Positive values specify output G;
 * nonpositive values infer G=floor((T+7*B)/8) from host shape metadata only.
 * Fixed Q uses B=ceil(T/ceil(max_q_seqlen/block_shape[0])); valid storage must fit.
 * @li max_logical_blocks: Optional int64, default -1. Positive values specify output
 * capacity U; nonpositive values infer U=(K+residual)*union_size. This is not a block-ID bound.
 * G/U must cover the union including residual tails. Extra groups
 * are empty (idx=-1, count=0, mask=0); insufficient U raises a device error.
 * Direct ACLNN takes G/U from the caller's output shapes and ignores these hints.
 * @li max_q_seqlen: Optional int64, default -1. Ignored for every int64 value when
 * either Q length tensor is present; otherwise a nonnegative fixed Q length.
 * @li max_kv_seqlen: Optional int64, default -1. Ignored for every int64 value when
 * either KV length tensor is present; otherwise a nonnegative fixed KV length,
 * with ceil(max_kv_seqlen/Y)<=INT32_MAX.
 * @li layout_sparse_pattern: Optional int64, default 4. Only layout 4 ([H,T,K]) is supported.
 * @li is_input_idx_sorted: Optional bool, default true. Promises ascending valid
 * input prefixes; false allows unsorted input. Duplicate valid IDs are merged.
 * @li residual_block_mode: Optional int64, default 0, uses explicit blocks only.
 * Mode 1 retains explicit blocks below floor(max(V,0)/Y) and inserts floor(V/Y)
 * when V>0 and V%Y!=0 before union. V=KV-Q+qLocal+1 for mask_type=1 and KV
 * for mask_type=0. Added tails affect IDs, counts and masks; complete tails are skipped.
 *
 * @par Outputs:
 * @li union_sparse_idx: INT32 ND tensor of shape [H,G,U]. Contains ascending unique
 * valid block IDs for each group, padded with -1. Invalid block IDs are excluded.
 * @li union_sparse_count: INT32 ND tensor of shape [H,G]. Actual union block counts.
 * @li exact_mask: INT32 ND tensor of shape [H,G,U*Y]. Bit i indicates that group-local
 * query i has explicit/residual membership and can see this KV token. Padding and tokens outside
 * effective KV length are zero. Duplicate explicit/tail IDs are merged.
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
