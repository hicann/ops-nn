# UnionSparseGen

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|---|:---:|
| <term>Ascend 960PR&960DT products</term> | × |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

Merges explicitly selected KV blocks from adjacent queries in one request into a sorted unique union. Returns union indices, counts and a token-wise mask that preserves each query's selection.

## Parameters

The table uses parameter names and tensor types from the operator prototype. See the respective API documents for ACLNN camelCase names and length-tensor conversion in the public Torch entry point.

| Parameter | Input/Output/Attribute | Description | Data type | Format |
|---|---|---|---|---|
| sparse_block_idx | Input | [H,T,K] logical KV block IDs selected by each query. | INT32 | ND |
| sparse_block_count | Input | [H,T] valid prefix length of each input row. | INT32 | ND |
| cu_sqlen_q | Optional input | [B+1] prefix sum of Q storage lengths. | INT64 | ND |
| cu_sqlen_kv | Optional input | [B+1] prefix sum of KV storage lengths. | INT64 | ND |
| seq_used_q | Optional input | [B] effective Q length of each request. | INT32 | ND |
| seq_used_kv | Optional input | [B] effective KV length of each request. | INT32 | ND |
| union_sparse_idx | Output | [H,G,U] sorted unique block union, padded with -1. | INT32 | ND |
| union_sparse_count | Output | [H,G] number of union blocks in each group. | INT32 | ND |
| exact_mask | Output | [H,G,U×128] per-token query bit mask, padded with 0. | INT32 | ND |
| block_shape | Attribute | Q/KV block sizes. Default and supported value: [1,128]. | ListInt | - |
| union_size | Attribute | Queries per group. Default and supported value: 8. | INT64 | - |
| mask_type | Attribute | 0: noncausal; 1: bottom-right causal. Default: 0. | INT64 | - |
| max_q_seqlen | Attribute | Default: -1. Positive values bound effective Q and specify fixed Q slot width without cuQ. | INT64 | - |
| max_kv_seqlen | Attribute | Default: -1. Does not supply or bound effective KV length. | INT64 | - |
| layout_sparse_pattern | Attribute | Sparse pattern layout. Default and supported value: 4. | INT64 | - |
| is_input_idx_sorted | Attribute | Promises sorted valid input prefixes. Default and supported value: true. | BOOL | - |
| residual_block_mode | Attribute | Default and supported value: 0, explicit selected-block lists. | INT64 | - |
| totalq_union_num | Attribute | Total group count G. Required by low-level calls; derived by the public Python API. | INT64 | - |
| max_logical_blocks | Attribute | Maximum logical block count Lmax. Required by low-level calls; derived by the public Python API. | INT64 | - |

## Constraints

- Supports [1,128] blocks, groups of 8, sorted explicit lists and mask types 0/1. Input duplicates are allowed; no implicit tail block is added.
- Q and KV resolve their prefix inputs independently. Without cuQ, seqUsedQ and positive maxQ are required, and T=B×maxQ. Without cuKV, seqUsedKV is required.
- Effective lengths prefer seqUsed over prefix differences. Fixed slots can have different effective lengths. No separate Q/KV layout attribute is used.
- G=Σceil(effectiveQ/8), Lmax=max(1,max ceil(effectiveKV/128)), U=min(8K,Lmax). Groups do not cross requests.
- Indices/counts and outputs are INT32; cu prefixes are INT64. Low-level callers provide correct device lengths and G/Lmax without a device-to-host readback.

## Invocation Description

| Invocation method | Sample | Description |
|---|---|---|
| aclnn API | [test_aclnn_union_sparse_gen](./examples/test_aclnn_union_sparse_gen.cpp) | [aclnnUnionSparseGen](./docs/aclnnUnionSparseGen_en.md). |
| PyTorch extension | - | [Torch API](./docs/torchapi_union_sparse_gen_en.md). |
