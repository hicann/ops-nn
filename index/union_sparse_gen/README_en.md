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
| exact_mask | Output | [H,G,U×Y] per-token query bit mask, padded with 0. | INT32 | ND |
| block_shape | Attribute | Q/KV block sizes. Default: [1,128]. Supports [1,Y] for Y in {4,8,16,128}. | ListInt | - |
| union_size | Attribute | Queries per group. Default and supported value: 8. | INT64 | - |
| mask_type | Attribute | 0: noncausal; 1: bottom-right causal. Default: 0. | INT64 | - |
| max_q_seqlen | Attribute | Default: -1. Ignored with either Q length tensor; otherwise the nonnegative fixed Q length. | INT64 | - |
| max_kv_seqlen | Attribute | Default: -1. Ignored with either KV length tensor; otherwise the nonnegative fixed KV length. | INT64 | - |
| layout_sparse_pattern | Attribute | Sparse pattern layout. Default and supported value: 4. | INT64 | - |
| is_input_idx_sorted | Attribute | Default true promises sorted valid prefixes; false accepts unsorted input. | BOOL | - |
| residual_block_mode | Attribute | Default 0 uses explicit selections; 1 adds the incomplete visible tail block into unionSparseIdx/unionSparseCount/exactMask. | INT64 | - |
| totalq_union_num | Attribute | G hint; tiling uses output G for -1 and requires every other value to match output G. | INT64 | - |
| max_logical_blocks | Attribute | Sorting/membership bound L: -1 infers the bound; positive values must be in [1, INT32_MAX] and also set graph U. | INT64 | - |

## Constraints

- Supports [1,Y] blocks (Y in {4,8,16,128}), groups of 8, sorted/unsorted input, mask types 0/1 and residual modes 0/1. Duplicate IDs are allowed.

Q and KV select length sources independently: seqUsed takes precedence over adjacent cu differences. If either tensor for a family is present, every int64 value of its max attribute is ignored. Only when both tensors are absent does max specify that family's fixed effective length, and max must be nonnegative.

| cuX | seqUsedX | Effective length | maxX |
|---|---|---|---|
| Present | Present | seqUsedX[b], bounded by the cuX difference | Ignored |
| Present | nullptr | cuX[b+1]−cuX[b] | Ignored |
| nullptr | Present | seqUsedX[b] | Ignored |
| nullptr | nullptr | maxX | Nonnegative fixed length |

With cuQ, Q rows use prefix offsets. With only seqUsedQ, B is its length, T must be divisible by B, slot=T/B, request b starts at b×slot, and seqUsedQ[b] must not exceed slot. Without either Q tensor, slot=maxQ, B=ceil(T/slot), request b starts at b×slot, and effectiveQ=min(slot,T-b×slot); the final slot may be partial. KV with only seqUsedKV has no Q-slot bound. Every present length tensor must describe the same B: prefixes have [B+1] elements and used lengths have [B]. When all four tensors are absent, positive maxQ determines B=ceil(T/maxQ), counting a partial final slot as one batch; maxQ=0 requires T=0 and denotes an empty batch. Present KV length tensors may instead establish a nonempty B with maxQ=0 and T=0. B=0 requires T=0 and has no actual groups; explicitly allocated extra groups are empty. Fixed KV requires ceil(maxKV/Y)≤INT32_MAX. KV need not exceed Q. The default max=-1 is legal only when ignored.

## Output shapes

Graph InferShape reads only host shape metadata and attributes, with no length-tensor data access or host/device synchronization. For input `[H,T,K]` (`[kvN,totalqBlockNum,maxKvBlockNum]`), outputs are `[H,G,U]`, `[H,G]`, and `[H,G,U*Y]`:

- Positive `totalq_union_num` is used as G; otherwise G=(T+7*B)//8.
- Positive `max_logical_blocks` is used as U; otherwise U=(K+residual)*union_size. Capacity is not clipped to the KV logical domain.
- B comes from length-tensor shapes. Fixed Q uses ceil(T/ceil(maxQSeqlen/block_shape[0])). With supported blockX=1, the final fixed slot may contain fewer than maxQ rows. A zero maxQ requires T=0 and retains the empty-batch/present-KV-tensor batch rules.
- The actual group count remains sum(ceil(effectiveQ/8)). Extra output groups contain idx=-1, count=0, mask=0. The caller must ensure G covers all actual groups; insufficient G is not guaranteed to raise an error and may leave valid groups unprocessed.
- U must cover every group's actual union. When the kernel detects that an actual union exceeds U, it triggers a device error instead of silently truncating that union. The error may surface during execution or stream synchronization; rejection at `aclnnUnionSparseGenGetWorkspaceSize` is not guaranteed. This check does not validate G or guarantee errors for every form of insufficient output capacity.
- The default U=(K+residual)*union_size reserves one additional tail candidate per query when residual=1; maskType=0 uses the same safe bound.

`maxLogicalBlocks` accepts `-1` or `[1, INT32_MAX]`. A positive value supplies the sorting/membership bound L and must cover the effective KV logical domain. With `-1`, variable KV uses `INT32_MAX`; fixed KV uses `max(1, ceil(maxKvSeqlen/Y))`. The host does not read back device lengths.

`totalqUnionNum=-1` uses output G; every other value must equal `unionSparseIdx.shape[1]`. ACLNN computes `U=min(unionSize*(K+residualBlockMode),L)` and checks output shapes `[H,G,U]`, `[H,G]`, and `[H,G,U*Y]`. Mismatches are rejected by GetWorkspaceSize, including when G=0.

U is storage capacity. Invalid block IDs (negative or outside the request's effective KV logical domain) are filtered out.

The Torch adapter retains its existing formulas: G=sum(ceil(effectiveQ/8)), L=max(1,max(ceil(effectiveKV/Y))), and U=min(8*(K+residual),L). It passes G/L and allocates outputs with that U, which ACLNN respects. Its low-level and Meta entries still require explicit G/L and do not implement GE's automatic shape rule. The CPU golden retains exact effective outputs; graph tests pad those outputs to G/U and check that no valid blocks are discarded.

- Indices/counts and outputs are INT32; cu prefixes are INT64. Low-level callers provide correct device lengths and G/Lmax without a device-to-host readback.

## Invocation Description

| Invocation method | Sample | Description |
|---|---|---|
| aclnn API | [test_aclnn_union_sparse_gen](./examples/test_aclnn_union_sparse_gen.cpp) | [aclnnUnionSparseGen](./docs/aclnnUnionSparseGen_en.md). |
| PyTorch extension | - | [Torch API](./docs/torchapi_union_sparse_gen_en.md). |
