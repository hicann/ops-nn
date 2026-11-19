# union_sparse_gen

English | [简体中文](./torchapi_union_sparse_gen.md)

## Function

Builds a sorted block union, its count and an exact per-token query bit mask. The mask preserves membership and the effective KV boundary; mask_type=1 also applies bottom-right causal visibility. It does not perform attention. H denotes KV heads, T stored query rows, and B requests. Block IDs are relative to each request's KV sequence, not global token offsets.

## Function Prototype

```python
def union_sparse_gen(
    sparse_block_idx, sparse_block_count,
    cu_sqlen_q=None, cu_sqlen_kv=None,
    seq_used_q=None, seq_used_kv=None,
    *, block_shape=(1, 128), union_size=8, mask_type=0,
    max_q_seqlen=-1, max_kv_seqlen=-1,
    layout_sparse_pattern=4, is_input_idx_sorted=True,
    residual_block_mode=0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

The selective extension package exports `cann_ops_nn_custom.union_sparse_gen`; the full package exports `cann_ops_nn.union_sparse_gen`. The examples use the alias `ops`.

## Parameters

| Parameter | Description |
|---|---|
| sparse_block_idx | [H,T,K] logical KV block IDs selected by each query. NPU INT32. |
| sparse_block_count | [H,T] valid prefix length of each input row. NPU INT32. |
| cu_sqlen_q | [B+1] prefix sum of Q storage lengths. CPU INT32/INT64 Tensor or None. |
| cu_sqlen_kv | [B+1] prefix sum of KV storage lengths. CPU INT32/INT64 Tensor or None. |
| seq_used_q | [B] effective Q length of each request. CPU INT32/INT64 Tensor or None. |
| seq_used_kv | [B] effective KV length of each request. CPU INT32/INT64 Tensor or None. |
| block_shape | Q/KV block sizes. Default: [1,128]. Supports [1,Y] for Y in {4,8,16,128}. |
| union_size | Queries per group. Default and supported value: 8. |
| mask_type | 0: noncausal; 1: bottom-right causal. Default: 0. |
| max_q_seqlen | Default: -1. Ignored with either Q length tensor; otherwise the nonnegative fixed Q length. |
| max_kv_seqlen | Default: -1. Ignored with either KV length tensor; otherwise the nonnegative fixed KV length. |
| layout_sparse_pattern | Sparse pattern layout. Default and supported value: 4. |
| is_input_idx_sorted | Default true promises sorted valid prefixes; false accepts unsorted input. |
| residual_block_mode | Default 0 uses explicit selections; 1 adds incomplete visible tails to IDs, counts and masks. |

## Return Value

Three NPU INT32 tensors, in order:

```text
union_sparse_idx   [H,G,U]
union_sparse_count [H,G]
exact_mask         [H,G,U×Y]
G = Σ_b ceil(effectiveQ_b/8)
Lmax = max(1, max_b ceil(effectiveKV_b/Y))
U = min(8*(K+residual_block_mode), Lmax)
```

Indices are ascending and unique, with -1 padding. Mask padding and out-of-range KV tokens are zero. Bit i corresponds to query i within the group. Noncausal masks preserve per-query membership; they are not all ones.

## Constraints

- The supported attributes are block_shape=[1,Y] (Y in {4,8,16,128}), union_size=8, mask_type=0/1, layout_sparse_pattern=4, sorted=true/false and residual=0/1.
- Valid input prefixes must be nondecreasing when is_input_idx_sorted=true; false accepts unsorted input. Both allow duplicates, and sparse_block_count bounds each prefix. Invalid block IDs are ignored. Mode 0 retains legal explicit blocks. Mode 1 restores the per-query selection rule: retain explicit blocks fully within the visible range (d < floor(max(V,0)/Y)), add floor(V/Y) when V>0 and V%Y!=0, then merge and deduplicate. Added tails participate in unionSparseIdx, unionSparseCount and exactMask. V is KV-Q+localQuery+1 for maskType=1 and the effective KV length for maskType=0. Complete tails are not automatically added; existing tails are not counted twice.

Q and KV select length sources independently: seqUsed takes precedence over adjacent cu differences. If either tensor for a family is present, every int64 value of its max attribute is ignored. Only when both tensors are absent does max specify that family's fixed effective length, and max must be nonnegative.

| cuX | seqUsedX | Effective length | maxX |
|---|---|---|---|
| Present | Present | seqUsedX[b], bounded by the cuX difference | Ignored |
| Present | nullptr | cuX[b+1]−cuX[b] | Ignored |
| nullptr | Present | seqUsedX[b] | Ignored |
| nullptr | nullptr | maxKV for KV; min(maxQ,T-b×maxQ) for Q | Nonnegative fixed slot width/length |

With cuQ, Q rows use prefix offsets. With only seqUsedQ, B is its length, T must be divisible by B, slot=T/B, request b starts at b×slot, and seqUsedQ[b] must not exceed slot. Without either Q tensor, slot=maxQ, B=ceil(T/maxQ), and effectiveQ=min(maxQ,T-b×maxQ), including a partial final slot. For example, T=10 and maxQ=6 give Q lengths [6,4]. KV with only seqUsedKV has no Q-slot bound. Every present length tensor must describe the same B: prefixes have [B+1] elements and used lengths have [B]; for fixed Q, KV descriptors must agree with ceil(T/maxQ). When all four tensors are absent, positive maxQ determines B=ceil(T/maxQ); maxQ=0 requires T=0 and denotes an empty batch. Present KV length tensors may instead establish a nonempty B with maxQ=0 and T=0. B=0 requires T=G=0. Fixed KV requires ceil(maxKV/Y)≤INT32_MAX. KV need not exceed Q. The default max=-1 is legal only when ignored.

- CPU lengths are validated before transfer. cu prefixes are converted to NPU INT64; seqUsed lengths are converted to NPU INT32. No separate Q/KV layout attribute is used.
- The public wrapper derives G/Lmax/U from CPU lengths and allocates outputs. Calls include validation and metadata transfers; first use may JIT-compile the C++ bridge and requires a working C++ compiler and ninja. If ninja is missing, install it in the active Python environment with `python -m pip install ninja`.

## Low-Level API

```python
torch.ops.cann_ops_nn.union_sparse_gen_launch(
    sparse_block_idx, sparse_block_count,
    cu_sqlen_q, cu_sqlen_kv, seq_used_q, seq_used_kv,
    block_shape, union_size, mask_type, max_q_seqlen, max_kv_seqlen,
    layout_sparse_pattern, is_input_idx_sorted, residual_block_mode,
    totalq_union_num, max_logical_blocks,
)
```

All non-null tensor inputs must be on the same NPU. cu inputs must be INT64; other inputs are INT32. The caller supplies exact G and Lmax and guarantees valid device length values. The low-level API does not read back device lengths. G=0 denotes a real empty task, not automatic inference. Fake/Meta uses the supplied geometry and does not compute length values. Output capacity is always derived.

The low-level defaults for both `totalq_union_num` and `max_logical_blocks` are `-1`. This sentinel detects omitted arguments; it does not request automatic inference. Omitting either argument, supplying a negative G, or supplying a nonpositive Lmax raises an error before output allocation and kernel launch. Lmax must also be at most INT32_MAX. The public `union_sparse_gen` wrapper continues to compute and pass exact geometry automatically; its calling convention is unchanged.

For directly constructed GE nodes, InferShape reads only host shapes and attributes, without length tensor values or device synchronization. Positive totalq_union_num specifies G; otherwise G=(T+7*B)//8. Positive max_logical_blocks specifies capacity U; otherwise U=8*(K+residual). B comes from length descriptors or ceil(T/maxQ) for fixed Q, including a partial final slot. Direct ACLNN respects output shapes; Torch retains the G/L/U allocation rules above.

## Example

With the extension and its matching ACLNN operator package installed:

```python
import torch
import torch_npu
import cann_ops_nn_custom as ops

idx = torch.tensor([[[0], [0]]], dtype=torch.int32, device="npu:0")
count = torch.ones((1, 2), dtype=torch.int32, device="npu:0")
cu_q = torch.tensor([0, 2], dtype=torch.int64)
cu_kv = torch.tensor([0, 128], dtype=torch.int64)
union_idx, union_count, mask = ops.union_sparse_gen(
    idx, count, cu_q, cu_kv, mask_type=1,
)
assert union_idx.shape == (1, 1, 1)
assert union_count.cpu().tolist() == [[1]]
assert torch.all(mask[0, 0, :127] == 3)
assert mask[0, 0, 127].item() == 2
```

For the C API and exact visibility formula, see [aclnnUnionSparseGen](./aclnnUnionSparseGen_en.md).
