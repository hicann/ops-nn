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
| block_shape | Q/KV block sizes. Default and supported value: [1,128]. |
| union_size | Queries per group. Default and supported value: 8. |
| mask_type | 0: noncausal; 1: bottom-right causal. Default: 0. |
| max_q_seqlen | Default: -1. Positive values bound effective Q and specify fixed Q slot width without cuQ. |
| max_kv_seqlen | Default: -1. Does not supply or bound effective KV length. |
| layout_sparse_pattern | Sparse pattern layout. Default and supported value: 4. |
| is_input_idx_sorted | Promises sorted valid input prefixes. Default and supported value: true. |
| residual_block_mode | Default and supported value: 0, explicit selected-block lists. |

## Return Value

Three NPU INT32 tensors, in order:

```text
union_sparse_idx   [H,G,U]
union_sparse_count [H,G]
exact_mask         [H,G,U×128]
G = Σ_b ceil(effectiveQ_b/8)
Lmax = max(1, max_b ceil(effectiveKV_b/128))
U = min(8K, Lmax)
```

Indices are ascending and unique, with -1 padding. Mask padding and out-of-range KV tokens are zero. Bit i corresponds to query i within the group. Noncausal masks preserve per-query membership; they are not all ones.

## Constraints

- The supported attributes are block_shape=[1,128], union_size=8, mask_type=0/1, layout_sparse_pattern=4, sorted=true and residual=0.
- Input prefixes are nondecreasing, may contain duplicates, and are bounded by sparse_block_count. Invalid block IDs are ignored; legal blocks are retained even if their causal mask is all zero.
- Q and KV resolve cu presence independently. With cuQ, request offsets use cuQ and effective Q prefers seqUsedQ over prefix differences. Without cuQ, seqUsedQ and positive maxQ are required, T=B×maxQ, and request b starts at b×maxQ.
- Effective KV prefers seqUsedKV, otherwise cuKV differences. Without cuKV, seqUsedKV is required. A seqUsed length cannot exceed its prefix capacity.
- Both max attributes must be at least -1. Positive maxQ bounds effective Q and, in fixed Q storage, must not exceed INT32_MAX. maxKV neither supplies nor limits effective KV.
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

For directly constructed GE nodes, both attributes are optional: `-1` derives the missing geometry from length tensor values, using `G=sum(ceil(effectiveQ[b]/8))` and `Lmax=max(1,max_b ceil(effectiveKV[b]/128))`. Present `seq_used` values take precedence over prefix differences. Fixed Q slots still require `seq_used_q` and a positive `max_q_seqlen`. Inference precedes exact output allocation; device-only lengths introduce a runtime value dependency. ACLNN and low-level Torch still require explicit geometry. The host geometry resolver does not actively read length values when both hints are explicit; GE controls framework-level value-dependency scheduling.

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
