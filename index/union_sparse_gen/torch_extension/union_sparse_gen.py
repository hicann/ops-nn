# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from typing import Optional, Tuple
import torch
import torch_npu  # noqa: F401
from torch.library import impl
from cann_ops_nn.op_builder import OpBuilder, get_as_library


_SUPPORTED_BLOCK_SHAPE = (1, 128)
_SUPPORTED_UNION_SIZE = 8
_SUPPORTED_LAYOUT = 4
_EXPLICIT_BLOCK_MODE = 0
_SPARSE_IDX_RANK = 3
_COUNT_RANK = 2
_SPARSE_ENTRY_AXIS = 2
_INT32_BYTES = 4
_INT32_MAX = torch.iinfo(torch.int32).max
_INT64_MAX = torch.iinfo(torch.int64).max


def _capacity(k, width, logical, residual):
    torch._check(
        width == _SUPPORTED_UNION_SIZE
        and residual == _EXPLICIT_BLOCK_MODE
        and 0 < k <= _INT64_MAX // width - residual
        and 0 < logical <= _INT32_MAX,
        lambda: "invalid width, residual mode, input capacity, or logical block bound",
    )
    return min(width * (k + residual), logical)


def _check_geometry(groups, logical):
    torch._check(
        groups >= 0 and 0 < logical <= _INT32_MAX,
        lambda: (
            "totalq_union_num and max_logical_blocks must be explicitly provided: "
            "G >= 0 and 1 <= Lmax <= INT32_MAX; -1 does not request automatic inference"
        ),
    )


class UnionSparseGenOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("union_sparse_gen_launch")

    def sources(self):
        return ["csrc/union_sparse_gen.cpp"]

    def schema(self):
        return (
            "union_sparse_gen_launch(Tensor sparse_block_idx, Tensor sparse_block_count, Tensor? cu_sqlen_q=None, "
            "Tensor? cu_sqlen_kv=None, Tensor? seq_used_q=None, Tensor? seq_used_kv=None, "
            "int[] block_shape=[1,128], int union_size=8, int mask_type=0, "
            "int max_q_seqlen=-1, int max_kv_seqlen=-1, int layout_sparse_pattern=4, "
            "bool is_input_idx_sorted=True, int residual_block_mode=0, "
            "int totalq_union_num=-1, int max_logical_blocks=-1) -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def meta(
            sparse_block_idx,
            sparse_block_count,
            cu_sqlen_q=None,
            cu_sqlen_kv=None,
            seq_used_q=None,
            seq_used_kv=None,
            block_shape=(1, 128),
            union_size=8,
            mask_type=0,
            max_q_seqlen=-1,
            max_kv_seqlen=-1,
            layout_sparse_pattern=4,
            is_input_idx_sorted=True,
            residual_block_mode=0,
            totalq_union_num=-1,
            max_logical_blocks=-1,
        ):
            _check_geometry(totalq_union_num, max_logical_blocks)
            torch._check(
                isinstance(block_shape, (list, tuple))
                and tuple(block_shape) == _SUPPORTED_BLOCK_SHAPE
                and union_size == _SUPPORTED_UNION_SIZE
                and mask_type in (0, 1)
                and is_input_idx_sorted
                and residual_block_mode == _EXPLICIT_BLOCK_MODE,
                lambda: "unsupported block/mask_type attributes",
            )
            torch._check(
                layout_sparse_pattern == _SUPPORTED_LAYOUT,
                lambda: "unsupported layout",
            )
            torch._check(
                cu_sqlen_kv is not None or seq_used_kv is not None,
                lambda: "KV lengths required",
            )
            torch._check(
                sparse_block_idx.dim() == _SPARSE_IDX_RANK,
                lambda: "invalid input/groups",
            )
            torch._check(
                sparse_block_idx.numel() <= _INT64_MAX // _INT32_BYTES,
                lambda: "input byte size overflow",
            )
            torch._check(
                sparse_block_idx.shape[0] > 0
                and sparse_block_count.shape == sparse_block_idx.shape[:_COUNT_RANK]
                and sparse_block_idx.dtype == torch.int32
                and sparse_block_count.dtype == torch.int32,
                lambda: "sparse tensors must be int32 [H,T,K] and [H,T], with H > 0",
            )
            torch._check(
                max_q_seqlen >= -1 and max_kv_seqlen >= -1,
                lambda: "max sequence lengths must be -1 or nonnegative",
            )
            if cu_sqlen_q is None:
                torch._check(
                    0 < max_q_seqlen <= _INT32_MAX,
                    lambda: "fixed Q requires positive max_q_seqlen",
                )
                torch._check(
                    seq_used_q is not None,
                    lambda: "fixed Q slots require seq_used_q",
                )
                torch._check(
                    seq_used_q.dim() == 1 and seq_used_q.dtype == torch.int32,
                    lambda: "seq_used_q must be int32 [B]",
                )
                batch = seq_used_q.numel()
                torch._check(
                    sparse_block_idx.shape[1] == 0
                    if batch == 0
                    else sparse_block_idx.shape[1] == batch * max_q_seqlen,
                    lambda: "fixed Q requires T=B*max_q_seqlen (T=0 when B=0)",
                )
            else:
                torch._check(
                    cu_sqlen_q.dim() == 1
                    and cu_sqlen_q.numel() >= 1
                    and cu_sqlen_q.dtype == torch.int64,
                    lambda: "cu_sqlen_q must be int64 [B+1]",
                )
                batch = cu_sqlen_q.numel() - 1
            for length, size, dtype in (
                (cu_sqlen_kv, batch + 1, torch.int64),
                (seq_used_q, batch, torch.int32),
                (seq_used_kv, batch, torch.int32),
            ):
                if length is not None:
                    torch._check(
                        length.dim() == 1
                        and length.numel() == size
                        and length.dtype == dtype,
                        lambda: "length tensors must have matching batch geometry and prefix/used dtype",
                    )
            torch._check(
                batch != 0 or totalq_union_num == 0, lambda: "B=0 requires G=0"
            )
            capacity = _capacity(
                sparse_block_idx.shape[_SPARSE_ENTRY_AXIS],
                union_size,
                max_logical_blocks,
                residual_block_mode,
            )
            torch._check(
                sparse_block_idx.shape[0] * totalq_union_num * capacity * block_shape[1]
                <= _INT64_MAX // _INT32_BYTES,
                lambda: "output byte size overflow",
            )
            options = dict(dtype=torch.int32, device="meta")
            return (
                torch.empty(
                    (sparse_block_idx.shape[0], totalq_union_num, capacity), **options
                ),
                torch.empty((sparse_block_idx.shape[0], totalq_union_num), **options),
                torch.empty(
                    (
                        sparse_block_idx.shape[0],
                        totalq_union_num,
                        capacity * block_shape[1],
                    ),
                    **options,
                ),
            )


union_sparse_gen_builder = UnionSparseGenOpBuilder()
union_sparse_gen_builder._ensure_initialized()


# This loader remains available until the module registers its native backend.
# The dedicated Meta implementation handles Fake calls without loading C++.
@impl(get_as_library(), union_sparse_gen_builder.name, "CompositeExplicitAutograd")
def union_sparse_gen_launch(
    sparse_block_idx,
    sparse_block_count,
    cu_sqlen_q=None,
    cu_sqlen_kv=None,
    seq_used_q=None,
    seq_used_kv=None,
    block_shape=(1, 128),
    union_size=8,
    mask_type=0,
    max_q_seqlen=-1,
    max_kv_seqlen=-1,
    layout_sparse_pattern=4,
    is_input_idx_sorted=True,
    residual_block_mode=0,
    totalq_union_num=-1,
    max_logical_blocks=-1,
):
    _check_geometry(totalq_union_num, max_logical_blocks)
    torch._check(
        sparse_block_idx.device.type in ("npu", "privateuseone"),
        lambda: "sparse_block_idx must be an NPU tensor",
    )
    torch._check(
        isinstance(block_shape, (list, tuple))
        and tuple(block_shape) == _SUPPORTED_BLOCK_SHAPE
        and union_size == _SUPPORTED_UNION_SIZE
        and mask_type in (0, 1)
        and layout_sparse_pattern == _SUPPORTED_LAYOUT
        and is_input_idx_sorted
        and residual_block_mode == _EXPLICIT_BLOCK_MODE,
        lambda: "supported spec: block_shape=[1,128]/W8/mask0/1/layout4/sorted=true/residual0",
    )
    return union_sparse_gen_builder.load().union_sparse_gen_launch(
        sparse_block_idx,
        sparse_block_count,
        cu_sqlen_q,
        cu_sqlen_kv,
        seq_used_q,
        seq_used_kv,
        block_shape,
        union_size,
        mask_type,
        max_q_seqlen,
        max_kv_seqlen,
        layout_sparse_pattern,
        is_input_idx_sorted,
        residual_block_mode,
        totalq_union_num,
        max_logical_blocks,
    )


def _cpu_lengths(name: str, tensor: torch.Tensor) -> list[int]:
    torch._check(
        tensor.device.type == "cpu",
        lambda: f"{name} must be CPU-visible metadata",
    )
    torch._check(
        tensor.dtype in (torch.int32, torch.int64),
        lambda: f"{name} must be int32 or int64",
    )
    torch._check(tensor.dim() == 1, lambda: f"{name} must be rank 1")
    values = [int(v) for v in tensor.tolist()]
    torch._check(
        all(
            0
            <= value
            <= (_INT64_MAX if name in ("cu_sqlen_q", "cu_sqlen_kv") else _INT32_MAX)
            for value in values
        ),
        lambda: f"{name} values must fit nonnegative {'int64' if name.startswith('cu_') else 'int32'}",
    )
    return values


def _cpu_geometry_lengths(total_q, cu_q, cu_kv, used_q, used_kv, max_q=-1, max_kv=-1):
    if max_q < -1 or max_kv < -1:
        torch._check(False, lambda: "max sequence lengths must be -1 or nonnegative")
    if cu_q is None:
        if used_q is None:
            torch._check(False, lambda: "fixed Q slots require seq_used_q")
        if not 0 < max_q <= _INT32_MAX:
            torch._check(False, lambda: "fixed Q slots require positive max_q_seqlen")
        q_lens = _cpu_lengths("seq_used_q", used_q)
        batch = len(q_lens)
        if total_q != batch * max_q:
            torch._check(False, lambda: "fixed Q storage must equal B * max_q_seqlen")
        if any(n < 0 or n > max_q for n in q_lens):
            torch._check(False, lambda: "seq_used_q must fit fixed Q capacity")
        q_offsets = [b * max_q for b in range(batch + 1)]
    else:
        q_offsets = _cpu_lengths("cu_sqlen_q", cu_q)
        if not q_offsets or q_offsets[0] != 0 or q_offsets[-1] != total_q:
            torch._check(False, lambda: "cu_sqlen_q must start at 0 and cover T")
        q_capacity = [b - a for a, b in zip(q_offsets, q_offsets[1:])]
        if any(n < 0 for n in q_capacity):
            torch._check(False, lambda: "Q prefix must be nondecreasing")
        batch = len(q_capacity)
        q_lens = q_capacity if used_q is None else _cpu_lengths("seq_used_q", used_q)
        if len(q_lens) != batch or any(
            n < 0 or n > c for n, c in zip(q_lens, q_capacity)
        ):
            torch._check(False, lambda: "seq_used_q must fit cu_sqlen_q storage")
    if max_q > 0 and any(n > max_q for n in q_lens):
        torch._check(False, lambda: "effective Q exceeds max_q_seqlen")
    if cu_kv is None:
        if used_kv is None:
            torch._check(False, lambda: "seq_used_kv is required without cu_sqlen_kv")
        kv_lens = _cpu_lengths("seq_used_kv", used_kv)
        if len(kv_lens) != batch or any(n < 0 for n in kv_lens):
            torch._check(
                False, lambda: "seq_used_kv must match batch with nonnegative lengths"
            )
    else:
        kv_offsets = _cpu_lengths("cu_sqlen_kv", cu_kv)
        if len(kv_offsets) != batch + 1 or kv_offsets[0] != 0:
            torch._check(False, lambda: "KV prefix must start at 0 and match Q batch")
        capacity = [b - a for a, b in zip(kv_offsets, kv_offsets[1:])]
        if any(n < 0 for n in capacity):
            torch._check(False, lambda: "KV prefix must be nondecreasing")
        kv_lens = capacity if used_kv is None else _cpu_lengths("seq_used_kv", used_kv)
        if len(kv_lens) != batch or any(
            n < 0 or n > c for n, c in zip(kv_lens, capacity)
        ):
            torch._check(False, lambda: "seq_used_kv must fit cu_sqlen_kv storage")
    return q_lens, kv_lens, batch


def union_sparse_gen(
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    cu_sqlen_q: Optional[torch.Tensor] = None,
    cu_sqlen_kv: Optional[torch.Tensor] = None,
    seq_used_q: Optional[torch.Tensor] = None,
    seq_used_kv: Optional[torch.Tensor] = None,
    *,
    block_shape: Tuple[int, int] = (1, 128),
    union_size: int = 8,
    mask_type: int = 0,
    max_q_seqlen: int = -1,
    max_kv_seqlen: int = -1,
    layout_sparse_pattern: int = 4,
    is_input_idx_sorted: bool = True,
    residual_block_mode: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the logical union and token-wise query bitmask metadata."""
    torch._check(
        sparse_block_idx.device.type == "npu",
        lambda: "sparse_block_idx must be an NPU tensor",
    )
    torch._check(
        sparse_block_count.device == sparse_block_idx.device,
        lambda: "sparse_block_count device mismatch",
    )
    torch._check(
        sparse_block_idx.dtype == torch.int32
        and sparse_block_count.dtype == torch.int32,
        lambda: "sparse_block_idx and sparse_block_count must be int32",
    )
    torch._check(
        sparse_block_idx.dim() == _SPARSE_IDX_RANK,
        lambda: "sparse_block_idx must be [H,T,K]",
    )
    torch._check(
        sparse_block_idx.numel() <= _INT64_MAX // _INT32_BYTES,
        lambda: "input byte size overflow",
    )
    torch._check(
        sparse_block_count.shape == sparse_block_idx.shape[:_COUNT_RANK],
        lambda: "sparse_block_count must be [H,T]",
    )
    # Validate scalar attributes together: separate symbolic checks add measurable
    # eager-call overhead for short kernels. Fake/host validation stays independent.
    torch._check(
        isinstance(block_shape, (list, tuple))
        and tuple(block_shape) == _SUPPORTED_BLOCK_SHAPE
        and union_size == _SUPPORTED_UNION_SIZE
        and mask_type in (0, 1)
        and is_input_idx_sorted
        and layout_sparse_pattern == _SUPPORTED_LAYOUT
        and residual_block_mode == _EXPLICIT_BLOCK_MODE,
        lambda: (
            f"unsupported attributes: block_shape={block_shape}, union_size={union_size}, "
            f"mask_type={mask_type}, layout_sparse_pattern={layout_sparse_pattern}, "
            f"residual_block_mode={residual_block_mode}, max_q_seqlen={max_q_seqlen}, max_kv_seqlen={max_kv_seqlen}"
        ),
    )
    q_lens, kv_lens, batch = _cpu_geometry_lengths(
        sparse_block_idx.shape[1],
        cu_sqlen_q,
        cu_sqlen_kv,
        seq_used_q,
        seq_used_kv,
        max_q_seqlen,
        max_kv_seqlen,
    )
    num_groups = sum((length + union_size - 1) // union_size for length in q_lens)
    max_logical_blocks = max(
        1,
        max(
            ((length + block_shape[1] - 1) // block_shape[1] for length in kv_lens),
            default=0,
        ),
    )
    capacity = _capacity(
        sparse_block_idx.shape[_SPARSE_ENTRY_AXIS],
        union_size,
        max_logical_blocks,
        residual_block_mode,
    )
    torch._check(
        sparse_block_idx.shape[0] * num_groups * capacity * block_shape[1]
        <= _INT64_MAX // _INT32_BYTES,
        lambda: "output byte size overflow",
    )

    def to_npu(tensor: torch.Tensor, dtype=torch.int32) -> torch.Tensor:
        return tensor.to(device=sparse_block_idx.device, dtype=dtype).contiguous()

    npu_cu_q = to_npu(cu_sqlen_q, torch.int64) if cu_sqlen_q is not None else None
    npu_cu_kv = to_npu(cu_sqlen_kv, torch.int64) if cu_sqlen_kv is not None else None
    npu_seq_q = to_npu(seq_used_q) if seq_used_q is not None else None
    npu_seq_kv = to_npu(seq_used_kv) if seq_used_kv is not None else None
    torch._check(len(q_lens) == batch, lambda: "unreachable batch mismatch")
    return torch.ops.cann_ops_nn.union_sparse_gen_launch(
        sparse_block_idx,
        sparse_block_count,
        npu_cu_q,
        npu_cu_kv,
        npu_seq_q,
        npu_seq_kv,
        block_shape,
        union_size,
        mask_type,
        max_q_seqlen,
        max_kv_seqlen,
        layout_sparse_pattern,
        is_input_idx_sorted,
        residual_block_mode,
        num_groups,
        max_logical_blocks,
    )
