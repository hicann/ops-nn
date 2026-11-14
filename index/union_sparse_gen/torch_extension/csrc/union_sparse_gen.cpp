/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/extension.h>
#include <torch/library.h>

#include "aclnnop/aclnn_union_sparse_gen.h"
#include "aclnn_common.h"
#include <algorithm>
#include <limits>

namespace cann_ops_nn {
namespace index {
namespace {
constexpr int64_t kBlockShapeDimensions = 2;
constexpr int64_t kBlockXAxis = 0, kBlockYAxis = 1;
constexpr int64_t kSupportedBlockX = 1, kSupportedBlockY = 128, kSupportedUnionSize = 8;
constexpr int64_t kSparseIdxRank = 3, kHeadAxis = 0, kSparseEntryAxis = 2;
constexpr int64_t kNonCausalMask = 0, kCausalMask = 1, kSparsePatternLayout = 4, kExplicitBlocks = 0;
constexpr int64_t kIndexBytes = sizeof(int32_t);
void CheckInputTensors(const at::Tensor& sparseBlockIdx, const at::Tensor& sparseBlockCount,
                       const c10::optional<at::Tensor>& cuSqlenQ, const c10::optional<at::Tensor>& cuSqlenKv,
                       const c10::optional<at::Tensor>& seqUsedQ, const c10::optional<at::Tensor>& seqUsedKv)
{
    TORCH_CHECK(sparseBlockIdx.device().type() == at::kPrivateUse1, "sparse_block_idx must be an NPU tensor");
    TORCH_CHECK(sparseBlockCount.device() == sparseBlockIdx.device(), "sparse_block_count device mismatch");
    TORCH_CHECK((!cuSqlenQ.has_value() || cuSqlenQ->device() == sparseBlockIdx.device()), "cu_sqlen_q device mismatch");
    TORCH_CHECK((!cuSqlenKv.has_value() || cuSqlenKv->device() == sparseBlockIdx.device()),
                "cu_sqlen_kv device mismatch");
    TORCH_CHECK((!seqUsedQ.has_value() || seqUsedQ->device() == sparseBlockIdx.device()), "seq_used_q device mismatch");
    TORCH_CHECK((!seqUsedKv.has_value() || seqUsedKv->device() == sparseBlockIdx.device()),
                "seq_used_kv device mismatch");
    TORCH_CHECK(sparseBlockIdx.scalar_type() == at::kInt, "sparse_block_idx must be int32");
    TORCH_CHECK(sparseBlockCount.scalar_type() == at::kInt, "sparse_block_count must be int32");
    TORCH_CHECK((!cuSqlenQ.has_value() || cuSqlenQ->scalar_type() == at::kLong), "cu_sqlen_q must be int64");
    TORCH_CHECK((!cuSqlenKv.has_value() || cuSqlenKv->scalar_type() == at::kLong), "cu_sqlen_kv must be int64");
    TORCH_CHECK(sparseBlockIdx.is_contiguous(), "sparse_block_idx must be contiguous");
    TORCH_CHECK(sparseBlockCount.is_contiguous(), "sparse_block_count must be contiguous");
    TORCH_CHECK((!cuSqlenQ.has_value() || cuSqlenQ->is_contiguous()), "cu_sqlen_q must be contiguous");
    TORCH_CHECK((!cuSqlenKv.has_value() || cuSqlenKv->is_contiguous()), "cu_sqlen_kv must be contiguous");
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> union_sparse_gen_launch(
    const at::Tensor& sparseBlockIdx, const at::Tensor& sparseBlockCount, const c10::optional<at::Tensor>& cuSqlenQ,
    const c10::optional<at::Tensor>& cuSqlenKv, const c10::optional<at::Tensor>& seqUsedQ,
    const c10::optional<at::Tensor>& seqUsedKv, at::IntArrayRef blockShape, int64_t unionSize, int64_t maskType,
    int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t layoutSparsePattern, bool isInputIdxSorted,
    int64_t residualBlockMode, int64_t totalqUnionNum, int64_t maxLogicalBlocks)
{
    CheckInputTensors(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv);
    TORCH_CHECK(blockShape.size() == kBlockShapeDimensions && blockShape[kBlockXAxis] == kSupportedBlockX &&
                    blockShape[kBlockYAxis] == kSupportedBlockY && unionSize == kSupportedUnionSize &&
                    (maskType == kNonCausalMask || maskType == kCausalMask) &&
                    layoutSparsePattern == kSparsePatternLayout && isInputIdxSorted &&
                    residualBlockMode == kExplicitBlocks,
                "supported spec: block_shape=[1,128]/W8/mask0/1/layout4/sorted=true/residual0");
    TORCH_CHECK(totalqUnionNum >= 0 && maxLogicalBlocks > 0 && maxLogicalBlocks <= std::numeric_limits<int32_t>::max(),
                "totalq_union_num and max_logical_blocks must be explicitly provided: "
                "G >= 0 and 1 <= Lmax <= INT32_MAX; -1 does not request automatic inference");
    const int64_t blockY = blockShape[kBlockYAxis];
    TORCH_CHECK(sparseBlockIdx.dim() == kSparseIdxRank && sparseBlockIdx.size(kSparseEntryAxis) > 0 &&
                    sparseBlockIdx.size(kSparseEntryAxis) <=
                        std::numeric_limits<int64_t>::max() / unionSize - residualBlockMode,
                "invalid capacity attributes");
    TORCH_CHECK(sparseBlockIdx.numel() <= std::numeric_limits<int64_t>::max() / kIndexBytes,
                "input byte size overflow");
    const int64_t maxUnionSparseCount = std::min(
        unionSize * (sparseBlockIdx.size(kSparseEntryAxis) + residualBlockMode), maxLogicalBlocks);
    TORCH_CHECK(blockY > 0 && maxUnionSparseCount <= std::numeric_limits<int64_t>::max() / blockY,
                "mask shape overflow");
    TORCH_CHECK(sparseBlockIdx.size(kHeadAxis) >= 0 &&
                    (sparseBlockIdx.size(kHeadAxis) == 0 ||
                     totalqUnionNum <= std::numeric_limits<int64_t>::max() / sparseBlockIdx.size(kHeadAxis) /
                                           kIndexBytes / blockY / std::max<int64_t>(1, maxUnionSparseCount)),
                "output byte size overflow");
    const auto heads = sparseBlockIdx.size(kHeadAxis);
    at::TensorOptions options = at::TensorOptions().dtype(at::kInt).device(sparseBlockIdx.device());
    at::Tensor unionIdx = at::empty({heads, totalqUnionNum, maxUnionSparseCount}, options);
    at::Tensor unionCount = at::empty({heads, totalqUnionNum}, options);
    at::Tensor exactMask = at::empty({heads, totalqUnionNum, maxUnionSparseCount * blockY}, options);
    ACLNN_CMD(aclnnUnionSparseGen, sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv,
              blockShape, unionSize, maskType, maxQSeqlen, maxKvSeqlen, layoutSparsePattern, isInputIdxSorted,
              residualBlockMode, totalqUnionNum, maxLogicalBlocks, unionIdx, unionCount, exactMask);
    return {unionIdx, unionCount, exactMask};
}

} // namespace index
} // namespace cann_ops_nn

// Register the backend when the lazily built module is loaded. Subsequent
// torch.ops calls enter C++ directly, without a Python/pybind round trip.
TORCH_LIBRARY_IMPL(cann_ops_nn, PrivateUse1, m)
{
    m.impl("union_sparse_gen_launch", TORCH_FN(cann_ops_nn::index::union_sparse_gen_launch));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("union_sparse_gen_launch", &cann_ops_nn::index::union_sparse_gen_launch, "UnionSparseGen ACLNN launcher");
}
