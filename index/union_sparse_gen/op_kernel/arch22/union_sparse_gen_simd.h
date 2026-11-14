/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_ARCH22_SIMD_H
#define UNION_SPARSE_GEN_ARCH22_SIMD_H
#include "kernel_operator.h"
#include "../union_sparse_gen_tiling_data.h"
#include "union_sparse_gen_membership_plan.h"

namespace UnionSparseGenArch22 {
using namespace AscendC;
namespace Plan = UnionSparseGenA2Plan;
constexpr int32_t kMetaWindow = kUnionSparseGenMetaWindow;
constexpr int32_t kMetaPrefixStride = kUnionSparseGenPrefixStride;
constexpr int32_t kMaxRows = Plan::kMaxGroupWidth;
constexpr int32_t kTableUnroll = 2;
constexpr int32_t kAdaptiveVectorWidth = Plan::kWideVectorLanes;
// Measured work crossover: larger domains prefer sorting when its UB fits.
using UnionSparseGenA2Plan::Execution;

template <int32_t GroupWidth = 0, int32_t VectorWidth = 0, Execution Mode = Execution::Ordered>
class UnionSparseGen {
public:
    __aicore__ inline void Init(GM_ADDR sparseBlockIdx, GM_ADDR sparseBlockCount, GM_ADDR cuQ, GM_ADDR cuKv,
                                GM_ADDR usedQ, GM_ADDR usedKv, GM_ADDR unionIdx, GM_ADDR unionCount, GM_ADDR exactMask,
                                GM_ADDR workspace, const UnionSparseGenTilingData& t)
    {
        (void)workspace;
        BindGlobalBuffers(sparseBlockIdx, sparseBlockCount, cuQ, cuKv, usedQ, usedKv, unionIdx, unionCount, exactMask);
        LoadPlan(t, cuQ);
        InitBuffers();
        InitViewsAndState();
    }

    __aicore__ inline void Process() { ProcessGroups<Mode != Execution::Ordered>(); }
    template <bool Extended>
    __aicore__ inline void ProcessGroups()
    {
        const int64_t block = GetBlockIdx();
        if (block >= usedCore_ || groups_ == 0) {
            return;
        }
        // Real prefill has a large count gradient across its query sequence.
        // Interleave a single request without paying multi-request relocation;
        // retain contiguous intervals for the general packed-request path.
        const bool strided = batch_ == 1 && groups_ >= usedCore_;
        const int64_t first = strided ? block : block * workPerCore_ + (block < workTail_ ? block : workTail_);
        const int64_t step = strided ? usedCore_ : 1;
        const int64_t n = workPerCore_ + (block < workTail_ ? 1 : 0);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t flat = first + i * step;
            head_ = flat / groups_;
            if (!Locate(flat % groups_)) {
                continue;
            }
            outputBase_ = flat * capacity_;
            if constexpr (Extended) {
                MergeMembership();
            } else {
                Merge();
            }
        }
        WaitSlot(0);
        WaitSlot(1);
        SetMaskNorm();
        ResetMask();
    }

private:
    __aicore__ inline void BindGlobalBuffers(GM_ADDR sparseBlockIdx, GM_ADDR sparseBlockCount, GM_ADDR cuQ,
                                             GM_ADDR cuKv, GM_ADDR usedQ, GM_ADDR usedKv, GM_ADDR unionIdx,
                                             GM_ADDR unionCount, GM_ADDR exactMask)
    {
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sparseBlockIdx));
        counts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sparseBlockCount));
        cuQ_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cuQ));
        cuKv_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cuKv));
        usedQ_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(usedQ));
        usedKv_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(usedKv));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(unionIdx));
        outCounts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(unionCount));
        mask_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(exactMask));
    }

    __aicore__ inline void LoadPlan(const UnionSparseGenTilingData& t, GM_ADDR cuQ)
    {
        batch_ = t.batchSize;
        totalQ_ = t.totalQ;
        topK_ = t.topK;
        groups_ = t.numGroups;
        capacity_ = t.unionCapacity;
        width_ = t.unionSize;
        by_ = t.blockY;
        nonCausal_ = t.maskType == 0;
        usedCore_ = t.usedCoreNum;
        workPerCore_ = t.workPerCore;
        workTail_ = t.workTail;
        hasQ_ = t.hasSeqUsedQ != 0;
        hasKv_ = t.hasSeqUsedKv != 0;
        hasCuQ_ = cuQ != nullptr;
        hasCuKv_ = t.hasCuSqlenKv != 0;
        residual_ = t.residualBlockMode != 0;
        window_ = t.simdInputWindow;
        tile_ = t.simdTileBlocks;
        algorithmBytes_ = t.simdAlgorithmBytes;
    }

    __aicore__ inline void InitBuffers()
    {
        pipe_.InitBuffer(metaBuf_, UnionSparseGenMetadataBytes(hasCuQ_, hasQ_, hasKv_));
        pipe_.InitBuffer(inputBuf_, InputRows() * window_ * sizeof(int32_t));
        pipe_.InitBuffer(countBuf_, kMaxRows * sizeof(int32_t));
        pipe_.InitBuffer(pairBuf_, Plan::kOutputSlots * tile_ * Plan::kPairWords * sizeof(int32_t));
        pipe_.InitBuffer(outputBuf_, Plan::kOutputSlots * tile_ * by_ * sizeof(int32_t));
        pipe_.InitBuffer(algorithmBuf_, algorithmBytes_ > 0 ? algorithmBytes_ : Plan::kDmaBlockBytes);
        if (by_ == Plan::kGatherBlockY) {
            pipe_.InitBuffer(broadcastBuf_, tile_ * Plan::kGatherBlockY * sizeof(uint32_t));
            broadcastOffsets_ = broadcastBuf_.Get<uint32_t>();
            for (int32_t i = 0; i < tile_ * Plan::kGatherBlockY; ++i) {
                broadcastOffsets_.SetValue(i, uint32_t(i / Plan::kGatherBlockY * Plan::kWordBytes));
            }
        }
    }

    __aicore__ inline void InitViewsAndState()
    {
        // Keep initialized full-buffer handles; derive subviews at each copy/read.
        // Persisting conditional LocalTensor subviews leaves private handle state
        // uninitialized in the instrumented device build.
        const int32_t prefixWindows = int32_t(hasCuQ_) + int32_t(!hasKv_);
        prefix_ = metaBuf_.Get<int64_t>();
        meta_ = metaBuf_.Get<int32_t>();
        kvPrefixOffset_ = hasCuQ_ ? kMetaPrefixStride : 0;
        usedMetaOffset_ = prefixWindows * Plan::kPrefixWords * kMetaPrefixStride;
        usedKvOffset_ = usedMetaOffset_ + (hasQ_ ? kMetaWindow : 0);
        rowData_ = inputBuf_.Get<int32_t>();
        cnt_ = countBuf_.Get<int32_t>();
        metaStart_ = -1;
        request_ = 0;
        requestGroup_ = 0;
        pending_[0] = false;
        pending_[1] = false;
        paddingReady_[0] = false;
        paddingReady_[1] = false;
        vectorOffsetsReady_ = false;
        vectorScratchOffset_ = -1;
    }

    __aicore__ inline void ApplyBoundaryMask(int32_t boundaryFirst, int32_t boundaryEnd)
    {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t i = boundaryFirst; i < boundaryEnd; ++i) {
            const int64_t p = int64_t(pairIdx_.GetValue(i)) * by_;
            if (p + by_ - 1 <= visibilityBase_ && p + by_ - 1 < kvLen_) {
                continue;
            }
            if (p >= visibilityBase_ + rows_ || p >= kvLen_) {
                continue;
            }
            const uint32_t membership = pairMask_.GetValue(i);
            for (int32_t t = 0; t < by_; ++t) {
                const int64_t shift = p + t - visibilityBase_;
                const uint32_t value = p + t >= kvLen_ || shift >= rows_ ?
                                           0U :
                                           (shift <= 0 ? membership :
                                                         membership & (Plan::kAllQueryBits << uint32_t(shift)));
                out_.SetValue(i * by_ + t, value);
            }
        }
    }

    __aicore__ inline void PreparePaddingTile(int32_t n)
    {
        // A fully padded slot is immutable across subsequent padding DMA
        // reads, including the next group. Valid writes invalidate it.
        if (!paddingReady_[slot_]) {
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Duplicate(pairIdx_, int32_t(-1), uint32_t(n));
            Duplicate(out_, uint32_t(0), uint32_t(n * by_));
            paddingReady_[slot_] = n == tile_;
        }
    }

    __aicore__ inline void ExpandOutputMask(int32_t n)
    {
        paddingReady_[slot_] = false;
        bool boundary = false;
        int32_t boundaryFirst = 0, boundaryEnd = n, zeroAligned = 0;
        // Sorted IDs delimit one full-keep prefix, <=31 transition
        // tokens and an all-zero suffix. No per-block scalar scan.
        boundaryFirst = VisibilityRank(n, false);
        boundaryEnd = VisibilityRank(n, true);
        boundary = boundaryFirst < boundaryEnd;
        zeroAligned = (boundaryEnd + Plan::kDmaWords - 1) / Plan::kDmaWords * Plan::kDmaWords;
        const int32_t end = zeroAligned < n ? zeroAligned : n;
        for (int32_t i = boundaryEnd; i < end; ++i) {
            pairMask_.SetValue(i, 0U);
        }
        const int32_t aligned = (n + Plan::kDmaWords - 1) / Plan::kDmaWords * Plan::kDmaWords;
        for (int32_t i = n; i < aligned; ++i) {
            pairMask_.SetValue(i, 0U);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        if (zeroAligned < aligned) {
            Duplicate(pairMask_[zeroAligned], uint32_t(0), uint32_t(aligned - zeroAligned));
            PipeBarrier<PIPE_V>();
        }
        if (by_ == Plan::kGatherBlockY) {
            Gather(out_, pairMask_, broadcastOffsets_, 0, uint32_t(aligned * Plan::kGatherBlockY));
        } else {
            const int32_t parts = by_ / Plan::kBroadcastLanes;
            for (int32_t part = 0; part < parts; ++part) {
                Brcb(out_[part * Plan::kBroadcastLanes], pairMask_, uint8_t(aligned / Plan::kBroadcastLanes),
                     BrcbRepeatParams{uint16_t(parts), uint16_t(parts * Plan::kBroadcastLanes)});
            }
        }
        if (boundary) {
            ApplyBoundaryMask(boundaryFirst, boundaryEnd);
        }
    }

    __aicore__ inline void BeginGroupMerge()
    {
        rank_ = 0;
        buffered_ = 0;
        slot_ = 0;
        BeginTile();
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        CopyInput(cnt_, counts_[head_ * totalQ_ + qStart_], rows_);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    __aicore__ inline void FinishGroupCount()
    {
        const int64_t count = rank_;
        while (rank_ < capacity_) {
            const int32_t n = capacity_ - rank_ < tile_ ? capacity_ - rank_ : tile_;
            Flush(n, true);
        }
        WaitSlot(0);
        WaitSlot(1);
        cnt_.SetValue(0, int32_t(count));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        Write(outCounts_[outputBase_ / capacity_], cnt_, 1);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    template <typename TableT>
    __aicore__ inline void AccumulateMembershipChunk(LocalTensor<TableT> table, int64_t base, int32_t length)
    {
        auto* address = reinterpret_cast<__ubuf__ TableT*>(table.GetPhyAddr());
        for (int32_t row = 0; row < rows_; ++row) {
            const int64_t visible = kvLen_ - qLen_ + q0_ + row + 1;
            const int64_t bound = residual_ ? (visible > 0 ? visible / by_ : 0) : logicalBlocks_;
            const uint32_t bit = 1U << uint32_t(row);
            const int64_t count = cnt_.GetValue(row);
            const int32_t limit = int32_t(count < 0 ? 0 : (count > topK_ ? topK_ : count));
            const int64_t inputBase = (head_ * totalQ_ + qStart_ + row) * topK_;
            const int32_t readWindow = int32_t(InputRows()) * window_;
            for (int64_t at = 0; at < limit; at += readWindow) {
                const int32_t n = int32_t(limit - at < readWindow ? limit - at : readWindow);
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                CopyInput(rowData_, input_[inputBase + at], n);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                auto* values = reinterpret_cast<__ubuf__ int32_t*>(rowData_.GetPhyAddr());
                UpdateMembership(values, n, address, uint32_t(base), uint32_t(length), uint32_t(bound), bit);
            }
            if (residual_ && visible > 0 && visible % by_ != 0) {
                const int64_t tail = visible / by_;
                if (tail >= base && tail < base + length) {
                    address[tail - base] |= TableT(bit);
                }
            }
        }
    }

    __aicore__ inline int64_t LoadMergeLimits(int32_t& countActive)
    {
        countActive = 0;
        int64_t entries = 0;
        for (int32_t r = 0; r < Width(); ++r) {
            pos_[r] = 0;
            limit_[r] = r < rows_ ? cnt_.GetValue(r) : 0;
            if (limit_[r] < 0) {
                limit_[r] = 0;
            }
            if (limit_[r] > topK_) {
                limit_[r] = topK_;
            }
            entries += limit_[r];
            if constexpr (VectorWidth > 0) {
                countActive += limit_[r] != 0;
            }
        }

        return entries;
    }

    __aicore__ inline bool TryBatchedVectorMerge(int64_t entries, int32_t countActive)
    {
        const bool vectorKeysValid = logicalBlocks_ <= kVectorMaxLogicalBlocks;
        int64_t inputRuns = 0, packedProposals = 0;
        bool vectorDone = entries == 0;

        const bool residentVector = CanVectorResident();
        if (vectorKeysValid && entries >= Plan::kMinBatchedEntries && countActive > 1) {
            for (int32_t r = 0; r < Width(); ++r) {
                inputRuns += (int64_t(limit_[r]) + kVectorLanes - 1) / kVectorLanes;
                packedProposals += (int64_t(limit_[r]) + Plan::kDmaWords - 1) / Plan::kDmaWords * Plan::kDmaWords;
            }
        }
        if (vectorKeysValid && entries >= Plan::kMinBatchedEntries && countActive > 1 &&
            (residentVector ? PreferVectorWork(entries, countActive, inputRuns, packedProposals, false) :
                              PreferStreamingVectorWork(entries, countActive, inputRuns, packedProposals))) {
            if (residentVector) {
                VectorResident();
                vectorDone = true;
            } else if (PreferVectorStream(entries, countActive) && VectorLeafSize() != 0) {
                VectorStream();
                vectorDone = true;
            }
        }

        return vectorDone;
    }

    __aicore__ inline void MergeMultipleRows(int64_t entries, int32_t active)
    {
        const bool vectorKeysValid = logicalBlocks_ <= kVectorMaxLogicalBlocks;

        // Bound the rounded proposal extent before multiplying it. A very
        // large legal storage K can otherwise overflow an ineligible plan.
        const int64_t maxProposalRow = algorithmBytes_ / (Width() * Plan::kResidentRecordBytes) /
                                       Plan::kSort32Elements * Plan::kSort32Elements;
        const bool proposalFits = topK_ <= maxProposalRow;
        const int64_t proposals = proposalFits ? Width() * ((topK_ + Plan::kSort32Elements - 1) /
                                                            Plan::kSort32Elements * Plan::kSort32Elements) :
                                                 0;
        const int64_t tableLength = (logicalBlocks_ + Plan::kTableAlignmentBlocks - 1) / Plan::kTableAlignmentBlocks *
                                        Plan::kTableAlignmentBlocks +
                                    Plan::kTableAlignmentBlocks;
        const int64_t tableBytes = tableLength * Plan::kWordBytes +
                                   ((tableLength / Plan::kBitsPerByte + Plan::kDmaBlockBytes - 1) /
                                    Plan::kDmaBlockBytes * Plan::kDmaBlockBytes);
        if (PreferTable(entries, active) && tableBytes <= algorithmBytes_) {
            Table();
        }
        // A table-ineligible request may use exact resident proposals only
        // when both proposal buffers fit. The ordinary table/tree region
        // keeps its previously measured dispatch; no preprocessing pass.
        else if (vectorKeysValid && tableBytes > algorithmBytes_ && entries >= Plan::kMinBatchedEntries &&
                 proposalFits && proposals < Plan::kMrgSortCountLimit) {
            VectorMerge();
        } else {
            Tree();
        }
    }

    __aicore__ inline void MergeScalarRows(int64_t entries)
    {
        int32_t active = 0, single = -1;

        for (int32_t r = 0; r < Width(); ++r) {
            Next(r);
            if (live_[r]) {
                ++active;
                single = r;
            }
        }
        if (active == 1) {
            while (live_[single]) {
                const int32_t id = value_[single];
                Emit(id, 1U << uint32_t(single));
                do {
                    Next(single);
                } while (live_[single] && value_[single] == id);
            }
        } else if (active > 1) {
            MergeMultipleRows(entries, active);
        }
    }

    __aicore__ inline int64_t Width() const { return GroupWidth == 0 ? width_ : int64_t(GroupWidth); }
    __aicore__ inline int64_t InputRows() const { return VectorWidth == 0 ? kMaxRows : Width(); }
    template <typename T>
    __aicore__ inline void CopyInput(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t n)
    {
        // All UB destinations are 32-byte aligned. GM addresses need only element alignment.
        DataCopyPad(dst, src, DataCopyExtParams{1, n * uint32_t(sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<T>{false, 0, 0, 0});
    }
    __aicore__ inline void Write(GlobalTensor<int32_t> dst, LocalTensor<int32_t> src, uint32_t n)
    {
        if (n) {
            DataCopyPad(dst, src, DataCopyExtParams{1, n * uint32_t(Plan::kWordBytes), 0, 0, 0});
        }
    }

    __aicore__ inline void WaitSlot(int32_t slot)
    {
        if (!pending_[slot]) {
            return;
        }
        const event_t event = slot == 0 ? EVENT_ID1 : EVENT_ID2;
        WaitFlag<HardEvent::MTE3_S>(event);
        WaitFlag<HardEvent::MTE3_V>(event);
        pending_[slot] = false;
    }
    __aicore__ inline void BeginTile()
    {
        WaitSlot(slot_);
        auto pairs = pairBuf_.Get<int32_t>();
        pairIdx_ = pairs[slot_ * tile_ * Plan::kPairWords];
        pairMask_ = pairs[slot_ * tile_ * Plan::kPairWords + tile_].ReinterpretCast<uint32_t>();
        out_ = outputBuf_.Get<uint32_t>()[slot_ * tile_ * by_];
    }
    __aicore__ inline void LoadMeta(int64_t request)
    {
        const int64_t start = (request / kMetaWindow) * kMetaWindow;
        if (start == metaStart_) {
            return;
        }
        const int32_t n = batch_ - start < kMetaWindow ? batch_ - start : kMetaWindow;
        // The allocated windows are disjoint; release old scalar reads once.
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        if (hasCuQ_) {
            CopyInput(prefix_, cuQ_[start], n + 1);
        }
        if (!hasKv_ && hasCuKv_) {
            CopyInput(prefix_[kvPrefixOffset_], cuKv_[start], n + 1);
        }
        if (hasQ_) {
            CopyInput(meta_[usedMetaOffset_], usedQ_[start], n);
        }
        if (hasKv_) {
            CopyInput(meta_[usedKvOffset_], usedKv_[start], n);
        }
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        metaStart_ = start;
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    __aicore__ inline bool Locate(int64_t target)
    {
        if (target < requestGroup_) {
            request_ = 0;
            requestGroup_ = 0;
        }
        while (request_ < batch_) {
            LoadMeta(request_);
            const int32_t r = request_ - metaStart_;
            const int64_t qBase = hasCuQ_ ? prefix_.GetValue(r) : request_ * (totalQ_ / batch_);
            qLen_ = hasQ_ ? meta_.GetValue(usedMetaOffset_ + r) : prefix_.GetValue(r + 1) - qBase;
            kvLen_ = hasKv_ ? meta_.GetValue(usedKvOffset_ + r) :
                              prefix_.GetValue(kvPrefixOffset_ + r + 1) - prefix_.GetValue(kvPrefixOffset_ + r);
            const int64_t n = (qLen_ + Width() - 1) / Width();
            if (target < requestGroup_ + n) {
                q0_ = (target - requestGroup_) * Width();
                rows_ = qLen_ - q0_ < Width() ? qLen_ - q0_ : Width();
                qStart_ = qBase + q0_;
                logicalBlocks_ = (kvLen_ + by_ - 1) / by_;
                // The common writer clips token p against KV and a per-query
                // visibility base. Noncausal uses KV-1 so every valid token
                // keeps membership; effective Q/group addressing stay unchanged.
                visibilityBase_ = nonCausal_ ? kvLen_ - 1 : kvLen_ - qLen_ + q0_;
                return true;
            }
            requestGroup_ += n;
            ++request_;
        }
        return false;
    }
    __aicore__ inline void Next(int32_t row)
    {
        live_[row] = false;
        while (pos_[row] < limit_[row]) {
            const int32_t s = pos_[row] % window_;
            if (s == 0) {
                const int32_t remain = limit_[row] - pos_[row];
                const int32_t n = remain < window_ ? remain : window_;
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                CopyInput(rowData_[row * window_], input_[(head_ * totalQ_ + qStart_ + row) * topK_ + pos_[row]], n);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            }
            const int32_t value = rowData_.GetValue(row * window_ + s);
            ++pos_[row];
            if (value >= 0 && int64_t(value) < logicalBlocks_) {
                value_[row] = value;
                live_[row] = true;
                return;
            }
        }
    }
    __aicore__ inline int32_t Winner(int32_t a, int32_t b)
    {
        if (a < 0) {
            return b;
        }
        if (b < 0) {
            return a;
        }
        return value_[a] <= value_[b] ? a : b;
    }
    __aicore__ inline void Update(int32_t row)
    {
        int32_t node = Width() + row;
        tree_[node] = live_[row] ? row : -1;
        for (node >>= 1; node; node >>= 1) {
            tree_[node] = Winner(tree_[node * Plan::kBinaryTreeArity], tree_[node * Plan::kBinaryTreeArity + 1]);
        }
    }
    __aicore__ inline int32_t VisibilityRank(int32_t n, bool zero)
    {
        int32_t low = 0, high = n;
        while (low < high) {
            const int32_t mid = (low + high) / 2;
            const int64_t p = int64_t(pairIdx_.GetValue(mid)) * by_;
            const bool before = zero ? p < visibilityBase_ + rows_ : p + by_ - 1 <= visibilityBase_;
            if (before) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }
    __aicore__ inline void Flush(int32_t n, bool padding = false)
    {
        if (padding) {
            PreparePaddingTile(n);
        } else {
            ExpandOutputMask(n);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        Write(indices_[outputBase_ + rank_], pairIdx_, n);
        Write(mask_[(outputBase_ + rank_) * by_], out_.ReinterpretCast<int32_t>(), n * by_);
        const event_t event = slot_ == 0 ? EVENT_ID1 : EVENT_ID2;
        SetFlag<HardEvent::MTE3_S>(event);
        SetFlag<HardEvent::MTE3_V>(event);
        pending_[slot_] = true;
        rank_ += n;
        buffered_ = 0;
        slot_ ^= 1;
        BeginTile();
    }
    __aicore__ inline void Emit(int32_t id, uint32_t membership)
    {
        pairIdx_.SetValue(buffered_, id);
        pairMask_.SetValue(buffered_, membership);
        ++buffered_;
        if (buffered_ == tile_) {
            Flush(buffered_);
        }
    }
    __aicore__ inline void Tree()
    {
        for (int32_t r = 0; r < Width(); ++r) {
            tree_[Width() + r] = live_[r] ? r : -1;
        }
        for (int32_t p = Width() - 1; p > 0; --p) {
            tree_[p] = Winner(tree_[Plan::kBinaryTreeArity * p], tree_[Plan::kBinaryTreeArity * p + 1]);
        }
        while (tree_[1] >= 0) {
            const int32_t id = value_[tree_[1]];
            uint32_t bits = 0;
            while (tree_[1] >= 0 && value_[tree_[1]] == id) {
                const int32_t r = tree_[1];
                bits |= 1U << uint32_t(r);
                do {
                    Next(r);
                } while (live_[r] && value_[r] == id);
                Update(r);
            }
            Emit(id, bits);
        }
    }
#include "union_sparse_gen_vector.h"
    __aicore__ inline void TableUpdate(int32_t id, __ubuf__ uint32_t* table, uint32_t upper, uint32_t bit,
                                       int32_t& last)
    {
        // Effective KV lengths are int32; supported blockY makes L <2^29.
        // One unsigned comparison also rejects every negative int32 input.
        if (uint32_t(id) < upper && id != last) {
            table[id] = table[id] | bit;
            last = id;
        }
    }
    __aicore__ inline void TableFast(LocalTensor<uint32_t> table)
    {
        auto* tableAddress = reinterpret_cast<__ubuf__ uint32_t*>(table.GetPhyAddr());
        const uint32_t upper = uint32_t(logicalBlocks_);
        const int32_t windowSize = window_;
        for (int32_t r = 0; r < rows_; ++r) {
            if (!live_[r]) {
                continue;
            }
            const uint32_t bit = 1U << uint32_t(r);
            int32_t last = value_[r];
            tableAddress[last] |= bit;
            auto row = rowData_[r * windowSize];
            auto* rowAddress = reinterpret_cast<__ubuf__ int32_t*>(row.GetPhyAddr());
            const int64_t inputBase = (head_ * totalQ_ + qStart_ + r) * topK_;
            const int32_t limit = limit_[r];
            int32_t at = pos_[r], loaded = (at - 1) / windowSize;
            while (at < limit) {
                const int32_t window = at / windowSize;
                const int32_t start = window * windowSize;
                const int32_t available = limit - start;
                const int32_t n = available < windowSize ? available : windowSize;
                if (window != loaded) {
                    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    CopyInput(row, input_[inputBase + start], n);
                    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    loaded = window;
                }
                int32_t i = at - start;
                for (; i + kTableUnroll <= n; i += kTableUnroll) {
#pragma unroll
                    for (int32_t j = 0; j < kTableUnroll; ++j) {
                        TableUpdate(rowAddress[i + j], tableAddress, upper, bit, last);
                    }
                }
                for (; i < n; ++i) {
                    TableUpdate(rowAddress[i], tableAddress, upper, bit, last);
                }
                at = start + n;
            }
        }
    }
    __aicore__ inline void Table()
    {
        if constexpr (VectorWidth > 0) {
            vectorOffsetsReady_ = false;
        }
        auto table = algorithmBuf_.Get<uint32_t>();
        // Bitmap reads can round beyond the logical domain; initialize the guard too.
        Duplicate(table, uint32_t(0),
                  uint32_t((logicalBlocks_ + Plan::kTableAlignmentBlocks - 1) / Plan::kTableAlignmentBlocks *
                               Plan::kTableAlignmentBlocks +
                           Plan::kTableAlignmentBlocks));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        TableFast(table);
        CompactTable(table);
    }
    __aicore__ inline void CompactTable(LocalTensor<uint32_t> table)
    {
        CompactTableRange(table, int32_t(logicalBlocks_), 0);
    }
    template <typename TableT>
    __aicore__ inline void CompactTableRange(LocalTensor<TableT> table, int32_t length, int64_t base)
    {
        const int32_t aligned = (length + Plan::kTableAlignmentBlocks - 1) / Plan::kTableAlignmentBlocks *
                                    Plan::kTableAlignmentBlocks +
                                Plan::kTableAlignmentBlocks;
        LocalTensor<uint8_t> selected = table[aligned].template ReinterpretCast<uint8_t>();
        if (length <= Plan::kSmallMembershipDomainBlocks) {
            Duplicate(selected.ReinterpretCast<uint16_t>(), uint16_t(0),
                      uint32_t(Plan::kDmaBlockBytes / Plan::kHalfwordBytes));
            PipeBarrier<PIPE_V>();
        }
        // The input windows are dead after the table update. Reuse their UB for
        // exact int32 IDs; no full-L sequence or additional persistent buffer.
        const int32_t chunk = tile_ < InputRows() * window_ ? tile_ : InputRows() * window_;
        for (int32_t start = 0; start < length; start += chunk) {
            const int32_t n = length - start < chunk ? length - start : chunk;
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            LocalTensor<uint32_t> values = table[start].template ReinterpretCast<uint32_t>();
            if constexpr (sizeof(TableT) == 1) {
                // W8 membership is exactly representable in half; expand only
                // the current compaction tile, preserving a byte per block.
                auto halves = rowData_[Plan::kCompactionHalfVector * tile_].ReinterpretCast<half>();
                values = rowData_[tile_].ReinterpretCast<uint32_t>();
                Cast(halves, table[start], RoundMode::CAST_NONE,
                     uint32_t((n + Plan::kCompareLanes - 1) / Plan::kCompareLanes * Plan::kCompareLanes));
                PipeBarrier<PIPE_V>();
                Cast(values.ReinterpretCast<int32_t>(), halves, RoundMode::CAST_RINT,
                     uint32_t((n + Plan::kCompareLanes - 1) / Plan::kCompareLanes * Plan::kCompareLanes));
                PipeBarrier<PIPE_V>();
            }
            // All rounded reads remain inside the initialized table, including
            // a last tile shorter than64 lanes. GatherMask consumes actual n.
            CompareScalar(selected, values.ReinterpretCast<int32_t>(), int32_t(0), CMPMODE::EQ,
                          uint32_t((n + Plan::kCompareLanes - 1) / Plan::kCompareLanes * Plan::kCompareLanes));
            PipeBarrier<PIPE_V>();
            auto shorts = selected.ReinterpretCast<uint16_t>();
            Not(shorts, shorts,
                uint32_t((n + Plan::kCompareLanes - 1) / Plan::kCompareLanes *
                         (Plan::kCompareLanes / Plan::kBitsPerHalfword)));
            PipeBarrier<PIPE_V>();
            CreateVecIndex(rowData_, int32_t(base + start), uint32_t(n));
            PipeBarrier<PIPE_V>();
            uint64_t indicesCount = 0, membershipCount = 0;
            GatherMask(pairIdx_, rowData_, selected.ReinterpretCast<uint32_t>(), true, uint32_t(n),
                       GatherMaskParams{1, 1, Plan::kVectorRepeatBlocks, Plan::kVectorRepeatBlocks}, indicesCount);
            PipeBarrier<PIPE_V>();
            GatherMask(pairMask_, values, selected.ReinterpretCast<uint32_t>(), true, uint32_t(n),
                       GatherMaskParams{1, 1, Plan::kVectorRepeatBlocks, Plan::kVectorRepeatBlocks}, membershipCount);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            ResetMask();
            // Identical bitmaps and lengths imply identical retained counts.
            if (indicesCount != 0) {
                buffered_ = int32_t(membershipCount);
                Flush(buffered_);
            }
        }
    }

    __aicore__ inline void BuildScalarProposals(LocalTensor<float> src, int32_t runLength, int32_t total)
    {
        auto words = src.ReinterpretCast<uint32_t>();
        Duplicate(words, uint32_t(0), total * Plan::kPairWords);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t r = 0; r < Width(); ++r) {
            int32_t j = 0;
            while (live_[r]) {
                const int32_t id = value_[r];
                // Exact finite-positive float bit encoding, not a numeric cast.
                words.SetValue((r * runLength + j) * Plan::kPairWords, kVectorKey - uint32_t(id));
                words.SetValue((r * runLength + j) * Plan::kPairWords + 1, 1U << uint32_t(r));
                ++j;
                do {
                    Next(r);
                } while (live_[r] && value_[r] == id);
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
    __aicore__ inline LocalTensor<float> MergeScalarProposals(LocalTensor<float> src, LocalTensor<float> dst,
                                                              int32_t runLength)
    {
        int32_t runs = Width(), run = runLength;
        while (runs > 1) {
            const int32_t nextRuns = (runs + Plan::kMergeFanIn - 1) / Plan::kMergeFanIn;
            for (int32_t p = 0; p < nextRuns; ++p) {
                const int32_t n = runs - p * Plan::kMergeFanIn < Plan::kMergeFanIn ? runs - p * Plan::kMergeFanIn :
                                                                                     Plan::kMergeFanIn;
                uint16_t lens[Plan::kMergeFanIn] = {};
                for (int32_t s = 0; s < n; ++s) {
                    lens[s] = run;
                }
                auto first = src[p * Plan::kMergeFanIn * run * Plan::kPairWords];
                MrgSortSrcList<float> list(
                    first,
                    src[(p * Plan::kMergeFanIn +
                         (n > Plan::kMergeSecondInput ? Plan::kMergeSecondInput : Plan::kMergeFirstInput)) *
                        run * Plan::kPairWords],
                    src[(p * Plan::kMergeFanIn +
                         (n > Plan::kMergeThirdInput ? Plan::kMergeThirdInput : Plan::kMergeFirstInput)) *
                        run * Plan::kPairWords],
                    src[(p * Plan::kMergeFanIn +
                         (n > Plan::kMergeFourthInput ? Plan::kMergeFourthInput : Plan::kMergeFirstInput)) *
                        run * Plan::kPairWords]);
                MrgSort(dst[p * Plan::kMergeFanIn * run * Plan::kPairWords], list,
                        MrgSort4Info(lens, false, uint16_t((1U << n) - 1), 1));
            }
            PipeBarrier<PIPE_V>();
            auto tmp = src;
            src = dst;
            dst = tmp;
            runs = nextRuns;
            run *= Plan::kMergeFanIn;
        }

        return src;
    }
    __aicore__ inline void EmitScalarProposals(LocalTensor<float> src, int32_t total)
    {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto words = src.ReinterpretCast<uint32_t>();
        int32_t at = 0;
        while (at < total && words.GetValue(at * Plan::kPairWords) != 0) {
            const uint32_t key = words.GetValue(at * Plan::kPairWords);
            uint32_t bits = 0;
            do {
                bits |= words.GetValue(at * Plan::kPairWords + 1);
                ++at;
            } while (at < total && words.GetValue(at * Plan::kPairWords) == key);
            Emit(int32_t(kVectorKey - key), bits);
        }
    }
    __aicore__ inline void VectorMerge()
    {
        if constexpr (VectorWidth > 0) {
            vectorOffsetsReady_ = false;
        }
        const int32_t runLength = (topK_ + Plan::kSort32Elements - 1) / Plan::kSort32Elements * Plan::kSort32Elements;
        const int32_t total = Width() * runLength;
        auto src = algorithmBuf_.Get<float>();
        auto dst = src[total * Plan::kPairWords];
        BuildScalarProposals(src, runLength, total);
        auto sorted = MergeScalarProposals(src, dst, runLength);
        EmitScalarProposals(sorted, total);
    }
    // Fixed-window proposal cascade. Each level exposes a sorted prefix and
    // resumes exactly at the four hardware-reported consumption counts. Leaves
    // hold a UB-budgeted power-of-two proposal window, up to256. Every level
    // occupies W*leaf proposals, so the complete cascade fits the runtime UB.
    // No global scratch, full-U intermediate, recursion or floating conversion.
    __aicore__ inline int32_t StreamStart(int32_t node)
    {
        if (node < Plan::kStreamLevel1Begin) {
            return node * streamLeaf_ * Plan::kPairWords;
        }
        if (node < Plan::kStreamLevel2Begin) {
            return Width() * streamLeaf_ * Plan::kPairWords +
                   (node - Plan::kStreamLevel1Begin) * streamLeaf_ * (Plan::kPairWords * Plan::kMergeFanIn);
        }
        if (node < Plan::kStreamRoot) {
            return Width() * streamLeaf_ * (Plan::kPairWords * Plan::kStreamLevel2Depth) +
                   (node - Plan::kStreamLevel2Begin) * streamLeaf_ *
                       (Plan::kPairWords * Plan::kMergeFanIn * Plan::kMergeFanIn);
        }
        return Width() * streamLeaf_ * (Plan::kPairWords * Plan::kStreamRootDepth);
    }
    __aicore__ inline void FillLeaf(int32_t row) { FillVectorLeaf(row); }
    __aicore__ inline void MergeStreams(int32_t node, int32_t first, int32_t children)
    {
        MergeVectorStreams(node, first, children);
    }
    __aicore__ inline void FillLevel1(int32_t group)
    {
        for (int32_t r = group * Plan::kMergeFanIn; r < group * Plan::kMergeFanIn + Plan::kMergeFanIn; ++r) {
            if (streamLength_[r] == 0) {
                FillLeaf(r);
            }
        }
        MergeStreams(Plan::kStreamLevel1Begin + group, group * Plan::kMergeFanIn, Plan::kMergeFanIn);
    }
    __aicore__ inline void FillLevel2(int32_t group)
    {
        const int32_t available = Width() / Plan::kMergeFanIn - group * Plan::kMergeFanIn;
        const int32_t n = available < Plan::kMergeFanIn ? available : Plan::kMergeFanIn;
        for (int32_t i = group * Plan::kMergeFanIn; i < group * Plan::kMergeFanIn + n; ++i) {
            if (streamLength_[Plan::kStreamLevel1Begin + i] == 0) {
                FillLevel1(i);
            }
        }
        MergeStreams(Plan::kStreamLevel2Begin + group, Plan::kStreamLevel1Begin + group * Plan::kMergeFanIn, n);
    }

    __aicore__ inline int64_t TableWork(int64_t entries, int32_t active) const
    {
        // The short-domain calibration accounts for at most two 64-ID blocks.
        // Both table/tree and vector/table decisions use the same estimate.
        if (logicalBlocks_ <= Plan::kSmallMembershipDomainBlocks) {
            return entries +
                   ((logicalBlocks_ + Plan::kTableAlignmentBlocks - 1) / Plan::kTableAlignmentBlocks *
                        Plan::kTableAlignmentBlocks +
                    Plan::kTableAlignmentBlocks) /
                       Plan::kTableScanBlocksPerWork +
                   Plan::kSmallTableRowWork * active;
        }
        return entries + (logicalBlocks_ + Plan::kTableScanBlocksPerWork - 1) / Plan::kTableScanBlocksPerWork +
               Plan::kLargeTableRowWork * active;
    }
    __aicore__ inline bool PreferTable(int64_t entries, int32_t active)
    {
        if (logicalBlocks_ > Plan::kSmallMembershipDomainBlocks && entries < Plan::kMinBatchedEntries) {
            return false;
        }
        return Plan::kWorkComparisonScale * TableWork(entries, active) <
               Plan::kPreferredWorkScale * (Plan::kScalarEntryWork * entries);
    }
#include "union_sparse_gen_unsorted.h"
    template <typename TableT>
    __aicore__ inline void UpdateMembership(__ubuf__ int32_t* values, int32_t n, __ubuf__ TableT* table, uint32_t base,
                                            uint32_t length, uint32_t upper, uint32_t bit)
    {
        if (base == 0 && upper <= length) {
            for (int32_t i = 0; i < n; ++i) {
                const uint32_t id = uint32_t(values[i]);
                if (id < upper) {
                    table[id] |= TableT(bit);
                }
            }
        } else {
            for (int32_t i = 0; i < n; ++i) {
                const uint32_t id = uint32_t(values[i]), offset = id - base;
                if (id < upper && offset < length) {
                    table[offset] |= TableT(bit);
                }
            }
        }
    }
    __attribute__((noinline)) __aicore__ void MergeMembership()
    {
        if constexpr (Mode == Execution::MembershipAdaptive) {
            // Host proves full sort residency; request-local lengths stay dynamic.
            if (logicalBlocks_ > UnionSparseGenA2Plan::kSmallMembershipDomainBlocks &&
                logicalBlocks_ <= kVectorMaxLogicalBlocks) {
                MergeUnsortedBatch(int32_t(UnionSparseGenA2Plan::MembershipSortStride(topK_, residual_)));
                return;
            }
        }
        // Membership width follows the group width, not a model shape.
        if (Width() == Plan::kMinGroupWidth) {
            MergeMembershipTyped<uint8_t>();
        } else {
            MergeMembershipTyped<uint32_t>();
        }
    }
    template <typename TableT>
    __aicore__ inline void MergeMembershipTyped()
    {
        BeginGroupMerge();
        const int64_t chunkCapacity = ((algorithmBytes_ - Plan::kTableScratchBytes) / sizeof(TableT) /
                                       Plan::kTableAlignmentBlocks) *
                                      Plan::kTableAlignmentBlocks;
        auto table = algorithmBuf_.Get<TableT>();
        for (int64_t base = 0; base < logicalBlocks_; base += chunkCapacity) {
            const int32_t length = int32_t(logicalBlocks_ - base < chunkCapacity ? logicalBlocks_ - base :
                                                                                   chunkCapacity);
            const int32_t aligned = (length + Plan::kTableAlignmentBlocks - 1) / Plan::kTableAlignmentBlocks *
                                        Plan::kTableAlignmentBlocks +
                                    Plan::kTableAlignmentBlocks;
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            // Table, rounded read guard and compaction bitmap are all initialized.
            Duplicate(table.template ReinterpretCast<uint32_t>(), uint32_t(0),
                      uint32_t((aligned * sizeof(TableT) + Plan::kTableGuardBytes + Plan::kDmaBlockBytes - 1) /
                               Plan::kDmaBlockBytes * Plan::kDmaWords));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            AccumulateMembershipChunk(table, base, length);
            CompactTableRange(table, length, base);
        }
        FinishGroupCount();
    }

    __aicore__ inline void Merge()
    {
        BeginGroupMerge();
        int32_t countActive = 0;
        const int64_t entries = LoadMergeLimits(countActive);
        // Vector preprocessing validates/deduplicates before any scalar cursor is primed.
        bool vectorDone = entries == 0;
        if constexpr (VectorWidth > 0) {
            vectorDone = TryBatchedVectorMerge(entries, countActive);
        }
        if (!vectorDone) {
            MergeScalarRows(entries);
        }
        if (buffered_) {
            Flush(buffered_);
        }
        FinishGroupCount();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> metaBuf_, inputBuf_, countBuf_, pairBuf_, outputBuf_, algorithmBuf_, broadcastBuf_;
    GlobalTensor<int32_t> input_, counts_, usedQ_, usedKv_, indices_, outCounts_, mask_;
    GlobalTensor<int64_t> cuQ_, cuKv_;
    LocalTensor<int64_t> prefix_;
    LocalTensor<int32_t> meta_, rowData_, cnt_, pairIdx_;
    LocalTensor<uint32_t> pairMask_, out_, broadcastOffsets_;
    int64_t batch_, totalQ_, topK_, groups_, capacity_, width_, by_, usedCore_, workPerCore_, workTail_;
    int64_t head_, qStart_, q0_, qLen_, kvLen_, logicalBlocks_, visibilityBase_, outputBase_, rank_;
    int64_t metaStart_, request_, requestGroup_, algorithmBytes_;
    int32_t rows_, window_, tile_, buffered_, slot_;
    int32_t kvPrefixOffset_, usedMetaOffset_, usedKvOffset_;
    bool hasQ_, hasKv_, hasCuQ_, hasCuKv_, residual_, pending_[Plan::kOutputSlots], paddingReady_[Plan::kOutputSlots],
        nonCausal_;
    int32_t pos_[kMaxRows], limit_[kMaxRows], value_[kMaxRows], tree_[Plan::kBinaryTreeArity * kMaxRows];
    bool live_[kMaxRows];
    LocalTensor<uint32_t> stream_;
    int32_t streamLength_[Plan::kStreamNodeCount], streamOffset_[Plan::kStreamNodeCount], streamLeaf_;
    bool vectorOffsetsReady_;
    int64_t vectorScratchOffset_;
};
} // namespace UnionSparseGenArch22
#endif
