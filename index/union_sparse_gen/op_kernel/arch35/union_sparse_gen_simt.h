/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_SIMT_H
#define UNION_SPARSE_GEN_SIMT_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "union_sparse_gen_a5_tiling_data.h"
#include "union_sparse_gen_a5_tiling_key.h"
#include "lib/sort/sort.h"

namespace UnionSparseGenNs {
using namespace AscendC;
namespace Plan = UnionSparseGenA5Plan;

constexpr int32_t kThreadNum = Plan::kThreadNum;

template <typename OffsetT>
__simt_callee__ __aicore__ inline uint32_t BuildCausalQueryMask(int32_t block, int32_t lane, OffsetT kvLen,
                                                                OffsetT qLen, OffsetT qLocalStart, int32_t rows,
                                                                int32_t blockY)
{
    const OffsetT kvPos = static_cast<OffsetT>(block) * blockY + lane;
    if (kvPos >= kvLen) {
        return 0U;
    }
    const OffsetT causalBase = kvLen - qLen + qLocalStart;
    OffsetT firstSlot = kvPos - causalBase;
    if (firstSlot < 0) {
        firstSlot = 0;
    }
    if (firstSlot >= rows) {
        return 0U;
    }
    const uint32_t rowMask = rows == Plan::kBitsPerWord ?
                                 Plan::kAllQueryBits :
                                 ((static_cast<uint32_t>(1) << static_cast<uint32_t>(rows)) - 1U);
    return rowMask & (Plan::kAllQueryBits << static_cast<uint32_t>(firstSlot));
}

template <typename OffsetT, int32_t ThreadNum, bool HasAnySeqUsed, int32_t GroupWidth, int32_t BlockY, bool Causal>
__simt_vf__ __aicore__ __launch_bounds__(ThreadNum) inline void UnionSparseGenWarpMergeCore(
    __gm__ int32_t* sparseBlockIdx, __gm__ int32_t* sparseBlockCount, __gm__ int64_t* cuSqlenQ,
    __gm__ int64_t* cuSqlenKv, __gm__ int32_t* seqUsedQ, __gm__ int32_t* seqUsedKv, __gm__ int32_t* unionIdx,
    __gm__ int32_t* unionCount, __gm__ int32_t* exactMask, OffsetT workStart, OffsetT workCount, OffsetT numGroups,
    OffsetT batchSize, OffsetT totalQ, OffsetT topK, OffsetT maxLogicalBlocks, OffsetT unionCapacity, OffsetT unionSize,
    OffsetT /*blockY*/, bool hasSeqUsedQ, bool hasSeqUsedKv)
{
    constexpr OffsetT blockY = BlockY;
    // Keep cursor/collective/writer control flow visible in one SIMT scope.
    // One lane maps to one query row. A warp is split into 1, 2, or 4
    // independent 32-, 16-, or 8-lane merge subgroups.
    constexpr int32_t kInactiveBlock = 0x7FFFFFFF;
    constexpr int32_t kGroupsPerWarp = Plan::kWarpLanes / GroupWidth;
    const OffsetT tid = static_cast<OffsetT>(threadIdx.x);
    const int32_t lane = static_cast<int32_t>(tid & Plan::kWarpLaneMask);
    const OffsetT warp = tid >> Plan::kWarpShift;
    const int32_t subgroup = lane / GroupWidth;
    const int32_t subgroupLane = lane - subgroup * GroupWidth;
    const OffsetT subgroupId = warp * kGroupsPerWarp + static_cast<OffsetT>(subgroup);
    const OffsetT subgroupStride = static_cast<OffsetT>((ThreadNum / Plan::kWarpLanes) * kGroupsPerWarp);
    __gm__ uint32_t* exactMaskWords = reinterpret_cast<__gm__ uint32_t*>(exactMask);

    for (OffsetT workId = workStart + subgroupId; workId < workStart + workCount; workId += subgroupStride) {
        const OffsetT head = workId / numGroups;
        const OffsetT globalGroup = workId - head * numGroups;
        // Locate the request and its effective lengths.
        OffsetT remaining = globalGroup;
        OffsetT request = 0;
        OffsetT qStart = 0;
        OffsetT qLen = 0;
        OffsetT localGroup = 0;
        for (OffsetT batch = 0; batch < batchSize; ++batch) {
            qStart = cuSqlenQ != nullptr ? static_cast<OffsetT>(cuSqlenQ[batch]) : batch * (totalQ / batchSize);
            const OffsetT qCapacity = cuSqlenQ != nullptr ?
                                          static_cast<OffsetT>(cuSqlenQ[batch + 1] - cuSqlenQ[batch]) :
                                          totalQ / batchSize;
            if constexpr (HasAnySeqUsed) {
                qLen = hasSeqUsedQ ? static_cast<OffsetT>(seqUsedQ[batch]) : qCapacity;
            } else {
                qLen = qCapacity;
            }
            const OffsetT groups = (qLen + unionSize - 1) / unionSize;
            if (remaining < groups) {
                request = batch;
                localGroup = remaining;
                break;
            }
            remaining -= groups;
        }

        OffsetT kvLen = 0;
        if constexpr (HasAnySeqUsed) {
            if (hasSeqUsedKv) {
                kvLen = static_cast<OffsetT>(seqUsedKv[request]);
            } else {
                kvLen = static_cast<OffsetT>(cuSqlenKv[request + 1] - cuSqlenKv[request]);
            }
        } else {
            kvLen = static_cast<OffsetT>(cuSqlenKv[request + 1] - cuSqlenKv[request]);
        }
        const OffsetT requestLogicalBlocks = (kvLen + blockY - 1) / blockY;
        const OffsetT effectiveLogicalBlocks = requestLogicalBlocks < maxLogicalBlocks ? requestLogicalBlocks :
                                                                                         maxLogicalBlocks;
        const OffsetT qLocalStart = localGroup * unionSize;
        const OffsetT remainingRows = qLen - qLocalStart;
        const int32_t rows = static_cast<int32_t>(remainingRows < unionSize ? remainingRows : unionSize);
        const OffsetT groupLinear = head * numGroups + globalGroup;
        const OffsetT unionBase = groupLinear * unionCapacity;
        const OffsetT maskBase = unionBase * blockY;
        const OffsetT rowLinear = head * totalQ + qStart + qLocalStart + subgroupLane;
        const OffsetT rowSparseBase = rowLinear * topK;
        const bool activeLane = subgroupLane < rows;
        int32_t count = 0;
        OffsetT slot = 0;
        if (activeLane) {
            count = sparseBlockCount[rowLinear];
            // Match the bounded valid-prefix handling of the membership path.
            count = count < 0 ? 0 : (static_cast<OffsetT>(count) > topK ? static_cast<int32_t>(topK) : count);
        }

        // Advance sorted row cursors together with subgroup selection and writeback.
        int32_t current = kInactiveBlock;
        OffsetT rank = 0;
        while (true) {
            int32_t next = kInactiveBlock;
            if (activeLane) {
                while (slot < static_cast<OffsetT>(count)) {
                    const int32_t value = sparseBlockIdx[rowSparseBase + slot];
                    if (value >= 0 && static_cast<OffsetT>(value) < effectiveLogicalBlocks) {
                        next = value;
                        break;
                    }
                    ++slot;
                }
            }
            current = next;

            int32_t block = current;
            int32_t other = asc_shfl_xor(block, Plan::kPairShuffleDistance, GroupWidth);
            block = other < block ? other : block;
            other = asc_shfl_xor(block, Plan::kQuadShuffleDistance, GroupWidth);
            block = other < block ? other : block;
            other = asc_shfl_xor(block, Plan::kOctetShuffleDistance, GroupWidth);
            block = other < block ? other : block;
            if constexpr (GroupWidth >= Plan::kHalfWarpLanes) {
                other = asc_shfl_xor(block, Plan::kHalfWarpShuffleDistance, GroupWidth);
                block = other < block ? other : block;
            }
            if constexpr (GroupWidth == Plan::kWarpLanes) {
                other = asc_shfl_xor(block, Plan::kFullWarpShuffleDistance, GroupWidth);
                block = other < block ? other : block;
            }
            if (block == kInactiveBlock) {
                break;
            }
            const uint32_t warpBallot = asc_ballot(current == block);
            uint32_t blockMask = 0U;
            if constexpr (GroupWidth == Plan::kWarpLanes) {
                blockMask = warpBallot;
            } else {
                blockMask = (warpBallot >> static_cast<uint32_t>(subgroup * GroupWidth)) &
                            ((static_cast<uint32_t>(1) << static_cast<uint32_t>(GroupWidth)) - 1U);
            }
            if (subgroupLane == 0) {
                unionIdx[unionBase + rank] = block;
            }
            if constexpr (!Causal) {
                const OffsetT first = static_cast<OffsetT>(block) * blockY;
                if (first + blockY <= kvLen) {
                    const uint4 fill = make_uint4(blockMask, blockMask, blockMask, blockMask);
                    auto* output = reinterpret_cast<__gm__ uint4*>(exactMaskWords + maskBase + rank * blockY);
                    for (int32_t vector = subgroupLane; vector < BlockY / Plan::kVectorWords; vector += GroupWidth) {
                        output[vector] = fill;
                    }
                } else {
                    for (int32_t token = subgroupLane; token < BlockY; token += GroupWidth) {
                        exactMaskWords[maskBase + rank * blockY + token] = first + token < kvLen ? blockMask : 0U;
                    }
                }
            } else if constexpr (BlockY <= Plan::kNarrowBlockY) {
                if (subgroupLane < BlockY) {
                    const uint32_t causalMask = BuildCausalQueryMask(block, subgroupLane, kvLen, qLen, qLocalStart,
                                                                     rows, BlockY);
                    exactMaskWords[maskBase + rank * blockY + subgroupLane] = blockMask & causalMask;
                }
            } else if constexpr (BlockY == Plan::kMediumBlockY) {
                // A block-wide branch costs more than the short scalar writer here.
                for (int32_t token = subgroupLane; token < BlockY; token += GroupWidth) {
                    const uint32_t causalMask = BuildCausalQueryMask(block, token, kvLen, qLen, qLocalStart, rows,
                                                                     BlockY);
                    exactMaskWords[maskBase + rank * blockY + token] = blockMask & causalMask;
                }
            } else {
                const OffsetT firstToken = static_cast<OffsetT>(block) * blockY;
                const OffsetT causalBase = kvLen - qLen + qLocalStart;
                const bool allAllowed = firstToken + blockY - 1 <= causalBase && firstToken + blockY - 1 < kvLen;
                const bool allMasked = firstToken >= kvLen || firstToken >= causalBase + rows;
                if (allAllowed || allMasked) {
                    // All lanes agree: keep merge membership, expand only at writeback.
                    const uint32_t value = allAllowed ? blockMask : 0U;
                    const uint4 fill = make_uint4(value, value, value, value);
                    auto* blockOutput = reinterpret_cast<__gm__ uint4*>(exactMaskWords + maskBase + rank * blockY);
                    for (int32_t vector = subgroupLane; vector < BlockY / Plan::kVectorWords; vector += GroupWidth) {
                        blockOutput[vector] = fill;
                    }
                } else {
                    for (int32_t token = subgroupLane; token < BlockY; token += GroupWidth) {
                        const uint32_t causalMask = BuildCausalQueryMask(block, token, kvLen, qLen, qLocalStart, rows,
                                                                         BlockY);
                        exactMaskWords[maskBase + rank * blockY + token] = blockMask & causalMask;
                    }
                }
            }
            ++rank;
            if (current == block) {
                ++slot;
                while (slot < static_cast<OffsetT>(count) && sparseBlockIdx[rowSparseBase + slot] == block) {
                    ++slot;
                }
            }
        }

        if (subgroupLane == 0) {
            unionCount[groupLinear] = static_cast<int32_t>(rank);
        }
        // Fill the unused output capacity after the final emitted block.
        const OffsetT tailRanks = unionCapacity - rank;
        for (OffsetT i = subgroupLane; i < tailRanks; i += GroupWidth) {
            unionIdx[unionBase + rank + i] = -1;
        }
        const OffsetT tailMaskElements = tailRanks * blockY;
        const OffsetT activeMaskElements = rank * blockY;
        const OffsetT tailMaskVectors = tailMaskElements >> Plan::kVectorShift;
        __gm__ uint4* tailMask = reinterpret_cast<__gm__ uint4*>(exactMaskWords + maskBase + activeMaskElements);
        const uint4 zero = make_uint4(0U, 0U, 0U, 0U);
        for (OffsetT i = subgroupLane; i < tailMaskVectors; i += GroupWidth) {
            tailMask[i] = zero;
        }
    }
}

#include "union_sparse_gen_extended.h"
static constexpr SortConfig kUnionRadixConfig{SortType::RADIX_SORT, false};

template <UnionSparseGenA5Algorithm Algorithm, int32_t OrderedWidth = 0>
class UnionSparseGen {
public:
    __aicore__ inline void Init(GM_ADDR sparseBlockIdx, GM_ADDR sparseBlockCount, GM_ADDR cuSqlenQ, GM_ADDR cuSqlenKv,
                                GM_ADDR seqUsedQ, GM_ADDR seqUsedKv, GM_ADDR unionIdx, GM_ADDR unionCount,
                                GM_ADDR exactMask, GM_ADDR workspace, const UnionSparseGenA5TilingData& payload)
    {
        const auto& tiling = payload.semantic;
        plan_ = &payload;
        sparseIdx_ = reinterpret_cast<__gm__ int32_t*>(sparseBlockIdx);
        sparseCount_ = reinterpret_cast<__gm__ int32_t*>(sparseBlockCount);
        cuSqlenQ_ = reinterpret_cast<__gm__ int64_t*>(cuSqlenQ);
        cuSqlenKv_ = reinterpret_cast<__gm__ int64_t*>(cuSqlenKv);
        seqUsedQ_ = reinterpret_cast<__gm__ int32_t*>(seqUsedQ);
        seqUsedKv_ = reinterpret_cast<__gm__ int32_t*>(seqUsedKv);
        unionIdx_ = reinterpret_cast<__gm__ int32_t*>(unionIdx);
        unionCount_ = reinterpret_cast<__gm__ int32_t*>(unionCount);
        exactMask_ = reinterpret_cast<__gm__ int32_t*>(exactMask);
        (void)workspace; // The generated entry supplies the library system workspace.
        tiling_ = &tiling;
        if constexpr (Algorithm != UnionSparseGenA5Algorithm::WarpMerge) {
            const int64_t scratch = Algorithm == UnionSparseGenA5Algorithm::RadixMembership ?
                                        payload.radixScratchBytes :
                                        payload.tableScratchBytes;
            pipe_.InitBuffer(extendedScratchBuffer_, uint32_t(scratch));
            pipe_.InitBuffer(extendedStateBuffer_, uint32_t(kUnionSparseGenA5StateBytes));
        }
    }

    __aicore__ inline void Process()
    {
        const int64_t core = GetBlockIdx();
        if (core >= tiling_->usedCoreNum || tiling_->numGroups == 0) {
            return;
        }
        const int64_t extra = core < tiling_->workTail ? 1 : 0;
        const int64_t start = core * tiling_->workPerCore + (core < tiling_->workTail ? core : tiling_->workTail);
        const int64_t count = tiling_->workPerCore + extra;

        if constexpr (Algorithm != UnionSparseGenA5Algorithm::WarpMerge) {
            if (tiling_->useInt32) {
                LaunchMembership<int32_t>(start, count);
            } else {
                LaunchMembership<int64_t>(start, count);
            }
        } else {
            static_assert(OrderedWidth == Plan::kMinGroupWidth || OrderedWidth == Plan::kHalfWarpLanes ||
                          OrderedWidth == Plan::kWarpLanes);
            LaunchWarpMergeWidth<OrderedWidth>(start, count);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    template <typename OffsetT>
    __aicore__ inline void LaunchMembership(int64_t start, int64_t count)
    {
        if (tiling_->maskType == 0) {
            LaunchMembershipMode<OffsetT, false>(start, count);
        } else {
            LaunchMembershipMode<OffsetT, true>(start, count);
        }
    }
    template <typename OffsetT, bool Causal>
    __aicore__ inline void LaunchRadixMembership(int64_t start, int64_t count, LocalTensor<uint32_t> scratch,
                                                 __ubuf__ UnionSparseGenA5GroupState* state)
    {
        const uint32_t n = uint32_t(plan_->radixCapacity);
        auto keys = scratch, bits = scratch[Plan::kInputMemberPlane * n],
             sortedKeys = scratch[Plan::kSortedKeyPlane * n], sortedBits = scratch[Plan::kSortedMemberPlane * n];
        keys.SetSize(n);
        bits.SetSize(n);
        sortedKeys.SetSize(n);
        sortedBits.SetSize(n);
        auto temporary = extendedScratchBuffer_.Get<uint8_t>()[n * Plan::kRadixRecordBytes];
        temporary.SetSize(uint32_t(plan_->radixTmpBytes));
        for (int64_t work = start; work < start + count; ++work) {
            asc_vf_call<PrepareRadixMembership<OffsetT, Causal>>(
                dim3(kThreadNum), sparseIdx_, sparseCount_, cuSqlenQ_, cuSqlenKv_, seqUsedQ_, seqUsedKv_, unionIdx_,
                unionCount_, exactMask_, reinterpret_cast<__ubuf__ uint32_t*>(keys.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(bits.GetPhyAddr()), state, OffsetT(work),
                OffsetT(tiling_->numGroups), OffsetT(tiling_->batchSize), OffsetT(tiling_->totalQ),
                OffsetT(tiling_->topK), OffsetT(tiling_->unionCapacity), OffsetT(tiling_->unionSize),
                OffsetT(tiling_->blockY), OffsetT(n), tiling_->residualBlockMode != 0);
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            auto stateWords = extendedStateBuffer_.Get<int64_t>();
            if (stateWords.GetValue(kUnionSparseGenA5InputCountWord) == 0) {
                continue;
            }
            const uint32_t actual = uint32_t(stateWords.GetValue(kUnionSparseGenA5RecordCountWord));
            AscendC::Sort<uint32_t, uint32_t, true, kUnionRadixConfig>(sortedKeys, sortedBits, keys, bits, temporary,
                                                                       actual);
            PipeBarrier<PIPE_ALL>();
            asc_vf_call<CompactRadixMembership<OffsetT, Causal>>(
                dim3(kThreadNum), reinterpret_cast<__ubuf__ uint32_t*>(sortedKeys.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(sortedBits.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(keys.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(bits.GetPhyAddr()), state, unionIdx_, unionCount_, exactMask_,
                OffsetT(work), OffsetT(tiling_->unionCapacity), OffsetT(tiling_->blockY));
            PipeBarrier<PIPE_ALL>();
        }
    }
    template <typename OffsetT, bool Causal>
    __aicore__ inline void LaunchPackedMembership(int64_t start, int64_t count, LocalTensor<uint32_t> scratch,
                                                  __ubuf__ UnionSparseGenA5GroupState* state)
    {
        const int64_t tableWords = plan_->tableBlockCapacity * tiling_->unionSize / Plan::kBitsPerWord;
        const int64_t bitmapWords = plan_->tableBlockCapacity / Plan::kBitsPerWord;
        auto* table = reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr());
        auto* presence = table + tableWords;
        auto* keys = presence + bitmapWords;
        auto* bits = keys + plan_->compactCapacity;
        asc_vf_call<PackedMembership<OffsetT, Causal>>(
            dim3(kThreadNum), sparseIdx_, sparseCount_, cuSqlenQ_, cuSqlenKv_, seqUsedQ_, seqUsedKv_, unionIdx_,
            unionCount_, exactMask_, table, presence, keys, bits, state, OffsetT(start), OffsetT(count),
            OffsetT(tiling_->numGroups), OffsetT(tiling_->batchSize), OffsetT(tiling_->totalQ), OffsetT(tiling_->topK),
            OffsetT(tiling_->unionCapacity), OffsetT(tiling_->unionSize), OffsetT(tiling_->blockY),
            OffsetT(plan_->tableBlockCapacity), OffsetT(plan_->compactCapacity), tiling_->residualBlockMode != 0);
    }
    template <typename OffsetT, bool Causal>
    __aicore__ inline void LaunchMembershipMode(int64_t start, int64_t count)
    {
        auto stateTensor = extendedStateBuffer_.Get<uint8_t>();
        auto* state = reinterpret_cast<__ubuf__ UnionSparseGenA5GroupState*>(stateTensor.GetPhyAddr());
        auto scratch = extendedScratchBuffer_.Get<uint32_t>();
        if constexpr (Algorithm == UnionSparseGenA5Algorithm::RadixMembership) {
            LaunchRadixMembership<OffsetT, Causal>(start, count, scratch, state);
        } else {
            LaunchPackedMembership<OffsetT, Causal>(start, count, scratch, state);
        }
    }
    template <int32_t GroupWidth>
    __aicore__ inline void LaunchWarpMergeWidth(int64_t start, int64_t count)
    {
        constexpr int32_t WarpThreadNum = kThreadNum;
        const bool hasAnySeqUsed = seqUsedQ_ != nullptr || seqUsedKv_ != nullptr;
        if (tiling_->useInt32 != 0) {
            if (hasAnySeqUsed) {
                LaunchWarpMergeTyped<int32_t, WarpThreadNum, true, GroupWidth>(start, count);
            } else {
                LaunchWarpMergeTyped<int32_t, WarpThreadNum, false, GroupWidth>(start, count);
            }
        } else if (hasAnySeqUsed) {
            LaunchWarpMergeTyped<int64_t, WarpThreadNum, true, GroupWidth>(start, count);
        } else {
            LaunchWarpMergeTyped<int64_t, WarpThreadNum, false, GroupWidth>(start, count);
        }
    }

    template <typename OffsetT, int32_t ThreadNum, bool HasAnySeqUsed, int32_t GroupWidth>
    __aicore__ inline void LaunchWarpMergeTyped(int64_t start, int64_t count)
    {
        if (tiling_->blockY == Plan::kGatherBlockY) {
            LaunchWarpMergeBlockY<OffsetT, ThreadNum, HasAnySeqUsed, GroupWidth, Plan::kGatherBlockY>(start, count);
        } else if (tiling_->blockY == Plan::kNarrowBlockY) {
            LaunchWarpMergeBlockY<OffsetT, ThreadNum, HasAnySeqUsed, GroupWidth, Plan::kNarrowBlockY>(start, count);
        } else if (tiling_->blockY == Plan::kMediumBlockY) {
            LaunchWarpMergeBlockY<OffsetT, ThreadNum, HasAnySeqUsed, GroupWidth, Plan::kMediumBlockY>(start, count);
        } else {
            LaunchWarpMergeBlockY<OffsetT, ThreadNum, HasAnySeqUsed, GroupWidth, Plan::kWideBlockY>(start, count);
        }
    }

    template <typename OffsetT, int32_t ThreadNum, bool HasAnySeqUsed, int32_t GroupWidth, int32_t BlockY>
    __aicore__ inline void LaunchWarpMergeBlockY(int64_t start, int64_t count)
    {
        if (tiling_->maskType == 0) {
            LaunchWarpMergeMode<OffsetT, ThreadNum, HasAnySeqUsed, GroupWidth, BlockY, false>(start, count);
        } else {
            LaunchWarpMergeMode<OffsetT, ThreadNum, HasAnySeqUsed, GroupWidth, BlockY, true>(start, count);
        }
    }
    template <typename OffsetT, int32_t ThreadNum, bool HasAnySeqUsed, int32_t GroupWidth, int32_t BlockY, bool Causal>
    __aicore__ inline void LaunchWarpMergeMode(int64_t start, int64_t count)
    {
        asc_vf_call<UnionSparseGenWarpMergeCore<OffsetT, ThreadNum, HasAnySeqUsed, GroupWidth, BlockY, Causal>>(
            dim3(ThreadNum), sparseIdx_, sparseCount_, cuSqlenQ_, cuSqlenKv_, seqUsedQ_, seqUsedKv_, unionIdx_,
            unionCount_, exactMask_, static_cast<OffsetT>(start), static_cast<OffsetT>(count),
            static_cast<OffsetT>(tiling_->numGroups), static_cast<OffsetT>(tiling_->batchSize),
            static_cast<OffsetT>(tiling_->totalQ), static_cast<OffsetT>(tiling_->topK),
            static_cast<OffsetT>(tiling_->maxLogicalBlocks), static_cast<OffsetT>(tiling_->unionCapacity),
            static_cast<OffsetT>(tiling_->unionSize), static_cast<OffsetT>(tiling_->blockY), seqUsedQ_ != nullptr,
            seqUsedKv_ != nullptr);
    }

private:
    __gm__ int32_t* sparseIdx_ = nullptr;
    __gm__ int32_t* sparseCount_ = nullptr;
    __gm__ int64_t* cuSqlenQ_ = nullptr;
    __gm__ int64_t* cuSqlenKv_ = nullptr;
    __gm__ int32_t* seqUsedQ_ = nullptr;
    __gm__ int32_t* seqUsedKv_ = nullptr;
    __gm__ int32_t* unionIdx_ = nullptr;
    __gm__ int32_t* unionCount_ = nullptr;
    __gm__ int32_t* exactMask_ = nullptr;
    const UnionSparseGenA5TilingData* plan_ = nullptr;
    const UnionSparseGenTilingData* tiling_ = nullptr;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> extendedScratchBuffer_, extendedStateBuffer_;
};
} // namespace UnionSparseGenNs

#endif
