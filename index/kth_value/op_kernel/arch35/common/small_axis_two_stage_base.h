/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file small_axis_two_stage_base.h
 * \brief Common SIMT kernels and constants shared by sort and kth_value small_axis_two_stage kernels.
 */

#ifndef SMALL_AXIS_TWO_STAGE_BASE_H
#define SMALL_AXIS_TWO_STAGE_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "signed_zero_sort_utils.h"
#include "util_type_simd.h"

namespace SmallAxisCommon {
using namespace AscendC;

constexpr uint32_t TWO_STAGE_THREAD_NUM = 1024;
using SignedZeroSortCommon::FloatBitsT;
using SignedZeroSortCommon::IS_FLOATING_POINT_V;

/**
 * @brief SIMT gather kernel for non-last-axis two-stage-sort batches.
 * @tparam T Input storage data type
 */
template <typename T>
__simt_vf__ LAUNCH_BOUND(TWO_STAGE_THREAD_NUM) __aicore__
    void LoadNonLastBatchSimt(uint32_t totalElems, uint32_t segmentLen, uint32_t validSegs, uint64_t outerBaseOffset,
                              uint64_t innerStart, uint64_t innerSize, __gm__ volatile T* input, __ubuf__ T* output)
{
    // Gather the original [axis, inner] tile into [inner segment, axis] order for two-stage sorting.
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < totalElems; idx += TWO_STAGE_THREAD_NUM) {
        uint32_t axis = idx / validSegs;
        uint32_t seg = idx - axis * validSegs;
        output[seg * segmentLen + axis] = input[outerBaseOffset + static_cast<uint64_t>(axis) * innerSize + innerStart +
                                                seg];
    }
}

// Scatter sorted values back to per-segment order via rank-inverse mapping.
// Example (2 segments x 3 elements, ascending sort):
//   Input: [[3, 1, 5], [4, 2, 6]] -> flatten [3, 1, 5, 4, 2, 6]
//   Stage1 sort: stage1Values=[1, 2, 3, 4, 5, 6], stage1Order=[1, 4, 0, 3, 2, 5]
//   Build rankInverse: rankInverse[flatIdx]=rank -> [2, 0, 4, 3, 1, 5]
//   Scatter to per-seg: flatIdx 0(rank=2)->seg0[1]=3, flatIdx 1(rank=0)->seg0[0]=1, ...
//   Final: values=[1, 3, 5, 2, 4, 6], indices=[1, 0, 2, 1, 0, 2]
// Uses 8-way unroll for the inner counting loop to reduce loop overhead and hide memory latency.
/**
 * @brief SIMT scatter kernel that builds per-row sorted values and original indices from rank inverse data.
 * @tparam T Sorted value data type
 * @tparam OutIdxT Final index data type stored in UB
 */
template <typename T, typename OutIdxT>
__simt_vf__ LAUNCH_BOUND(TWO_STAGE_THREAD_NUM) __aicore__
    void RankInverseScatter(uint32_t totalElems, uint32_t segmentLen, __ubuf__ T* stage1ValuePtr,
                            __ubuf__ uint32_t* stage1OrderPtr, __ubuf__ uint32_t* signFlagPtr,
                            __ubuf__ uint32_t* rankInversePtr, __ubuf__ T* finalValuePtr, __ubuf__ OutIdxT* finalIdxPtr)
{
    // Build inverse mapping: for each flattened index, store its rank from stage-1 sort.
    // rankInverse[stage1Order[rank]] = rank, e.g. stage1Order=[1, 4, 0, 3, 2, 5]
    // gives rankInverse=[2, 0, 4, 3, 1, 5].
    for (uint32_t rank = static_cast<uint32_t>(threadIdx.x); rank < totalElems; rank += TWO_STAGE_THREAD_NUM) {
        uint32_t flatIdx = stage1OrderPtr[rank];
        if constexpr (IS_FLOATING_POINT_V<T>) {
            using BitsT = FloatBitsT<T>;
            __ubuf__ BitsT* signFlagBits = reinterpret_cast<__ubuf__ BitsT*>(signFlagPtr);
            if (signFlagBits[flatIdx] != static_cast<BitsT>(0U)) {
                constexpr BitsT signBit = static_cast<BitsT>(BitsT{1} << (sizeof(T) * 8U - 1U));
                __ubuf__ BitsT* stage1ValueBits = reinterpret_cast<__ubuf__ BitsT*>(stage1ValuePtr);
                stage1ValueBits[rank] = signBit;
            }
        }
        rankInversePtr[flatIdx] = rank;
    }
    asc_syncthreads();

    // Compute final position for each element by counting how many same-segment elements have lower rank.
    for (uint32_t flat = static_cast<uint32_t>(threadIdx.x); flat < totalElems; flat += TWO_STAGE_THREAD_NUM) {
        uint32_t flatIdx = flat;
        uint32_t segId = flatIdx / segmentLen;
        uint32_t laneId = flatIdx - segId * segmentLen;

        uint32_t rank = rankInversePtr[flatIdx];
        uint32_t segBase = segId * segmentLen;
        // 8-way unroll: reduces loop overhead and hides memory latency via independent accumulators.
        uint32_t pos0 = 0;
        uint32_t pos1 = 0;
        uint32_t pos2 = 0;
        uint32_t pos3 = 0;
        uint32_t pos4 = 0;
        uint32_t pos5 = 0;
        uint32_t pos6 = 0;
        uint32_t pos7 = 0;
        uint32_t lane = 0;
        for (; lane + 7U < segmentLen; lane += 8U) {
            uint32_t scanBase = segBase + lane;
            pos0 += (rankInversePtr[scanBase] < rank) ? 1U : 0U;
            pos1 += (rankInversePtr[scanBase + 1U] < rank) ? 1U : 0U;
            pos2 += (rankInversePtr[scanBase + 2U] < rank) ? 1U : 0U;
            pos3 += (rankInversePtr[scanBase + 3U] < rank) ? 1U : 0U;
            pos4 += (rankInversePtr[scanBase + 4U] < rank) ? 1U : 0U;
            pos5 += (rankInversePtr[scanBase + 5U] < rank) ? 1U : 0U;
            pos6 += (rankInversePtr[scanBase + 6U] < rank) ? 1U : 0U;
            pos7 += (rankInversePtr[scanBase + 7U] < rank) ? 1U : 0U;
        }
        uint32_t pos = pos0 + pos1 + pos2 + pos3 + pos4 + pos5 + pos6 + pos7;
        for (; lane < segmentLen; ++lane) {
            pos += (rankInversePtr[segBase + lane] < rank) ? 1U : 0U;
        }

        uint32_t outOffset = segBase + pos;
        finalValuePtr[outOffset] = stage1ValuePtr[rank];
        finalIdxPtr[outOffset] = static_cast<OutIdxT>(laneId);
    }
}

// Build packed keys for stage-2 sort: key = segId * totalElems + rank.
// Example after stage1Order=[1, 4, 0, 3, 2, 5] for 2x3 input:
//   flatIdx 1 -> seg0 lane1 rank0 -> key 0*6+0 = 0, stage1Order becomes laneId 1
//   flatIdx 4 -> seg1 lane1 rank1 -> key 1*6+1 = 7, stage1Order becomes laneId 1
//   stage2Keys=[0, 7, 2, 9, 4, 11], stage1Order=[1, 1, 0, 0, 2, 2]
/**
 * @brief SIMT helper that packs stage-2 sort keys and rewrites stage-1 order to row-local lane ids.
 */
inline __simt_vf__ LAUNCH_BOUND(TWO_STAGE_THREAD_NUM) __aicore__
    void BuildStage2KeysSimt(uint32_t totalElems, uint32_t segmentLen, __ubuf__ uint32_t* stage1OrderPtr,
                             __ubuf__ uint16_t* stage2KeyPtr)
{
    for (uint32_t rank = static_cast<uint32_t>(threadIdx.x); rank < totalElems; rank += TWO_STAGE_THREAD_NUM) {
        uint32_t flatIdx = stage1OrderPtr[rank];
        uint32_t segId = flatIdx / segmentLen;
        uint32_t laneId = flatIdx - segId * segmentLen;
        stage2KeyPtr[rank] = static_cast<uint16_t>(segId * totalElems + rank);
        stage1OrderPtr[rank] = laneId;
    }
}

/**
 * @brief CRTP base class for small-axis two-stage-sort kernels.
 *        Contains shared Process loop, UB buffer setup, contiguous/non-last load, and the two-stage sort pipeline.
 *        Derived classes provide batch mapping details and output store behavior.
 * @tparam Derived CRTP derived type
 * @tparam T Input/output value data type
 * @tparam FinalIdxT Final index data type stored in UB before output
 * @tparam IsDescend Sort order: true for descending, false for ascending
 */
template <typename Derived, typename T, typename FinalIdxT, bool IsDescend>
class SmallAxisTwoStageBase {
public:
    __aicore__ inline void Process()
    {
        Derived* op = static_cast<Derived*>(this);
        if (op->IsProcessInvalid()) {
            return;
        }
        for (uint32_t batchId = blockIdx_; batchId < batchNum_; batchId += blockDim_) {
            uint32_t validSegs = op->ComputeValidSegs(batchId);
            if (validSegs == 0U) {
                continue;
            }
            op->ProcessBatch(batchId, validSegs);
        }
    }

    __aicore__ inline void InitSortBuffers(TPipe* pipe, uint32_t maxFlatElems, uint32_t tmpUbSize, bool useRankInverse,
                                           uint32_t finalIdxElemBytes)
    {
        pipe_ = pipe;
        maxFlatElems_ = maxFlatElems;
        useRankInverse_ = useRankInverse;

        pipe_->InitBuffer(inputBuf_, ROUND_UP_AGLIN(maxFlatElems_ * sizeof(T)));
        pipe_->InitBuffer(stage1ValueBuf_, ROUND_UP_AGLIN(maxFlatElems_ * sizeof(T)));
        pipe_->InitBuffer(stage1OrderBuf_, ROUND_UP_AGLIN(maxFlatElems_ * sizeof(uint32_t)));
        pipe_->InitBuffer(finalIdxBuf_, ROUND_UP_AGLIN(maxFlatElems_ * finalIdxElemBytes));
        pipe_->InitBuffer(rankInverseBuf_, ROUND_UP_AGLIN(maxFlatElems_ * sizeof(uint32_t)));
        if (!useRankInverse_) {
            pipe_->InitBuffer(stage2OrderBuf_, ROUND_UP_AGLIN(maxFlatElems_ * sizeof(uint32_t)));
        }
        pipe_->InitBuffer(tmpBuf_, tmpUbSize);

        inputValues_ = inputBuf_.template Get<T>();
        stage1Values_ = stage1ValueBuf_.template Get<T>();
        stage1Order_ = stage1OrderBuf_.template Get<uint32_t>();
        // finalValues_ intentionally aliases inputValues_. After Stage1Sort finishes, inputValues_ is no longer
        // read in the same batch, so the input buffer can be reused as the final output buffer without overlap.
        finalValues_ = inputValues_;
        finalIdx_ = finalIdxBuf_.template Get<FinalIdxT>();
        // finalIdx_ is not live until rank-inverse scatter, so its B32-or-wider
        // storage safely carries the per-source -0 marker during Stage1Sort.
        stage1SignFlags_ = finalIdxBuf_.template Get<uint32_t>();
        if (useRankInverse_) {
            rankInverse_ = rankInverseBuf_.template Get<uint32_t>();
        } else {
            // The two-stage-sort path reuses finalIdxBuf_ as stage2KeysIn_ (uint16) before final indices are built.
            // The host UB estimate sizes finalIdxBuf_ by max(sizeof(FinalIdxT), sizeof(uint16_t)) for this alias.
            stage2KeysIn_ = finalIdxBuf_.template Get<uint16_t>();
            // stage2KeysOut_ is compact uint16_t while holding sorted keys. BuildOutputs later reuses the same
            // rankInverseBuf_ storage as uint32_t gather-offset scratch after the key values are no longer needed.
            stage2KeysOut_ = rankInverseBuf_.template Get<uint16_t>();
            stage2Order_ = stage2OrderBuf_.template Get<uint32_t>();
        }
        tmp_ = tmpBuf_.template Get<uint8_t>();
    }

    __aicore__ inline void LoadContiguousBatch(GlobalTensor<T>& inputGm, int64_t inputStart, uint32_t totalElems)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParam{1, static_cast<uint32_t>(totalElems * sizeof(T)), 0, 0, 0};
        DataCopyPad(inputValues_, inputGm[inputStart], copyParam, padParams);
        event_t eventIdMte2ToV = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }

    __aicore__ inline void LoadNonLastBatch(GlobalTensor<T>& inputGm, uint64_t outerBaseOffset, uint64_t innerStart,
                                            uint64_t innerSize, uint32_t validSegs, uint32_t totalElems)
    {
        asc_vf_call<LoadNonLastBatchSimt<T>>(
            dim3(TWO_STAGE_THREAD_NUM), totalElems, segmentLen_, validSegs, outerBaseOffset, innerStart, innerSize,
            (__gm__ volatile T*)inputGm.GetPhyAddr(), (__ubuf__ T*)inputValues_.GetPhyAddr());
    }

    __aicore__ inline void RunTwoStageSort(uint32_t totalElems)
    {
        Stage1Sort(totalElems);
        if (useRankInverse_) {
            ScatterOutputs(totalElems);
        } else {
            BuildStage2Keys(totalElems);
            Stage2Sort(totalElems);
            BuildOutputs(totalElems);
        }
    }

protected:
    TPipe* pipe_ = nullptr;

    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> stage1ValueBuf_;
    TBuf<TPosition::VECCALC> stage1OrderBuf_;
    TBuf<TPosition::VECCALC> finalIdxBuf_;
    TBuf<TPosition::VECCALC> rankInverseBuf_;
    TBuf<TPosition::VECCALC> stage2OrderBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    LocalTensor<T> inputValues_;
    LocalTensor<T> stage1Values_;
    LocalTensor<uint32_t> stage1Order_;
    LocalTensor<uint32_t> rankInverse_;
    LocalTensor<uint16_t> stage2KeysIn_;
    LocalTensor<uint16_t> stage2KeysOut_;
    LocalTensor<uint32_t> stage2Order_;
    LocalTensor<T> finalValues_;
    LocalTensor<FinalIdxT> finalIdx_;
    LocalTensor<uint32_t> stage1SignFlags_;
    LocalTensor<uint8_t> tmp_;

    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t batchNum_ = 0;
    uint32_t segmentLen_ = 0;
    uint32_t maxFlatElems_ = 0;
    bool useRankInverse_ = false;

private:
    // Two-stage sort strategy for small-axis sorting:
    //   Stage 1: Sort all rows in the current batch as one flattened array. The Sort API returns values
    //            plus original flattened positions in stage1Order_.
    //   Stage 2: Restore row-local ordering. The normal path sorts packed uint32 keys
    //            (segId * totalElems + rank) to regroup ranks by segment; the rank-inverse path avoids
    //            this second sort and computes each element's position inside its original segment.
    static constexpr SortConfig kStage1SortConfig{SortType::RADIX_SORT, IsDescend};
    static constexpr SortConfig kStage2SortConfig{SortType::RADIX_SORT, false};

    __aicore__ inline void Stage1Sort(uint32_t totalElems)
    {
        if constexpr (IS_FLOATING_POINT_V<T>) {
            if (useRankInverse_) {
                SignedZeroSortCommon::PrepareSignedZeroKeysAndFlagsVec(inputValues_, stage1SignFlags_, totalElems);
                AscendC::Sort<T, false, kStage1SortConfig>(stage1Values_, stage1Order_, inputValues_, tmp_, totalElems);
                return;
            }
        }
        AscendC::Sort<T, false, kStage1SortConfig>(stage1Values_, stage1Order_, inputValues_, tmp_, totalElems);
    }

    __aicore__ inline void ScatterOutputs(uint32_t totalElems)
    {
        asc_vf_call<RankInverseScatter<T, FinalIdxT>>(
            dim3(TWO_STAGE_THREAD_NUM), totalElems, segmentLen_, (__ubuf__ T*)stage1Values_.GetPhyAddr(),
            (__ubuf__ uint32_t*)stage1Order_.GetPhyAddr(), (__ubuf__ uint32_t*)stage1SignFlags_.GetPhyAddr(),
            (__ubuf__ uint32_t*)rankInverse_.GetPhyAddr(), (__ubuf__ T*)finalValues_.GetPhyAddr(),
            (__ubuf__ FinalIdxT*)finalIdx_.GetPhyAddr());
    }

    __aicore__ inline void BuildStage2Keys(uint32_t totalElems)
    {
        asc_vf_call<BuildStage2KeysSimt>(dim3(TWO_STAGE_THREAD_NUM), totalElems, segmentLen_,
                                         (__ubuf__ uint32_t*)stage1Order_.GetPhyAddr(),
                                         (__ubuf__ uint16_t*)stage2KeysIn_.GetPhyAddr());
    }

    __aicore__ inline void Stage2Sort(uint32_t totalElems)
    {
        AscendC::Sort<uint16_t, false, kStage2SortConfig>(stage2KeysOut_, stage2Order_, stage2KeysIn_, tmp_,
                                                          totalElems);
    }

    __aicore__ inline void BuildOutputs(uint32_t totalElems)
    {
        // Full pipeline example (2x3, ascending):
        //   Input [[3,1,5], [4,2,6]] -> flatten [3,1,5,4,2,6]
        //   Stage1Sort:
        //     stage1Values=[1,2,3,4,5,6]
        //     stage1Order =[1,4,0,3,2,5]  // original flatIdx per rank
        //   BuildStage2Keys:
        //     stage2Keys=[0,7,2,9,4,11]   // key = segId * 6 + rank
        //     stage1Order=[1,1,0,0,2,2]   // rewritten to row-local lane ids
        //   Stage2Sort:
        //     stage2KeysOut=[0,2,4,7,9,11]
        //     stage2Order  =[0,2,4,1,3,5] // final permutation into per-segment order
        //   BuildOutputs gathers stage1Values/stage1Order via stage2Order:
        //     values =[1,3,5,2,4,6], indices=[1,0,2,1,0,2]
        // Reuse rankInverseBuf_ as uint32_t offset scratch. stage2KeysOut_ itself is uint16_t compact storage;
        // the following Muls operations overwrite this buffer before Gather reads it as byte offsets.
        LocalTensor<uint32_t> gatherOffsets = stage2KeysOut_.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> gatherOffsetsInt = gatherOffsets.template ReinterpretCast<int32_t>();

        Muls(gatherOffsetsInt, stage2Order_.template ReinterpretCast<int32_t>(), static_cast<int32_t>(sizeof(T)),
             totalElems);
        Gather(finalValues_, stage1Values_, gatherOffsets, 0, totalElems);

        Muls(gatherOffsetsInt, stage2Order_.template ReinterpretCast<int32_t>(), static_cast<int32_t>(sizeof(uint32_t)),
             totalElems);
        if constexpr (IsSameType<FinalIdxT, int64_t>::value) {
            LocalTensor<int32_t> gatheredIdxInt32 = stage2Order_.template ReinterpretCast<int32_t>();
            Gather(gatheredIdxInt32, stage1Order_.template ReinterpretCast<int32_t>(), gatherOffsets, 0, totalElems);
            Cast(finalIdx_, gatheredIdxInt32, RoundMode::CAST_NONE, Ops::Base::CeilAlign(totalElems, 4u));
        } else {
            Gather(finalIdx_.template ReinterpretCast<int32_t>(), stage1Order_.template ReinterpretCast<int32_t>(),
                   gatherOffsets, 0, totalElems);
        }
    }
};

} // namespace SmallAxisCommon

#endif
