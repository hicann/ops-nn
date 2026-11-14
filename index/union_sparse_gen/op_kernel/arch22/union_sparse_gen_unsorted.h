/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Included in the SIMD class. Two budgeted UB buffers carry exact encoded IDs
// and membership payloads. No top-k truncation or GM intermediate is used.
__aicore__ inline bool LoadMembershipInput()
{
    bool resident = rows_ > 0 && topK_ % Plan::kDmaWords == 0 && topK_ <= InputRows() * window_ / rows_;
    for (int32_t row = 0; row < rows_ && resident; ++row) {
        resident = cnt_.GetValue(row) >= topK_;
    }
    if (resident) {
        // Only complete count prefixes are contiguous here. Reuse input UB;
        // algorithm/sort scratch and their budget remain unchanged.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        CopyInput(rowData_, input_[(head_ * totalQ_ + qStart_) * topK_], uint32_t(rows_ * topK_));
        // This event stays outstanding while independent sort scratch is initialized.
        SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
    }

    return resident;
}

__aicore__ inline void InitMembershipOffsets(LocalTensor<uint32_t> scratch, int32_t allocated)
{
    const bool initialize = !vectorOffsetsReady_ ||
                            vectorScratchOffset_ != int64_t(allocated) * Plan::kResidentRecordWords;
    InitVectorOffsets(scratch, int64_t(allocated) * Plan::kResidentRecordWords);
    if (initialize) {
        auto shifts = scratch[kPackingVector * kVectorStride + Plan::kPairWords * kVectorLanes];
        for (int32_t stage = kBasePrefixStages; stage < kExtendedPrefixStages; ++stage) {
            for (int32_t i = 0; i < kVectorLanes; ++i) {
                shifts.SetValue((stage + kExtendedShiftBias) * kVectorLanes + i,
                                uint32_t((i >= (1 << stage) ? i - (1 << stage) : kVectorLanes) * Plan::kWordBytes));
            }
        }
    }
}

__aicore__ inline void PackMembershipRows(LocalTensor<uint32_t> words, LocalTensor<uint32_t> temporary,
                                          LocalTensor<uint32_t> scratch, int32_t stride, int32_t allocated,
                                          bool resident)
{
    for (int32_t row = 0; row < rows_; ++row) {
        const int64_t visible = kvLen_ - qLen_ + q0_ + row + 1;
        const int64_t upper = residual_ ? (visible > 0 ? visible / by_ : 0) : logicalBlocks_;
        const int64_t count = cnt_.GetValue(row);
        const int32_t limit = int32_t(count < 0 ? 0 : (count > topK_ ? topK_ : count));
        uint32_t previous = 0;
        if (upper > 0) {
            for (int32_t at = 0; at < limit; at += kVectorLanes) {
                const int32_t n = limit - at < kVectorLanes ? limit - at : kVectorLanes;
                const int32_t capacity = (n + Plan::kDmaWords - 1) / Plan::kDmaWords * Plan::kDmaWords;
                PackVectorWindow<true>(temporary, scratch, (head_ * totalQ_ + qStart_ + row) * topK_ + at, n, capacity,
                                       1U << uint32_t(row), previous, -1, upper,
                                       resident ? int32_t(row * topK_ + at) : -1);
                uint64_t extracted = 0;
                GatherMask(words[row * stride + at], temporary, uint8_t(Plan::kGatherKeyPattern), true,
                           uint32_t(capacity * Plan::kPairWords),
                           GatherMaskParams{1, 1, Plan::kVectorRepeatBlocks, Plan::kVectorRepeatBlocks}, extracted);
                GatherMask(words[allocated + row * stride + at], temporary, uint8_t(Plan::kGatherMemberPattern), true,
                           uint32_t(capacity * Plan::kPairWords),
                           GatherMaskParams{1, 1, Plan::kVectorRepeatBlocks, Plan::kVectorRepeatBlocks}, extracted);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
            }
        }
        if (residual_ && visible > 0 && visible % by_ != 0) {
            words.SetValue(row * stride + int32_t(topK_), kVectorKey - uint32_t(visible / by_));
            words.SetValue(allocated + row * stride + int32_t(topK_), 1U << uint32_t(row));
        }
    }
}

__aicore__ inline void MergeMembershipSortTail(LocalTensor<float> dst, LocalTensor<float> src, int32_t allocated,
                                               int32_t complete, int32_t runs, int32_t run)
{
    const int32_t first = complete * Plan::kMergeFanIn, count = runs - first;
    uint16_t lens[Plan::kMergeFanIn] = {};
    for (int32_t i = 0; i < count; ++i) {
        lens[i] = uint16_t(allocated - (first + i) * run < run ? allocated - (first + i) * run : run);
    }
    const int32_t offset = first * run * Plan::kPairWords;
    if (count == 1) {
        // A pure V copy is bit-exact and requires no MTE->V fence.
        Adds(dst[offset].ReinterpretCast<int32_t>(), src[offset].ReinterpretCast<int32_t>(), int32_t(0),
             uint32_t(lens[0] * Plan::kPairWords));
    } else {
        auto a = src[offset], b = src[(first + 1) * run * Plan::kPairWords];
        auto c = src[(first + (count > Plan::kMergeThirdInput ? Plan::kMergeThirdInput : Plan::kMergeFirstInput)) *
                     run * Plan::kPairWords],
             d = src[(first + (count > Plan::kMergeFourthInput ? Plan::kMergeFourthInput : Plan::kMergeFirstInput)) *
                     run * Plan::kPairWords];
        MrgSort(dst[offset], MrgSortSrcList<float>(a, b, c, d),
                MrgSort4Info(lens, false, uint16_t((1U << count) - 1), 1));
    }
}

__aicore__ inline LocalTensor<float> MergeMembershipRuns(LocalTensor<float> src, LocalTensor<float> dst,
                                                         int32_t allocated)
{
    int32_t runs = allocated / Plan::kSort32Elements, run = Plan::kSort32Elements;
    while (runs > 1) {
        // Batch all complete four-way groups in one vector instruction.
        const int32_t complete = (runs - 1) / Plan::kMergeFanIn;
        if (complete) {
            uint16_t lens[Plan::kMergeFanIn] = {uint16_t(run), uint16_t(run), uint16_t(run), uint16_t(run)};
            MrgSort(dst,
                    MrgSortSrcList<float>(src, src[run * Plan::kPairWords],
                                          src[run * (Plan::kPairWords * Plan::kMergeThirdInput)],
                                          src[run * (Plan::kPairWords * Plan::kMergeFourthInput)]),
                    MrgSort4Info(lens, false, (1U << Plan::kMergeFanIn) - 1, uint8_t(complete)));
        }
        MergeMembershipSortTail(dst, src, allocated, complete, runs, run);
        PipeBarrier<PIPE_V>();
        auto swap = src;
        src = dst;
        dst = swap;
        runs = (runs + Plan::kMergeFanIn - 1) / Plan::kMergeFanIn;
        run *= Plan::kMergeFanIn;
    }

    return src;
}

__aicore__ inline void EmitMembershipRuns(LocalTensor<uint32_t> sorted, LocalTensor<uint32_t> scratch,
                                          int32_t allocated)
{
    uint32_t pendingKey = 0, pendingBits = 0;
    const int32_t outputChunk = tile_ < kVectorLanes ? tile_ : kVectorLanes;
    for (int32_t at = 0; at < allocated; at += outputChunk) {
        if (!ReduceVectorPart<true>(sorted[at * Plan::kPairWords],
                                    allocated - at < outputChunk ? allocated - at : outputChunk, scratch, pendingKey,
                                    pendingBits)) {
            break;
        }
    }
    if (pendingKey) {
        Emit(int32_t(kVectorKey - pendingKey), pendingBits);
    }
    if (buffered_) {
        Flush(buffered_);
    }
    FinishGroupCount();
}
__aicore__ inline void MergeUnsortedBatch(int32_t stride)
{
    BeginGroupMerge();
    const int32_t allocated = int32_t(Width()) * stride;
    const bool resident = LoadMembershipInput();
    auto words = algorithmBuf_.Get<uint32_t>();
    auto temporary = words[allocated * Plan::kPairWords];
    auto scratch = words[allocated * Plan::kResidentRecordWords];
    InitMembershipOffsets(scratch, allocated);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(words, uint32_t(0), uint32_t(allocated * Plan::kPairWords));
    PipeBarrier<PIPE_V>();
    if (resident) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
    }
    PackMembershipRows(words, temporary, scratch, stride, allocated, resident);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Sort32(temporary.ReinterpretCast<float>(), words.ReinterpretCast<float>(), words[allocated],
           uint8_t(allocated / Plan::kSort32Elements));
    PipeBarrier<PIPE_V>();
    LocalTensor<float> sorted = MergeMembershipRuns(temporary.ReinterpretCast<float>(), words.ReinterpretCast<float>(),
                                                    allocated);
    EmitMembershipRuns(sorted.ReinterpretCast<uint32_t>(), scratch, allocated);
}
