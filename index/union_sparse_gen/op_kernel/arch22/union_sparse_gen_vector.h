/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Included in UnionSparseGen's private class scope. Production eligibility
// includes full UB budgeting and a conservative request-local cost model.
static constexpr int32_t kVectorLanes = VectorWidth != 0 ? VectorWidth : UnionSparseGenA2Plan::kDefaultVectorLanes;
static constexpr int32_t kVectorStride = kVectorLanes + Plan::kDmaWords;
// Scratch: six lane arrays, interleaved-pair offsets, then prefix-shift tables.
static constexpr int32_t kEncodedVector = 2, kShiftedKeyVector = 3, kShiftedMemberVector = 4;
static constexpr int32_t kCombinedMemberVector = 5, kPackingVector = 6;
static constexpr int32_t kBasePrefixStages = 6, kBaseShiftTables = 8;
static constexpr int32_t kExtendedPrefixStages = kBasePrefixStages + Plan::kExtraPrefixTables;
static constexpr int32_t kExtendedShiftTables = kBaseShiftTables + Plan::kExtraPrefixTables;
static constexpr int32_t kIdentityShiftTable = kBaseShiftTables - 1;
static constexpr int32_t kExtendedShiftBias = 2; // Skip next-key and identity tables.
static constexpr int32_t kSelectionBitmaps = 3, kDifferentBitmap = 2;
static constexpr int32_t kVectorBitmapBytes = (kVectorLanes + Plan::kDmaBlockBytes * Plan::kBitsPerByte - 1) /
                                              (Plan::kDmaBlockBytes * Plan::kBitsPerByte) * Plan::kDmaBlockBytes;
static constexpr int32_t kVectorScratchBytes = UnionSparseGenA2Plan::VectorScratchBytes(kVectorLanes);
static constexpr uint32_t kVectorKey = 0x7effffffU;
// Keep every encoded ID in the positive normal FP32 range. The integer
// merge remains valid beyond this limit; FP32 sort/filter does not.
static constexpr uint32_t kFp32MinNormalBits = 0x00800000U;
static constexpr int64_t kVectorMaxLogicalBlocks = int64_t(kVectorKey) - kFp32MinNormalBits + 1;

__aicore__ inline bool CanVectorResident()
{
    const int64_t runs = Width() * ((topK_ + kVectorLanes - 1) / kVectorLanes);
    return tile_ >= Plan::kDmaWords && Width() * window_ >= Plan::kInputPrefetchSlots * kVectorLanes &&
           runs <= Plan::kMaxResidentRuns && algorithmBytes_ >= kVectorScratchBytes &&
           ((topK_ + Plan::kDmaWords - 1) / Plan::kDmaWords * Plan::kDmaWords) <=
               (algorithmBytes_ - kVectorScratchBytes) / (Width() * Plan::kResidentRecordBytes);
}
__aicore__ inline float KeyFloat(uint32_t word)
{
    union {
        uint32_t word;
        float value;
    } bits;
    bits.word = word;
    return bits.value;
}
__aicore__ inline void InitVectorOffsets(LocalTensor<uint32_t> scratch, int64_t layoutWords)
{
    if (vectorOffsetsReady_ && vectorScratchOffset_ == layoutWords) {
        return;
    }
    vectorScratchOffset_ = layoutWords;
    auto packing = scratch[kPackingVector * kVectorStride];
    auto shifts = packing[Plan::kPairWords * kVectorLanes];
    for (int32_t i = 0; i < kVectorLanes; ++i) {
        packing.SetValue(Plan::kPairWords * i, uint32_t(i * Plan::kWordBytes));
        packing.SetValue(Plan::kPairWords * i + 1, uint32_t((kVectorStride + i) * Plan::kWordBytes));
        // Table0 shifts to the next key; the other tables shift backwards.
        shifts.SetValue(i, uint32_t((i + 1) * Plan::kWordBytes));
        shifts.SetValue(kIdentityShiftTable * kVectorLanes + i, uint32_t(i * Plan::kWordBytes));
        for (int32_t stage = 0; stage < kBasePrefixStages; ++stage) {
            const int32_t distance = 1 << stage;
            shifts.SetValue((stage + 1) * kVectorLanes + i,
                            uint32_t((i >= distance ? i - distance : kVectorLanes) * Plan::kWordBytes));
        }
    }
    vectorOffsetsReady_ = true;
}
__aicore__ inline void QueueVectorInput(int32_t slot, int64_t offset, int32_t n)
{
    // The previous Pack finishes V->S before this slot is reused. The other
    // input slot is disjoint and can arrive while current vector work runs.
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    DataCopyPad(rowData_[slot * kVectorLanes], input_[offset],
                DataCopyExtParams{1, uint32_t(n * Plan::kWordBytes), 0, 0, 0},
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(slot == 0 ? EVENT_ID3 : EVENT_ID4);
}
__aicore__ inline LocalTensor<int32_t> LoadVectorWindow(int64_t inputOffset, int32_t n, int32_t inputSlot,
                                                        int32_t residentOffset)
{
    auto ids = rowData_;
    if (residentOffset >= 0) {
        ids = rowData_[residentOffset];
    } else if (inputSlot >= 0) {
        ids = rowData_[inputSlot * kVectorLanes];
        WaitFlag<HardEvent::MTE2_V>(inputSlot == 0 ? EVENT_ID3 : EVENT_ID4);
    } else {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        CopyInput(ids, input_[inputOffset], n);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    return ids;
}

template <bool Unsorted = false>
__aicore__ inline void FilterVectorWindow(LocalTensor<uint32_t> scratch, LocalTensor<int32_t> ids, int32_t n,
                                          uint32_t& previous, int64_t blockLimit)
{
    auto encoded = scratch[kEncodedVector * kVectorStride];
    auto shiftedKeys = scratch[kShiftedKeyVector * kVectorStride];
    auto packing = scratch[kPackingVector * kVectorStride];
    auto shifts = packing[Plan::kPairWords * kVectorLanes];
    auto selected = shifts[(Unsorted ? kExtendedShiftTables : kBaseShiftTables) * kVectorLanes]
                        .ReinterpretCast<uint8_t>();
    auto valid = selected[kVectorBitmapBytes];
    auto different = selected[kDifferentBitmap * kVectorBitmapBytes];
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(encoded.ReinterpretCast<int32_t>(), int32_t(kVectorKey), kVectorStride);
    PipeBarrier<PIPE_V>();
    // Vector int32 subtraction preserves all key bits, including
    // modulo encodings of invalid inputs; it is not a float cast.
    Sub(encoded.ReinterpretCast<int32_t>(), encoded.ReinterpretCast<int32_t>(), ids, n);
    PipeBarrier<PIPE_V>();
    // Legal IDs map bijectively to this positive finite-normal
    // interval. Every negative/out-of-range int32 maps outside it
    // (or to a NaN, whose ordered comparison is false).
    // dav_c220 L2 comparison emits count*sizeof(T)/256 repeats.
    // Round initialized input to64 lanes; GatherMask still consumes
    // only actual n. Otherwise a one-ID tail reuses a stale bitmap.
    const uint32_t compared = uint32_t((n + Plan::kCompareLanes - 1) / Plan::kCompareLanes * Plan::kCompareLanes);
    CompareScalar(selected, encoded.ReinterpretCast<float>(),
                  KeyFloat(kVectorKey - uint32_t((blockLimit < 0 ? logicalBlocks_ : blockLimit) - 1)), CMPMODE::GE,
                  compared);
    CompareScalar(valid, encoded.ReinterpretCast<float>(), KeyFloat(kVectorKey), CMPMODE::LE, compared);
    PipeBarrier<PIPE_V>();
    And(selected.ReinterpretCast<uint16_t>(), selected.ReinterpretCast<uint16_t>(), valid.ReinterpretCast<uint16_t>(),
        kVectorLanes / Plan::kBitsPerHalfword);
    Gather(shiftedKeys, encoded, shifts[kVectorLanes], 0, kVectorLanes);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    shiftedKeys.SetValue(0, previous);
    previous = encoded.GetValue(n - 1);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Compare(different, encoded.ReinterpretCast<int32_t>(), shiftedKeys.ReinterpretCast<int32_t>(), CMPMODE::EQ,
            compared);
    PipeBarrier<PIPE_V>();
    Not(different.ReinterpretCast<uint16_t>(), different.ReinterpretCast<uint16_t>(),
        kVectorLanes / Plan::kBitsPerHalfword);
    PipeBarrier<PIPE_V>();
    And(selected.ReinterpretCast<uint16_t>(), selected.ReinterpretCast<uint16_t>(),
        different.ReinterpretCast<uint16_t>(), kVectorLanes / Plan::kBitsPerHalfword);
    PipeBarrier<PIPE_V>();
}

template <bool Unsorted = false>
__aicore__ inline int32_t PackVectorWindow(LocalTensor<uint32_t> packed, LocalTensor<uint32_t> scratch,
                                           int64_t inputOffset, int32_t n, int32_t capacity, uint32_t bit,
                                           uint32_t& previous, int32_t inputSlot = -1, int64_t blockLimit = -1,
                                           int32_t residentOffset = -1)
{
    auto keys = scratch;
    auto members = scratch[kVectorStride];
    auto encoded = scratch[kEncodedVector * kVectorStride];
    auto packing = scratch[kPackingVector * kVectorStride];
    auto shifts = packing[Plan::kPairWords * kVectorLanes];
    auto selected = shifts[(Unsorted ? kExtendedShiftTables : kBaseShiftTables) * kVectorLanes]
                        .ReinterpretCast<uint8_t>();
    uint64_t kept = 0;
    Duplicate(selected.ReinterpretCast<uint16_t>(), uint16_t(0),
              kSelectionBitmaps * kVectorBitmapBytes / Plan::kHalfwordBytes);
    Duplicate(keys, uint32_t(0), kVectorStride);
    Duplicate(members, bit, kVectorLanes);
    if (n != 0) {
        auto ids = LoadVectorWindow(inputOffset, n, inputSlot, residentOffset);
        FilterVectorWindow<Unsorted>(scratch, ids, n, previous, blockLimit);
        GatherMask(keys, encoded, selected.ReinterpretCast<uint32_t>(), true, uint32_t(n),
                   GatherMaskParams{1, 1, Plan::kVectorRepeatBlocks, Plan::kVectorRepeatBlocks}, kept);
    }
    PipeBarrier<PIPE_V>();
    Gather(packed, keys, packing, 0, uint32_t(capacity * Plan::kPairWords));
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return int32_t(kept);
}

// Append compacted pairs in aligned vector batches. At most seven old tail
// elements are carried scalarly; all new IDs/memberships move through Gather.
// Stage buffers contain an eight-word guard before their compacted payload.
__aicore__ inline void AppendVectorPairs(LocalTensor<uint32_t> idx, LocalTensor<uint32_t> bits,
                                         LocalTensor<uint32_t> linear, int32_t count)
{
    if (buffered_ + count > tile_) {
        Flush(buffered_);
    }
    const int32_t prefix = buffered_ % Plan::kDmaWords, aligned = buffered_ - prefix;
    // ReduceVectorPart completed V->S before this sole call site. If Flush
    // ran above, buffered_ (and therefore prefix) is zero: no scalar prefix
    // read can depend on the vector work issued by Flush.
    for (int32_t j = 0; j < prefix; ++j) {
        idx.SetValue(Plan::kDmaWords - prefix + j, uint32_t(pairIdx_.GetValue(aligned + j)));
        bits.SetValue(Plan::kDmaWords - prefix + j, pairMask_.GetValue(aligned + j));
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    for (int32_t i = 0; i < prefix + count; i += kVectorLanes) {
        const int32_t n = prefix + count - i < kVectorLanes ? prefix + count - i : kVectorLanes;
        Gather(pairIdx_[aligned + i].ReinterpretCast<uint32_t>(), idx, linear,
               uint32_t((Plan::kDmaWords - prefix + i) * Plan::kWordBytes), uint32_t(n));
        Gather(pairMask_[aligned + i], bits, linear, uint32_t((Plan::kDmaWords - prefix + i) * Plan::kWordBytes),
               uint32_t(n));
    }
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    buffered_ += count;
    if (buffered_ == tile_) {
        Flush(buffered_);
    }
}
template <bool Unsorted = false>
__aicore__ inline bool LoadVectorReduction(LocalTensor<uint32_t> packed, LocalTensor<uint32_t> scratch, int32_t n,
                                           uint32_t pendingKey, uint32_t pendingBits)
{
    auto keys = scratch;
    auto members = scratch[kVectorStride];
    auto shifts = scratch[kPackingVector * kVectorStride + Plan::kPairWords * kVectorLanes];
    auto selected = shifts[(Unsorted ? kExtendedShiftTables : kBaseShiftTables) * kVectorLanes]
                        .ReinterpretCast<uint8_t>();
    auto shiftedKeys = scratch[kShiftedKeyVector * kVectorStride];
    uint64_t extracted = 0;
    Duplicate(selected.ReinterpretCast<uint16_t>(), uint16_t(0),
              kSelectionBitmaps * kVectorBitmapBytes / Plan::kHalfwordBytes);
    Duplicate(keys, uint32_t(0), kVectorStride);
    Duplicate(members, uint32_t(0), kVectorStride);
    PipeBarrier<PIPE_V>();
    GatherMask(keys, packed, uint8_t(Plan::kGatherKeyPattern), true, uint32_t(n * Plan::kPairWords),
               GatherMaskParams{1, 1, Plan::kVectorRepeatBlocks, Plan::kVectorRepeatBlocks}, extracted);
    GatherMask(members, packed, uint8_t(Plan::kGatherMemberPattern), true, uint32_t(n * Plan::kPairWords),
               GatherMaskParams{1, 1, Plan::kVectorRepeatBlocks, Plan::kVectorRepeatBlocks}, extracted);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if (keys.GetValue(0) == 0) {
        return false;
    }
    if (pendingKey != 0) {
        if (keys.GetValue(0) == pendingKey) {
            members.SetValue(0, members.GetValue(0) | pendingBits);
        } else {
            Emit(int32_t(kVectorKey - pendingKey), pendingBits);
        }
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(shiftedKeys, uint32_t(0), kVectorStride);
    PipeBarrier<PIPE_V>();

    return true;
}

template <bool Unsorted = false>
__aicore__ inline bool PrepareVectorRunCheck(LocalTensor<uint32_t> scratch, int32_t n, uint32_t& lastKey)
{
    auto keys = scratch;
    auto members = scratch[kVectorStride];
    auto shifts = scratch[kPackingVector * kVectorStride + Plan::kPairWords * kVectorLanes];
    auto selected = shifts[(Unsorted ? kExtendedShiftTables : kBaseShiftTables) * kVectorLanes]
                        .ReinterpretCast<uint8_t>();
    auto valid = selected[kVectorBitmapBytes];
    const uint32_t compared = uint32_t((n + Plan::kCompareLanes - 1) / Plan::kCompareLanes * Plan::kCompareLanes);
    bool checkRuns = false;
    if constexpr (Unsorted) {
        lastKey = keys.GetValue(n - 1);
        checkRuns = n == kVectorLanes && keys.GetValue(0) != lastKey;
        if (checkRuns) {
            CompareScalar(valid, keys.ReinterpretCast<int32_t>(), int32_t(0), CMPMODE::EQ, compared);
            PipeBarrier<PIPE_V>();
            Not(valid.ReinterpretCast<uint16_t>(), valid.ReinterpretCast<uint16_t>(),
                kVectorLanes / Plan::kBitsPerHalfword);
            PipeBarrier<PIPE_V>();
        }
    }

    return checkRuns;
}

__aicore__ inline bool HasSpanningVectorRun(LocalTensor<uint8_t> selected, LocalTensor<uint8_t> valid, int32_t n)
{
    PipeBarrier<PIPE_V>();
    And(selected.ReinterpretCast<uint16_t>(), selected.ReinterpretCast<uint16_t>(), valid.ReinterpretCast<uint16_t>(),
        kVectorLanes / Plan::kBitsPerHalfword);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    auto flags = selected.ReinterpretCast<uint32_t>();
    uint32_t spanning = 0;
    for (int32_t word = 0; word < (n + Plan::kBitsPerWord - 1) / Plan::kBitsPerWord; ++word) {
        spanning |= flags.GetValue(word);
    }
    // No live equal key spans this distance: prior prefix stages
    // already completed every run, including the incoming carry.
    if (spanning == 0) {
        return false;
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);

    return true;
}

template <bool Unsorted = false>
__aicore__ inline void AccumulateVectorMembership(LocalTensor<uint32_t> scratch, int32_t n, bool checkRuns)
{
    auto keys = scratch;
    auto members = scratch[kVectorStride];
    auto shifts = scratch[kPackingVector * kVectorStride + Plan::kPairWords * kVectorLanes];
    auto selected = shifts[(Unsorted ? kExtendedShiftTables : kBaseShiftTables) * kVectorLanes]
                        .ReinterpretCast<uint8_t>();
    auto valid = selected[kVectorBitmapBytes];
    const uint32_t compared = uint32_t((n + Plan::kCompareLanes - 1) / Plan::kCompareLanes * Plan::kCompareLanes);
    auto shiftedKeys = scratch[kShiftedKeyVector * kVectorStride];
    auto shiftedBits = scratch[kShiftedMemberVector * kVectorStride];
    auto combined = scratch[kCombinedMemberVector * kVectorStride];
    // Legacy sorted input is deduplicated per row and needs log2(W) steps.
    // Unsorted proposals may have arbitrary duplicate runs and scan the whole
    // tile. Carry preserves the OR when a run crosses a tile boundary.
    for (int32_t stage = 0; (1 << stage) < (Unsorted ? n : Width()); ++stage) {
        auto offsets = shifts[(Unsorted && stage >= kBasePrefixStages ? stage + kExtendedShiftBias : stage + 1) *
                              kVectorLanes];
        Gather(shiftedKeys, keys, offsets, 0, uint32_t(n));
        Gather(shiftedBits, members, offsets, 0, uint32_t(n));
        PipeBarrier<PIPE_V>();
        Compare(selected, keys.ReinterpretCast<int32_t>(), shiftedKeys.ReinterpretCast<int32_t>(), CMPMODE::EQ,
                compared);
        if constexpr (Unsorted) {
            if (checkRuns && (1 << stage) >= Width()) {
                if (!HasSpanningVectorRun(selected, valid, n)) {
                    break;
                }
            }
        }
        Or(combined.ReinterpretCast<uint16_t>(), members.ReinterpretCast<uint16_t>(),
           shiftedBits.ReinterpretCast<uint16_t>(), uint32_t(n * Plan::kHalfwordsPerWord));
        PipeBarrier<PIPE_V>();
        Select(members.ReinterpretCast<float>(), selected, combined.ReinterpretCast<float>(),
               members.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, uint32_t(n));
        PipeBarrier<PIPE_V>();
    }
}

template <bool Unsorted = false>
__aicore__ inline void SaveVectorCarry(LocalTensor<uint32_t> scratch, LocalTensor<uint8_t> selected, int32_t n,
                                       uint32_t lastKey, uint32_t& pendingKey, uint32_t& pendingBits)
{
    auto keys = scratch;
    auto members = scratch[kVectorStride];
    if constexpr (Unsorted) {
        pendingKey = lastKey;
    } else {
        pendingKey = keys.GetValue(n - 1);
    }
    pendingBits = members.GetValue(n - 1);
    auto flags = selected.ReinterpretCast<uint32_t>();
    const int32_t word = (n - 1) / Plan::kBitsPerWord;
    flags.SetValue(word, flags.GetValue(word) & ~(1U << uint32_t((n - 1) % Plan::kBitsPerWord)));
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
}

template <bool Unsorted = false>
__aicore__ inline void CompactVectorReduction(LocalTensor<uint32_t> scratch, int32_t n, uint32_t lastKey,
                                              uint32_t& pendingKey, uint32_t& pendingBits)
{
    auto keys = scratch;
    auto members = scratch[kVectorStride];
    auto shifts = scratch[kPackingVector * kVectorStride + Plan::kPairWords * kVectorLanes];
    auto selected = shifts[(Unsorted ? kExtendedShiftTables : kBaseShiftTables) * kVectorLanes]
                        .ReinterpretCast<uint8_t>();
    auto encoded = scratch[kEncodedVector * kVectorStride];
    auto shiftedKeys = scratch[kShiftedKeyVector * kVectorStride];
    auto shiftedBits = scratch[kShiftedMemberVector * kVectorStride];
    auto valid = selected[kVectorBitmapBytes];
    const uint32_t compared = uint32_t((n + Plan::kCompareLanes - 1) / Plan::kCompareLanes * Plan::kCompareLanes);
    Gather(shiftedKeys, keys, shifts, 0, uint32_t(n));
    PipeBarrier<PIPE_V>();
    Compare(selected, keys.ReinterpretCast<int32_t>(), shiftedKeys.ReinterpretCast<int32_t>(), CMPMODE::EQ, compared);
    CompareScalar(valid, keys.ReinterpretCast<int32_t>(), int32_t(0), CMPMODE::EQ, compared);
    PipeBarrier<PIPE_V>();
    Not(selected.ReinterpretCast<uint16_t>(), selected.ReinterpretCast<uint16_t>(),
        kVectorLanes / Plan::kBitsPerHalfword);
    Not(valid.ReinterpretCast<uint16_t>(), valid.ReinterpretCast<uint16_t>(), kVectorLanes / Plan::kBitsPerHalfword);
    PipeBarrier<PIPE_V>();
    And(selected.ReinterpretCast<uint16_t>(), selected.ReinterpretCast<uint16_t>(), valid.ReinterpretCast<uint16_t>(),
        kVectorLanes / Plan::kBitsPerHalfword);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    SaveVectorCarry<Unsorted>(scratch, selected, n, lastKey, pendingKey, pendingBits);
    Duplicate(encoded.ReinterpretCast<int32_t>(), int32_t(kVectorKey), kVectorLanes);
    PipeBarrier<PIPE_V>();
    Sub(encoded.ReinterpretCast<int32_t>(), encoded.ReinterpretCast<int32_t>(), keys.ReinterpretCast<int32_t>(),
        uint32_t(n));
    PipeBarrier<PIPE_V>();
    uint64_t kept = 0, keptBits = 0;
    // Gather may fetch masked lanes in its last vector repeat. Initialize
    // both compacted staging tails as well as the logical valid prefix.
    Duplicate(shiftedKeys, uint32_t(0), kVectorStride);
    Duplicate(shiftedBits, uint32_t(0), kVectorStride);
    PipeBarrier<PIPE_V>();
    GatherMask(shiftedKeys[Plan::kDmaWords].ReinterpretCast<int32_t>(), encoded.ReinterpretCast<int32_t>(),
               selected.ReinterpretCast<uint32_t>(), true, uint32_t(n),
               GatherMaskParams{1, 1, Plan::kVectorRepeatBlocks, Plan::kVectorRepeatBlocks}, kept);
    GatherMask(shiftedBits[Plan::kDmaWords], members, selected.ReinterpretCast<uint32_t>(), true, uint32_t(n),
               GatherMaskParams{1, 1, Plan::kVectorRepeatBlocks, Plan::kVectorRepeatBlocks}, keptBits);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if (kept != 0) {
        AppendVectorPairs(shiftedKeys, shiftedBits, shifts[kIdentityShiftTable * kVectorLanes], int32_t(kept));
    }
}

template <bool Unsorted = false>
__aicore__ inline bool ReduceVectorPart(LocalTensor<uint32_t> packed, int32_t n, LocalTensor<uint32_t> scratch,
                                        uint32_t& pendingKey, uint32_t& pendingBits)
{
    if (!LoadVectorReduction<Unsorted>(packed, scratch, n, pendingKey, pendingBits)) {
        return false;
    }
    uint32_t lastKey = 0;
    const bool checkRuns = PrepareVectorRunCheck<Unsorted>(scratch, n, lastKey);
    AccumulateVectorMembership<Unsorted>(scratch, n, checkRuns);
    CompactVectorReduction<Unsorted>(scratch, n, lastKey, pendingKey, pendingBits);
    return true;
}

__aicore__ inline void PrefetchNextVectorInput(int32_t row, int32_t at, int32_t limit, int32_t inputSlot)
{
    int32_t nextRow = row, nextAt = at + kVectorLanes;
    if (nextAt >= limit) {
        ++nextRow;
        nextAt = 0;
        while (nextRow < Width() && limit_[nextRow] == 0) {
            ++nextRow;
        }
    }
    if (nextRow < Width()) {
        const int32_t nextN = limit_[nextRow] - nextAt < kVectorLanes ? limit_[nextRow] - nextAt : kVectorLanes;
        QueueVectorInput(inputSlot ^ 1, (head_ * totalQ_ + qStart_ + nextRow) * topK_ + nextAt, nextN);
    }
}

__aicore__ inline void PackResidentRows(LocalTensor<uint32_t> source, LocalTensor<uint32_t> scratch, int32_t* lengths,
                                        int32_t& runs, int32_t& proposals, bool prefetch)
{
    int32_t inputSlot = 0;
    if (prefetch) {
        int32_t first = 0;
        while (first < Width() && limit_[first] == 0) {
            ++first;
        }
        if (first < Width()) {
            QueueVectorInput(0, (head_ * totalQ_ + qStart_ + first) * topK_,
                             limit_[first] < kVectorLanes ? limit_[first] : kVectorLanes);
        }
    }
    for (int32_t row = 0; row < Width(); ++row) {
        uint32_t previous = 0;
        const int32_t limit = limit_[row];
        const int64_t base = (head_ * totalQ_ + qStart_ + row) * topK_;
        for (int32_t at = 0; at < limit; at += kVectorLanes) {
            const int32_t n = limit - at < kVectorLanes ? limit - at : kVectorLanes;
            if (prefetch) {
                PrefetchNextVectorInput(row, at, limit, inputSlot);
            }
            const int32_t capacity = (n + Plan::kDmaWords - 1) / Plan::kDmaWords * Plan::kDmaWords;
            auto packed = source[proposals * Plan::kPairWords];
            if (PackVectorWindow(packed, scratch, base + at, n, capacity, 1U << uint32_t(row), previous,
                                 prefetch ? inputSlot : -1)) {
                lengths[runs++] = capacity;
                proposals += capacity;
            }
            inputSlot ^= 1;
        }
    }
}

__aicore__ inline LocalTensor<float> MergeResidentRuns(LocalTensor<float> src, LocalTensor<float> dst, int32_t* lengths,
                                                       int32_t runs)
{
    while (runs > 1) {
        const int32_t next = (runs + Plan::kMergeFanIn - 1) / Plan::kMergeFanIn;
        int32_t sourceOffset = 0, destinationOffset = 0;
        for (int32_t group = 0; group < next; ++group) {
            const int32_t count = runs - group * Plan::kMergeFanIn < Plan::kMergeFanIn ?
                                      runs - group * Plan::kMergeFanIn :
                                      Plan::kMergeFanIn;
            uint16_t lens[Plan::kMergeFanIn] = {};
            int32_t offsets[Plan::kMergeFanIn] = {}, length = 0;
            for (int32_t i = 0; i < count; ++i) {
                offsets[i] = sourceOffset;
                lens[i] = uint16_t(lengths[group * Plan::kMergeFanIn + i]);
                sourceOffset += int32_t(lens[i]) * Plan::kPairWords;
                length += lens[i];
            }
            if (count == 1) {
                DataCopy(dst[destinationOffset], src[offsets[0]], uint32_t(length * Plan::kPairWords));
            } else {
                auto a = src[offsets[0]], b = src[offsets[1]];
                auto c = src[offsets[count > Plan::kMergeThirdInput ? Plan::kMergeThirdInput : Plan::kMergeFirstInput]],
                     d = src[offsets[count > Plan::kMergeFourthInput ? Plan::kMergeFourthInput :
                                                                       Plan::kMergeFirstInput]];
                MrgSort(dst[destinationOffset], MrgSortSrcList<float>(a, b, c, d),
                        MrgSort4Info(lens, false, uint16_t((1U << count) - 1), 1));
            }
            lengths[group] = length;
            destinationOffset += length * Plan::kPairWords;
        }
        PipeBarrier<PIPE_V>();
        auto temporary = src;
        src = dst;
        dst = temporary;
        runs = next;
    }

    return src;
}

__aicore__ inline void EmitResidentRuns(LocalTensor<uint32_t> sorted, LocalTensor<uint32_t> scratch, int32_t proposals)
{
    uint32_t pendingKey = 0, pendingBits = 0;
    const int32_t outputChunk = tile_ < kVectorLanes ? tile_ : kVectorLanes;
    for (int32_t at = 0; at < proposals; at += outputChunk) {
        if (!ReduceVectorPart(sorted[at * Plan::kPairWords],
                              proposals - at < outputChunk ? proposals - at : outputChunk, scratch, pendingKey,
                              pendingBits)) {
            break;
        }
    }
    if (pendingKey != 0) {
        Emit(int32_t(kVectorKey - pendingKey), pendingBits);
    }
}
__aicore__ inline void VectorResident()
{
    const int32_t allocated = int32_t(Width() * ((topK_ + Plan::kDmaWords - 1) / Plan::kDmaWords * Plan::kDmaWords));
    auto source = algorithmBuf_.Get<uint32_t>();
    auto destination = source[allocated * Plan::kPairWords];
    auto scratch = source[allocated * Plan::kResidentRecordWords];
    InitVectorOffsets(scratch, int64_t(allocated) * Plan::kResidentRecordWords);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    int32_t lengths[Plan::kMaxResidentRuns], runs = 0, proposals = 0;
    const bool prefetch = Width() * window_ >= Plan::kInputPrefetchSlots * kVectorLanes;
    PackResidentRows(source, scratch, lengths, runs, proposals, prefetch);
    if (runs == 0) {
        return;
    }
    LocalTensor<float> sorted = MergeResidentRuns(source.ReinterpretCast<float>(), destination.ReinterpretCast<float>(),
                                                  lengths, runs);
    EmitResidentRuns(sorted.ReinterpretCast<uint32_t>(), scratch, proposals);
}

__aicore__ inline int32_t VectorLeafSize()
{
    int32_t leaf = window_ < kVectorLanes ? window_ : kVectorLanes;
    const int32_t levels = Width() <= Plan::kNarrowCascadeMaxWidth ? Plan::kNarrowCascadeLevels :
                                                                     Plan::kWideCascadeLevels;
    while (leaf >= Plan::kSort32Elements &&
           Width() * leaf * (Plan::kPairWords * Plan::kWordBytes) * levels + kVectorScratchBytes > algorithmBytes_) {
        leaf /= Plan::kBudgetShrinkFactor;
    }
    return leaf >= Plan::kSort32Elements ? leaf : 0;
}
// Recreate the view from the owning buffer rather than copying uninitialized
// LocalTensor shape metadata through a long-lived private tensor member.
__aicore__ inline LocalTensor<uint32_t> VectorScratch() { return algorithmBuf_.Get<uint32_t>()[vectorScratchOffset_]; }
__aicore__ inline void FillVectorLeaf(int32_t row)
{
    int32_t kept = 0;
    uint32_t previous = uint32_t(value_[row]);
    while (kept == 0 && pos_[row] < limit_[row]) {
        const int32_t n = limit_[row] - pos_[row] < streamLeaf_ ? limit_[row] - pos_[row] : streamLeaf_;
        const int64_t offset = (head_ * totalQ_ + qStart_ + row) * topK_ + pos_[row];
        kept = PackVectorWindow(stream_[StreamStart(row)], VectorScratch(), offset, n, streamLeaf_, 1U << uint32_t(row),
                                previous);
        pos_[row] += n;
    }
    value_[row] = int32_t(previous);
    streamLength_[row] = kept;
    streamOffset_[row] = 0;
}
__aicore__ inline void AlignVectorStreamSuffix(int32_t base, int32_t offset, int32_t n)
{
    auto temporary = VectorScratch()[kCombinedMemberVector * kVectorStride];
    auto linear = VectorScratch()[kPackingVector * kVectorStride + Plan::kPairWords * kVectorLanes +
                                  kIdentityShiftTable * kVectorLanes];
    // Read an unaligned suffix through Gather into a disjoint aligned
    // tile, then copy back. No overlapping UB-to-UB DMA or per-ID scalar.
    if (offset != 0) {
        for (int32_t i = 0; i < n * Plan::kPairWords; i += kVectorLanes) {
            const int32_t words = n * Plan::kPairWords - i < kVectorLanes ? n * Plan::kPairWords - i : kVectorLanes;
            Duplicate(temporary, uint32_t(0), kVectorLanes);
            PipeBarrier<PIPE_V>();
            Gather(temporary, stream_[base], linear, uint32_t((offset * Plan::kPairWords + i) * Plan::kWordBytes),
                   uint32_t(words));
            PipeBarrier<PIPE_V>();
            DataCopy(stream_[base + i], temporary,
                     uint32_t((words + Plan::kDmaWords - 1) / Plan::kDmaWords * Plan::kDmaWords));
            PipeBarrier<PIPE_V>();
        }
    }
}
__aicore__ inline void MergeVectorStreams(int32_t node, int32_t first, int32_t children)
{
    int32_t sources[Plan::kMergeFanIn] = {};
    uint16_t lens[Plan::kMergeFanIn] = {};
    int32_t active = 0;
    auto floats = stream_.ReinterpretCast<float>();
    for (int32_t child = first; child < first + children; ++child) {
        const int32_t n = streamLength_[child];
        if (n == 0) {
            continue;
        }
        const int32_t base = StreamStart(child), offset = streamOffset_[child];
        AlignVectorStreamSuffix(base, offset, n);
        streamOffset_[child] = 0;
        sources[active] = child;
        lens[active++] = uint16_t(n);
    }
    streamLength_[node] = 0;
    streamOffset_[node] = 0;
    if (active == 0) {
        return;
    }
    const int32_t destination = StreamStart(node);
    if (active == 1) {
        DataCopy(stream_[destination], stream_[StreamStart(sources[0])],
                 uint32_t((lens[0] * Plan::kPairWords + Plan::kDmaWords - 1) / Plan::kDmaWords * Plan::kDmaWords));
        PipeBarrier<PIPE_V>();
        streamLength_[node] = lens[0];
        streamLength_[sources[0]] = 0;
        return;
    }
    auto a = floats[StreamStart(sources[0])], b = floats[StreamStart(sources[1])];
    auto c = floats[StreamStart(
        sources[active > Plan::kMergeThirdInput ? Plan::kMergeThirdInput : Plan::kMergeFirstInput])];
    auto d = floats[StreamStart(
        sources[active > Plan::kMergeFourthInput ? Plan::kMergeFourthInput : Plan::kMergeFirstInput])];
    MrgSort(floats[destination], MrgSortSrcList<float>(a, b, c, d),
            MrgSort4Info(lens, true, uint16_t((1U << active) - 1), 1));
    PipeBarrier<PIPE_V>();
    uint16_t consumed[Plan::kMergeFanIn] = {};
    GetMrgSortResult(consumed[Plan::kMergeFirstInput], consumed[Plan::kMergeSecondInput],
                     consumed[Plan::kMergeThirdInput], consumed[Plan::kMergeFourthInput]);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    int32_t n = 0;
    for (int32_t i = 0; i < active; ++i) {
        streamLength_[sources[i]] -= consumed[i];
        streamOffset_[sources[i]] = consumed[i];
        n += consumed[i];
    }
    streamLength_[node] = n;
}

__aicore__ inline void VectorStream()
{
    streamLeaf_ = VectorLeafSize();
    const int32_t words = int32_t(Width()) * streamLeaf_ * Plan::kPairWords *
                          (Width() <= Plan::kNarrowCascadeMaxWidth ? Plan::kNarrowCascadeLevels :
                                                                     Plan::kWideCascadeLevels);
    stream_ = algorithmBuf_.Get<uint32_t>();
    InitVectorOffsets(algorithmBuf_.Get<uint32_t>()[words], words);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(stream_, uint32_t(0), uint32_t(words));
    PipeBarrier<PIPE_V>();
    for (int32_t i = 0; i < Plan::kStreamNodeCount; ++i) {
        streamLength_[i] = 0;
        streamOffset_[i] = 0;
    }
    for (int32_t r = 0; r < Width(); ++r) {
        pos_[r] = 0;
        value_[r] = 0;
    }
    const int32_t root = Width() <= Plan::kNarrowCascadeMaxWidth ? Plan::kStreamLevel2Begin : Plan::kStreamRoot;
    uint32_t pendingKey = 0, pendingBits = 0;
    while (true) {
        if (Width() <= Plan::kNarrowCascadeMaxWidth) {
            FillLevel2(0);
        } else {
            for (int32_t i = 0; i < Plan::kStreamLevel2Count; ++i) {
                if (streamLength_[Plan::kStreamLevel2Begin + i] == 0) {
                    FillLevel2(i);
                }
            }
            MergeVectorStreams(Plan::kStreamRoot, Plan::kStreamLevel2Begin, Plan::kStreamLevel2Count);
        }
        const int32_t n = streamLength_[root];
        if (n == 0) {
            break;
        }
        const int32_t base = StreamStart(root);
        const int32_t tile = tile_ < kVectorLanes ? tile_ : kVectorLanes;
        for (int32_t i = 0; i < n; i += tile) {
            const int32_t count = n - i < tile ? n - i : tile;
            ReduceVectorPart(stream_[base + i * Plan::kPairWords], count, VectorScratch(), pendingKey, pendingBits);
        }
        streamLength_[root] = 0;
    }
    if (pendingKey != 0) {
        Emit(int32_t(kVectorKey - pendingKey), pendingBits);
    }
}

__aicore__ inline bool PreferVectorStream(int64_t entries, int32_t active)
{
    // Whole-operator A2 windows reject the four-level cascade and small work:
    // their setup/compaction dominates table/tree processing. Uncalibrated
    // resource regions retain the existing exact fallback, with no shape ban.
    // W8 streaming already consumes min(tile_, kVectorLanes) output chunks.
    // Its UB/work gates also apply when a scalar table fits.
    if (Width() > Plan::kNarrowCascadeMaxWidth || (Width() != Plan::kPreferredStreamWidth && tile_ < kVectorLanes) ||
        entries < Width() * kVectorLanes) {
        return false;
    }
    const int64_t length = (logicalBlocks_ + Plan::kTableAlignmentBlocks - 1) / Plan::kTableAlignmentBlocks *
                               Plan::kTableAlignmentBlocks +
                           Plan::kTableAlignmentBlocks;
    const int64_t tableBytes = length * Plan::kWordBytes + (length / Plan::kBitsPerByte + Plan::kDmaBlockBytes - 1) /
                                                               Plan::kDmaBlockBytes * Plan::kDmaBlockBytes;
    return Width() == Plan::kPreferredStreamWidth || tableBytes > algorithmBytes_ || !PreferTable(entries, active);
}

__aicore__ inline bool PreferVectorWork(int64_t entries, int32_t active, int64_t inputRuns, int64_t proposals,
                                        bool streaming)
{
    // Integer work units calibrated from complete A2 calls. Counts/run totals
    // come from the existing O(W) metadata pass, including their cost. The
    // common full-output term prevents padding-heavy calls from claiming a
    // large benefit solely from the algorithm core. Require a10% margin.
    const int64_t length = (logicalBlocks_ + Plan::kTableAlignmentBlocks - 1) / Plan::kTableAlignmentBlocks *
                               Plan::kTableAlignmentBlocks +
                           Plan::kTableAlignmentBlocks;
    const int64_t tableBytes = length * Plan::kWordBytes + (length / Plan::kBitsPerByte + Plan::kDmaBlockBytes - 1) /
                                                               Plan::kDmaBlockBytes * Plan::kDmaBlockBytes;
    // Calibrate the shared batched writer in coarser units than scalar ID work.
    // Larger public group widths use the shared direct merge layout.
    const int64_t common = Plan::kWriterBaseWork + capacity_ * (by_ + 1) / Plan::kWriterWorkScale;
    // Use the cheaper scan lower bound when full-output work dominates.
    // Repeated rows can terminate the old merge cheaply in that region;
    // applying that bound globally incorrectly rejects useful streaming.
    const int64_t scalar = common +
                           (tableBytes <= algorithmBytes_ && PreferTable(entries, active) ?
                                TableWork(entries, active) :
                                (common > Plan::kOutputDominanceRatio * entries ? entries :
                                                                                  Plan::kScalarEntryWork * entries));
    const int64_t vectorTiles = (proposals + kVectorLanes - 1) / kVectorLanes;
    // Resident tails pay per-run setup plus occupied lanes, not a full tile.
    const int64_t runWork = streaming ? Plan::kStreamingRunWork * inputRuns :
                                        Plan::kResidentRunWork * (inputRuns + vectorTiles);
    int64_t vector = Plan::kVectorRowWork * active + runWork + Plan::kVectorTileWork * vectorTiles;
    if (streaming) {
        vector *= Plan::kStreamingWorkPenalty;
    }
    return Plan::kWorkComparisonScale * (common + vector) < Plan::kPreferredWorkScale * scalar;
}

// Streaming has a larger live state: isolate its scalar planning so those
// temporaries do not extend the vector merge register-allocation region.
__attribute__((noinline)) __aicore__ bool PreferStreamingVectorWork(int64_t entries, int32_t active, int64_t inputRuns,
                                                                    int64_t proposals)
{
    return PreferVectorWork(entries, active, inputRuns, proposals, true);
}
