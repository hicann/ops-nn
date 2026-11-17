/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCATTER_ELEMENTS_V2_AMAX_H
#define SCATTER_ELEMENTS_V2_AMAX_H
#include "kernel_operator.h"

namespace ScatterElementsV2Amax {
using namespace AscendC;
__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

// Output ownership with row-broadcast indices. A task buffers a group of rows
// and scans compact indices once for the entire group. Padded UB row strides
// allow vector updates for odd channel counts; GM remains densely packed.
template <typename T>
class ScatterReduceRowOwnerKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR src, GM_ADDR out, uint32_t edges, uint32_t nodes,
                                uint32_t channels, uint32_t rowsPerTask, uint32_t indexTile, bool includeSelf)
    {
        indexTile_ = indexTile;
        edges_ = edges;
        nodes_ = nodes;
        columns_ = channels;
        rowsPerTask_ = rowsPerTask;
        stride_ = (channels + 63U) / 64U * 64U;
        includeSelf_ = includeSelf;
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(index));
        pipe_.InitBuffer(acc_, rowsPerTask_ * stride_ * sizeof(T));
        pipe_.InitBuffer(seen_, rowsPerTask_ * sizeof(uint32_t));
        pipe_.InitBuffer(value_, stride_ * sizeof(T));
        if constexpr (!IsSameType<T, float>::value) {
            pipe_.InitBuffer(af_, stride_ * sizeof(float));
            pipe_.InitBuffer(vf_, stride_ * sizeof(float));
        }
        pipe_.InitBuffer(m0_, stride_ / 8);
        pipe_.InitBuffer(m1_, stride_ / 8);
        pipe_.InitBuffer(m2_, stride_ / 8);
        pipe_.InitBuffer(idx_, indexTile_ * sizeof(int64_t));
        pipe_.InitBuffer(fi_, indexTile_ * sizeof(float));
        pipe_.InitBuffer(pos_, indexTile_ * sizeof(int32_t));
        pipe_.InitBuffer(selected_, indexTile_ * sizeof(int32_t));
        pipe_.InitBuffer(im0_, indexTile_ / 8);
        pipe_.InitBuffer(im1_, indexTile_ / 8);
        mv_ = pipe_.FetchEventID(HardEvent::MTE2_V);
        vm_ = pipe_.FetchEventID(HardEvent::V_MTE2);
        vs_ = pipe_.FetchEventID(HardEvent::V_S);
        sv_ = pipe_.FetchEventID(HardEvent::S_V);
        vo_ = pipe_.FetchEventID(HardEvent::V_MTE3);
        ov_ = pipe_.FetchEventID(HardEvent::MTE3_V);
    }

    __aicore__ inline void Process()
    {
        auto indices = idx_.Get<int64_t>();
        auto converted = fi_.Get<float>();
        auto selected = selected_.Get<int32_t>();
        auto seen = seen_.Get<uint32_t>();
        CreateVecIndex(pos_.Get<int32_t>(), int32_t(0), indexTile_);
        const uint32_t tasks = (nodes_ + rowsPerTask_ - 1) / rowsPerTask_;
        for (uint32_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
            const uint32_t first = task * rowsPerTask_;
            const uint32_t rows = MinU32(rowsPerTask_, nodes_ - first);
            const uint32_t gapBlocks = (stride_ * sizeof(T) - (columns_ * sizeof(T) + 31U) / 32U * 32U) / 32U;
            SyncVM();
            DataCopyPad(acc_.Get<T>(), self_[uint64_t(first) * columns_],
                        {static_cast<uint16_t>(rows), uint32_t(columns_ * sizeof(T)), 0, gapBlocks, 0},
                        {false, 0, 0, T(0)});
            SyncMV();
            Duplicate(seen, includeSelf_ ? 1U : 0U, (rows + 7U) / 8U * 8U);
            PipeBarrier<PIPE_V>();
            for (uint64_t base = 0; base < edges_; base += indexTile_) {
                const uint32_t count = MinU32(indexTile_, edges_ - base);
                const uint32_t aligned = (count + 63U) / 64U * 64U;
                SyncVM();
                DataCopyPad(indices, index_[base], {1, count * 8U, 0, 0, 0}, {false, 0, 0, int64_t(0)});
                SyncMV();
                Cast(converted, indices, RoundMode::CAST_RINT, count);
                PipeBarrier<PIPE_V>();
                CompareScalar(im0_.Get<uint8_t>(), converted, float(int32_t(first)), CMPMODE::GE, aligned);
                CompareScalar(im1_.Get<uint8_t>(), converted, float(int32_t(first + rows)), CMPMODE::LT, aligned);
                PipeBarrier<PIPE_V>();
                And(im0_.Get<uint16_t>(), im0_.Get<uint16_t>(), im1_.Get<uint16_t>(), indexTile_ / 16);
                PipeBarrier<PIPE_V>();
                const uint64_t kept = Gather(count);
                for (uint32_t k = 0; k < kept; ++k) {
                    const uint32_t position = selected.GetValue(k);
                    const uint32_t row = static_cast<uint32_t>(indices.GetValue(position)) - first;
                    const uint32_t offset = row * stride_;
                    SyncVM();
                    DataCopyPad(value_.Get<T>(), src_[uint64_t(base + position) * columns_],
                                {1, uint32_t(columns_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, T(0)});
                    SyncMV();
                    if (seen.GetValue(row) == 0) {
                        DataCopy(acc_.Get<T>()[offset], value_.Get<T>(), stride_);
                        seen.SetValue(row, 1);
                    } else {
                        Update(offset, stride_);
                    }
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::S_V>(sv_);
                WaitFlag<HardEvent::S_V>(sv_);
            }
            SetFlag<HardEvent::V_MTE3>(vo_);
            WaitFlag<HardEvent::V_MTE3>(vo_);
            DataCopyPad(out_[uint64_t(first) * columns_], acc_.Get<T>(),
                        {static_cast<uint16_t>(rows), uint32_t(columns_ * sizeof(T)), gapBlocks, 0, 0});
            SetFlag<HardEvent::MTE3_V>(ov_);
            WaitFlag<HardEvent::MTE3_V>(ov_);
        }
        SetMaskNorm();
        ResetMask();
    }

private:
    __aicore__ inline void SyncVM()
    {
        SetFlag<HardEvent::V_MTE2>(vm_);
        WaitFlag<HardEvent::V_MTE2>(vm_);
    }
    __aicore__ inline void SyncMV()
    {
        SetFlag<HardEvent::MTE2_V>(mv_);
        WaitFlag<HardEvent::MTE2_V>(mv_);
    }
    __aicore__ inline uint64_t Gather(uint32_t count)
    {
        uint64_t kept = 0;
        GatherMask(selected_.Get<int32_t>(), pos_.Get<int32_t>(), im0_.Get<uint32_t>(), true, count, {1, 1, 8, 0},
                   kept);
        SetFlag<HardEvent::V_S>(vs_);
        WaitFlag<HardEvent::V_S>(vs_);
        return kept;
    }
    __aicore__ inline void Update(uint32_t offset, uint32_t count)
    {
        LocalTensor<float> a;
        LocalTensor<float> v;
        if constexpr (IsSameType<T, float>::value) {
            a = acc_.Get<float>()[offset];
            v = value_.Get<float>();
        } else {
            a = af_.Get<float>();
            v = vf_.Get<float>();
            Cast(a, acc_.Get<T>()[offset], RoundMode::CAST_NONE, count);
            Cast(v, value_.Get<T>(), RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
        // Replace iff accumulator is not NaN and (source is NaN or greater).
        // Equal values retain the first source, including the sign of zero.
        Compare(m0_.Get<uint8_t>(), v, a, CMPMODE::GT, count);
        Compare(m1_.Get<uint8_t>(), v, v, CMPMODE::EQ, count);
        Compare(m2_.Get<uint8_t>(), a, a, CMPMODE::EQ, count);
        PipeBarrier<PIPE_V>();
        // Complement EQ to include unordered (NaN) source values.
        Not(m1_.Get<uint16_t>(), m1_.Get<uint16_t>(), count / 16);
        PipeBarrier<PIPE_V>();
        Or(m0_.Get<uint16_t>(), m0_.Get<uint16_t>(), m1_.Get<uint16_t>(), count / 16);
        PipeBarrier<PIPE_V>();
        And(m0_.Get<uint16_t>(), m0_.Get<uint16_t>(), m2_.Get<uint16_t>(), count / 16);
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, float>::value) {
            Select(acc_.Get<float>()[offset], m0_.Get<uint8_t>(), value_.Get<float>(), acc_.Get<float>()[offset],
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        } else {
            // Select bits through half for BF16 too; no conversion of stored values.
            Select(acc_.Get<half>()[offset], m0_.Get<uint8_t>(), value_.Get<half>(), acc_.Get<half>()[offset],
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        }
        PipeBarrier<PIPE_V>();
    }
    TPipe pipe_;
    TBuf<TPosition::VECCALC> acc_, seen_, value_, af_, vf_, m0_, m1_, m2_;
    TBuf<TPosition::VECCALC> idx_, fi_, pos_, selected_, im0_, im1_;
    GlobalTensor<T> self_, src_, out_;
    GlobalTensor<int64_t> index_;
    TEventID mv_, vm_, vs_, sv_, vo_, ov_;
    uint32_t edges_, nodes_, columns_, rowsPerTask_, stride_, indexTile_;
    bool includeSelf_;
};

// Generic output-owner path for small, unaligned, or large-output tensors.
template <typename T>
class ScatterReduceOwnerKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR src, GM_ADDR out, uint32_t inputRows,
                                uint32_t outputRows, uint32_t columns, uint32_t chunkLength, uint32_t tileLength,
                                bool includeSelf, bool compactRows = false)
    {
        inputRows_ = inputRows;
        compactRows_ = compactRows;
        outputRows_ = outputRows;
        columns_ = columns;
        tileLength_ = tileLength;
        filtered_ = !compactRows && outputRows <= (1U << 24);
        includeSelf_ = includeSelf;
        const uint32_t outputElements = outputRows_ * columns_;
        start_ = GetBlockIdx() * chunkLength;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), outputElements);
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(index),
                                 compactRows_ ? inputRows_ : inputRows_ * columns_);
        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src), inputRows_ * columns_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), outputElements);
        chunkLength_ = chunkLength;
        pipe_.InitBuffer(outBuffer_, chunkLength * sizeof(T));
        pipe_.InitBuffer(seenBuffer_, chunkLength * sizeof(uint32_t));
        pipe_.InitBuffer(indexBuffer_, tileLength_ * sizeof(int64_t));
        pipe_.InitBuffer(srcBuffer_, tileLength_ * sizeof(T));
        {
            pipe_.InitBuffer(targetBuffer_, tileLength_ * 4);
            pipe_.InitBuffer(positionBuffer_, tileLength_ * 4);
            pipe_.InitBuffer(selectedBuffer_, tileLength_ * 4);
            pipe_.InitBuffer(maskBuffer_, tileLength_ / 4);
            mte2ToVector_ = pipe_.FetchEventID(HardEvent::MTE2_V);
            vectorToScalar_ = pipe_.FetchEventID(HardEvent::V_S);
        }
        mte2ToScalar_ = pipe_.FetchEventID(HardEvent::MTE2_S);
        scalarToMte2_ = pipe_.FetchEventID(HardEvent::S_MTE2);
        scalarToMte3_ = pipe_.FetchEventID(HardEvent::S_MTE3);
        mte3ToScalar_ = pipe_.FetchEventID(HardEvent::MTE3_S);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<T> output = outBuffer_.Get<T>();
        LocalTensor<uint16_t> outputBits16 = outBuffer_.Get<uint16_t>();
        LocalTensor<uint32_t> outputBits32 = outBuffer_.Get<uint32_t>();
        LocalTensor<uint32_t> seen = seenBuffer_.Get<uint32_t>();
        LocalTensor<int64_t> indices = indexBuffer_.Get<int64_t>();
        LocalTensor<T> values = srcBuffer_.Get<T>();
        LocalTensor<uint16_t> valueBits16 = srcBuffer_.Get<uint16_t>();
        LocalTensor<uint32_t> valueBits32 = srcBuffer_.Get<uint32_t>();
        const uint32_t totalOutput = outputRows_ * columns_;
        const uint32_t totalInput = inputRows_ * columns_;
        if (filtered_) {
            auto positions = positionBuffer_.Get<int32_t>();
            CreateVecIndex(positions, int32_t(0), tileLength_);
            Duplicate(targetBuffer_.Get<float>(), float(0), tileLength_);
            Duplicate(selectedBuffer_.Get<int32_t>(), int32_t(0), tileLength_);
            Duplicate(maskBuffer_.Get<uint16_t>(), uint16_t(0), tileLength_ / 8);
            SetFlag<HardEvent::V_S>(vectorToScalar_);
            WaitFlag<HardEvent::V_S>(vectorToScalar_);
        }
        for (; start_ < totalOutput; start_ += GetBlockNum() * chunkLength_) {
            const uint32_t count = MinU32(chunkLength_, totalOutput - start_);
            DataCopyPad(output, selfGm_[start_], {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, T(0)});
            SetFlag<HardEvent::MTE2_S>(mte2ToScalar_);
            WaitFlag<HardEvent::MTE2_S>(mte2ToScalar_);
            for (uint32_t j = 0; j < count; ++j)
                seen.SetValue(j, includeSelf_ ? 1U : 0U);
            for (uint64_t offset = 0; offset < totalInput; offset += tileLength_) {
                const uint32_t valid = MinU32(tileLength_, totalInput - offset);
                const uint32_t indexStart = compactRows_ ? offset / columns_ : offset;
                const uint32_t indexCount = compactRows_ ? (offset + valid - 1) / columns_ - indexStart + 1 : valid;
                DataCopyPad(indices, indexGm_[indexStart], {1, indexCount * 8U, 0, 0, 0}, {false, 0, 0, int64_t(0)});
                DataCopyPad(values, srcGm_[offset], {1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0},
                            {false, 0, 0, T(0)});
                SetFlag<HardEvent::MTE2_S>(mte2ToScalar_);
                WaitFlag<HardEvent::MTE2_S>(mte2ToScalar_);
                uint32_t column = offset % columns_;
                const uint64_t selected = filtered_ ? SelectOwned(valid, count) : valid;
                for (uint32_t k = 0; k < selected; ++k) {
                    const uint32_t j = filtered_ ? selectedBuffer_.Get<int32_t>().GetValue(k) : k;
                    if (filtered_)
                        column = (offset + j) % columns_;
                    const uint32_t indexPos = compactRows_ ? (offset + j) / columns_ - indexStart : j;
                    const int64_t row = indices.GetValue(indexPos);
                    const uint64_t target = static_cast<uint64_t>(row) * columns_ + column;
                    if (row >= 0 && row < outputRows_ && target >= start_ && target < start_ + count) {
                        const uint32_t pos = static_cast<uint32_t>(target) - start_;
                        const T rawValue = values.GetValue(j);
                        bool valueIsNan = false;
                        bool previousIsNan = false;
                        bool valueIsGreater = false;
                        if constexpr (IsSameType<T, float>::value) {
                            const uint32_t valueBits = valueBits32.GetValue(j);
                            const uint32_t previousBits = outputBits32.GetValue(pos);
                            valueIsNan = (valueBits & 0x7f800000U) == 0x7f800000U && (valueBits & 0x007fffffU) != 0;
                            previousIsNan = (previousBits & 0x7f800000U) == 0x7f800000U &&
                                            (previousBits & 0x007fffffU) != 0;
                            valueIsGreater = rawValue > output.GetValue(pos);
                        } else {
                            constexpr uint16_t exponentMask = IsSameType<T, half>::value ? 0x7c00U : 0x7f80U;
                            constexpr uint16_t mantissaMask = IsSameType<T, half>::value ? 0x03ffU : 0x007fU;
                            const uint16_t valueBits = valueBits16.GetValue(j);
                            const uint16_t previousBits = outputBits16.GetValue(pos);
                            valueIsNan = (valueBits & exponentMask) == exponentMask && (valueBits & mantissaMask) != 0;
                            previousIsNan = (previousBits & exponentMask) == exponentMask &&
                                            (previousBits & mantissaMask) != 0;
                            const uint16_t valueKey = (valueBits & 0x8000U) ?
                                                          static_cast<uint16_t>(~valueBits) :
                                                          static_cast<uint16_t>(valueBits ^ 0x8000U);
                            const uint16_t previousKey = (previousBits & 0x8000U) ?
                                                             static_cast<uint16_t>(~previousBits) :
                                                             static_cast<uint16_t>(previousBits ^ 0x8000U);
                            const bool bothZero = (valueBits & 0x7fffU) == 0 && (previousBits & 0x7fffU) == 0;
                            valueIsGreater = !bothZero && valueKey > previousKey;
                        }
                        if (!seen.GetValue(pos) || (!previousIsNan && (valueIsNan || valueIsGreater))) {
                            output.SetValue(pos, rawValue);
                        }
                        seen.SetValue(pos, 1);
                    }
                    if (++column == columns_)
                        column = 0;
                }
                SetFlag<HardEvent::S_MTE2>(scalarToMte2_);
                WaitFlag<HardEvent::S_MTE2>(scalarToMte2_);
            }
            SetFlag<HardEvent::S_MTE3>(scalarToMte3_);
            WaitFlag<HardEvent::S_MTE3>(scalarToMte3_);
            DataCopyPad(outGm_[start_], output, {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_S>(mte3ToScalar_);
            WaitFlag<HardEvent::MTE3_S>(mte3ToScalar_);
        }
        SetMaskNorm();
        ResetMask();
    }

private:
    __aicore__ inline uint64_t SelectOwned(uint32_t valid, uint32_t count)
    {
        auto target = targetBuffer_.Get<float>();
        auto masks = maskBuffer_.Get<uint8_t>();
        const uint32_t maskBytes = tileLength_ / 8;
        SetFlag<HardEvent::MTE2_V>(mte2ToVector_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector_);
        Cast(target, indexBuffer_.Get<int64_t>(), RoundMode::CAST_RINT, valid);
        PipeBarrier<PIPE_V>();
        // CompareScalar needs complete 256-byte repeats; GatherMask excludes padding.
        const uint32_t compareCount = (valid + 63) / 64 * 64;
        // A flattened output tile can split a row. Select that enclosing row
        // range, then check the exact integer target in the scalar loop.
        const uint32_t firstRow = start_ / columns_;
        const uint32_t endRow = (start_ + count - 1U) / columns_ + 1U;
        CompareScalar(masks, target, float(int32_t(firstRow)), CMPMODE::GE, compareCount);
        CompareScalar(masks[maskBytes], target, float(int32_t(endRow)), CMPMODE::LT, compareCount);
        PipeBarrier<PIPE_V>();
        And(masks.ReinterpretCast<uint16_t>(), masks.ReinterpretCast<uint16_t>(),
            masks[maskBytes].ReinterpretCast<uint16_t>(), maskBytes / 2);
        PipeBarrier<PIPE_V>();
        uint64_t selected = 0;
        GatherMask(selectedBuffer_.Get<int32_t>(), positionBuffer_.Get<int32_t>(), masks.ReinterpretCast<uint32_t>(),
                   true, valid, {1, 1, 8, 0}, selected);
        SetFlag<HardEvent::V_S>(vectorToScalar_);
        WaitFlag<HardEvent::V_S>(vectorToScalar_);
        return selected;
    }
    TBuf<TPosition::VECCALC> targetBuffer_, positionBuffer_, selectedBuffer_, maskBuffer_;
    TEventID mte2ToVector_, vectorToScalar_;
    bool filtered_ = false;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> outBuffer_, seenBuffer_, indexBuffer_, srcBuffer_;
    GlobalTensor<T> selfGm_, srcGm_, outGm_;
    GlobalTensor<int64_t> indexGm_;
    TEventID scalarToMte3_, mte3ToScalar_, mte2ToScalar_, scalarToMte2_;
    bool compactRows_ = false;
    bool includeSelf_ = false;
    uint32_t chunkLength_ = 0, inputRows_ = 0, outputRows_ = 0;
    uint32_t columns_ = 0, tileLength_ = 0;
    uint64_t start_ = 0;
};

// Fast graph-attention path: input is partitioned across cores and scanned
// exactly once. Each core reduces into UB, then contiguous atomic-max writes
// merge partials. State in workspace preserves untouched values, NaNs and
// infinities: BF16 AtomicMax on 910B saturates both infinities to finite values.
template <typename T>
class ScatterReduceAtomicKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR src, GM_ADDR out, GM_ADDR workspace,
                                uint32_t inputRows, uint32_t outputRows, uint32_t columns, uint32_t inputChunk,
                                uint32_t tileLength, bool includeSelf, bool compactRows)
    {
        inputRows_ = inputRows;
        compactRows_ = compactRows;
        outputRows_ = outputRows;
        columns_ = columns;
        totalInput_ = inputRows_ * columns_;
        totalOutput_ = outputRows_ * columns_;
        inputStart_ = GetBlockIdx() * inputChunk;
        inputCount_ = inputStart_ < totalInput_ ? MinU32(inputChunk, totalInput_ - inputStart_) : 0;
        tileLength_ = tileLength;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), totalOutput_);
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(index), compactRows_ ? inputRows_ : totalInput_);
        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src), totalInput_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), totalOutput_);
        stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), totalOutput_);
        zeroGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + totalOutput_ * sizeof(int32_t)),
                                totalOutput_);
        selfBackupGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace + totalOutput_ * 2U * sizeof(int32_t)),
                                      totalOutput_);
        inPlace_ = self == out;
        includeSelf_ = includeSelf;
        pipe_.InitBuffer(partialBuffer_, totalOutput_ * sizeof(T));
        pipe_.InitBuffer(stateBuffer_, totalOutput_ * sizeof(int32_t));
        pipe_.InitBuffer(zeroBuffer_, totalOutput_ * sizeof(int32_t));
        pipe_.InitBuffer(indexBuffer_, tileLength_ * sizeof(int64_t));
        pipe_.InitBuffer(valueBuffer_, tileLength_ * sizeof(T));
        mte2ToScalar_ = pipe_.FetchEventID(HardEvent::MTE2_S);
        scalarToMte2_ = pipe_.FetchEventID(HardEvent::S_MTE2);
        scalarToMte3_ = pipe_.FetchEventID(HardEvent::S_MTE3);
        vectorToMte3_ = pipe_.FetchEventID(HardEvent::V_MTE3);
        mte3ToScalar_ = pipe_.FetchEventID(HardEvent::MTE3_S);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<T> partial = partialBuffer_.Get<T>();
        LocalTensor<int32_t> state = stateBuffer_.Get<int32_t>();
        LocalTensor<int64_t> indices = indexBuffer_.Get<int64_t>();
        LocalTensor<T> values = valueBuffer_.Get<T>();
        InitLocal(partial, state);
        InitGlobal(partial, state, values);
        SyncAll<true>();
        ReduceLocal(partial, state, indices, values);
        MergePartials(partial, state);
        SyncAll<true>();
        Finalize(partial, state, values);
    }

private:
    // AtomicMax merges states by priority. In particular, an all--inf reduction
    // must remain distinguishable from a real finite dtype minimum, even when
    // another core contributes only the -inf reduction identity.
    static constexpr int32_t kUnseen = 0;
    static constexpr int32_t kNegativeInfOnly = 1;
    static constexpr int32_t kFinite = 2;
    static constexpr int32_t kPositiveInf = 3;
    static constexpr int32_t kNan = 4;

    __aicore__ inline void InitLocal(LocalTensor<T> partial, LocalTensor<int32_t> state)
    {
        constexpr uint32_t kVectorChunk = 8192;
        for (uint32_t offset = 0; offset < totalOutput_; offset += kVectorChunk) {
            const uint32_t count = MinU32(kVectorChunk, totalOutput_ - offset);
            Duplicate(partial[offset], T(-__builtin_inff()), count);
            Duplicate(state[offset], kUnseen, count);
            Duplicate(zeroBuffer_.Get<int32_t>()[offset], INT32_MAX, count);
        }
    }

    __aicore__ inline void InitGlobal(LocalTensor<T> partial, LocalTensor<int32_t> state, LocalTensor<T> scratch)
    {
        const uint32_t alignment = 32U / sizeof(T);
        const uint32_t groups = (totalOutput_ + alignment - 1U) / alignment;
        const uint32_t chunk = (groups + GetBlockNum() - 1U) / GetBlockNum() * alignment;
        const uint32_t start = GetBlockIdx() * chunk;
        const uint32_t count = start < totalOutput_ ? MinU32(chunk, totalOutput_ - start) : 0;
        SetFlag<HardEvent::V_MTE3>(vectorToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vectorToMte3_);
        if (count > 0) {
            if (inPlace_) {
                for (uint32_t offset = 0; offset < count; offset += tileLength_) {
                    const uint32_t valid = MinU32(tileLength_, count - offset);
                    DataCopyPad(scratch, selfGm_[start + offset],
                                {1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0}, {false, 0, 0, T(0)});
                    SetFlag<HardEvent::MTE2_S>(mte2ToScalar_);
                    WaitFlag<HardEvent::MTE2_S>(mte2ToScalar_);
                    SetFlag<HardEvent::S_MTE3>(scalarToMte3_);
                    WaitFlag<HardEvent::S_MTE3>(scalarToMte3_);
                    DataCopyPad(selfBackupGm_[start + offset], scratch,
                                {1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0});
                    SetFlag<HardEvent::MTE3_S>(mte3ToScalar_);
                    WaitFlag<HardEvent::MTE3_S>(mte3ToScalar_);
                }
            }
            DataCopyPad(outGm_[start], partial, {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0});
            DataCopyPad(stateGm_[start], state, {1, count * 4U, 0, 0, 0});
            DataCopyPad(zeroGm_[start], zeroBuffer_.Get<int32_t>(), {1, count * 4U, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE3_S>(mte3ToScalar_);
        WaitFlag<HardEvent::MTE3_S>(mte3ToScalar_);
    }

    __aicore__ inline void ReduceLocal(LocalTensor<T> partial, LocalTensor<int32_t> state, LocalTensor<int64_t> indices,
                                       LocalTensor<T> values)
    {
        LocalTensor<uint16_t> partialBits16 = partialBuffer_.Get<uint16_t>();
        LocalTensor<uint16_t> valueBits16 = valueBuffer_.Get<uint16_t>();
        LocalTensor<uint32_t> partialBits32 = partialBuffer_.Get<uint32_t>();
        LocalTensor<uint32_t> valueBits32 = valueBuffer_.Get<uint32_t>();
        const uint32_t end = inputStart_ + inputCount_;
        for (uint32_t offset = inputStart_; offset < end; offset += tileLength_) {
            const uint32_t valid = MinU32(tileLength_, end - offset);
            const uint32_t indexStart = compactRows_ ? offset / columns_ : offset;
            const uint32_t indexCount = compactRows_ ? (offset + valid - 1) / columns_ - indexStart + 1 : valid;
            DataCopyPad(indices, indexGm_[indexStart], {1, indexCount * 8U, 0, 0, 0}, {false, 0, 0, int64_t(0)});
            DataCopyPad(values, srcGm_[offset], {1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, T(0)});
            SetFlag<HardEvent::MTE2_S>(mte2ToScalar_);
            WaitFlag<HardEvent::MTE2_S>(mte2ToScalar_);
            uint32_t column = offset % columns_;
            for (uint32_t j = 0; j < valid; ++j) {
                const int64_t row = indices.GetValue(compactRows_ ? (offset + j) / columns_ - indexStart : j);
                if (row >= 0 && row < outputRows_) {
                    const uint32_t target = static_cast<uint32_t>(row) * columns_ + column;
                    const T value = values.GetValue(j);
                    bool valueIsNan = false;
                    bool valueIsPositiveInf = false;
                    bool valueIsNegativeInf = false;
                    bool valueIsGreater = false;
                    if constexpr (IsSameType<T, float>::value) {
                        const uint32_t bits = valueBits32.GetValue(j);
                        valueIsNan = (bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU) != 0;
                        valueIsPositiveInf = bits == 0x7f800000U;
                        valueIsNegativeInf = bits == 0xff800000U;
                        valueIsGreater = value > partial.GetValue(target);
                    } else {
                        constexpr uint16_t exponentMask = IsSameType<T, half>::value ? 0x7c00U : 0x7f80U;
                        constexpr uint16_t mantissaMask = IsSameType<T, half>::value ? 0x03ffU : 0x007fU;
                        const uint16_t valueBits = valueBits16.GetValue(j);
                        const uint16_t previousBits = partialBits16.GetValue(target);
                        valueIsNan = (valueBits & exponentMask) == exponentMask && (valueBits & mantissaMask) != 0;
                        valueIsPositiveInf = valueBits == exponentMask;
                        valueIsNegativeInf = valueBits == (exponentMask | 0x8000U);
                        const uint16_t valueKey = (valueBits & 0x8000U) ? static_cast<uint16_t>(~valueBits) :
                                                                          static_cast<uint16_t>(valueBits ^ 0x8000U);
                        const uint16_t previousKey = (previousBits & 0x8000U) ?
                                                         static_cast<uint16_t>(~previousBits) :
                                                         static_cast<uint16_t>(previousBits ^ 0x8000U);
                        const bool bothZero = (valueBits & 0x7fffU) == 0 && (previousBits & 0x7fffU) == 0;
                        valueIsGreater = !bothZero && valueKey > previousKey;
                    }
                    // Atomic max does not order equal signed zeros. Carry the
                    // first source position and sign through an atomic min.
                    const uint32_t bits = IsSameType<T, float>::value ? valueBits32.GetValue(j) :
                                                                        valueBits16.GetValue(j);
                    const uint32_t magnitude = IsSameType<T, float>::value ? 0x7fffffffU : 0x7fffU;
                    if ((bits & magnitude) == 0) {
                        const uint32_t sign = bits >> (IsSameType<T, float>::value ? 31 : 15);
                        const int32_t code = static_cast<int32_t>((offset + j) * 2U + sign);
                        auto zeros = zeroBuffer_.Get<int32_t>();
                        if (code < zeros.GetValue(target))
                            zeros.SetValue(target, code);
                    }
                    if (valueIsNan) {
                        state.SetValue(target, kNan);
                    } else if (valueIsPositiveInf) {
                        partial.SetValue(target, value);
                        if (state.GetValue(target) < kPositiveInf)
                            state.SetValue(target, kPositiveInf);
                    } else {
                        if (valueIsGreater)
                            partial.SetValue(target, value);
                        const int32_t valueState = valueIsNegativeInf ? kNegativeInfOnly : kFinite;
                        if (state.GetValue(target) < valueState)
                            state.SetValue(target, valueState);
                    }
                }
                if (++column == columns_)
                    column = 0;
            }
            SetFlag<HardEvent::S_MTE2>(scalarToMte2_);
            WaitFlag<HardEvent::S_MTE2>(scalarToMte2_);
        }
    }

    __aicore__ inline void MergePartials(LocalTensor<T> partial, LocalTensor<int32_t> state)
    {
        SetFlag<HardEvent::S_MTE3>(scalarToMte3_);
        WaitFlag<HardEvent::S_MTE3>(scalarToMte3_);
        // The atomic route guarantees 32-byte aligned output and state buffers.
        SetAtomicMax<T>();
        DataCopy(outGm_, partial, totalOutput_);
        SetAtomicNone();
        SetAtomicMax<int32_t>();
        DataCopy(stateGm_, state, totalOutput_);
        SetAtomicNone();
        SetAtomicMin<int32_t>();
        DataCopy(zeroGm_, zeroBuffer_.Get<int32_t>(), totalOutput_);
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_S>(mte3ToScalar_);
        WaitFlag<HardEvent::MTE3_S>(mte3ToScalar_);
    }

    __aicore__ inline void Finalize(LocalTensor<T> output, LocalTensor<int32_t> state, LocalTensor<T> self)
    {
        const uint32_t alignment = 32U / sizeof(T);
        const uint32_t groups = (totalOutput_ + alignment - 1U) / alignment;
        const uint32_t chunk = (groups + GetBlockNum() - 1U) / GetBlockNum() * alignment;
        const uint32_t ownedStart = GetBlockIdx() * chunk;
        const uint32_t ownedCount = ownedStart < totalOutput_ ? MinU32(chunk, totalOutput_ - ownedStart) : 0;
        for (uint32_t localOffset = 0; localOffset < ownedCount; localOffset += tileLength_) {
            const uint32_t valid = MinU32(tileLength_, ownedCount - localOffset);
            const uint32_t globalOffset = ownedStart + localOffset;
            DataCopyPad(output, outGm_[globalOffset], {1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, T(0)});
            DataCopyPad(state, stateGm_[globalOffset], {1, valid * 4U, 0, 0, 0}, {false, 0, 0, int32_t(0)});
            DataCopyPad(zeroBuffer_.Get<int32_t>(), zeroGm_[globalOffset], {1, valid * 4U, 0, 0, 0},
                        {false, 0, 0, int32_t(0)});
            if (inPlace_) {
                DataCopyPad(self, selfBackupGm_[globalOffset], {1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0},
                            {false, 0, 0, T(0)});
            } else {
                DataCopyPad(self, selfGm_[globalOffset], {1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0},
                            {false, 0, 0, T(0)});
            }
            SetFlag<HardEvent::MTE2_S>(mte2ToScalar_);
            WaitFlag<HardEvent::MTE2_S>(mte2ToScalar_);
            for (uint32_t j = 0; j < valid; ++j) {
                const int32_t itemState = state.GetValue(j);
                if (itemState == kUnseen)
                    output.SetValue(j, self.GetValue(j));
                else if (itemState == kNegativeInfOnly)
                    output.SetValue(j, T(-__builtin_inff()));
                else if (itemState == kPositiveInf)
                    output.SetValue(j, T(__builtin_inff()));
                else if (itemState == kNan)
                    output.SetValue(j, T(__builtin_nanf("")));
                uint32_t magnitude;
                if constexpr (IsSameType<T, float>::value) {
                    magnitude = output.template ReinterpretCast<uint32_t>().GetValue(j) & 0x7fffffffU;
                } else {
                    magnitude = output.template ReinterpretCast<uint16_t>().GetValue(j) & 0x7fffU;
                }
                if (itemState == kFinite && magnitude == 0) {
                    const int32_t code = zeroBuffer_.Get<int32_t>().GetValue(j);
                    if constexpr (IsSameType<T, float>::value) {
                        output.template ReinterpretCast<uint32_t>().SetValue(j, uint32_t(code & 1) << 31);
                    } else {
                        output.template ReinterpretCast<uint16_t>().SetValue(j, uint16_t(code & 1) << 15);
                    }
                }
                if (includeSelf_ && itemState != kUnseen) {
                    // The original self participates once, after restoring the
                    // source reduction's special values. Ties retain self.
                    bool selfIsNan;
                    bool outputIsNan;
                    bool selfIsGreaterOrEqual;
                    if constexpr (IsSameType<T, float>::value) {
                        const uint32_t x = self.template ReinterpretCast<uint32_t>().GetValue(j);
                        const uint32_t y = output.template ReinterpretCast<uint32_t>().GetValue(j);
                        selfIsNan = (x & 0x7fffffffU) > 0x7f800000U;
                        outputIsNan = (y & 0x7fffffffU) > 0x7f800000U;
                        selfIsGreaterOrEqual = self.GetValue(j) >= output.GetValue(j);
                    } else {
                        constexpr uint16_t infinity = IsSameType<T, half>::value ? 0x7c00U : 0x7f80U;
                        const uint16_t x = self.template ReinterpretCast<uint16_t>().GetValue(j);
                        const uint16_t y = output.template ReinterpretCast<uint16_t>().GetValue(j);
                        selfIsNan = (x & 0x7fffU) > infinity;
                        outputIsNan = (y & 0x7fffU) > infinity;
                        const uint16_t xKey = (x & 0x8000U) ? uint16_t(~x) : uint16_t(x ^ 0x8000U);
                        const uint16_t yKey = (y & 0x8000U) ? uint16_t(~y) : uint16_t(y ^ 0x8000U);
                        const bool bothZero = (x & 0x7fffU) == 0 && (y & 0x7fffU) == 0;
                        selfIsGreaterOrEqual = bothZero || xKey >= yKey;
                    }
                    if (selfIsNan || (!outputIsNan && selfIsGreaterOrEqual)) {
                        output.SetValue(j, self.GetValue(j));
                    }
                }
            }
            SetFlag<HardEvent::S_MTE3>(scalarToMte3_);
            WaitFlag<HardEvent::S_MTE3>(scalarToMte3_);
            DataCopyPad(outGm_[globalOffset], output, {1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_S>(mte3ToScalar_);
            WaitFlag<HardEvent::MTE3_S>(mte3ToScalar_);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> partialBuffer_, stateBuffer_, indexBuffer_, valueBuffer_, zeroBuffer_;
    GlobalTensor<T> selfGm_, srcGm_, outGm_, selfBackupGm_;
    GlobalTensor<int64_t> indexGm_;
    GlobalTensor<int32_t> stateGm_, zeroGm_;
    TEventID mte2ToScalar_, scalarToMte2_, scalarToMte3_, vectorToMte3_, mte3ToScalar_;
    uint32_t inputRows_ = 0, outputRows_ = 0, columns_ = 0;
    uint32_t totalInput_ = 0, totalOutput_ = 0, inputStart_ = 0, inputCount_ = 0, tileLength_ = 0;
    bool compactRows_ = false;
    bool inPlace_ = false;
    bool includeSelf_ = false;
};

template <typename T, uint32_t Algorithm>
__aicore__ inline void Run(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace,
                           const ScatterElementsV2TilingData* tiling)
{
    if constexpr (Algorithm == 2) {
        ScatterReduceAtomicKernel<T> kernel;
        kernel.Init(var, indices, updates, var, GetUserWorkspace(workspace), tiling->amaxInputRows,
                    tiling->amaxOutputRows, tiling->amaxColumns, tiling->amaxChunkLength, tiling->amaxTileLength,
                    tiling->includeSelf != 0, tiling->amaxCompactRows != 0);
        kernel.Process();
    } else if constexpr (Algorithm == 1) {
        ScatterReduceRowOwnerKernel<T> kernel;
        kernel.Init(var, indices, updates, var, tiling->amaxInputRows, tiling->amaxOutputRows, tiling->amaxColumns,
                    tiling->amaxChunkLength, tiling->amaxTileLength, tiling->includeSelf != 0);
        kernel.Process();
    } else {
        ScatterReduceOwnerKernel<T> kernel;
        kernel.Init(var, indices, updates, var, tiling->amaxInputRows, tiling->amaxOutputRows, tiling->amaxColumns,
                    tiling->amaxChunkLength, tiling->amaxTileLength, tiling->includeSelf != 0,
                    tiling->amaxCompactRows != 0);
        kernel.Process();
    }
}
} // namespace ScatterElementsV2Amax
#endif
