/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Parallel ownership of original column chunks. Each chunk publishes four
// 32B slots (CE/FW high and low), then every core merges in original order.
#ifndef SOFTMAX_FOCAL_LOSS_PARALLEL_H
#define SOFTMAX_FOCAL_LOSS_PARALLEL_H
#include "softmax_focal_loss_nd_vec.h"
namespace SoftmaxFocalLoss {
template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline bool SoftmaxFocalLossND<T, TW, hasWeight>::UseExactSum(int64_t len) const
{
    int64_t blocks = (len + static_cast<int64_t>(VL_FP32) - 1) / static_cast<int64_t>(VL_FP32);
    int64_t logLen = 0;
    while ((static_cast<int64_t>(1) << logLen) < len) {
        ++logLen;
    }
    return blocks > 1 && blocks > logLen;
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::StagePartial(int64_t chunk, bool exact)
{
    // Four 32B slots: CE hi, FW hi, CE lo, FW lo. Store the raw chunk expansion,
    // before MergeRowSlots2; merging it into zero here would add a rounding step.
    LocalTensor<float> source = exact ? dfBuf_.template Get<float>() : accBuf_.template Get<float>();
    uint32_t sourceStride = exact ? VL_FP32 * ROW_SLOT_ELEM : static_cast<uint32_t>(accStride_);
    auto src = (__ubuf__ float*)source.GetPhyAddr();
    LocalTensor<float> stage = yQue_.template AllocTensor<float>();
    auto dst = (__ubuf__ float*)stage.GetPhyAddr();
    uint32_t slotCount = ROW_SLOT_ELEM;
    __VEC_SCOPE__
    {
        RegTensor<float> value;
        MaskReg mask = UpdateMask<float>(slotCount);
        for (uint16_t slot = 0; slot < 4; ++slot) {
            if (exact || slot < 2) {
                AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(value, src + slot * sourceStride);
            } else {
                AscendC::Reg::Duplicate(value, 0.0f);
            }
            AscendC::Reg::StoreAlign(dst + slot * ROW_SLOT_ELEM, value, mask);
        }
    }
    yQue_.template EnQue<float>(stage);
    stage = yQue_.template DeQue<float>();
    DataCopy(partialGm_[chunk * 4 * ROW_SLOT_ELEM], stage, 4 * ROW_SLOT_ELEM);
    yQue_.FreeTensor(stage);
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::MergeCompensated(RegTensor<float>& hi,
                                                                              RegTensor<float>& lo,
                                                                              RegTensor<float>& partHi,
                                                                              RegTensor<float>& partLo, MaskReg mask)
{
    // Same ordered operations and NaN residual guards as MergeRowSlots2.
    RegTensor<float> sum;
    RegTensor<float> error;
    RegTensor<float> back;
    RegTensor<float> temp;
    RegTensor<float> zero;
    MaskReg finite;
    AscendC::Reg::Duplicate(zero, 0.0f);
    AscendC::Reg::Add(sum, hi, partHi, mask);
    AscendC::Reg::Sub(back, sum, hi, mask);
    AscendC::Reg::Sub(temp, sum, back, mask);
    AscendC::Reg::Sub(temp, hi, temp, mask);
    AscendC::Reg::Sub(error, partHi, back, mask);
    AscendC::Reg::Add(error, temp, error, mask);
    AscendC::Reg::Add(error, error, lo, mask);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(finite, error, error, mask);
    AscendC::Reg::Select(error, error, zero, finite);
    AscendC::Reg::Add(hi, sum, error, mask);
    AscendC::Reg::Sub(back, hi, sum, mask);
    AscendC::Reg::Sub(temp, hi, back, mask);
    AscendC::Reg::Sub(temp, sum, temp, mask);
    AscendC::Reg::Sub(lo, error, back, mask);
    AscendC::Reg::Add(lo, temp, lo, mask);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(finite, lo, lo, mask);
    AscendC::Reg::Select(lo, lo, zero, finite);
    AscendC::Reg::Add(sum, hi, partLo, mask);
    AscendC::Reg::Sub(back, sum, hi, mask);
    AscendC::Reg::Sub(temp, sum, back, mask);
    AscendC::Reg::Sub(temp, hi, temp, mask);
    AscendC::Reg::Sub(error, partLo, back, mask);
    AscendC::Reg::Add(error, temp, error, mask);
    AscendC::Reg::Add(error, error, lo, mask);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(finite, error, error, mask);
    AscendC::Reg::Select(error, error, zero, finite);
    AscendC::Reg::Add(hi, sum, error, mask);
    AscendC::Reg::Sub(back, hi, sum, mask);
    AscendC::Reg::Sub(temp, hi, back, mask);
    AscendC::Reg::Sub(temp, sum, temp, mask);
    AscendC::Reg::Sub(lo, error, back, mask);
    AscendC::Reg::Add(lo, temp, lo, mask);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(finite, lo, lo, mask);
    AscendC::Reg::Select(lo, lo, zero, finite);
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::MergePartialBatch(__ubuf__ float* partials,
                                                                               __ubuf__ float* acc, uint16_t count,
                                                                               bool plainChunks, bool plainTail)
{
    uint32_t lanes = 2; // CE and FW use the same merge, independently in two lanes.
    __ubuf__ float* accLo = acc + 4 * accStride_;
    __VEC_SCOPE__
    {
        MaskReg mask = UpdateMask<float>(lanes);
        RegTensor<int32_t> index;
        RegTensor<float> hi;
        RegTensor<float> lo;
        RegTensor<float> partHi;
        RegTensor<float> partLo;
        RegTensor<int32_t> accIndex;
        AscendC::Reg::Arange(index, 0);
        AscendC::Reg::Muls(accIndex, index, static_cast<int32_t>(accStride_), mask);
        AscendC::Reg::Gather(hi, acc, (RegTensor<uint32_t>&)accIndex, mask);
        AscendC::Reg::Gather(lo, accLo, (RegTensor<uint32_t>&)accIndex, mask);
        AscendC::Reg::Muls(index, index, static_cast<int32_t>(ROW_SLOT_ELEM), mask);
        for (uint16_t i = 0; i < count; ++i) {
            __ubuf__ float* chunk = partials + i * 4 * ROW_SLOT_ELEM;
            AscendC::Reg::Gather(partHi, chunk, (RegTensor<uint32_t>&)index, mask);
            if (plainChunks || (plainTail && i == count - 1)) {
                // Keep the ordinary ReduceSum/Add path for short chunks, including a short tail.
                AscendC::Reg::Add(hi, hi, partHi, mask);
            } else {
                AscendC::Reg::Gather(partLo, chunk + 2 * ROW_SLOT_ELEM, (RegTensor<uint32_t>&)index, mask);
                MergeCompensated(hi, lo, partHi, partLo, mask);
            }
        }
        AscendC::Reg::Scatter(acc, hi, (RegTensor<uint32_t>&)accIndex, mask);
        AscendC::Reg::Scatter(accLo, lo, (RegTensor<uint32_t>&)accIndex, mask);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
template <bool compensateFw>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::MergePartials()
{
    LocalTensor<float> acc = accBuf_.template Get<float>();
    Duplicate(acc, 0.0f, 6 * accStride_);
    // Reuse the now-free input queue. No extra UB allocation or change to the original chunk width.
    int64_t queueBytes = (rowsPerTile_ * colsPerChunk_ + VL_FP32) * sizeof(T);
    int64_t batchChunks = queueBytes / (4 * BLOCK_SIZE);
    constexpr int64_t MAX_BATCH_CHUNKS = 64;
    batchChunks = batchChunks > MAX_BATCH_CHUNKS ? MAX_BATCH_CHUNKS : batchChunks;
    bool plainChunks = !compensateFw && !UseExactSum(colsPerChunk_);
    for (int64_t first = 0; first < chunkNum_; first += batchChunks) {
        int64_t count = chunkNum_ - first;
        count = count > batchChunks ? batchChunks : count;
        LocalTensor<float> parts = predQue_.template AllocTensor<float>();
        DataCopy(parts, partialGm_[first * 4 * ROW_SLOT_ELEM], static_cast<uint32_t>(count * 4 * ROW_SLOT_ELEM));
        predQue_.template EnQue<float>(parts);
        parts = predQue_.template DeQue<float>();
        bool plainTail = !compensateFw && first + count == chunkNum_ &&
                         !UseExactSum(r_ - (chunkNum_ - 1) * colsPerChunk_);
        MergePartialBatch((__ubuf__ float*)parts.GetPhyAddr(), (__ubuf__ float*)acc.GetPhyAddr(),
                          static_cast<uint16_t>(count), plainChunks, plainTail);
        predQue_.FreeTensor(parts);
    }

    PipeBarrier<PIPE_V>();
    if constexpr (compensateFw) {
        ComputeCompensatedProduct();
    } else {
        Mul(acc[3 * accStride_], acc, acc[accStride_], 1);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
template <bool compensateFw>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::ProcessSplitRow()
{
    int64_t block = static_cast<int64_t>(GetBlockIdx());
    int64_t baseChunks = chunkNum_ / realCoreNum_;
    int64_t extraCores = chunkNum_ % realCoreNum_;
    int64_t first = block * baseChunks + (block < extraCores ? block : extraCores);
    int64_t count = baseChunks + (block < extraCores ? 1 : 0);
    LocalTensor<float> ce = ceBuf_.template Get<float>();
    LocalTensor<float> fw = fwBuf_.template Get<float>();
    LocalTensor<float> acc = accBuf_.template Get<float>();
    for (int64_t c = first; c < first + count; ++c) {
        int64_t len = r_ - c * colsPerChunk_;
        len = len > colsPerChunk_ ? colsPerChunk_ : len;
        CopyIn(0, c * colsPerChunk_, 1, len);
        ComputeCeFw(1, len);
        bool exact = compensateFw || UseExactSum(len);
        if (exact) {
            ExactRowSum2<compensateFw, false>(
                (__ubuf__ float*)ce.GetPhyAddr(), (__ubuf__ float*)fw.GetPhyAddr(), (__ubuf__ float*)acc.GetPhyAddr(),
                (__ubuf__ float*)acc[4 * accStride_].GetPhyAddr(), (__ubuf__ float*)acc[accStride_].GetPhyAddr(),
                (__ubuf__ float*)acc[5 * accStride_].GetPhyAddr(), 1, len);
        } else {
            uint32_t shape[2] = {1, static_cast<uint32_t>(StrideOfF32(len))};
            ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(acc, ce, shape, false);
            ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(acc[accStride_], fw, shape, false);
        }
        StagePartial(c, exact);
    }
    // Every launched core participates, after its disjoint workspace writes complete.
    PipeBarrier<PIPE_MTE3>();
    SyncAll();
    MergePartials<compensateFw>();
    for (int64_t c = first; c < first + count; ++c) {
        int64_t len = r_ - c * colsPerChunk_;
        len = len > colsPerChunk_ ? colsPerChunk_ : len;
        CopyOutRow(0, c * colsPerChunk_, 1, len);
    }
}

} // namespace SoftmaxFocalLoss
#endif
