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
 * \file normalize_bbox.h
 * \brief normalize_bbox arch35 kernel (MemBase / SIMD level-2 route)
 *
 * y = boxes / [h, w, h, w]  (per-batch (h,w) from shape_hw int32 -> float)
 * boxes rank 2-8: dim0=batch, coord=4 anchored at last dim (normal) / dim1 (reversed),
 *   middle dims flatten into num (frames per batch). Tiling carries batch/num/coordNum.
 * reversedBox=0 (normal):   layout (batch, ..., 4), divisor period-2 [h,w,h,w,...]
 * reversedBox=1 (reversed): layout (batch, 4, ...), rows 0/2 / h, rows 1/3 / w
 */

#ifndef NORMALIZE_BBOX_H
#define NORMALIZE_BBOX_H

#include "kernel_operator.h"
#include "normalize_bbox_tiling_data.h"

namespace NormalizeBBox {
using namespace AscendC;

constexpr int64_t NB_BLOCK_SIZE = 32;
constexpr int64_t NB_VREPEAT_SIZE = 256;
constexpr int64_t NB_DOUBLE_BUFFER = 2;
constexpr uint64_t NB_ODD_LANE_MASK = 0xAAAAAAAAAAAAAAAAULL; // odd lanes (x-system coords) -> w
constexpr int64_t NB_MASK_ELEM_WIDTH = 128;                  // mask Duplicate covers 128 CompT elements per repeat
constexpr size_t NB_MASK_WORD_COUNT = 2;
constexpr uint8_t NB_DUPLICATE_REPEAT_COUNT = 1;
constexpr uint8_t NB_DUPLICATE_BLOCK_STRIDE = 1;
constexpr uint8_t NB_DUPLICATE_REPEAT_STRIDE = 8;

// Compute type: boxes dtype is half or float, Div runs in-place (CompT == T).
template <typename T>
struct DivCompute {
    using Type = T;
};

template <typename T, bool reversedBox>
class NormalizeBBoxKernel {
public:
    __aicore__ inline NormalizeBBoxKernel(TPipe& tpipe, const NormalizeBBoxTilingData& tilingData)
        : tpipe_(tpipe), tiling_(tilingData){};
    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR shape_hw, GM_ADDR y);
    __aicore__ inline void Process();

    using CompT = typename DivCompute<T>::Type;

private:
    __aicore__ inline void LoadHW(int64_t b, CompT& h, CompT& w);
    __aicore__ inline void BuildDivisorNormal(const CompT& h, const CompT& w, int32_t divWidth);
    __aicore__ inline void BuildConstBlock(const LocalTensor<CompT>& dst, const CompT& v, int32_t width);
    __aicore__ inline void ProcessBatchNormal(int64_t b, int64_t elemStart, int64_t elemCount);
    __aicore__ inline void ProcessBatchReversed(int64_t b, int64_t frameStart, int64_t frameCount);
    __aicore__ inline void CopyDivide(int64_t gmOffset, int64_t processLen, const LocalTensor<CompT>& divisor);

    __aicore__ inline int64_t CeilAlign(int64_t a, int64_t b) { return b == 0 ? a : (a + b - 1) / b * b; }

    TPipe& tpipe_;
    const NormalizeBBoxTilingData& tiling_;

    // basic params
    int64_t blockIdx_ = 0;
    int64_t batch_ = 0;
    int64_t num_ = 0;
    int64_t coordNum_ = 0;
    int64_t splitMode_ = 0;
    int64_t tileLen_ = 0;

    // this core's batch range (splitMode==1)
    int64_t batchStart_ = 0;
    int64_t batchCount_ = 0;

    TQue<QuePosition::VECIN, NB_DOUBLE_BUFFER> boxesInQue_;
    TQue<QuePosition::VECOUT, NB_DOUBLE_BUFFER> boxesOutQue_;
    TQue<QuePosition::VECIN, 1> shapeInQue_; // shape_hw 3 int32 per batch
    TBuf<TPosition::VECCALC> hwCastBuf_;     // int32 -> f32 [-> half] scratch for h/w
    TBuf<TPosition::VECCALC> divisorBuf_;    // divisor block(s), in CompT

    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> shapeGm_;
};

template <typename T, bool reversedBox>
__aicore__ inline void NormalizeBBoxKernel<T, reversedBox>::Init(GM_ADDR boxes, GM_ADDR shape_hw, GM_ADDR y)
{
    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    batch_ = tiling_.batch;
    num_ = tiling_.num;
    coordNum_ = tiling_.coordNum;
    splitMode_ = tiling_.splitMode;
    tileLen_ = tiling_.tileLen;

    // determine this core's work range
    if (splitMode_ == 1) {
        // split by batch: front bigCoreNum cores handle batchPerCore, rest handle (batchPerCore-1)
        int64_t bigCoreNum = static_cast<int64_t>(tiling_.bigCoreNum);
        int64_t batchPerCore = static_cast<int64_t>(tiling_.batchPerCore);
        int64_t tailBatchNum = static_cast<int64_t>(tiling_.tailBatchNum);
        if (blockIdx_ < bigCoreNum) {
            batchStart_ = blockIdx_ * batchPerCore;
            batchCount_ = batchPerCore;
        } else {
            batchStart_ = bigCoreNum * batchPerCore + (blockIdx_ - bigCoreNum) * tailBatchNum;
            batchCount_ = tailBatchNum;
        }
    } else {
        // split by num (batch == 1): handled inside Process via per-core num range
        batchStart_ = 0;
        batchCount_ = (blockIdx_ == 0 || batch_ == 1) ? batch_ : 0;
    }

    boxesGm_.SetGlobalBuffer((__gm__ T*)boxes, tiling_.totalElements);
    yGm_.SetGlobalBuffer((__gm__ T*)y, tiling_.totalElements);
    shapeGm_.SetGlobalBuffer((__gm__ int32_t*)shape_hw, tiling_.shapeElements);

    tpipe_.InitBuffer(boxesInQue_, NB_DOUBLE_BUFFER, tiling_.boxesBufferBytes);
    tpipe_.InitBuffer(boxesOutQue_, NB_DOUBLE_BUFFER, tiling_.boxesBufferBytes);
    tpipe_.InitBuffer(shapeInQue_, 1, tiling_.shapeBufferBytes); // 3 int32 fits in one 32B block
    tpipe_.InitBuffer(hwCastBuf_, tiling_.hwCastBufferBytes);    // f32 + half scratch
    // divisor is in CompT: normal needs 1 full-width block; reversed needs divH + divW (2 blocks)
    tpipe_.InitBuffer(divisorBuf_, tiling_.divisorBufferBytes);
}

// Load shape_hw[b] = [h, w, *] (int32) -> Cast to CompT scalars h, w.
// CompT is float for float and half for half. The divisor is built in CompT, so
// the whole divide runs in CompT (cast chain: int32->f32->[f16 for half boxes]->divide).
template <typename T, bool reversedBox>
__aicore__ inline void NormalizeBBoxKernel<T, reversedBox>::LoadHW(int64_t b, CompT& h, CompT& w)
{
    LocalTensor<int32_t> shapeLocal = shapeInQue_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(3 * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(shapeLocal, shapeGm_[b * 3], copyParams, padParams); // 3 int32 per batch
    shapeInQue_.EnQue(shapeLocal);
    shapeLocal = shapeInQue_.DeQue<int32_t>();

    LocalTensor<float> hwF32 = hwCastBuf_.Get<float>();
    Cast(hwF32, shapeLocal, RoundMode::CAST_RINT, 3); // int32 -> float32 (clean path)
    PipeBarrier<PIPE_V>();

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    if constexpr (std::is_same_v<CompT, half>) {
        LocalTensor<half> hwHalf = hwCastBuf_.Get<half>()[16]; // separate 32B block (byte 32)
        Cast(hwHalf, hwF32, RoundMode::CAST_RINT, 3);          // float32 -> half
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        h = hwHalf.GetValue(0);
        w = hwHalf.GetValue(1);
    } else {
        // CompT == float: read h/w directly as float
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        h = hwF32.GetValue(0);
        w = hwF32.GetValue(1);
    }
    shapeInQue_.FreeTensor(shapeLocal);
}

// normal layout divisor: period-2 [h,w,h,w,...] covering divWidth (aligned to a repeat), in CompT
template <typename T, bool reversedBox>
__aicore__ inline void NormalizeBBoxKernel<T, reversedBox>::BuildDivisorNormal(const CompT& h, const CompT& w,
                                                                               int32_t divWidth)
{
    LocalTensor<CompT> divisor = divisorBuf_.Get<CompT>();
    Duplicate(divisor, h, divWidth); // all lanes = h
    PipeBarrier<PIPE_V>();
    // overwrite odd lanes (x-system coords) with w via masked Duplicate.
    // DAV_3510 Vector lane count is dtype-dependent: 128 for 2-byte (half), 64 for 4-byte (float).
    // mask[2] has 128 bits, but only the first 64 are effective for 4-byte CompT on 3510,
    // so elements 64+ would keep h instead of w if we used a single 128-element repeat.
    // Process in chunks of 256/sizeof(CompT) elements (64 for float, 128 for half) to cover all dtypes.
    constexpr int32_t maskElemWidth = NB_VREPEAT_SIZE / static_cast<int32_t>(sizeof(CompT));
    uint64_t mask[NB_MASK_WORD_COUNT] = {NB_ODD_LANE_MASK, NB_ODD_LANE_MASK};
    for (int32_t off = 0; off < divWidth; off += maskElemWidth) {
        Duplicate(divisor[off], w, mask, NB_DUPLICATE_REPEAT_COUNT, NB_DUPLICATE_BLOCK_STRIDE,
                  NB_DUPLICATE_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
    }
}

// reversed layout divisor: single constant block (all h or all w), in CompT
template <typename T, bool reversedBox>
__aicore__ inline void NormalizeBBoxKernel<T, reversedBox>::BuildConstBlock(const LocalTensor<CompT>& dst,
                                                                            const CompT& v, int32_t width)
{
    Duplicate(dst, v, width);
}

// CopyIn one tile -> Div by divisor -> CopyOut.
// half/float: Div runs directly in T (CompT == T).
template <typename T, bool reversedBox>
__aicore__ inline void NormalizeBBoxKernel<T, reversedBox>::CopyDivide(int64_t gmOffset, int64_t processLen,
                                                                       const LocalTensor<CompT>& divisor)
{
    LocalTensor<T> boxesIn = boxesInQue_.template AllocTensor<T>();
    DataCopyExtParams inParams{1, static_cast<uint32_t>(processLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(boxesIn, boxesGm_[gmOffset], inParams, padParams);
    boxesInQue_.EnQue(boxesIn);

    boxesIn = boxesInQue_.template DeQue<T>();
    LocalTensor<T> boxesOut = boxesOutQue_.template AllocTensor<T>();
    Div(boxesOut, boxesIn, divisor, static_cast<int32_t>(processLen));
    boxesInQue_.FreeTensor(boxesIn);
    boxesOutQue_.EnQue(boxesOut);

    boxesOut = boxesOutQue_.template DeQue<T>();
    DataCopyExtParams outParams{1, static_cast<uint32_t>(processLen * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm_[gmOffset], boxesOut, outParams);
    boxesOutQue_.FreeTensor(boxesOut);
}

// normal layout: process elemCount elements of batch b starting at elemStart (elements within batch)
template <typename T, bool reversedBox>
__aicore__ inline void NormalizeBBoxKernel<T, reversedBox>::ProcessBatchNormal(int64_t b, int64_t elemStart,
                                                                               int64_t elemCount)
{
    if (elemCount == 0) {
        return;
    }
    CompT h, w;
    LoadHW(b, h, w);
    int64_t coverLen = static_cast<int64_t>(elemCount) < static_cast<int64_t>(tileLen_) ?
                           static_cast<int64_t>(elemCount) :
                           static_cast<int64_t>(tileLen_);
    int32_t divWidth = static_cast<int32_t>(CeilAlign(coverLen, NB_MASK_ELEM_WIDTH));
    if (divWidth > static_cast<int32_t>(tileLen_)) {
        // FP32 的最小 tile 为 64 元素，小于 128-bit mask 表示范围；直接截到 tile
        // 仍满足 dtype 对应的 256B repeat 对齐，不能向下按 128 元素取整成 0。
        divWidth = static_cast<int32_t>(tileLen_);
    }
    BuildDivisorNormal(h, w, divWidth);
    LocalTensor<CompT> divisor = divisorBuf_.Get<CompT>();
    int64_t base = b * num_ * coordNum_ + elemStart;
    for (int64_t off = 0; off < elemCount;) {
        int64_t processLen = (elemCount - off) < tileLen_ ? (elemCount - off) : tileLen_;
        CopyDivide(base + off, processLen, divisor);
        off += processLen;
    }
}

// reversed layout: process frameCount frames of batch b starting at frameStart; rows 0/2 / h, rows 1/3 / w
//
// [iter3 cross-row 32B residue — SAFE, no fix needed]
// In the num-split path (batch==1) every core owns a 32B-aligned num-segment [frameStart, frameStart+frameCount)
// and writes it into all 4 rows. When num*sizeof(T) is not a 32B multiple, a row boundary row*num is itself
// non-32B-aligned, so the last core's row-r tail write and core-0's row-(r+1) head write land in the *same*
// 32B GM block but cover *disjoint* byte ranges. This is safe: DataCopyPad UB->GM on DAV_3510 writes exactly
// blockLen bytes (byte-accurate, no 32B-granule read-modify-write), so disjoint sub-block writes from different
// cores never clobber each other. Empirically confirmed by the real-NPU ST cases that exercise the identical
// cross-core misaligned DataCopyPad write mechanism: normal (205,269,4) fp32 (batch-stride 4304B, mod 32 = 16,
// 220580/220580 correct) and reversed (2,4,7) fp32 (batch-stride 112B, mod 32 = 16, 56/56 correct).
template <typename T, bool reversedBox>
__aicore__ inline void NormalizeBBoxKernel<T, reversedBox>::ProcessBatchReversed(int64_t b, int64_t frameStart,
                                                                                 int64_t frameCount)
{
    if (frameCount == 0) {
        return;
    }
    CompT h, w;
    LoadHW(b, h, w);
    int64_t coverLen = static_cast<int64_t>(frameCount) < static_cast<int64_t>(tileLen_) ?
                           static_cast<int64_t>(frameCount) :
                           static_cast<int64_t>(tileLen_);
    int32_t divWidth = static_cast<int32_t>(CeilAlign(coverLen, NB_MASK_ELEM_WIDTH));
    if (divWidth > static_cast<int32_t>(tileLen_)) {
        divWidth = static_cast<int32_t>(tileLen_);
    }
    LocalTensor<CompT> divH = divisorBuf_.Get<CompT>();
    LocalTensor<CompT> divW = divisorBuf_.Get<CompT>()[tileLen_];
    BuildConstBlock(divH, h, divWidth);
    BuildConstBlock(divW, w, divWidth);
    PipeBarrier<PIPE_V>();
    for (int64_t row = 0; row < coordNum_; row++) {
        LocalTensor<CompT> divisor = (row % 2 == 0) ? divH : divW;
        int64_t rowBase = b * coordNum_ * num_ + row * num_ + frameStart;
        for (int64_t off = 0; off < frameCount;) {
            int64_t processLen = (frameCount - off) < tileLen_ ? (frameCount - off) : tileLen_;
            CopyDivide(rowBase + off, processLen, divisor);
            off += processLen;
        }
    }
}

template <typename T, bool reversedBox>
__aicore__ inline void NormalizeBBoxKernel<T, reversedBox>::Process()
{
    if (blockIdx_ >= static_cast<int64_t>(tiling_.usedCoreNum)) {
        return;
    }
    if (splitMode_ == 1) {
        // split by batch (batch > 1)
        for (int64_t i = 0; i < batchCount_; i++) {
            int64_t b = batchStart_ + i;
            if constexpr (!reversedBox) {
                ProcessBatchNormal(b, 0, num_ * coordNum_);
            } else {
                ProcessBatchReversed(b, 0, num_);
            }
        }
    } else {
        // split by num (batch == 1): each core handles a num range
        int64_t numBigCore = static_cast<int64_t>(tiling_.numBigCore);
        int64_t numPerCore = static_cast<int64_t>(tiling_.numPerCore);
        int64_t tailNumCore = static_cast<int64_t>(tiling_.tailNumCore);
        int64_t numStart;
        int64_t numCnt;
        if (blockIdx_ < numBigCore) {
            numStart = blockIdx_ * numPerCore;
            numCnt = numPerCore;
        } else {
            numStart = numBigCore * numPerCore + (blockIdx_ - numBigCore) * tailNumCore;
            numCnt = tailNumCore;
        }
        if (numStart >= num_) {
            numCnt = 0;
        } else if (numStart + numCnt > num_) {
            numCnt = num_ - numStart;
        }
        if constexpr (!reversedBox) {
            ProcessBatchNormal(0, numStart * coordNum_, numCnt * coordNum_);
        } else {
            ProcessBatchReversed(0, numStart, numCnt);
        }
    }
}
} // namespace NormalizeBBox
#endif // NORMALIZE_BBOX_H
