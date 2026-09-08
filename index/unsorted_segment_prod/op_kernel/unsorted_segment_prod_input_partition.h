/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNSORTED_SEGMENT_PROD_INPUT_PARTITION_H
#define UNSORTED_SEGMENT_PROD_INPUT_PARTITION_H

#include "../unsorted_segment_common/arch35/unsorted_segment_base.h"

namespace UnsortedSegmentProd {
using namespace AscendC;

constexpr uint32_t IP_DB_BUF = 2;
constexpr uint32_t IP_FLAG_STRIDE = 16;

template <typename X_T, typename IDS_T, typename InitValueType, typename VectorComputeFunc>
class KernelProdInputPartition {
public:
    __aicore__ inline KernelProdInputPartition(const UnsortedSegment::UnsortedSegmentProdInputPartTilingData* tiling,
                                               TPipe* pipe)
        : td_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputePartial();
    __aicore__ inline void MergePartial();
    __aicore__ inline void WriteInitOutput();

    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<X_T> yGm_;
    AscendC::GlobalTensor<X_T> wsGm_;
    AscendC::GlobalTensor<IDS_T> idsGm_;
    AscendC::GlobalTensor<int32_t> flagGm_;
    TQue<QuePosition::VECIN, IP_DB_BUF> xQue_;
    TQue<QuePosition::VECIN, IP_DB_BUF> idsQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TQue<QuePosition::VECIN, IP_DB_BUF> mergeQue_;
    TQue<QuePosition::VECOUT, 1> mergeOutQue_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> flagBuf_;
    TPipe* pipe_ = nullptr;
    const UnsortedSegment::UnsortedSegmentProdInputPartTilingData* td_;

    uint64_t innerAlign_ = 0;
    uint64_t ySize_ = 0;
    uint64_t ySizeAlignBytes_ = 0;
    uint32_t flagCount_ = 0;
    bool anyValidGm_ = false;
};

template <typename X_T, typename IDS_T, typename InitValueType, typename VectorComputeFunc>
__aicore__ inline void KernelProdInputPartition<X_T, IDS_T, InitValueType, VectorComputeFunc>::Init(GM_ADDR x,
                                                                                                    GM_ADDR segmentIds,
                                                                                                    GM_ADDR output,
                                                                                                    GM_ADDR workspace)
{
    xGm_.SetGlobalBuffer((__gm__ X_T*)(x));
    idsGm_.SetGlobalBuffer((__gm__ IDS_T*)(segmentIds));
    yGm_.SetGlobalBuffer((__gm__ X_T*)(output));
    wsGm_.SetGlobalBuffer((__gm__ X_T*)(workspace));
    innerAlign_ = UnsortedSegment::Aligned(td_->innerDim * sizeof(X_T),
                                           static_cast<uint64_t>(UnsortedSegment::ONE_BLOCK_SIZE)) /
                  sizeof(X_T);
    ySize_ = td_->outputOuterDim * innerAlign_;
    ySizeAlignBytes_ = UnsortedSegment::Aligned(static_cast<uint64_t>(td_->partCoreNum) * ySize_ * sizeof(X_T),
                                                static_cast<uint64_t>(UnsortedSegment::ONE_BLOCK_SIZE));
    flagGm_.SetGlobalBuffer((__gm__ int32_t*)(workspace + ySizeAlignBytes_));
    flagCount_ = static_cast<uint32_t>(GetBlockNum());

    pipe_->InitBuffer(xQue_, IP_DB_BUF, td_->baseS * innerAlign_ * sizeof(X_T));
    pipe_->InitBuffer(
        idsQue_, IP_DB_BUF,
        UnsortedSegment::Aligned(td_->baseS * sizeof(IDS_T), static_cast<uint64_t>(UnsortedSegment::ONE_BLOCK_SIZE)));
    pipe_->InitBuffer(yQue_, 1, ySize_ * sizeof(X_T));
    pipe_->InitBuffer(mergeQue_, IP_DB_BUF, td_->mergeChunk * sizeof(X_T));
    pipe_->InitBuffer(mergeOutQue_, 1, td_->mergeChunk * sizeof(X_T));
    pipe_->InitBuffer(reduceBuf_,
                      UnsortedSegment::AllIdsInvalidReduceBufBytes<IDS_T>(static_cast<uint32_t>(td_->baseS)));
    pipe_->InitBuffer(flagBuf_, flagCount_ * IP_FLAG_STRIDE * sizeof(int32_t));
}

template <typename X_T, typename IDS_T, typename InitValueType, typename VectorComputeFunc>
__aicore__ inline void KernelProdInputPartition<X_T, IDS_T, InitValueType, VectorComputeFunc>::ComputePartial()
{
    uint64_t blockId = GetBlockIdx();
    uint64_t rowStart = blockId * td_->normRowNum;
    if (rowStart >= td_->inputOuterDim) {
        LocalTensor<X_T> yInit = yQue_.AllocTensor<X_T>();
        Duplicate(yInit, InitValueType::Get(), ySize_);
        yQue_.EnQue<X_T>(yInit);
        yInit = yQue_.DeQue<X_T>();
        UnsortedSegment::CopyOut(wsGm_, yInit, blockId * ySize_, 1, ySize_, 0, 0);
        yQue_.FreeTensor(yInit);
        return;
    }
    uint64_t rowEnd = rowStart + td_->normRowNum;
    if (rowEnd > td_->inputOuterDim) {
        rowEnd = td_->inputOuterDim;
    }
    uint64_t myRows = rowEnd - rowStart;
    bool anyValidIds = false;

    LocalTensor<X_T> yLocal = yQue_.AllocTensor<X_T>();
    Duplicate(yLocal, InitValueType::Get(), ySize_);
    yQue_.EnQue<X_T>(yLocal);
    yLocal = yQue_.DeQue<X_T>();

    uint64_t sLoopNum = Ops::Base::CeilDiv(myRows, td_->baseS);
    for (uint64_t sLoop = 0; sLoop < sLoopNum; sLoop++) {
        uint64_t rowOff = sLoop * td_->baseS;
        uint64_t rows = (rowOff + td_->baseS > myRows) ? (myRows - rowOff) : td_->baseS;

        LocalTensor<IDS_T> idsLocal = idsQue_.AllocTensor<IDS_T>();
        UnsortedSegment::CopyIn(idsLocal, idsGm_, rowStart + rowOff, 1, rows, 0);
        idsQue_.EnQue<IDS_T>(idsLocal);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        idsLocal = idsQue_.DeQue<IDS_T>();

        LocalTensor<IDS_T> reduceDst = reduceBuf_.Get<IDS_T>();
        bool hasValidIds = !UnsortedSegment::AllIdsInvalidVectorized(
            idsLocal, static_cast<uint32_t>(rows), static_cast<int64_t>(td_->outputOuterDim), reduceDst);
        if (!hasValidIds) {
            idsQue_.FreeTensor(idsLocal);
            continue;
        }
        anyValidIds = true;

        LocalTensor<X_T> xLocal = xQue_.AllocTensor<X_T>();
        if (innerAlign_ == td_->innerDim) {
            UnsortedSegment::CopyIn(xLocal, xGm_, (rowStart + rowOff) * td_->innerDim, 1, rows * td_->innerDim, 0, 0);
        } else {
            for (uint64_t r = 0; r < rows; r++) {
                UnsortedSegment::CopyIn(xLocal[r * innerAlign_], xGm_, (rowStart + rowOff + r) * td_->innerDim, 1,
                                        td_->innerDim, 0, 0);
            }
        }
        xQue_.EnQue<X_T>(xLocal);

        xLocal = xQue_.DeQue<X_T>();
        for (uint64_t i = 0; i < rows; i++) {
            int64_t dstIdx = static_cast<int64_t>(idsLocal.GetValue(i));
            if (dstIdx < 0 || dstIdx >= static_cast<int64_t>(td_->outputOuterDim)) {
                continue;
            }
            uint64_t dstOffset = static_cast<uint64_t>(dstIdx) * innerAlign_;
            VectorComputeFunc()(yLocal, xLocal, dstOffset, i * innerAlign_, td_->innerDim);
        }
        xQue_.FreeTensor(xLocal);
        idsQue_.FreeTensor(idsLocal);
    }

    yQue_.EnQue<X_T>(yLocal);
    yLocal = yQue_.DeQue<X_T>();
    UnsortedSegment::CopyOut(wsGm_, yLocal, blockId * ySize_, 1, ySize_, 0, 0);
    yQue_.FreeTensor(yLocal);
    anyValidGm_ = anyValidIds;
}

template <typename X_T, typename IDS_T, typename InitValueType, typename VectorComputeFunc>
__aicore__ inline void KernelProdInputPartition<X_T, IDS_T, InitValueType, VectorComputeFunc>::WriteInitOutput()
{
    uint64_t blockId = GetBlockIdx();
    uint64_t mergeStart = blockId * td_->mergeNormNum;
    if (mergeStart >= ySize_) {
        return;
    }
    uint64_t mergeEnd = mergeStart + td_->mergeNormNum;
    if (mergeEnd > ySize_) {
        mergeEnd = ySize_;
    }

    for (uint64_t off = mergeStart; off < mergeEnd; off += td_->mergeChunk) {
        uint64_t len = (off + td_->mergeChunk > mergeEnd) ? (mergeEnd - off) : td_->mergeChunk;

        LocalTensor<X_T> initLocal = mergeOutQue_.AllocTensor<X_T>();
        Duplicate(initLocal, InitValueType::Get(), len);
        mergeOutQue_.EnQue<X_T>(initLocal);
        initLocal = mergeOutQue_.DeQue<X_T>();
        if (innerAlign_ == td_->innerDim) {
            UnsortedSegment::CopyOut(yGm_, initLocal, off, 1, len, 0, 0);
        } else {
            uint64_t rowBegin = off / innerAlign_;
            uint64_t rowCnt = len / innerAlign_;
            for (uint64_t r = 0; r < rowCnt; r++) {
                UnsortedSegment::CopyOut(yGm_, initLocal[r * innerAlign_], (rowBegin + r) * td_->innerDim, 1,
                                         td_->innerDim, 0, 0);
            }
        }
        mergeOutQue_.FreeTensor(initLocal);
    }
}

template <typename X_T, typename IDS_T, typename InitValueType, typename VectorComputeFunc>
__aicore__ inline void KernelProdInputPartition<X_T, IDS_T, InitValueType, VectorComputeFunc>::MergePartial()
{
    uint64_t blockId = GetBlockIdx();
    uint64_t mergeStart = blockId * td_->mergeNormNum;
    if (mergeStart >= ySize_) {
        return;
    }
    uint64_t mergeEnd = mergeStart + td_->mergeNormNum;
    if (mergeEnd > ySize_) {
        mergeEnd = ySize_;
    }

    for (uint64_t off = mergeStart; off < mergeEnd; off += td_->mergeChunk) {
        uint64_t len = (off + td_->mergeChunk > mergeEnd) ? (mergeEnd - off) : td_->mergeChunk;

        LocalTensor<X_T> acc = mergeOutQue_.AllocTensor<X_T>();
        UnsortedSegment::CopyIn(acc, wsGm_, off, 1, len, 0, 0);
        mergeOutQue_.EnQue<X_T>(acc);
        acc = mergeOutQue_.DeQue<X_T>();

        for (uint64_t c = 1; c < td_->partCoreNum; c++) {
            LocalTensor<X_T> part = mergeQue_.AllocTensor<X_T>();
            UnsortedSegment::CopyIn(part, wsGm_, c * ySize_ + off, 1, len, 0, 0);
            mergeQue_.EnQue<X_T>(part);
            part = mergeQue_.DeQue<X_T>();
            VectorComputeFunc()(acc, part, static_cast<uint64_t>(0), static_cast<uint64_t>(0), len);
            mergeQue_.FreeTensor(part);
        }

        mergeOutQue_.EnQue<X_T>(acc);
        acc = mergeOutQue_.DeQue<X_T>();
        if (innerAlign_ == td_->innerDim) {
            UnsortedSegment::CopyOut(yGm_, acc, off, 1, len, 0, 0);
        } else {
            uint64_t rowBegin = off / innerAlign_;
            uint64_t rowCnt = len / innerAlign_;
            for (uint64_t r = 0; r < rowCnt; r++) {
                UnsortedSegment::CopyOut(yGm_, acc[r * innerAlign_], (rowBegin + r) * td_->innerDim, 1, td_->innerDim,
                                         0, 0);
            }
        }
        mergeOutQue_.FreeTensor(acc);
    }
}

template <typename X_T, typename IDS_T, typename InitValueType, typename VectorComputeFunc>
__aicore__ inline void KernelProdInputPartition<X_T, IDS_T, InitValueType, VectorComputeFunc>::Process()
{
    ComputePartial();
    flagGm_.SetValue(static_cast<uint32_t>(GetBlockIdx()) * IP_FLAG_STRIDE, anyValidGm_ ? 1 : 0);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        flagGm_);
    SyncAll();
    LocalTensor<int32_t> flagUb = flagBuf_.Get<int32_t>();
    UnsortedSegment::CopyIn(flagUb, flagGm_, 0, 1, flagCount_ * IP_FLAG_STRIDE, 0, 0);
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    bool globalAnyValid = false;
    for (uint32_t b = 0; b < flagCount_; b++) {
        if (flagUb.GetValue(b * IP_FLAG_STRIDE) != 0) {
            globalAnyValid = true;
            break;
        }
    }
    if (!globalAnyValid) {
        WriteInitOutput();
        return;
    }
    MergePartial();
}

} // namespace UnsortedSegmentProd

#endif
