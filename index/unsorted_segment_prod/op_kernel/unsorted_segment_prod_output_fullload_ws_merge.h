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
 * \file unsorted_segment_prod_output_fullload_ws_merge.h
 * \brief unsorted_segment_prod_output_fullload_ws_merge.h
 */

#ifndef UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_WS_MERGE_H
#define UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_WS_MERGE_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "./unsorted_segment_prod_base.h"

namespace UnsortedSegmentProd {
constexpr uint32_t OFW_MERGE_CHUNK = 4096;

template <typename TX, typename Index, typename SimtGatherFunc, typename InitValueType, typename VectorComputeFunc>
class KernelUnsortedSegmentProdOutFlWs {
public:
    __aicore__ inline KernelUnsortedSegmentProdOutFlWs(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output, GM_ADDR workspace,
                                const UnsortedSegment::UnsortedSegmentOutFlTilingData* tiling)
    {
        xGm.SetGlobalBuffer((__gm__ TX*)(x));
        segmentIdsGm.SetGlobalBuffer((__gm__ Index*)(segmentIds));
        outputGm.SetGlobalBuffer((__gm__ TX*)(output));

        inputOuterDimSize = static_cast<uint32_t>(tiling->inputOuterDim);
        outputOuterDimSize = static_cast<uint32_t>(tiling->outputOuterDim);
        innerDimSize = static_cast<uint32_t>(tiling->innerDim);
        maxIndexNum = static_cast<uint32_t>(tiling->maxIndexNum);
        oneCoreUbLoop = static_cast<uint32_t>(tiling->oneCoreUbLoopTimes);
        rowNumOneUb = static_cast<uint32_t>(tiling->rowNumUb);
        innerDimSizeAlign = UnsortedSegment::RoundUpOneBlock(innerDimSize * sizeof(TX)) / sizeof(TX);

        oneRowOutNum = innerDimSize * outputOuterDimSize;
        wsStride = UnsortedSegment::RoundUpOneBlock(oneRowOutNum * sizeof(TX)) / sizeof(TX);

        wsGm.SetGlobalBuffer((__gm__ TX*)(workspace));

        uint32_t currUbRowNum = (rowNumOneUb < maxIndexNum) ? rowNumOneUb : maxIndexNum;
        pipe_->InitBuffer(inQueueX, UnsortedSegment::BUFFER_ADD_NUM,
                          UnsortedSegment::RoundUpOneBlock(innerDimSize * sizeof(TX) * currUbRowNum));
        pipe_->InitBuffer(inQueueIndex, UnsortedSegment::BUFFER_ADD_NUM,
                          UnsortedSegment::RoundUpOneBlock(sizeof(Index) * currUbRowNum));
        pipe_->InitBuffer(tmpBuf, UnsortedSegment::RoundUpOneBlock(innerDimSize * sizeof(TX) * outputOuterDimSize) *
                                      UnsortedSegment::ROW_NUM);
        mergeChunk = (oneRowOutNum < OFW_MERGE_CHUNK) ? oneRowOutNum : OFW_MERGE_CHUNK;
        pipe_->InitBuffer(mergeQue, UnsortedSegment::BUFFER_ADD_NUM, mergeChunk * sizeof(TX));
        pipe_->InitBuffer(mergeOutQue, 1, mergeChunk * sizeof(TX));
        pipe_->InitBuffer(reduceBuf, UnsortedSegment::AllIdsInvalidReduceBufBytes<Index>(maxIndexNum));
    }

    __aicore__ inline void CopyInIndex(uint64_t offset, uint32_t extent)
    {
        LocalTensor<Index> indexLocal = inQueueIndex.AllocTensor<Index>();
        DataCopyExtParams extParams{static_cast<uint16_t>(1), static_cast<uint32_t>(extent * sizeof(Index)), 0, 0, 0};
        DataCopyPadExtParams<Index> padParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, segmentIdsGm[offset], extParams, padParams);
        inQueueIndex.EnQue(indexLocal);
    }

    __aicore__ inline void CopyInX(uint64_t offset, uint32_t extent)
    {
        LocalTensor<TX> xLocal = inQueueX.AllocTensor<TX>();
        DataCopyPadExtParams<TX> padParams{false, 0, 0, static_cast<TX>(0)};
        DataCopyExtParams dataCopyParam{static_cast<uint16_t>(1),
                                        static_cast<uint32_t>(extent * innerDimSize * sizeof(TX)), 0, 0, 0};
        DataCopyPad(xLocal, xGm[offset], dataCopyParam, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Process()
    {
        if (inputOuterDimSize * innerDimSize == 0) {
            return;
        }
        if (GetBlockIdx() >= GetBlockNum()) {
            return;
        }

        uint32_t oneRowOutNumAlign = UnsortedSegment::RoundUpOneBlock(oneRowOutNum * sizeof(TX)) / sizeof(TX);
        uint32_t bufAlign32 = oneRowOutNumAlign * UnsortedSegment::ROW_NUM;
        LocalTensor<TX> midRes = tmpBuf.Get<TX>();
        Duplicate(midRes, InitValueType::Get(), bufAlign32);
        PipeBarrier<PIPE_V>();
        for (uint32_t loop = 0; loop < oneCoreUbLoop; ++loop) {
            int64_t startIndex = GetBlockIdx() * maxIndexNum + loop * rowNumOneUb;
            int64_t start = startIndex * innerDimSize;
            int64_t remain;
            if (GetBlockIdx() == GetBlockNum() - 1) {
                remain = inputOuterDimSize - startIndex;
            } else {
                remain = maxIndexNum - loop * rowNumOneUb;
            }
            int64_t currTileIndex = (rowNumOneUb < remain) ? rowNumOneUb : remain;
            int64_t needIndexOneUb = (currTileIndex < maxIndexNum) ? currTileIndex : maxIndexNum;
            if (remain > 0) {
                CopyInIndex(startIndex, needIndexOneUb);
            } else {
                break;
            }
            event_t eventIDMTE2ToV = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            LocalTensor<Index> indexUb = inQueueIndex.DeQue<Index>();
            LocalTensor<Index> reduceDst = reduceBuf.Get<Index>();
            bool hasValidIds = !UnsortedSegment::AllIdsInvalidVectorized(
                indexUb, static_cast<uint32_t>(needIndexOneUb), static_cast<int64_t>(outputOuterDimSize), reduceDst);
            if (!hasValidIds) {
                inQueueIndex.FreeTensor(indexUb);
                continue;
            }
            CopyInX(start, needIndexOneUb);
            LocalTensor<TX> xUbLocal = inQueueX.DeQue<TX>();
            __ubuf__ TX* xUbLocalPtr = (__ubuf__ TX*)xUbLocal.GetPhyAddr();
            __ubuf__ TX* midResPtr = (__ubuf__ TX*)midRes.GetPhyAddr();
            DataSyncBarrier<MemDsbT::UB>();
            asc_vf_call<UnsortedSegment::SimtGatherValue<TX, Index, SimtGatherFunc>>(
                dim3{innerDimSize, UnsortedSegment::ROW_NUM}, midResPtr, xUbLocalPtr,
                (__ubuf__ Index*)indexUb.GetPhyAddr(), outputOuterDimSize, innerDimSize, needIndexOneUb,
                oneRowOutNumAlign, UnsortedSegment::ROW_NUM);
            inQueueX.FreeTensor(xUbLocal);
            inQueueIndex.FreeTensor(indexUb);
        }
        int32_t stepStride = UnsortedSegment::ROW_NUM / UnsortedSegment::TWO;
        for (int32_t i = 0; i < UnsortedSegment::HALFTIME; ++i) {
            for (int32_t first = 0; first < stepStride; ++first) {
                int32_t oneOffset = first * oneRowOutNumAlign;
                int32_t twoOffset = (first + stepStride) * oneRowOutNumAlign;
                VectorComputeFunc()(midRes, midRes, oneOffset, twoOffset, oneRowOutNumAlign);
            }
            stepStride >>= 1;
        }

        event_t eventIdVToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        UnsortedSegment::CopyOut(wsGm, midRes, static_cast<uint64_t>(GetBlockIdx()) * wsStride, 1, oneRowOutNum, 0, 0);
        AscendC::DataCacheCleanAndInvalid<TX, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
            wsGm);
        SyncAll();

        MergeWsPartial();
    }

private:
    __aicore__ inline void MergeWsPartial()
    {
        uint32_t blockNum = GetBlockNum();
        uint64_t mergeNorm = (oneRowOutNum + blockNum - 1) / blockNum;
        uint64_t mergeStart = static_cast<uint64_t>(GetBlockIdx()) * mergeNorm;
        if (mergeStart >= oneRowOutNum) {
            return;
        }
        uint64_t mergeEnd = mergeStart + mergeNorm;
        if (mergeEnd > oneRowOutNum) {
            mergeEnd = oneRowOutNum;
        }
        for (uint64_t off = mergeStart; off < mergeEnd; off += mergeChunk) {
            uint64_t len = (off + mergeChunk > mergeEnd) ? (mergeEnd - off) : mergeChunk;
            LocalTensor<TX> acc = mergeOutQue.AllocTensor<TX>();
            UnsortedSegment::CopyIn(acc, wsGm, off, 1, static_cast<uint32_t>(len), 0, 0);
            mergeOutQue.EnQue<TX>(acc);
            acc = mergeOutQue.DeQue<TX>();
            for (uint32_t c = 1; c < blockNum; ++c) {
                LocalTensor<TX> part = mergeQue.AllocTensor<TX>();
                UnsortedSegment::CopyIn(part, wsGm, static_cast<uint64_t>(c) * wsStride + off, 1,
                                        static_cast<uint32_t>(len), 0, 0);
                mergeQue.EnQue<TX>(part);
                part = mergeQue.DeQue<TX>();
                VectorComputeFunc()(acc, part, static_cast<uint64_t>(0), static_cast<uint64_t>(0),
                                    static_cast<uint64_t>(len));
                mergeQue.FreeTensor(part);
            }
            mergeOutQue.EnQue<TX>(acc);
            acc = mergeOutQue.DeQue<TX>();
            UnsortedSegment::CopyOut(outputGm, acc, off, 1, static_cast<uint32_t>(len), 0, 0);
            mergeOutQue.FreeTensor(acc);
        }
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, UnsortedSegment::BUFFER_ADD_NUM> inQueueX;
    TQue<QuePosition::VECIN, UnsortedSegment::BUFFER_ADD_NUM> inQueueIndex;
    TQue<QuePosition::VECIN, UnsortedSegment::BUFFER_ADD_NUM> mergeQue;
    TQue<QuePosition::VECOUT, 1> mergeOutQue;
    AscendC::GlobalTensor<TX> xGm;
    AscendC::GlobalTensor<TX> outputGm;
    AscendC::GlobalTensor<TX> wsGm;
    AscendC::GlobalTensor<Index> segmentIdsGm;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> reduceBuf;

    uint32_t inputOuterDimSize{1};
    uint32_t outputOuterDimSize{1};
    uint32_t innerDimSize{1};
    uint32_t innerDimSizeAlign{1};
    uint32_t maxIndexNum{1};
    uint32_t oneCoreUbLoop{1};
    uint32_t rowNumOneUb{1};
    uint32_t oneRowOutNum{1};
    uint32_t wsStride{1};
    uint32_t mergeChunk{1};
};
} // namespace UnsortedSegmentProd

#endif // UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_WS_MERGE_H
