/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNSORTED_SEGMENT_OUT_FULL_LOAD_H
#define UNSORTED_SEGMENT_OUT_FULL_LOAD_H

#include "unsorted_segment_base.h"

namespace UnsortedSegment
{
using namespace AscendC;

template <typename TX, typename Index, uint8_t Mode>
class KernelUnsortedSegmentFL
{
public:
    __aicore__ inline KernelUnsortedSegmentFL(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output,
                                const UnsortedSegmentOutFlTilingData* tiling)
    {
        xGm.SetGlobalBuffer((__gm__ TX*)(x));
        segmentIdsGm.SetGlobalBuffer((__gm__ Index*)(segmentIds));
        outputGm.SetGlobalBuffer((__gm__ TX*)(output));

        this->inputOuterDimSize = tiling->inputOuterDim;
        this->outputOuterDimSize = tiling->outputOuterDim;
        this->innerDimSize = tiling->innerDim;
        this->maxIndexNum = tiling->maxIndexNum;
        this->oneCoreUbLoop = tiling->oneCoreUbLoopTimes;
        this->rowNumOneUb = tiling->rowNumUb;
        this->innerDimSizeAlign = RoundUpOneBlock(innerDimSize * sizeof(TX)) / sizeof(TX);

        if (inputOuterDimSize * innerDimSize == 0) {
            InitGm<TX, Mode>(output, outputOuterDimSize * innerDimSize);
            return;
        }

        InitGm<TX, Mode>(output, outputOuterDimSize * innerDimSize);

        uint32_t currUbRowNum = (rowNumOneUb < maxIndexNum) ? rowNumOneUb : maxIndexNum;
        pipe_->InitBuffer(inQueueX, BUFFER_ADD_NUM, RoundUpOneBlock(innerDimSize * sizeof(TX) * currUbRowNum));
        pipe_->InitBuffer(inQueueIndex, BUFFER_ADD_NUM, RoundUpOneBlock(sizeof(Index) * currUbRowNum));
        pipe_->InitBuffer(tmpBuf, RoundUpOneBlock(innerDimSize * sizeof(TX) * outputOuterDimSize) * ROW_NUM);
    }

    __aicore__ inline void CopyInIndex(uint64_t offset, uint32_t extent)
    {
        LocalTensor<Index> indexLocal = inQueueIndex.AllocTensor<Index>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(1),                       // blockCount
            static_cast<uint32_t>(extent * sizeof(Index)),  // blockLen
            0,                                              // srcStride
            0,                                              // dstStride
            0                                               // rsv
        };
        DataCopyPadExtParams<Index> padParams{
            false,  // isPad
            0,      // leftPadding
            0,      // rightPadding
            0       // paddingValue
        };
        DataCopyPad(indexLocal, segmentIdsGm[offset], extParams, padParams);
        inQueueIndex.EnQue(indexLocal);
    }

    __aicore__ inline void CopyInX(uint64_t offset, uint32_t extent)
    {
        LocalTensor<TX> xLocal = inQueueX.AllocTensor<TX>();
        DataCopyPadExtParams<TX> padParams;
        padParams.isPad = false;
        padParams.rightPadding = 0;
        padParams.paddingValue = static_cast<TX>(0);
        DataCopyExtParams dataCopyParam;
        dataCopyParam.blockCount = 1;
        dataCopyParam.blockLen = extent * innerDimSize * sizeof(TX);
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 0;
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

        __gm__ TX* output = (__gm__ TX*)outputGm.GetPhyAddr();
        uint32_t oneRowOutNum = innerDimSize * outputOuterDimSize;
        uint32_t oneRowOutNumAlign = RoundUpOneBlock(oneRowOutNum * sizeof(TX)) / sizeof(TX);
        uint32_t bufAlign32 = oneRowOutNumAlign * ROW_NUM;
        LocalTensor<TX> midRes = tmpBuf.Get<TX>();
        if constexpr (Mode == 0) {
            Duplicate(midRes, GetDtypeMax<TX>(), bufAlign32);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t loop = 0; loop < oneCoreUbLoop; ++loop) {
            int64_t startIndex = block_idx * maxIndexNum + loop * rowNumOneUb;
            int64_t start = startIndex * innerDimSize;
            int64_t remain;
            if (block_idx == block_num - 1) {
                remain = inputOuterDimSize - startIndex;
            } else {
                remain = maxIndexNum - loop * rowNumOneUb;
            }
            int64_t currTileIndex = (rowNumOneUb < remain) ? rowNumOneUb : remain;
            int64_t needIndexOneUb = (currTileIndex < maxIndexNum) ? currTileIndex : maxIndexNum;
            if (remain > 0) {
                CopyInX(start, needIndexOneUb);
                CopyInIndex(startIndex, needIndexOneUb);
            } else {
                break;
            }
            LocalTensor<TX> xUbLocal = inQueueX.DeQue<TX>();
            LocalTensor<Index> indexUb = inQueueIndex.DeQue<Index>();
            __local_mem__ TX* xUbLocalPtr = (__local_mem__ TX*)xUbLocal.GetPhyAddr();
            __local_mem__ TX* midResPtr = (__local_mem__ TX*)midRes.GetPhyAddr();
            DataSyncBarrier<MemDsbT::UB>();
            AscendC::Simt::VF_CALL<SimtGatherValue<TX, Index, Mode>>(
                AscendC::Simt::Dim3{static_cast<uint32_t>(innerDimSize), static_cast<uint32_t>(ROW_NUM)},
                midResPtr, xUbLocalPtr, (__ubuf__ Index*)indexUb.GetPhyAddr(), outputOuterDimSize, innerDimSize,
                needIndexOneUb, oneRowOutNumAlign);
            inQueueX.FreeTensor(xUbLocal);
            inQueueIndex.FreeTensor(indexUb);
        }
        // rownum lines reduces to one line
        int32_t stepStride = ROW_NUM / TWO;
        for (int32_t i = 0; i < HALFTIME; ++i) {
            for (int32_t first = 0; first < stepStride; ++first) {
                int32_t OneOffset = first * oneRowOutNumAlign;
                int32_t TwoOffset = (first + stepStride) * oneRowOutNumAlign;
                if constexpr (Mode == 0) {
                    AscendC::Min(midRes[OneOffset], midRes[OneOffset], midRes[TwoOffset], oneRowOutNumAlign);
                }
            }
            stepStride /= TWO;
        }
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        if constexpr (Mode == 0) {
            SetAtomicMin<TX>();
        }
        DataCopyExtParams extParams{
            static_cast<uint16_t>(1),         // blockCount
            static_cast<uint32_t>(oneRowOutNum * sizeof(TX)),  // blockLen
            0,                                                 // srcStride
            0,                                                 // dstStride
            0                                                  // rsv
        };
        DataCopyPad(outputGm, midRes, extParams);
        SetAtomicNone();
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUFFER_ADD_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_ADD_NUM> inQueueIndex;
    AscendC::GlobalTensor<TX> xGm;
    AscendC::GlobalTensor<TX> outputGm;
    AscendC::GlobalTensor<Index> segmentIdsGm;
    TBuf<TPosition::VECCALC> tmpBuf;

    uint32_t inputOuterDimSize{1};
    uint32_t outputOuterDimSize{1};
    uint32_t innerDimSize{1};
    uint32_t innerDimSizeAlign{1};
    uint32_t maxIndexNum{1};
    uint32_t oneCoreUbLoop{1};
    uint32_t rowNumOneUb{1};
};
}  // namespace UnsortedSegmentFL
#endif
