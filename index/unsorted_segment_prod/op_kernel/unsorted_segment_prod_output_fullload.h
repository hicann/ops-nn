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
 * \file unsorted_segment_prod_output_fullload.h
 * \brief unsorted_segment_prod_output_fullload.h
 */

#ifndef UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_H
#define UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "./unsorted_segment_prod_base.h"

namespace UnsortedSegmentProd {
constexpr uint32_t OUTFL_MERGE_THREAD_DIM = 256;

template <typename TX>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_DIM_LAUNCH_BOUND) inline void SimtOutFlMergeProd(
    __ubuf__ const TX* midResUb, __gm__ TX* outputGm, const uint32_t n)
{
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const TX val = midResUb[i];
        if constexpr (sizeof(TX) >= 4) {
            while (true) {
                TX oldVal = outputGm[i];
                TX newVal = oldVal * val;
                TX cas = Simt::AtomicCas(outputGm + i, oldVal, newVal);
                if (cas == oldVal) {
                    break;
                }
            }
        } else {
            uint32_t pairIndex = i & ~1U;
            bool isLow = (i & 1U) == 0;
            while (true) {
                TX oldVal1 = outputGm[pairIndex];
                TX oldVal2 = outputGm[pairIndex + 1];
                uint32_t oldU32;
                (reinterpret_cast<TX*>(&oldU32))[0] = oldVal1;
                (reinterpret_cast<TX*>(&oldU32))[1] = oldVal2;
                TX oldVal = isLow ? oldVal1 : oldVal2;
                TX neighbor = isLow ? oldVal2 : oldVal1;
                TX newVal = oldVal * val;
                uint32_t newU32;
                if (isLow) {
                    (reinterpret_cast<TX*>(&newU32))[0] = newVal;
                    (reinterpret_cast<TX*>(&newU32))[1] = neighbor;
                } else {
                    (reinterpret_cast<TX*>(&newU32))[0] = neighbor;
                    (reinterpret_cast<TX*>(&newU32))[1] = newVal;
                }
                uint32_t casU32 = Simt::AtomicCas(reinterpret_cast<__gm__ uint32_t*>(outputGm + pairIndex), oldU32,
                                                  newU32);
                if (casU32 == oldU32) {
                    break;
                }
            }
        }
    }
}

template <typename TX, typename Index, typename SimtGatherFunc, typename InitValueType, typename VectorComputeFunc>
class KernelUnsortedSegmentProdOutFl {
public:
    __aicore__ inline KernelUnsortedSegmentProdOutFl(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output,
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

        InitGmProd<TX, InitGmOneValue<TX>>(output, static_cast<uint64_t>(outputOuterDimSize) * innerDimSize);

        if (inputOuterDimSize * innerDimSize == 0) {
            return;
        }

        uint32_t currUbRowNum = (rowNumOneUb < maxIndexNum) ? rowNumOneUb : maxIndexNum;
        pipe_->InitBuffer(inQueueX, UnsortedSegment::BUFFER_ADD_NUM,
                          UnsortedSegment::RoundUpOneBlock(innerDimSize * sizeof(TX) * currUbRowNum));
        pipe_->InitBuffer(inQueueIndex, UnsortedSegment::BUFFER_ADD_NUM,
                          UnsortedSegment::RoundUpOneBlock(sizeof(Index) * currUbRowNum));
        pipe_->InitBuffer(tmpBuf, UnsortedSegment::RoundUpOneBlock(innerDimSize * sizeof(TX) * outputOuterDimSize) *
                                      UnsortedSegment::ROW_NUM);
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

        uint32_t oneRowOutNum = innerDimSize * outputOuterDimSize;
        uint32_t oneRowOutNumAlign = UnsortedSegment::RoundUpOneBlock(oneRowOutNum * sizeof(TX)) / sizeof(TX);
        uint32_t bufAlign32 = oneRowOutNumAlign * UnsortedSegment::ROW_NUM;
        LocalTensor<TX> midRes = tmpBuf.Get<TX>();
        Duplicate(midRes, InitValueType::Get(), bufAlign32);
        PipeBarrier<PIPE_V>();
        bool anyValidIds = false;
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
            anyValidIds = true;
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
        if (!anyValidIds) {
            return;
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
        DataSyncBarrier<MemDsbT::UB>();
        __ubuf__ TX* mergeMidResPtr = (__ubuf__ TX*)midRes.GetPhyAddr();
        __gm__ TX* outputPtr = (__gm__ TX*)outputGm.GetPhyAddr();
        asc_vf_call<SimtOutFlMergeProd<TX>>(dim3{OUTFL_MERGE_THREAD_DIM}, mergeMidResPtr, outputPtr, oneRowOutNum);
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, UnsortedSegment::BUFFER_ADD_NUM> inQueueX;
    TQue<QuePosition::VECIN, UnsortedSegment::BUFFER_ADD_NUM> inQueueIndex;
    AscendC::GlobalTensor<TX> xGm;
    AscendC::GlobalTensor<TX> outputGm;
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
};
} // namespace UnsortedSegmentProd

#endif // UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_H
