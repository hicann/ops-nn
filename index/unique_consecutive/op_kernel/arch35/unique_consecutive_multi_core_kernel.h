/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unique_consecutive_multi_core_kernel.h
 * \brief
 */

#ifndef UNIQUE_CONSECUTIVE_MULTI_CORE_KERNEL_H
#define UNIQUE_CONSECUTIVE_MULTI_CORE_KERNEL_H

#include "unique_consecutive_helper.h"

constexpr int64_t MAGIC_GM_PAGE_SIZE = 128;
constexpr int32_t BUFFER_NUM_MULTI = 1;

template <typename T, typename T1, typename DTYPE_COUNT, bool COUNT_OUT, bool ISINT64>
class UniqueConsecutiveMutilCoreKerenl {
public:
    __aicore__ inline UniqueConsecutiveMutilCoreKerenl(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR count, GM_ADDR shape_out, GM_ADDR workspace,
                                const UniqueConsecutiveTilingData* tilingData)
    {
        coreIdx_ = GetBlockIdx();
        isFinalCore_ = (coreIdx_ == GetBlockNum() - 1);

        coreTileLength_ = (isFinalCore_) ? tilingData->tileLengthTailCore : tilingData->tileLengthPerCore;
        totalSize_ = tilingData->totalSize;
        baseCount_ = 1 + coreIdx_ * tilingData->tileLengthPerCore;
        tileLengthPerCore_ = tilingData->tileLengthPerCore;
        idxQueueSize_ = tilingData->idxQueueSize;
        adjUbTileLength_ = tilingData->adjUbTileLength;
        ubTileLength_ = adjUbTileLength_ - 1;

        xGm_.SetGlobalBuffer((__gm__ T*)(x) + tilingData->tileLengthPerCore * coreIdx_);
        yGm_.SetGlobalBuffer((__gm__ T*)(y));

        if (isFinalCore_) {
            shapeGm_.SetGlobalBuffer((__gm__ uint64_t*)shape_out);
        }

        pipe_->InitBuffer(xQueue_, BUFFER_NUM_MULTI, tilingData->valueQueueSize);
        pipe_->InitBuffer(yQueue_, BUFFER_NUM_MULTI, tilingData->valueQueueSize);

        countGm_.SetGlobalBuffer((__gm__ int32_t*)(count));
        if constexpr (COUNT_OUT) {
            pipe_->InitBuffer(countQueue_, BUFFER_NUM_MULTI, tilingData->countQueueSize);
            pipe_->InitBuffer(idxQueue_, BUFFER_NUM_MULTI, tilingData->idxQueueSize);
        }

        pipe_->InitBuffer(collectingCntBuf_, tilingData->collectingCntBufSize);
        pipe_->InitBuffer(offsetCntBuf_, tilingData->offsetCntBufSize);
        pipe_->InitBuffer(prevIdxBuf_, tilingData->prevIdxBufSize);
        pipe_->InitBuffer(shapeBuf_, tilingData->shapeBufSize);

        coreCollectWorkspaceGm_.SetGlobalBuffer((__gm__ int64_t*)(workspace));
        idxGm_.SetGlobalBuffer((__gm__ int32_t*)(idx) + tilingData->tileLengthPerCore * coreIdx_);
    }

    __aicore__ inline void Process()
    {
        if (isFinalCore_) {
            LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();
            Duplicate(shapeTensor, (uint64_t)1, SHAPE_LEN);
        }

        int64_t coreCollectNums = 0;
        int64_t lastUniqueIdx = 0;
        ProcessCollecting(coreCollectNums, lastUniqueIdx);

        PipeBarrier<PIPE_ALL>();
        SyncAll();

        if (coreCollectNums == 0) {
            return;
        }

        int64_t yOffset = 0;
        int64_t prevTailCount = 0;
        FindOffset(yOffset, prevTailCount);

        ProcessMerging(yOffset, prevTailCount);

        if (isFinalCore_) {
            coreCollectNums += yOffset;
            CopyOutShape(coreCollectNums, coreCollectNums);
        }
    }

    __aicore__ inline void ProcessMerging(int64_t copyOutOffset, int64_t prevTailCount)
    {
        int64_t ubLoops = CEIL_DIV(coreTileLength_, ubTileLength_) - 1;
        int64_t ubMainLength = ubTileLength_ + 1;
        int64_t ubTailLength = coreTileLength_ - ubTileLength_ * ubLoops;
        ubTailLength = (isFinalCore_) ? ubTailLength : ubTailLength + 1;

        int64_t offsetXGm = 0;
        int64_t gatherCnt = 0;
        int64_t innerBaseCount = baseCount_;

        for (int64_t i = 0; i < ubLoops; ++i) {
            CopyInX(offsetXGm, ubMainLength);
            CollectUniqueAndCount<false>(innerBaseCount, ubMainLength, gatherCnt, i, prevTailCount, copyOutOffset);
            offsetXGm += ubTileLength_;
            copyOutOffset += gatherCnt;
            innerBaseCount += ubTileLength_;
        }

        CopyInX(offsetXGm, ubTailLength);
        if (isFinalCore_) {
            CollectUniqueAndCount<true>(innerBaseCount, ubTailLength, gatherCnt, ubLoops, prevTailCount, copyOutOffset);
        } else {
            CollectUniqueAndCount<false>(innerBaseCount, ubTailLength, gatherCnt, ubLoops, prevTailCount,
                                         copyOutOffset);
        }
    }

    __aicore__ inline void ProcessCollecting(int64_t& coreCollectNums, int64_t& lastUniqueIdx)
    {
        int64_t ubLoops = CEIL_DIV(coreTileLength_, ubTileLength_) - 1;
        int64_t ubMainLength = ubTileLength_ + 1;
        int64_t ubTailLength = coreTileLength_ - ubTileLength_ * ubLoops;
        ubTailLength = (isFinalCore_) ? ubTailLength : ubTailLength + 1;

        int64_t offsetXGm = 0;
        int64_t gatherCnt = 0;
        int64_t innerBaseCount = baseCount_;

        for (int64_t i = 0; i < ubLoops; ++i) {
            CopyInX(offsetXGm, ubMainLength);
            CollectNumsAndLastIdx<false>(innerBaseCount, ubMainLength, gatherCnt, lastUniqueIdx, i);
            offsetXGm += ubTileLength_;
            coreCollectNums += gatherCnt;
            innerBaseCount += ubTileLength_;
        }

        CopyInX(offsetXGm, ubTailLength);
        if (isFinalCore_) {
            CollectNumsAndLastIdx<true>(innerBaseCount, ubTailLength, gatherCnt, lastUniqueIdx, ubLoops);
        } else {
            CollectNumsAndLastIdx<false>(innerBaseCount, ubTailLength, gatherCnt, lastUniqueIdx, ubLoops);
        }
        coreCollectNums += gatherCnt;

        CopyOutCnt2Workspace(coreCollectNums, lastUniqueIdx);
    }

    template <bool TAIL_LOOP>
    __aicore__ inline void CollectNumsAndLastIdx(int64_t innerBaseCount, int64_t nums, int64_t& gatherCnt,
                                                 int64_t& lastUniqueIdx, int64_t i)
    {
        LocalTensor<T> xLocal = xQueue_.template DeQue<T>();
        LocalTensor<int32_t> outTensor = yQueue_.template AllocTensor<int32_t>();
        uint64_t reduceCntValue = -1;
        CountAdjacentNe<T, T1, TAIL_LOOP>(outTensor, xLocal, nums, reduceCntValue);
        gatherCnt = reduceCntValue;
        yQueue_.FreeTensor(outTensor);

        if constexpr (COUNT_OUT) {
            if (gatherCnt > 0) {
                if constexpr (ISINT64) {
                    int64_t offset = coreIdx_ * tileLengthPerCore_ + i * ubTileLength_;
                    LocalTensor<int32_t> idxLocal = idxQueue_.template AllocTensor<int32_t>();
                    uint64_t reduceCntIdx = -1;
                    uint64_t alignPosition = idxQueueSize_ / MIDPOINT_DIVIDER / sizeof(int32_t);
                    CollectPostUniqueIdx<T, T1, TAIL_LOOP>(idxLocal, xLocal, 1, nums, nums, reduceCntIdx,
                                                           alignPosition);
                    LocalTensor<int64_t> idxLocalInt64 = idxLocal.template ReinterpretCast<int64_t>();
                    CastAndAddsOffsets(idxLocalInt64, idxLocal, reduceCntIdx, alignPosition, offset);
                    SimpleNativePipeSync<HardEvent::V_S>();
                    lastUniqueIdx = idxLocalInt64.GetValue(gatherCnt - 1);
                    idxQueue_.FreeTensor(idxLocal);
                } else {
                    LocalTensor<int32_t> idxLocal = idxQueue_.template AllocTensor<int32_t>();
                    uint64_t reduceCntIdx = -1;
                    CollectPostUniqueIdx<T, T1, TAIL_LOOP>(idxLocal, xLocal, innerBaseCount, totalSize_, nums,
                                                           reduceCntIdx, START_POSITION);
                    SimpleNativePipeSync<HardEvent::V_S>();
                    lastUniqueIdx = static_cast<int64_t>(idxLocal.GetValue(gatherCnt - 1));
                    idxQueue_.FreeTensor(idxLocal);
                }
            }
        }
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t copyLen)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>((copyLen) * sizeof(T)), 0, 0, 0};

        LocalTensor<T> xLocal = xQueue_.template AllocTensor<T>();
        DataCopyPad(xLocal, xGm_[offset], dataCopyParams, padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOutCnt2Workspace(int64_t coreCollectNums, int64_t lastUniqueIdx)
    {
        LocalTensor<int64_t> countLocal = collectingCntBuf_.Get<int64_t>();
        countLocal.SetValue(0, coreCollectNums);
        int64_t copyNums = 1;
        if constexpr (COUNT_OUT) {
            countLocal.SetValue(1, lastUniqueIdx);
            copyNums = 2;
        }
        SimpleNativePipeSync<HardEvent::S_MTE3>();
        Copy2GmEx<int64_t>(coreCollectWorkspaceGm_[CORE_COLLECT_NUM_OFFSET * coreIdx_], countLocal, 1, copyNums, 0, 0);
    }

    __aicore__ inline void CopyInCounts(LocalTensor<int64_t>& countLocal)
    {
        DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = coreIdx_ + 1;
        dataCopyParams.blockLen = sizeof(int64_t);
        dataCopyParams.srcStride = MAGIC_GM_PAGE_SIZE - sizeof(int64_t);
        dataCopyParams.dstStride = 0;

        DataCopyPad<int64_t, PaddingMode::Compact>(countLocal, coreCollectWorkspaceGm_, dataCopyParams, padParams);
    }

    __aicore__ inline void FindOffset(int64_t& yOffset, int64_t& prevTail)
    {
        LocalTensor<int64_t> countLocal = collectingCntBuf_.Get<int64_t>();
        LocalTensor<int64_t> offsetLocal = offsetCntBuf_.Get<int64_t>();
        CopyInCounts(countLocal);
        SimpleNativePipeSync<HardEvent::MTE2_V>();
        ReduceSum(offsetLocal, countLocal, offsetLocal, coreIdx_);
        SimpleNativePipeSync<HardEvent::V_S>();
        yOffset = offsetLocal.GetValue(0);
        if constexpr (COUNT_OUT) {
            bool first = true;
            for (int i = coreIdx_ - 1; i >= 0; i--) {
                int64_t coreCount = countLocal.GetValue(i);
                if (coreCount != 0) {
                    first = false;
                    CopyInCoreFinal(i, prevTail);
                    break;
                }
            }
            if (first) {
                prevTail = 0;
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInCoreFinal(int64_t prevCoreIdx, int64_t& prevTail)
    {
        LocalTensor<int64_t> prevIdxLocal = prevIdxBuf_.Get<int64_t>();
        DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(int64_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(prevIdxLocal,
                    coreCollectWorkspaceGm_[CORE_COLLECT_NUM_OFFSET * prevCoreIdx + CORE_LAST_UNIQUE_IDX_OFFSET],
                    dataCopyParams, padParams);
        SimpleNativePipeSync<HardEvent::MTE2_S>();
        prevTail = prevIdxLocal.GetValue(0);
    }

    template <bool TAIL_LOOP>
    __aicore__ inline void CollectUniqueAndCount(int64_t innerBaseCount, int64_t nums, int64_t& gatherCnt, int64_t i,
                                                 int64_t& prevTailCount, int64_t copyOutOffset)
    {
        LocalTensor<T> xLocal = xQueue_.template DeQue<T>();
        LocalTensor<T> outTensor = yQueue_.template AllocTensor<T>();
        uint64_t reduceCntValue = -1;
        CollectPostUniqueValue<T, T1, TAIL_LOOP>(outTensor, xLocal, nums, reduceCntValue);
        gatherCnt = reduceCntValue;
        SimpleNativePipeSync<HardEvent::S_MTE3>();
        Copy2GmEx<T>(yGm_[copyOutOffset], outTensor, 1, gatherCnt, 0, 0);
        yQueue_.FreeTensor(outTensor);

        if constexpr (COUNT_OUT) {
            if (gatherCnt > 0) {
                if constexpr (sizeof(DTYPE_COUNT) == sizeof(int64_t)) {
                    int64_t offset = coreIdx_ * tileLengthPerCore_ + i * ubTileLength_;
                    LocalTensor<int32_t> idxLocal = idxQueue_.template AllocTensor<int32_t>();
                    uint64_t reduceCntIdx = -1;
                    uint64_t alignPosition = idxQueueSize_ / MIDPOINT_DIVIDER / sizeof(int32_t);
                    CollectPostUniqueIdx<T, T1, TAIL_LOOP>(idxLocal, xLocal, 1, nums, nums, reduceCntIdx,
                                                           alignPosition);
                    LocalTensor<int64_t> idxLocalInt64 = idxLocal.template ReinterpretCast<int64_t>();
                    CastAndAddsOffsets(idxLocalInt64, idxLocal, reduceCntIdx, alignPosition, offset);
                    SimpleNativePipeSync<HardEvent::V_S>();
                    int64_t firstValue = idxLocalInt64.GetValue(0) - prevTailCount;
                    prevTailCount = idxLocalInt64.GetValue(gatherCnt - 1);
                    LocalTensor<int64_t> outCount = countQueue_.template AllocTensor<int64_t>();
                    PostAdjDiff<int64_t>(outCount, idxLocalInt64, firstValue, gatherCnt, START_POSITION);
                    countQueue_.EnQue(outCount);
                    idxQueue_.FreeTensor(idxLocal);
                    CopyOutCount(copyOutOffset, gatherCnt);
                } else {
                    LocalTensor<int32_t> idxLocal = idxQueue_.template AllocTensor<int32_t>();
                    uint64_t reduceCntIdx = -1;
                    CollectPostUniqueIdx<T, T1, TAIL_LOOP>(idxLocal, xLocal, innerBaseCount, totalSize_, nums,
                                                           reduceCntIdx, START_POSITION);
                    SimpleNativePipeSync<HardEvent::V_S>();
                    int32_t prevTailIdx = static_cast<int32_t>(prevTailCount);
                    int32_t firstValue = idxLocal.GetValue(0) - prevTailIdx;
                    prevTailCount = idxLocal.GetValue(gatherCnt - 1);
                    LocalTensor<int32_t> outCount = countQueue_.template AllocTensor<int32_t>();
                    PostAdjDiff<int32_t>(outCount, idxLocal, firstValue, gatherCnt, START_POSITION);
                    countQueue_.EnQue(outCount);
                    idxQueue_.FreeTensor(idxLocal);
                    CopyOutCount(copyOutOffset, gatherCnt);
                }
            }
        }
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutCount(int64_t offset, int64_t copyLen)
    {
        if constexpr (sizeof(DTYPE_COUNT) == sizeof(int64_t)) {
            LocalTensor<int32_t> outCount = countQueue_.template DeQue<int32_t>();
            Copy2GmEx<int32_t>(countGm_[offset * DOUBLE_OFFSET], outCount, 1, copyLen * DOUBLE_OFFSET, 0, 0);
            countQueue_.FreeTensor(outCount);
        } else {
            LocalTensor<int32_t> outCount = countQueue_.template DeQue<int32_t>();
            Copy2GmEx<int32_t>(countGm_[offset], outCount, 1, copyLen, 0, 0);
            countQueue_.FreeTensor(outCount);
        }
    }

    __aicore__ inline void CopyOutShape(uint64_t dimNumValue, uint64_t dimNumIdx)
    {
        LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();

        shapeTensor.SetValue(SHAPE0_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
        shapeTensor.SetValue(SHAPE0_DIM0_IDX, dimNumValue);

        shapeTensor.SetValue(SHAPE1_SIZE_IDX, 1);
        shapeTensor.SetValue(SHAPE1_DIM0_IDX, 1);

        shapeTensor.SetValue(SHAPE2_SIZE_IDX, 1);
        if constexpr (COUNT_OUT) {
            shapeTensor.SetValue(SHAPE2_DIM0_IDX, dimNumIdx);
        } else {
            shapeTensor.SetValue(SHAPE2_DIM0_IDX, 1);
        }

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = SHAPE_LEN * sizeof(uint64_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        SimpleNativePipeSync<HardEvent::S_MTE3>();
        DataCopyPad(shapeGm_, shapeTensor, dataCopyParams);
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM_MULTI> xQueue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM_MULTI> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_MULTI> countQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_MULTI> idxQueue_;

    TBuf<TPosition::VECCALC> collectingCntBuf_;
    TBuf<TPosition::VECCALC> offsetCntBuf_;
    TBuf<TPosition::VECCALC> prevIdxBuf_;
    TBuf<TPosition::VECCALC> shapeBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> countGm_;
    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<uint64_t> shapeGm_;

    GlobalTensor<int64_t> coreCollectWorkspaceGm_;

    int64_t ubTileLength_;
    int64_t adjUbTileLength_;
    int64_t coreIdx_;
    bool isFinalCore_;
    int64_t coreTileLength_;
    int64_t totalSize_;
    int64_t baseCount_;

    int64_t tileLengthPerCore_;
    int64_t idxQueueSize_;

    TPipe* pipe_ = nullptr;
};
#endif // UNIQUE_CONSECUTIVE_MULTI_CORE_KERNEL_H