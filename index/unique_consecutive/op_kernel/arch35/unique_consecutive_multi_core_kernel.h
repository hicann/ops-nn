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
class UniqueConsecutiveMutilCoreKerenl
{
public:
    __aicore__ inline UniqueConsecutiveMutilCoreKerenl(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR count, GM_ADDR shape_out, GM_ADDR workspace,
                                const UniqueConsecutiveTilingData* tilingData)
    {
        coreIdx_ = GetBlockIdx();
        isFinalCore_ = (coreIdx_ == GetBlockNum() - 1);

        coreTileLength_ = (isFinalCore_) ? tilingData->tileLengthTailCore : tilingData->tileLengthPerCore;
        totalSize_ = tilingData->totalSize;
        baseCount_ = 1 + coreIdx_ * tilingData->tileLengthPerCore;
        tileLengthPerCore_ = tilingData->tileLengthPerCore;
        countQueueSize_ = tilingData->countQueueSize;
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
        pipe_->InitBuffer(countQueue_, BUFFER_NUM_MULTI, tilingData->countQueueSize);
        pipe_->InitBuffer(idxCopyInQueue_, BUFFER_NUM_MULTI, tilingData->idxCopyInQueueSize);

        pipe_->InitBuffer(collectingCntBuf_, tilingData->collectingCntBufSize);
        pipe_->InitBuffer(offsetCntBuf_, tilingData->offsetCntBufSize);
        pipe_->InitBuffer(prevIdxBuf_, tilingData->prevIdxBufSize);
        pipe_->InitBuffer(shapeBuf_, tilingData->shapeBufSize);

        // init workspace
        collectNumGm_.SetGlobalBuffer((__gm__ int64_t*)(workspace));
        valueWorkspaceGm_.SetGlobalBuffer((__gm__ T*)(workspace + MAGIC_GM_PAGE_SIZE * GetBlockNum()) +
                                          coreIdx_ * tilingData->tileLengthPerCore);
        if constexpr (ISINT64) {
            idxWorkspaceGm_.SetGlobalBuffer(
            (__gm__ int32_t*)(workspace + MAGIC_GM_PAGE_SIZE * GetBlockNum() + tilingData->totalSize * sizeof(T)) +
            coreIdx_ * tilingData->tileLengthPerCore * DOUBLE_OFFSET);
        } else {
            idxWorkspaceGm_.SetGlobalBuffer(
            (__gm__ int32_t*)(workspace + MAGIC_GM_PAGE_SIZE * GetBlockNum() + tilingData->totalSize * sizeof(T)) +
            coreIdx_ * tilingData->tileLengthPerCore);
        }
        idxWorkspaceStartGm_.SetGlobalBuffer(
            (__gm__ int32_t*)(workspace + MAGIC_GM_PAGE_SIZE * GetBlockNum() + tilingData->totalSize * sizeof(T)));

        idxGm_.SetGlobalBuffer((__gm__ int32_t*)(idx) + tilingData->tileLengthPerCore * coreIdx_);
    }

    __aicore__ inline void Process()
    {
        if (isFinalCore_) {
            LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();
            Duplicate(shapeTensor, (uint64_t)1, SHAPE_LEN);
        }

        int64_t coreCollectNums = 0;
        ProcessCollecting(coreCollectNums);

        PipeBarrier<PIPE_ALL>();
        SyncAll();

        if (coreCollectNums == 0) {
            return;
        }

        int64_t yOffset = 0;
        int64_t prevTailCount = 0;
        FindOffset(yOffset, prevTailCount, coreCollectNums);  // Get merging offset and prevTailCount

        ProcessMerging(yOffset, coreCollectNums, prevTailCount);

        if (isFinalCore_) {
            coreCollectNums += yOffset;
            CopyOutShape(coreCollectNums, coreCollectNums);
        }
    }

    __aicore__ inline void ProcessMerging(int64_t yOffset, int64_t collectNums, int64_t prevTailCount)
    {
        int64_t mergingLoops = collectNums / ubTileLength_;
        int64_t mergingTails = collectNums % ubTileLength_;

        int64_t copyInOffset = 0;
        int64_t copyOutOffset = yOffset;

        for (int64_t i = 0; i < mergingLoops; ++i) {
            CopyInAndComputeCount(copyInOffset, copyOutOffset, ubTileLength_,
                                  prevTailCount);  // find unique, store into workspace.
            copyInOffset += ubTileLength_;
            copyOutOffset += ubTileLength_;
        }

        // tail process
        if (mergingTails > 0) {
            CopyInAndComputeCount(copyInOffset, copyOutOffset, mergingTails,
                                  prevTailCount);  // find unique, store into workspace.
        }
    }

    __aicore__ inline void ProcessCollecting(int64_t& coreCollectNums)
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
            CollectUniques<false>(innerBaseCount, ubMainLength, gatherCnt, i);  // find unique, store into workspace.
            CopyOutCollecteds2Worksapce(coreCollectNums, gatherCnt);
            offsetXGm += ubTileLength_;
            coreCollectNums += gatherCnt;
            innerBaseCount += ubTileLength_;
        }
        // alg always have tails
        CopyInX(offsetXGm, ubTailLength);
        if (isFinalCore_) {
            CollectUniques<true>(innerBaseCount, ubTailLength, gatherCnt, ubLoops);
        } else {
            CollectUniques<false>(innerBaseCount, ubTailLength, gatherCnt, ubLoops);
        }

        CopyOutCollecteds2Worksapce(coreCollectNums, gatherCnt);
        coreCollectNums += gatherCnt;

        CopyOutCnt2Workspace(coreCollectNums);
    }

    template <bool TAIL_LOOP>
    __aicore__ inline void CollectUniques(int64_t innerBaseCount, int64_t nums, int64_t& gatherCnt, int64_t i)
    {
        LocalTensor<T> xLocal = xQueue_.template DeQue<T>();
        LocalTensor<T> outTensor = yQueue_.template AllocTensor<T>();
        uint64_t reduceCntValue = -1;
        CollectPostUniqueValue<T, T1, TAIL_LOOP>(outTensor, xLocal, nums, reduceCntValue);
        yQueue_.EnQue(outTensor);

        if constexpr (COUNT_OUT) {
            if constexpr (ISINT64) {
                int64_t offset = coreIdx_ * tileLengthPerCore_ + i * ubTileLength_;
                LocalTensor<int32_t> countLocal = countQueue_.template AllocTensor<int32_t>();
                uint64_t reduceCntIdx = -1;   // use for debug only
                uint64_t alignPosition = countQueueSize_ / MIDPOINT_DIVIDER / sizeof(int32_t);
                CollectPostUniqueIdx<T, T1, TAIL_LOOP>(countLocal, xLocal, 1, nums, nums, reduceCntIdx, alignPosition);
                LocalTensor<int64_t> outCount = countLocal.template ReinterpretCast<int64_t>();
                CastAndAddsOffsets(outCount, countLocal, reduceCntIdx, alignPosition, offset);
                countQueue_.EnQue(outCount);
            } else {
                LocalTensor<int32_t> outCount = countQueue_.template AllocTensor<int32_t>();
                uint64_t reduceCntIdx = -1;  // use for debug only
                CollectPostUniqueIdx<T, T1, TAIL_LOOP>(outCount, xLocal, innerBaseCount, totalSize_, nums, reduceCntIdx, START_POSITION);
                countQueue_.EnQue(outCount);
            }
        }

        gatherCnt = reduceCntValue;
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

    __aicore__ inline void CopyOutCollecteds2Worksapce(int64_t offset, int64_t copyLen)
    {
        LocalTensor<T> yLocal = yQueue_.template DeQue<T>();

        if constexpr (COUNT_OUT) {
            if constexpr (ISINT64) {
                LocalTensor<int32_t> countLocal = countQueue_.template DeQue<int32_t>();
                Copy2GmEx<int32_t>(idxWorkspaceGm_[offset * DOUBLE_OFFSET], countLocal, 1, copyLen * DOUBLE_OFFSET, 0, 0);
                countQueue_.FreeTensor(countLocal);
            } else {
                LocalTensor<int32_t> countLocal = countQueue_.template DeQue<int32_t>();
                Copy2GmEx<int32_t>(idxWorkspaceGm_[offset], countLocal, 1, copyLen, 0, 0);
                countQueue_.FreeTensor(countLocal);
            }
        }

        Copy2GmEx<T>(valueWorkspaceGm_[offset], yLocal, 1, copyLen, 0, 0);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutCnt2Workspace(int64_t coreCollectNums)
    {
        LocalTensor<int64_t> countLocal = collectingCntBuf_.Get<int64_t>();
        countLocal.SetValue(0, coreCollectNums);
        SimpleNativePipeSync<HardEvent::S_MTE3>();
        Copy2GmEx<int64_t>(collectNumGm_[(MAGIC_GM_PAGE_SIZE / sizeof(int64_t)) * coreIdx_], countLocal, 1, 1, 0, 0);
    }

    __aicore__ inline void CopyInCounts(LocalTensor<int64_t>& countLocal)
    {
        DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = coreIdx_ + 1;
        dataCopyParams.blockLen = sizeof(int64_t);
        dataCopyParams.srcStride = MAGIC_GM_PAGE_SIZE - sizeof(int64_t);
        dataCopyParams.dstStride = 0;

        DataCopyPad<int64_t, PaddingMode::Compact>(countLocal, collectNumGm_, dataCopyParams, padParams);
    }

    __aicore__ inline void FindOffset(int64_t& yOffset, int64_t& prevTail, int64_t coreCollectNums)
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
                    CopyInCoreFinal(i, coreCount, prevTail);
                    break;
                }
            }
            if (first) {
                prevTail = 0;
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInCoreFinal(int64_t prevCoreIdx, int64_t prevCoreCount, int64_t& prevTail)
    {
        // we use MTE2 here, maybe better to use GlobalTenosr.GetValue()
        int64_t offset = prevCoreIdx * tileLengthPerCore_ + prevCoreCount - 1;
        LocalTensor<int32_t> prevIdxLocal = prevIdxBuf_.Get<int32_t>();
        if constexpr (ISINT64) {
            DataCopyPrevIdx(prevIdxLocal, offset, DOUBLE_OFFSET);
            SimpleNativePipeSync<HardEvent::MTE2_S>();
            LocalTensor<int64_t> prevIdxLocalInt64 = prevIdxLocal.template ReinterpretCast<int64_t>();
            prevTail = prevIdxLocalInt64.GetValue(0);
        } else {
            DataCopyPrevIdx(prevIdxLocal, offset, SINGLE_OFFSET);
            SimpleNativePipeSync<HardEvent::MTE2_S>();
            prevTail = static_cast<int64_t>(prevIdxLocal.GetValue(0));
        }
    }

    __aicore__ inline void DataCopyPrevIdx(LocalTensor<int32_t> prevIdxLocal, int64_t offset, int64_t copyLen)
    {
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(int32_t) * copyLen;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(prevIdxLocal, idxWorkspaceStartGm_[offset * copyLen], dataCopyParams, padParams);
    }

    __aicore__ inline void CopyInAndComputeCount(int64_t wsOffset, int64_t yOffset, int64_t nums,
                                                 int64_t& prevTailCount)
    {
        LocalTensor<T> xLocal = xQueue_.template AllocTensor<T>();
        CopyInUniqueValues(xLocal, wsOffset, nums);
        if constexpr (COUNT_OUT) {
            if constexpr (ISINT64) {
                CopyInUniqueIdx(wsOffset * DOUBLE_OFFSET, nums * DOUBLE_OFFSET);
                LocalTensor<int64_t> idxLocal = idxCopyInQueue_.template DeQue<int64_t>();
                LocalTensor<int64_t> outCount = countQueue_.template AllocTensor<int64_t>();
                int64_t firstValue = idxLocal.GetValue(0) - prevTailCount;
                prevTailCount = idxLocal.GetValue(nums - 1);
                PostAdjDiff<int64_t>(outCount, idxLocal, firstValue, nums, START_POSITION);
                idxCopyInQueue_.FreeTensor(idxLocal);
                countQueue_.EnQue(outCount);
                CopyOutCount(yOffset, nums);
            } else {
                if constexpr (sizeof(DTYPE_COUNT) == sizeof(int64_t) ) {
                    CopyInUniqueIdx(wsOffset, nums);
                    LocalTensor<int32_t> idxLocal = idxCopyInQueue_.template DeQue<int32_t>();
                    LocalTensor<int32_t> countLocal = countQueue_.template AllocTensor<int32_t>();
                    int32_t firstValue = idxLocal.GetValue(0) - prevTailCount;
                    prevTailCount = idxLocal.GetValue(nums - 1);
                    uint64_t alignPosition = countQueueSize_ / MIDPOINT_DIVIDER / sizeof(int32_t);
                    PostAdjDiff<int32_t>(countLocal, idxLocal, firstValue, nums, alignPosition);
                    idxCopyInQueue_.FreeTensor(idxLocal);
                    LocalTensor<int64_t> outCount = countLocal.template ReinterpretCast<int64_t>();
                    Cast(outCount, countLocal[alignPosition], RoundMode::CAST_NONE, nums);
                    countQueue_.EnQue(outCount);
                    CopyOutCount(yOffset, nums);
                } else {
                    CopyInUniqueIdx(wsOffset, nums);
                    LocalTensor<int32_t> idxLocal = idxCopyInQueue_.template DeQue<int32_t>();
                    LocalTensor<int32_t> outCount = countQueue_.template AllocTensor<int32_t>();
                    int32_t firstValue = idxLocal.GetValue(0) - prevTailCount;
                    prevTailCount = idxLocal.GetValue(nums - 1);
                    PostAdjDiff<int32_t>(outCount, idxLocal, firstValue, nums, START_POSITION);
                    idxCopyInQueue_.FreeTensor(idxLocal);
                    countQueue_.EnQue(outCount);
                    CopyOutCount(yOffset, nums);
                } 
            }
        }
        SimpleNativePipeSync<HardEvent::MTE2_MTE3>();
        Copy2GmEx<T>(yGm_[yOffset], xLocal, 1, nums, 0, 0);
        SimpleNativePipeSync<HardEvent::MTE3_MTE2>();
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

    __aicore__ inline void CopyInUniqueValues(LocalTensor<T>& xLocal, int64_t offset, int64_t nums)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = nums * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(xLocal, valueWorkspaceGm_[offset], dataCopyParams, padParams);
    }

    __aicore__ inline void CopyInUniqueIdx(int64_t offset, int64_t nums)
    {
        LocalTensor<int32_t> idxLocal = idxCopyInQueue_.template AllocTensor<int32_t>();
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = nums * sizeof(int32_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(idxLocal, idxWorkspaceGm_[offset], dataCopyParams, padParams);
        idxCopyInQueue_.EnQue(idxLocal);
        SimpleNativePipeSync<HardEvent::MTE2_S>();
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
    TQue<QuePosition::VECIN, BUFFER_NUM_MULTI> idxCopyInQueue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM_MULTI> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_MULTI> countQueue_;

    TBuf<TPosition::VECCALC> collectingCntBuf_;
    TBuf<TPosition::VECCALC> offsetCntBuf_;
    TBuf<TPosition::VECCALC> prevIdxBuf_;
    TBuf<TPosition::VECCALC> shapeBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> countGm_;
    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<uint64_t> shapeGm_;

    GlobalTensor<int64_t> collectNumGm_;
    GlobalTensor<T> valueWorkspaceGm_;
    GlobalTensor<int32_t> idxWorkspaceGm_;
    GlobalTensor<int32_t> idxWorkspaceStartGm_;

    int64_t ubTileLength_;
    int64_t adjUbTileLength_;
    int64_t coreIdx_;
    bool isFinalCore_;
    int64_t coreTileLength_;
    int64_t totalSize_;
    int64_t baseCount_;

    int64_t tileLengthPerCore_;
    int64_t countQueueSize_;

    TPipe* pipe_ = nullptr;
};
#endif  // UNIQUE_CONSECUTIVE_MULTI_CORE_KERNEL_H