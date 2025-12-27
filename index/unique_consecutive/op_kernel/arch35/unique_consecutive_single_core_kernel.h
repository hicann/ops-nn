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
 * \file unique_consecutive_single_core_kernel.h
 * \brief
 */

#ifndef UNIQUE_CONSECUTIVE_SINGLE_CORE_KERNEL_H
#define UNIQUE_CONSECUTIVE_SINGLE_CORE_KERNEL_H

#include "unique_consecutive_helper.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t ALIGNMENT_SCALE_FACTOR = 2;

template <typename T, typename T1, typename DTYPE_COUNT, bool COUNT_OUT, bool ISINT64>
class UniqueConsecutiveSingleCoreKerenl
{
public:
    __aicore__ inline UniqueConsecutiveSingleCoreKerenl(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR count, GM_ADDR shape_out, GM_ADDR workspace,
                                const UniqueConsecutiveTilingData* tilingData)
    {
        totalLength_ = tilingData->totalSize;
        blockSize_ = GetDataBlockSizeInBytes();

        xGm_.SetGlobalBuffer((__gm__ T*)(x));
        yGm_.SetGlobalBuffer((__gm__ T*)(y));
        shapeGm_.SetGlobalBuffer((__gm__ uint64_t*)shape_out);

        valueQueueSize_ = CEIL_DIV(totalLength_ * sizeof(T), blockSize_) * blockSize_;
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, valueQueueSize_);
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, valueQueueSize_);

        countGm_.SetGlobalBuffer((__gm__ int32_t*)(count));
        if constexpr (sizeof(DTYPE_COUNT) == (sizeof(int64_t))) {
            countQueueSize_ = CEIL_DIV(totalLength_ * sizeof(int64_t), blockSize_ * ALIGNMENT_SCALE_FACTOR) * blockSize_ * ALIGNMENT_SCALE_FACTOR;
            pipe_->InitBuffer(countQueue_, BUFFER_NUM, totalLength_ * sizeof(int64_t));
        } else {
            countQueueSize_ = CEIL_DIV(totalLength_ * sizeof(int32_t), blockSize_) * blockSize_;
            pipe_->InitBuffer(countQueue_, BUFFER_NUM, totalLength_ * sizeof(int32_t));
        }
        pipe_->InitBuffer(idxBuf_, CEIL_DIV(totalLength_ * sizeof(int32_t), blockSize_) * blockSize_);
        pipe_->InitBuffer(shapeBuf_, CEIL_DIV(SHAPE_LEN * sizeof(uint64_t), blockSize_) * blockSize_);

        idxGm_.SetGlobalBuffer((__gm__ int32_t*)(idx));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();
        Duplicate(shapeTensor, (uint64_t)1, SHAPE_LEN);
        CopyInX();
        ComputeAndCopyOut();
    }

    __aicore__ inline void CopyInX()
    {
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = (totalLength_) * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        LocalTensor<T> xLocal = xQueue_.template AllocTensor<T>();

        DataCopyPad(xLocal, xGm_, dataCopyParams, padParams);

        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeAndCopyOut()
    {
        LocalTensor<int32_t> idxTensor = idxBuf_.Get<int32_t>();
        LocalTensor<T> xLocal = xQueue_.template DeQue<T>();
        LocalTensor<T> outTensor = yQueue_.template AllocTensor<T>();

        uint64_t reduceCntValue = -1;
        uint64_t reduceCntIdx = -1;

        CollectPostUniqueValue<T, T1, true>(outTensor, xLocal, totalLength_, reduceCntValue);

        if constexpr (COUNT_OUT) {
            CollectPostUniqueIdx<T, T1, true>(idxTensor, xLocal, 1, totalLength_, totalLength_, reduceCntIdx, START_POSITION);
        }

        xQueue_.FreeTensor(xLocal);

        if constexpr (COUNT_OUT) {
            if constexpr (sizeof(DTYPE_COUNT) == sizeof(int64_t)) {
                LocalTensor<int32_t> countLocal = countQueue_.template AllocTensor<int32_t>();
                uint64_t alignPosition = countQueueSize_ / DOUBLE_OFFSET / sizeof(int32_t);
                PostAdjDiff<int32_t>(countLocal, idxTensor, idxTensor.GetValue(0), reduceCntIdx, alignPosition);
                LocalTensor<int64_t> outCount = countLocal.template ReinterpretCast<int64_t>();
                Cast(outCount, countLocal[alignPosition], RoundMode::CAST_NONE, reduceCntIdx);
                countQueue_.EnQue(outCount);
            } else {
                LocalTensor<int32_t> outCount = countQueue_.template AllocTensor<int32_t>();
                PostAdjDiff<int32_t>(outCount, idxTensor, idxTensor.GetValue(0), reduceCntIdx, START_POSITION);
                countQueue_.EnQue(outCount);
            }
        }

        yQueue_.EnQue(outTensor);

        CopyOut(reduceCntValue, reduceCntIdx);
        CopyOutShape(reduceCntValue, reduceCntIdx);
    }

    __aicore__ inline void CopyOut(int32_t copyLenValue, int32_t copyLenIdx)
    {
        DataCopyExtParams dataCopyParamsValue;
        dataCopyParamsValue.blockCount = 1;
        dataCopyParamsValue.blockLen = copyLenValue * sizeof(T);
        dataCopyParamsValue.srcStride = 0;
        dataCopyParamsValue.dstStride = 0;

        LocalTensor<T> yLocal = yQueue_.template DeQue<T>();

        if constexpr (COUNT_OUT) {
            if constexpr (sizeof(DTYPE_COUNT) == sizeof(int64_t)) {
                LocalTensor<int32_t> outCount = countQueue_.template DeQue<int32_t>();
                Copy2GmEx<int32_t>(countGm_, outCount, 1, copyLenIdx * DOUBLE_OFFSET, 0, 0);
                countQueue_.FreeTensor(outCount);
            } else {
                LocalTensor<int32_t> outCount = countQueue_.template DeQue<int32_t>();
                Copy2GmEx<int32_t>(countGm_, outCount, 1, copyLenIdx, 0, 0 );
                countQueue_.FreeTensor(outCount);
            } 
        }

        DataCopyPad(yGm_, yLocal, dataCopyParamsValue);
        yQueue_.FreeTensor(yLocal);
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
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> countQueue_;

    TBuf<TPosition::VECCALC> idxBuf_;
    TBuf<TPosition::VECCALC> shapeBuf_;

    GlobalTensor<T> xGm_;

    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> countGm_;
    GlobalTensor<int32_t> idxGm_;

    GlobalTensor<uint64_t> shapeGm_;

    uint32_t totalLength_;
    uint32_t blockSize_;
    int64_t countQueueSize_;
    int64_t valueQueueSize_;

    TPipe* pipe_ = nullptr;
};
#endif  // UNIQUE_CONSECUTIVE_SINGLE_CORE_KERNEL_H