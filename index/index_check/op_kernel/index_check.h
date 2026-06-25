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
 * \file index_check.h
 * \brief
 */
#ifndef INDEX_CHECK_H
#define INDEX_CHECK_H
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr int32_t SIGN_BIT_SHIFT = 31;
constexpr uint32_t MAX_TENSOR_NUM = 8;

template <typename T>
class IndexCheckKernel
{
public:
    __aicore__ inline IndexCheckKernel() = delete;
    __aicore__ inline IndexCheckKernel(
        GM_ADDR bounds, GM_ADDR indexList, GM_ADDR workSpace, const IndexCheckTilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling, indexList);
        InitBounds(bounds);
        InitBuffers(pipe);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < tensorId_; i++) {
            uint64_t tensorLen = tensorLens_[i];
            if (tensorLen == 0) {
                continue;
            }
            indexGm_.SetGlobalBuffer(GetTensorAddr(indexList_, i));

            uint64_t formerCoreNum = tensorLen % usedCoreNum_;
            uint64_t formerCoreDataNum = (tensorLen + usedCoreNum_ - 1) / usedCoreNum_;
            uint64_t tailCoreDataNum = tensorLen / usedCoreNum_;

            uint64_t dataNum = 0;
            uint64_t idxAddrOffset = 0;
            if (blockIdx_ < formerCoreNum) {
                dataNum = formerCoreDataNum;
                idxAddrOffset = formerCoreDataNum * blockIdx_;
            } else {
                dataNum = tailCoreDataNum;
                idxAddrOffset = formerCoreDataNum * formerCoreNum +
                                tailCoreDataNum * (blockIdx_ - formerCoreNum);
            }

            if (dataNum == 0) {
                continue;
            }

            uint64_t processed = 0;
            uint32_t blockElementCount = BLOCK_BYTES / sizeof(T);
            if (dataNum < blockElementCount) {
                ProcessScalar(i, static_cast<uint32_t>(dataNum), idxAddrOffset);
            } else {
                while (processed < dataNum) {
                    uint64_t remaining = dataNum - processed;
                    uint64_t currentBatch = (remaining > maxBatchSize_) ? maxBatchSize_ : remaining;
                    ProcessBatch(i, static_cast<uint32_t>(currentBatch), idxAddrOffset + processed);
                    processed += currentBatch;
                }
            }
        }
    }

private:
    __aicore__ inline void InitParams(const IndexCheckTilingData& tiling, GM_ADDR indexList)
    {
        indexList_ = indexList;
        blockIdx_ = GetBlockIdx();
        usedCoreNum_ = tiling.params.usedCoreNum;
        tensorId_ = tiling.params.tensorId;
        maxBatchSize_ = tiling.params.maxBatchSize;
        for (uint64_t i = 0; i < tensorId_; i++) {
            tensorLens_[i] = tiling.params.tensorLens[i];
        }
    }

    __aicore__ inline void InitBounds(GM_ADDR bounds)
    {
        boundsGm_.SetGlobalBuffer((__gm__ int64_t*)bounds);
        for (uint64_t i = 0; i < tensorId_; i++) {
            bounds_[i] = boundsGm_.GetValue(i);
        }
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        uint32_t idxAlignUnit = BLOCK_BYTES / sizeof(int32_t);
        alignedBatch_ = ((maxBatchSize_ + idxAlignUnit - 1) / idxAlignUnit) * idxAlignUnit;

        pipe.InitBuffer(indicesQue_, BUFFER_NUM, alignedBatch_ * sizeof(T));
        pipe.InitBuffer(workingQue_, BUFFER_NUM, alignedBatch_ * sizeof(int32_t));
        pipe.InitBuffer(tempQue_, BUFFER_NUM, alignedBatch_ * sizeof(int32_t));
        pipe.InitBuffer(floatQue_, BUFFER_NUM, alignedBatch_ * sizeof(float));
        pipe.InitBuffer(workQue_, BUFFER_NUM, alignedBatch_ * sizeof(float));
        pipe.InitBuffer(boundsQue_, BUFFER_NUM, alignedBatch_ * sizeof(int32_t));
    }

    __aicore__ inline void ProcessScalar(uint64_t tensorIdx, uint32_t count, uint64_t offset)
    {
        int64_t bound = bounds_[tensorIdx];
        for (uint32_t i = 0; i < count; i++) {
            T idxVal = indexGm_.GetValue(offset + i);
            int64_t idx = static_cast<int64_t>(idxVal);
            if (idx < 0) {
                idx += bound;
            }
            ascendc_assert((idx >= 0),
                "Index out of range in dimension %lu: index value %ld is too negative for bounds %ld!\n",
                tensorIdx, static_cast<int64_t>(idxVal), bound);
            ascendc_assert((idx < bound),
                "Index out of range in dimension %lu: index value %ld exceeds bounds %ld!\n",
                tensorIdx, static_cast<int64_t>(idxVal), bound);
        }
    }

    __aicore__ inline void ProcessBatch(uint64_t tensorIdx, uint32_t count, uint64_t offset)
    {
        LocalTensor<T> indexLocal = indicesQue_.AllocTensor<T>();
        LocalTensor<int32_t> workingLocal = workingQue_.AllocTensor<int32_t>();
        LocalTensor<int32_t> tempLocal = tempQue_.AllocTensor<int32_t>();
        LocalTensor<float> floatLocal = floatQue_.AllocTensor<float>();
        LocalTensor<float> workLocal = workQue_.AllocTensor<float>();
        LocalTensor<int32_t> boundsLocal = boundsQue_.AllocTensor<int32_t>();

        uint32_t copyAlignUnit = BLOCK_BYTES / sizeof(T);
        uint32_t countCopyAligned = ((count + copyAlignUnit - 1) / copyAlignUnit) * copyAlignUnit;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        uint8_t rightPadCount = static_cast<uint8_t>(countCopyAligned - count);
        DataCopyPadExtParams<T> padParams{true, 0, rightPadCount, static_cast<T>(0)};
        DataCopyPad(indexLocal, indexGm_[offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        if constexpr (std::is_same<T, int64_t>::value) {
            Cast(workingLocal, indexLocal, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        } else {
            auto indexInt32Local = indexLocal.template ReinterpretCast<int32_t>();
            DataCopy(workingLocal, indexInt32Local, countCopyAligned);
            PipeBarrier<PIPE_V>();
        }

        ShiftRight(tempLocal, workingLocal, SIGN_BIT_SHIFT, count);
        PipeBarrier<PIPE_V>();
        Muls(tempLocal, tempLocal, static_cast<int32_t>(bounds_[tensorIdx]), count);
        PipeBarrier<PIPE_V>();
        Sub(workingLocal, workingLocal, tempLocal, count);
        PipeBarrier<PIPE_V>();

        Cast(floatLocal, workingLocal, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();

        ReduceMin(floatLocal, floatLocal, workLocal, static_cast<int32_t>(count), false);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float minValue = floatLocal.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        ascendc_assert((minValue >= 0.0f),
               "Index out of range in dimension %lu: index value %f is too negative for bounds %ld!\n",
               tensorIdx, minValue, bounds_[tensorIdx]);

        CheckUpperBound(floatLocal, workLocal, workingLocal, boundsLocal, tensorIdx, count);

        indicesQue_.FreeTensor(indexLocal);
        workingQue_.FreeTensor(workingLocal);
        tempQue_.FreeTensor(tempLocal);
        floatQue_.FreeTensor(floatLocal);
        workQue_.FreeTensor(workLocal);
        boundsQue_.FreeTensor(boundsLocal);
    }

    __aicore__ inline void CheckUpperBound(LocalTensor<float>& floatLocal,
        LocalTensor<float>& workLocal, LocalTensor<int32_t>& workingLocal,
        LocalTensor<int32_t>& boundsLocal, uint64_t tensorIdx, uint32_t count)
    {
        Duplicate(boundsLocal, static_cast<int32_t>(bounds_[tensorIdx]), count);
        PipeBarrier<PIPE_V>();
        Sub(workingLocal, workingLocal, boundsLocal, count);
        PipeBarrier<PIPE_V>();

        Cast(floatLocal, workingLocal, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();

        ReduceMax(floatLocal, floatLocal, workLocal, static_cast<int32_t>(count), false);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float maxValue = floatLocal.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        ascendc_assert((maxValue < 0.0f),
               "Index out of range in dimension %lu: index value %f exceeds bounds %ld!\n",
               tensorIdx, maxValue, bounds_[tensorIdx]);
    }

    __aicore__ inline __gm__ T* GetTensorAddr(GM_ADDR indexListPtr, const uint64_t offset)
    {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(indexListPtr);
        uint64_t tensorPtrOffset = *dataAddr;
        __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ T*>(*(tensorPtr + offset));
    }

private:
    GM_ADDR indexList_;
    GlobalTensor<T> indexGm_;
    GlobalTensor<int64_t> boundsGm_;

    TQue<TPosition::VECCALC, BUFFER_NUM> indicesQue_;
    TQue<TPosition::VECCALC, BUFFER_NUM> workingQue_;
    TQue<TPosition::VECCALC, BUFFER_NUM> tempQue_;
    TQue<TPosition::VECCALC, BUFFER_NUM> floatQue_;
    TQue<TPosition::VECCALC, BUFFER_NUM> workQue_;
    TQue<TPosition::VECCALC, BUFFER_NUM> boundsQue_;

    uint64_t blockIdx_ = 0;
    uint64_t usedCoreNum_ = 0;
    uint64_t tensorId_ = 0;
    uint64_t maxBatchSize_ = 0;
    uint32_t alignedBatch_ = 0;
    uint64_t tensorLens_[MAX_TENSOR_NUM];
    int64_t bounds_[MAX_TENSOR_NUM];
};

#endif // INDEX_CHECK_H
