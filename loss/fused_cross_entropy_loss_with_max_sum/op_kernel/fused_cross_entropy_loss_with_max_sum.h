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
 * \file FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM.h
 * \brief
 */

#ifndef FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_H_DETERMINIST_H
#define FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_H_DETERMINIST_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

constexpr int64_t BT_BLOCK_SIZE = 512 / 4;
constexpr int64_t CACEH_LINE = 512;
constexpr int64_t TENSOR_COUNT = 6;
constexpr int64_t DOUBLE_BUFFER = 2;

namespace AscendC {
template <typename T>
class FusedCrossEntropyLossWithMaxSum {
public:
    __aicore__ inline FusedCrossEntropyLossWithMaxSum() = delete;
    __aicore__ inline FusedCrossEntropyLossWithMaxSum(
        GM_ADDR inputTensors[TENSOR_COUNT], const FusedCrossEntropyLossWithMaxSumTilingData& tiling, TPipe& pipe)
    {
        InitParams(inputTensors, tiling, pipe);
    }

    __aicore__ inline void Process()
    {
        for (int i = 0; i < btCountTime; i++) {
            BtCompute(BT_BLOCK_SIZE);
            baseBtOffset += BT_BLOCK_SIZE;
        }
        if (reservedbtNum > 0) {
            BtCompute(reservedbtNum);
        }
    }

    __aicore__ inline void ProcessForMemory()
    {
        for (int i = 0; i < btCountTime; i++) {
            BtComputeForMemory(elementsNumber);
            baseBtOffset += elementsNumber;
        }
        if (reservedbtNum > 0) {
            BtComputeForMemory(reservedbtNum);
        }
    }

private:
    __aicore__ inline void SyncM2toV()
    {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    };
    __aicore__ inline void SyncM2toS()
    {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
    };

    __aicore__ inline void SyncM3toV()
    {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    };

    __aicore__ inline void SyncM3toM2()
    {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    };

    __aicore__ inline void SyncVtoM3()
    {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void InitParams(
        GM_ADDR inputTensors[TENSOR_COUNT], const FusedCrossEntropyLossWithMaxSumTilingData& tiling, TPipe& pipe)
    {
        pipe_ = &pipe;
        auto blockIdx_ = GetBlockIdx();
        vLen = static_cast<int64_t>(tiling.vLen);

        if (blockIdx_ < tiling.formerCoreNum) {
            btCountLen = tiling.formerbtCountLen;
            btCountTime = tiling.formerbtCountTime;
            reservedbtNum = tiling.formerCoreReservedbtNum;
            baseBtOffset = blockIdx_ * static_cast<int64_t>(btCountLen);
        } else {
            btCountLen = tiling.latterbtCountLen;
            btCountTime = tiling.latterbtCountTime;
            reservedbtNum = tiling.latterCoreReservedbtNum;
            baseBtOffset = static_cast<int64_t>(tiling.formerCoreNum) + blockIdx_ * static_cast<int64_t>(btCountLen);
        }
        singleCalLen = tiling.singleCalculationQuantity;
        singleCalReservedLen = tiling.singleCalculationReservedQuantity;
        elementsNumber = tiling.elementsNumber;

        vocabParallelLogitsGm_.SetGlobalBuffer((__gm__ T*)inputTensors[0]);
        logitsMaxGm_.SetGlobalBuffer((__gm__ float*)inputTensors[1]);
        sumExpLogitsGm_.SetGlobalBuffer((__gm__ float*)inputTensors[2]);
        predictedLogitsGm_.SetGlobalBuffer((__gm__ float*)inputTensors[3]);
        softmaxLogitsGm_.SetGlobalBuffer((__gm__ float*)inputTensors[4]);
        lossGm_.SetGlobalBuffer((__gm__ float*)inputTensors[5]);

        if (vLen > 0) {
            pipe.InitBuffer(vocabParallelLogitsQue_, DOUBLE_BUFFER, elementsNumber * sizeof(float));
            pipe.InitBuffer(logitsMaxQue_, 1, CACEH_LINE);
            pipe.InitBuffer(sumExpLogitsQue_, 1, CACEH_LINE);
            pipe.InitBuffer(predictedLogitsQue_, 1, CACEH_LINE);
            pipe.InitBuffer(logQue_, 1, CACEH_LINE);
            pipe.InitBuffer(softmaxLogitsQue_, 1, elementsNumber * sizeof(float));

            vocabParallelLogitsTensor = vocabParallelLogitsQue_.AllocTensor<float>();
            logitsMaxTensor = logitsMaxQue_.AllocTensor<float>();
            sumExpLogitsTensor = sumExpLogitsQue_.AllocTensor<float>();
            predictedLogitsTensor = predictedLogitsQue_.AllocTensor<float>();
            logTensor = logQue_.AllocTensor<float>();
            softmaxLogitsTensor = softmaxLogitsQue_.AllocTensor<float>();
        } else {
            pipe.InitBuffer(sumExpLogitsQue_, 1, elementsNumber * sizeof(float));
            pipe.InitBuffer(predictedLogitsQue_, 1, elementsNumber * sizeof(float));
            sumExpLogitsTensor = sumExpLogitsQue_.AllocTensor<float>();
            predictedLogitsTensor = predictedLogitsQue_.AllocTensor<float>();
        }
    }

    template <typename C>
    __aicore__ inline void GMToUB(
        GlobalTensor<C>& gm_, LocalTensor<float>& tensor_, int64_t copyOffset_, int64_t length)
    {
        LocalTensor<C> tmpTensor = tensor_.template ReinterpretCast<C>();
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(length * sizeof(C)), 0, 0, 0};
        DataCopyPadExtParams<C> padParams = {true, 0, 0, 0};
        DataCopyPad(tmpTensor[elementsNumber], gm_[copyOffset_], copyParams, padParams);
        SyncM2toV();
        Cast(tensor_, tmpTensor[elementsNumber], RoundMode::CAST_NONE, length);
    }

    __aicore__ inline void GMToUB(
        GlobalTensor<float>& gm_, LocalTensor<float>& tensor_, int64_t copyOffset_, int64_t length)
    {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(length * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams = {true, 0, 0, 0};
        DataCopyPad(tensor_, gm_[copyOffset_], copyParams, padParams);
        SyncM2toV();
    }

    __aicore__ inline void UBToGM(
        GlobalTensor<float>& gm_, LocalTensor<float>& tensor_, int64_t copyOffset_, int32_t reallength)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(reallength * sizeof(float)), 0, 0, 0};
        DataCopyPad(gm_[copyOffset_], tensor_, copyParams);
        SyncM3toM2();
    }

    __aicore__ inline void BtCompute(int64_t length)
    {
        GMToUB(sumExpLogitsGm_, sumExpLogitsTensor, baseBtOffset, length);
        GMToUB(predictedLogitsGm_, predictedLogitsTensor, baseBtOffset, length);
        GMToUB(logitsMaxGm_, logitsMaxTensor, baseBtOffset, length);
        Log(logTensor, sumExpLogitsTensor, length);
        Sub(predictedLogitsTensor, logTensor, predictedLogitsTensor, length);
        SyncVtoM3();
        UBToGM(lossGm_, predictedLogitsTensor, baseBtOffset, length);
        SyncM2toS();
        for (int j = 0; j < length; j++) {
            vocOffset = (baseBtOffset + j) * vLen;
            for (int k = 0; k < singleCalLen; k++) {
                Compute(-logitsMaxTensor.GetValue(j), 1.0f / sumExpLogitsTensor.GetValue(j), elementsNumber);
                vocOffset += elementsNumber;
            }
        }
        if (singleCalReservedLen > 0) {
            vocOffset = baseBtOffset * vLen + singleCalLen * elementsNumber;
            for (int j = 0; j < length; j++) {
                Compute(-logitsMaxTensor.GetValue(j), 1.0f / sumExpLogitsTensor.GetValue(j), singleCalReservedLen);
                vocOffset = vocOffset + vLen;
            }
        }
    }

    __aicore__ inline void BtComputeForMemory(int64_t length)
    {
        GMToUB(sumExpLogitsGm_, sumExpLogitsTensor, baseBtOffset, length);
        GMToUB(predictedLogitsGm_, predictedLogitsTensor, baseBtOffset, length);
        Log(sumExpLogitsTensor, sumExpLogitsTensor, length);
        Sub(predictedLogitsTensor, sumExpLogitsTensor, predictedLogitsTensor, length);
        SyncVtoM3();
        UBToGM(lossGm_, predictedLogitsTensor, baseBtOffset, length);
    }

    __aicore__ inline void Compute(float logicsMax, float sumExpLogis, int64_t length)
    {
        GMToUB(vocabParallelLogitsGm_, vocabParallelLogitsTensor, vocOffset, length);
        Adds(vocabParallelLogitsTensor, vocabParallelLogitsTensor, logicsMax, length);

        Exp(vocabParallelLogitsTensor, vocabParallelLogitsTensor, length);
        SyncM3toV();
        Muls(softmaxLogitsTensor, vocabParallelLogitsTensor, sumExpLogis, length);
        SyncVtoM3();
        UBToGM(softmaxLogitsGm_, softmaxLogitsTensor, vocOffset, length);
    }

private:
    TPipe* pipe_;
    GlobalTensor<T> vocabParallelLogitsGm_;
    GlobalTensor<float> logitsMaxGm_;
    GlobalTensor<float> sumExpLogitsGm_;
    GlobalTensor<float> predictedLogitsGm_;
    GlobalTensor<float> softmaxLogitsGm_;
    GlobalTensor<float> lossGm_;

    TQue<TPosition::VECIN, DOUBLE_BUFFER> vocabParallelLogitsQue_;
    TQue<TPosition::VECIN, 1> logitsMaxQue_;
    TQue<TPosition::VECIN, 1> sumExpLogitsQue_;
    TQue<TPosition::VECIN, 1> predictedLogitsQue_;
    TQue<TPosition::VECOUT, 1> logQue_;
    TQue<TPosition::VECOUT, 1> softmaxLogitsQue_;

    LocalTensor<float> vocabParallelLogitsTensor;
    LocalTensor<float> logitsMaxTensor;
    LocalTensor<float> sumExpLogitsTensor;
    LocalTensor<float> predictedLogitsTensor;
    LocalTensor<float> logTensor;
    LocalTensor<float> softmaxLogitsTensor;

    uint32_t btCountTime = 0;
    uint32_t btCountLen = 0;
    uint32_t reservedbtNum = 0;
    uint32_t singleCalLen = 0;
    uint32_t singleCalReservedLen = 0;
    uint32_t elementsNumber = 0;

    int64_t vLen = 0;
    int64_t baseBtOffset = 0;
    int64_t vocOffset = 0;
};
} // namespace AscendC

#endif // FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_H_DETERMINIST_H