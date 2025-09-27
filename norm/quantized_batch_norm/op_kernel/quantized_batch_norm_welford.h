/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantized_batch_norm_welford.h
 * \brief
 */
#ifndef QUANTIZED_BATCH_NORM_WELFORD_H
#define QUANTIZED_BATCH_NORM_WELFORD_H

#include "quantized_batch_norm_base.h"

namespace QuantizedBatchNormOps {
using namespace AscendC;

template <typename T1, typename T2, int SPLIT_MODE, int R0_ALIGN_MODE>
class QuantizedBatchNormWelford : public QuantizedBatchNormBase<T1, T2> {
public:
    __aicore__ inline QuantizedBatchNormWelford(TPipe* pipe)
    {
        this->pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR mean, GM_ADDR var, GM_ADDR input_scale, GM_ADDR input_zero_point, GM_ADDR output_scale,
        GM_ADDR output_zero_point, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
        const QuantizedBatchNormWelfordTilingData* __restrict tilingData)
    {
        patternR1 = tilingData->patternR1;
        patternA = tilingData->patternA;
        patternR0 = tilingData->patternR0;
        patternR0Align = tilingData->patternR0Align;
        aUbFactor = tilingData->aUbFactor;

        if (this->blockIdx == this->useCoreNum - 1) {
            aUbLoop = tilingData->tailCoreAUbLoop;
            aUbTail = tilingData->tailCoreAUbTail;
        } else {
            aUbLoop = tilingData->aUbLoop;
            aUbTail = tilingData->aUbTail;
        }
        nR0Loop = tilingData->nR0Loop;
        lastLoopNR0 = tilingData->lastLoopNR0;
        procNR0 = tilingData->procNR0;
        r0UbFactor = tilingData->r0UbFactor;
        r0UbLoop = tilingData->r0UbLoop;
        r0UbTail = tilingData->r0UbTail;

        this->epsilon = tilingData->epsilon;

        uint64_t aGmBlockOffset =
            static_cast<uint64_t>(this->blockIdx) * static_cast<uint64_t>(tilingData->blockFactor);
        uint64_t aR0GmBlockOffset = aGmBlockOffset * patternR0;
        uint64_t weightGmBlockOffset = aGmBlockOffset;
        this->xGm.SetGlobalBuffer((__gm__ T1*)x + aR0GmBlockOffset);
        this->weightGm.SetGlobalBuffer((__gm__ float*)weight + weightGmBlockOffset);
        this->biasGm.SetGlobalBuffer((__gm__ float*)bias + weightGmBlockOffset);
        this->meanGm.SetGlobalBuffer((__gm__ float*)mean + weightGmBlockOffset);
        this->varGm.SetGlobalBuffer((__gm__ float*)var + weightGmBlockOffset);
        this->inScaleGm.SetGlobalBuffer((__gm__ float*)input_scale);
        this->inZeroPointGm.SetGlobalBuffer((__gm__ int32_t*)input_zero_point);
        this->outScaleGm.SetGlobalBuffer((__gm__ float*)output_scale);
        this->outZeroPointGm.SetGlobalBuffer((__gm__ int32_t*)output_zero_point);

        this->yGm.SetGlobalBuffer((__gm__ T1*)y + aR0GmBlockOffset);

        this->pipe_->InitBuffer(xQueueHalfDequant0, r0UbFactor * HALF_SIZE);
        this->pipe_->InitBuffer(xQueueDequant0, r0UbFactor * FLOAT_SIZE);

        // 输入que
        this->pipe_->InitBuffer(xQueue, DOUBLE_BUFFER, r0UbFactor * sizeof(T1));
        this->pipe_->InitBuffer(
            wbmvsQueue, 1, aUbFactor * FLOAT_SIZE * MERGE_INPUT_NUM + MIN_BUFFER * FLOAT_SIZE * TWO_NUM);
        this->pipe_->InitBuffer(zeroPointQueue, 1, MIN_BUFFER * INT32_SIZE * TWO_NUM);

        // 输出que
        this->pipe_->InitBuffer(yQueue, DOUBLE_BUFFER, r0UbFactor * sizeof(T1));
    }

    __aicore__ inline void Process()
    {
        for (int64_t aUbLoopIdx = 0; aUbLoopIdx < aUbLoop; aUbLoopIdx++) {
            aUbLoopNowStartIdx = aUbLoopIdx * aUbFactor;
            if (unlikely(aUbLoopIdx == aUbLoop - 1)) {
                aProcNum = aUbTail;
            } else {
                aProcNum = aUbFactor;
            }

            int32_t calcount = static_cast<int32_t>(aProcNum);

            wbmvsTensor = wbmvsQueue.AllocTensor<float>();
            meanInTensor = wbmvsTensor[MEAN_OFFSET];
            varInTensor = wbmvsTensor[VAR_OFFSET * aUbFactor];
            weightTensor = wbmvsTensor[WEIGHT_OFFSET * aUbFactor];
            biasTensor = wbmvsTensor[BIAS_OFFSET * aUbFactor];
            inScaleTensor = wbmvsTensor[INSCALE_OFFSET * aUbFactor];
            outScaleTensor = wbmvsTensor[INSCALE_OFFSET * aUbFactor + OUTSCALE_OFFSET * MIN_BUFFER];
            zeroPointTensor = zeroPointQueue.AllocTensor<int32_t>();
            inZeroPointTensor = zeroPointTensor[INZP_OFFSET];
            outZeroPointTensor = zeroPointTensor[OUTZP_OFFSET * MIN_BUFFER];
            CopyInMeanVar(meanInTensor, varInTensor, aProcNum, aUbLoopNowStartIdx);
            CopyInWeightBias(weightTensor, biasTensor, aProcNum, aUbLoopNowStartIdx);
            CopyInScaleZeroPoint(1, 0);
            CalAlphaAndBetaWithQuantTensor();
            // 计算y
            CalcYAndCopyOut();

            wbmvsQueue.FreeTensor(wbmvsTensor);
            zeroPointQueue.FreeTensor(zeroPointTensor);
        }
    }

    __aicore__ inline void CalAlphaAndBetaWithQuantTensor()
    {
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventID);
        TEventID eventIDSV = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(eventIDSV);
        WaitFlag<HardEvent::S_V>(eventIDSV);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(eventIDSV);
        Adds(varInTensor, varInTensor, this->epsilon, aProcNum);
        PipeBarrier<PIPE_V>();
        Sqrt(varInTensor, varInTensor, aProcNum);
        PipeBarrier<PIPE_V>();
        Div(varInTensor, weightTensor, varInTensor, aProcNum);
        PipeBarrier<PIPE_V>();
        Muls(weightTensor, varInTensor, static_cast<float>(-1.0f), aProcNum);
        PipeBarrier<PIPE_V>();
        Muls(varInTensor, varInTensor, inputScale / outputScale, aProcNum);
        PipeBarrier<PIPE_V>();
        Mul(weightTensor, weightTensor, meanInTensor, aProcNum);
        PipeBarrier<PIPE_V>();
        Add(weightTensor, weightTensor, biasTensor, aProcNum);
        PipeBarrier<PIPE_V>();
        Muls(weightTensor, weightTensor, static_cast<float>(1.0f / outputScale), aProcNum);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CalcYAndCopyOut()
    {
        PipeBarrier<PIPE_ALL>();
        for (int64_t aNum = 0; aNum < aProcNum; aNum++) {
            alphaValue = varInTensor.GetValue(aNum);
            betaValue = weightTensor.GetValue(aNum);
            if constexpr (SPLIT_MODE == R0_SPLIT_NOT_ALIGN_MODE || SPLIT_MODE == R0_SPLIT_ALIGN_MODE) {
                CalcYAndCopyOutR0SPLIT(aNum);
            } else {
                CalcYAndCopyOutR0NOTSPLIT(aNum);
            }
        }
    }

private:
    __aicore__ inline void CopyInXAndDequant(
        LocalTensor<T1>& xTensor, const int64_t copyInSize, const uint64_t copyInGmOffset)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(copyInSize * sizeof(T1));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventID);
        DataCopyPad(xTensor, this->xGm[copyInGmOffset], intriParams, padParams);
        DequantX(xTensor, intriParams.blockCount * copyInSize);
    }

    __aicore__ inline void CopyInNR0AndDequant(
        LocalTensor<T1>& xTensor, const int64_t copyInR0LineNum, const uint64_t copyInGmOffset)
    {
        DataCopyPadExtParams<T1> padParams = {true, 0, static_cast<uint8_t>(patternR0Align - patternR0), 0};
        if constexpr (R0_ALIGN_MODE == R0_ALIGN) {
            padParams = {false, 0, 0, 0};
        }
        DataCopyExtParams intriParams;
        intriParams.blockCount = static_cast<uint16_t>(copyInR0LineNum);      // 多少batch
        intriParams.blockLen = static_cast<uint32_t>(patternR0 * sizeof(T1)); // 每个batch多少个
        intriParams.srcStride = static_cast<uint32_t>((patternA - 1) * patternR0 * sizeof(T1));
        intriParams.dstStride = 0;
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventID);
        DataCopyPad(xTensor, this->xGm[copyInGmOffset], intriParams, padParams);
        DequantNR0(xTensor, intriParams.blockCount * patternR0Align);
    }

    __aicore__ inline void CopyOutY(const int64_t copyOutSize, const uint64_t copyOutGmOffset)
    {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(copyOutSize * sizeof(T1));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventID);
        DataCopyPad(this->yGm[copyOutGmOffset], xTensor, intriParams);
    }

    __aicore__ inline void CopyOutNR0Y(const int64_t copyOutLineNum, const uint64_t copyOutGmOffset)
    {
        DataCopyExtParams intriParams;
        intriParams.blockCount = static_cast<uint16_t>(copyOutLineNum);
        intriParams.blockLen = static_cast<uint32_t>(patternR0 * sizeof(T1));
        intriParams.srcStride = 0;
        intriParams.dstStride = static_cast<uint32_t>((patternA - 1) * patternR0 * sizeof(T1));
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventID);
        DataCopyPad(this->yGm[copyOutGmOffset], xTensor, intriParams);
    }

    __aicore__ inline void CopyInWeightBias(
        LocalTensor<float>& weightTensor, LocalTensor<float>& biasTensor, const int64_t copyInSize,
        const uint64_t copyInGmOffset)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(copyInSize * sizeof(float));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(weightTensor, this->weightGm[copyInGmOffset], intriParams, padParams);
        DataCopyPad(biasTensor, this->biasGm[copyInGmOffset], intriParams, padParams);
    }

    __aicore__ inline void CopyInMeanVar(
        LocalTensor<float>& meanInTensor, LocalTensor<float>& varInTensor, const int64_t copyInSize,
        const uint64_t copyInGmOffset)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(copyInSize * sizeof(float));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(meanInTensor, this->meanGm[copyInGmOffset], intriParams, padParams);
        DataCopyPad(varInTensor, this->varGm[copyInGmOffset], intriParams, padParams);
    }

    __aicore__ inline void CopyInScaleZeroPoint(const int64_t copyInSize, const uint64_t copyInGmOffset)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint16_t>(copyInSize * sizeof(float));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(inScaleTensor, this->inScaleGm[copyInGmOffset], intriParams, padParams);
        DataCopyPad(inZeroPointTensor, this->inZeroPointGm[copyInGmOffset], intriParams, padParams);
        DataCopyPad(outScaleTensor, this->outScaleGm[copyInGmOffset], intriParams, padParams);
        DataCopyPad(outZeroPointTensor, this->outZeroPointGm[copyInGmOffset], intriParams, padParams);
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>();
        SetFlag<HardEvent::MTE2_S>(eventID);
        WaitFlag<HardEvent::MTE2_S>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventID);
        inputScale = inScaleTensor.GetValue(0);
        inputZeroPoint = inZeroPointTensor.GetValue(0);
        outputScale = outScaleTensor.GetValue(0);
        outputZeroPoint = outZeroPointTensor.GetValue(0);
    }

    __aicore__ inline void QuantX(LocalTensor<float>& xQueueDequantTensor, uint32_t len)
    {
        PipeBarrier<PIPE_V>();
        Adds(xQueueDequantTensor, xQueueDequantTensor, static_cast<float>(outputZeroPoint), len);
        ClearInvalidNum(xQueueDequantTensor, len);
        if constexpr (!IsSameType<T1, int32_t>::value) {
            PipeBarrier<PIPE_V>();
            Cast(xQueueHalfDequantTensor1, xQueueDequantTensor, RoundMode::CAST_NONE, len);
            PipeBarrier<PIPE_V>();
            Cast(xTensor, xQueueHalfDequantTensor1, RoundMode::CAST_NONE, len);
        } else {
            PipeBarrier<PIPE_V>();
            Cast(xTensor, xQueueDequantTensor, RoundMode::CAST_ROUND, len);
        }
        xQueueDequant0.FreeTensor(xQueueDequantTensor);
    }

    __aicore__ inline void ClearInvalidNum(LocalTensor<float>& xQueueDequantTensor, uint32_t len)
    {
        LocalTensor<int32_t> xDequantInt32Tensor = xQueueDequantTensor.ReinterpretCast<int32_t>();
        PipeBarrier<PIPE_V>();
        Cast(xDequantInt32Tensor, xQueueDequantTensor, RoundMode::CAST_ROUND, len);
        PipeBarrier<PIPE_V>();
        Cast(xQueueDequantTensor, xDequantInt32Tensor, RoundMode::CAST_NONE, len);
    }

    __aicore__ inline void DequantX(LocalTensor<T1>& xTensor, uint32_t len)
    {
        xQueueDequantTensor1 = xQueueDequant0.Get<float>();
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventID);
        if constexpr (!IsSameType<T1, int32_t>::value) {
            xQueueHalfDequantTensor1 = xQueueHalfDequant0.Get<half>();
            PipeBarrier<PIPE_V>();
            Cast(xQueueHalfDequantTensor1, xTensor, RoundMode::CAST_NONE, len);
            PipeBarrier<PIPE_V>();
            Cast(xQueueDequantTensor1, xQueueHalfDequantTensor1, RoundMode::CAST_NONE, len);
            PipeBarrier<PIPE_V>();
        } else {
            PipeBarrier<PIPE_V>();
            Cast(xQueueDequantTensor1, xTensor, RoundMode::CAST_NONE, len);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
        Adds(xQueueDequantTensor1, xQueueDequantTensor1, static_cast<float>(-1.0f * inputZeroPoint), len);
    }

    __aicore__ inline void DequantNR0(LocalTensor<T1>& xTensor, uint32_t len)
    {
        xQueueDequantTensor1 = xQueueDequant0.Get<float>();
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventID);
        if constexpr (!IsSameType<T1, int32_t>::value) {
            xQueueHalfDequantTensor1 = xQueueHalfDequant0.Get<half>();
            Cast(xQueueHalfDequantTensor1, xTensor, RoundMode::CAST_NONE, len);
            PipeBarrier<PIPE_V>();
            Cast(xQueueDequantTensor1, xQueueHalfDequantTensor1, RoundMode::CAST_NONE, len);
        } else {
            PipeBarrier<PIPE_V>();
            Cast(xQueueDequantTensor1, xTensor, RoundMode::CAST_NONE, len);
        }
        PipeBarrier<PIPE_V>();
        Adds(xQueueDequantTensor1, xQueueDequantTensor1, static_cast<float>(-1.0f * inputZeroPoint), len);
    }

    __aicore__ inline void CalcYAndCopyOutR0SPLIT(int64_t aNum)
    {
        for (int64_t r1LoopIdx = 0; r1LoopIdx < patternR1; r1LoopIdx++) {
            r0ProcNum = r0UbFactor;
            for (int64_t r0LoopIdx = 0; r0LoopIdx < r0UbLoop; r0LoopIdx++) {
                xGmOffset =
                    r1LoopIdx * patternA * patternR0 + (aUbLoopNowStartIdx + aNum) * patternR0 + r0LoopIdx * r0UbFactor;
                if (unlikely(r0LoopIdx == r0UbLoop - 1)) {
                    r0ProcNum = r0UbTail;
                }
                xTensor = xQueue.AllocTensor<T1>();
                CopyInXAndDequant(xTensor, r0ProcNum, xGmOffset);
                PipeBarrier<PIPE_V>();
                TEventID sveventID = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
                SetFlag<HardEvent::S_V>(sveventID);
                WaitFlag<HardEvent::S_V>(sveventID);
                GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(sveventID);
                Muls(xQueueDequantTensor1, xQueueDequantTensor1, alphaValue, r0ProcNum);
                PipeBarrier<PIPE_V>();
                Adds(xQueueDequantTensor1, xQueueDequantTensor1, betaValue, r0ProcNum);
                PipeBarrier<PIPE_V>();
                QuantX(xQueueDequantTensor1, r0ProcNum);
                CopyOutY(r0ProcNum, xGmOffset);
                xQueue.FreeTensor(xTensor);
            }
        }
    }

    __aicore__ inline void CalcYAndCopyOutR0NOTSPLIT(int64_t aNum)
    {
        for (int64_t nR0LoopIdx = 0; nR0LoopIdx < nR0Loop; nR0LoopIdx++) {
            r0ProcNum = procNR0 * patternR0Align;
            int64_t prcoR0LineNum = procNR0;
            xGmOffset = nR0LoopIdx * procNR0 * patternA * patternR0 + (aUbLoopNowStartIdx + aNum) * patternR0;
            if (unlikely(nR0LoopIdx == nR0Loop - 1)) {
                r0ProcNum = lastLoopNR0 * patternR0Align;
                prcoR0LineNum = lastLoopNR0;
            }
            xTensor = xQueue.AllocTensor<T1>();
            CopyInNR0AndDequant(xTensor, prcoR0LineNum, xGmOffset);
            PipeBarrier<PIPE_V>();
            TEventID s2veventID = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
            SetFlag<HardEvent::S_V>(s2veventID);
            WaitFlag<HardEvent::S_V>(s2veventID);
            GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(s2veventID);
            Muls(xQueueDequantTensor1, xQueueDequantTensor1, alphaValue, r0ProcNum);
            PipeBarrier<PIPE_V>();
            Adds(xQueueDequantTensor1, xQueueDequantTensor1, betaValue, r0ProcNum);
            PipeBarrier<PIPE_V>();
            QuantX(xQueueDequantTensor1, r0ProcNum);
            CopyOutNR0Y(prcoR0LineNum, xGmOffset);
            xQueue.FreeTensor(xTensor);
        }
    }

private:
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static uint32_t X_NUM_PER_BLOCK = BLOCK_SIZE / sizeof(T1);
    constexpr static uint32_t HALF_SIZE = 2;
    constexpr static uint32_t FLOAT_SIZE = 4;
    constexpr static uint32_t INT8_SIZE = 1;
    constexpr static uint32_t INT16_SIZE = 2;
    constexpr static uint32_t INT32_SIZE = 4;
    constexpr static uint32_t TWO_NUM = 2;
    constexpr static uint32_t DOUBLE_BUFFER = 2;
    constexpr static uint32_t MERGE_INPUT_NUM = 4;
    constexpr static uint32_t ELEM_PER_REP_FP32 = 64;
    constexpr static uint32_t BLOCK_NUM_PER_REP = 8;
    constexpr static uint32_t B32_BLOCK_ALIGN_NUM = 8;
    constexpr static uint32_t UINT8_MAX_NUM = 255;
    constexpr static int R0_SPLIT_NOT_ALIGN_MODE = 0;
    constexpr static int R0_SPLIT_ALIGN_MODE = 1;
    constexpr static int R1_SPLIT_NOT_ALIGN_MODE = 2;
    constexpr static int R1_SPLIT_ALIGN_MODE = 3;
    constexpr static int R0_NOT_ALIGN = 0;
    constexpr static int R0_ALIGN = 1;
    constexpr static int MIN_BUFFER = 8;
    constexpr static int32_t MEAN_OFFSET = 0;
    constexpr static int32_t VAR_OFFSET = 1;
    constexpr static int32_t WEIGHT_OFFSET = 2;
    constexpr static int32_t BIAS_OFFSET = 3;
    constexpr static int32_t INSCALE_OFFSET = 4;
    constexpr static int32_t OUTSCALE_OFFSET = 1;
    constexpr static int32_t INZP_OFFSET = 0;
    constexpr static int32_t OUTZP_OFFSET = 1;

    /* shape variable*/
    int64_t patternR1;
    int64_t patternA;
    int64_t patternR0;
    int64_t patternR0Align;
    /* spilt variable */
    int64_t aUbFactor;
    int64_t aUbLoop;
    int64_t aUbTail;
    int64_t r0UbFactor;
    int64_t r0UbLoop;
    int64_t r0UbTail;
    int64_t procNR0;
    int64_t nR0Loop;
    int64_t lastLoopNR0;

    int64_t aProcNum;
    int64_t r0ProcNum;
    /* offset variable*/
    uint64_t aUbLoopNowStartIdx;
    uint64_t xGmOffset;

    /* calculate variable*/
    float count;
    uint64_t acc_val = 0;
    float alphaValue = 0.0;
    float betaValue = 0.0;
    float inputScale;
    int32_t inputZeroPoint;
    float outputScale;
    int32_t outputZeroPoint;
    /* ascendc variable */
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQueue;
    TQue<QuePosition::VECIN, 1> wbmvsQueue;
    TQue<QuePosition::VECIN, 1> zeroPointQueue;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQueue;
    TBuf<TPosition::VECCALC> xQueueDequant0;
    TBuf<TPosition::VECCALC> xQueueHalfDequant0;

    LocalTensor<float> wbmvsTensor;
    LocalTensor<float> meanInTensor;
    LocalTensor<float> varInTensor;
    LocalTensor<float> meanOutTensor;
    LocalTensor<float> varOutTensor;
    LocalTensor<float> saveMeanTensor;
    LocalTensor<float> saveVarTensor;
    LocalTensor<float> momentumMeanTensor;
    LocalTensor<float> momentumVarTensor;
    LocalTensor<float> weightTensor;
    LocalTensor<float> biasTensor;
    LocalTensor<float> inScaleTensor;
    LocalTensor<float> outScaleTensor;
    LocalTensor<int32_t> zeroPointTensor;
    LocalTensor<int32_t> inZeroPointTensor;
    LocalTensor<int32_t> outZeroPointTensor;

    LocalTensor<float> meanTensor;
    LocalTensor<float> m2Tensor;
    LocalTensor<T1> xTensor;
    LocalTensor<half> xQueueHalfDequantTensor1;
    LocalTensor<float> xQueueDequantTensor1;
    LocalTensor<T1> yTensor;
};
} // namespace QuantizedBatchNormOps
#endif // QUANTIZED_BATCH_NORM_WELFORD_H