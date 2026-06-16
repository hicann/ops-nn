/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_
#define RMS_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_

#include "rms_norm_dynamic_quant_base.h"

template <typename T, typename T_Y, int TILING_KEY, int BUFFER_NUM = 1>
class KernelRmsNormDynamicQuantSingleRow : public KernelRmsNormDynamicQuantBase<T, T_Y, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelRmsNormDynamicQuantSingleRow(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR smooth, GM_ADDR beta, GM_ADDR y, GM_ADDR outScale,
        GM_ADDR workspace, const RmsNormDynamicQuantTilingData* tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x, gamma, smooth, beta);
        this->InitOutGlobalTensors(y, outScale);

        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, 2 * this->numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(yQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T));

        Ppipe->InitBuffer(xBufFp32, this->numLastDimAligned * sizeof(float));
        Ppipe->InitBuffer(yBufFp32, this->numLastDimAligned * sizeof(float));
        Ppipe->InitBuffer(smoothBuf, this->numLastDimAligned * sizeof(T));

        Ppipe->InitBuffer(scalesQue, BUFFER_NUM, ROW_FACTOR * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->smoothExist) {
            LocalTensor<T> smoothLocal = smoothBuf.template Get<T>();
            DataCopyEx(smoothLocal, this->smoothGm, this->numLastDim);
        }

        int32_t outLoopCount = this->rowWork / ROW_FACTOR;
        int32_t outLoopTail = this->rowWork % ROW_FACTOR;
        uint32_t gmOffset = 0;
        uint32_t gmOffsetReduce = 0;

        LocalTensor<float> scalesLocalOut;

        for (int32_t loopIdx = 0; loopIdx < outLoopCount; ++loopIdx) {
            scalesLocalOut = scalesQue.template AllocTensor<float>();
            for (int32_t innerIdx = 0; innerIdx < ROW_FACTOR; ++innerIdx) {
                CopyInXAndGamma(gmOffset);
                ComputeRmsNorm(gmOffset);
                ComputeDynamicQuant(innerIdx, scalesLocalOut);
                CopyOut(gmOffset);
                gmOffset += this->numLastDim;
            }
            scalesQue.EnQue(scalesLocalOut);
            CopyOutScale(gmOffsetReduce, ROW_FACTOR);
            gmOffsetReduce += ROW_FACTOR;
        }
        {
            scalesLocalOut = scalesQue.template AllocTensor<float>();
            for (int32_t innerIdx = 0; innerIdx < outLoopTail; ++innerIdx) {
                CopyInXAndGamma(gmOffset);
                ComputeRmsNorm(gmOffset);
                ComputeDynamicQuant(innerIdx, scalesLocalOut);
                CopyOut(gmOffset);
                gmOffset += this->numLastDim;
            }
            scalesQue.EnQue(scalesLocalOut);
            CopyOutScale(gmOffsetReduce, outLoopTail);
        }
    }

private:
    __aicore__ inline void ComputeRmsNorm(int32_t gmOffset)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<T> iputLocal = inRowsQue.template DeQue<T>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        Cast(xLocalFp32, iputLocal, RoundMode::CAST_NONE, this->numLastDim);

        Mul(yLocalFp32, xLocalFp32, xLocalFp32, this->numLastDim);
        PipeBarrier<PIPE_V>();

        float squareSumTemp = ReduceSumHalfInterval(yLocalFp32, this->numLastDim);
        float rstdLocalTemp = 1 / sqrt(squareSumTemp * this->aveNum + this->eps);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(xLocalFp32, xLocalFp32, rstdLocalTemp, this->numLastDim);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> gammaLocal = iputLocal[this->numLastDimAligned];
        Cast(yLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->numLastDim);

        inRowsQue.FreeTensor(iputLocal);
        Mul(xLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);
        PipeBarrier<PIPE_V>();
        if (this->betaFlag == 1) {
            CopyInBeta();
            LocalTensor<T> betaLocal = inRowsQue.template DeQue<T>();
            Cast(yLocalFp32, betaLocal, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Add(xLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);
            PipeBarrier<PIPE_V>();
            inRowsQue.FreeTensor(betaLocal);
        }
    }

    __aicore__ inline void ComputeDynamicQuant(int32_t idx, LocalTensor<float>& scalesLocalOut)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<T_Y> yLocal = yQue.template AllocTensor<T_Y>();
        auto yOutLocal = yLocal[0];

        if (this->smoothExist) {
            LocalTensor<T> smoothLocal = smoothBuf.template Get<T>();
            Cast(yLocalFp32, smoothLocal, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Mul(yLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);
            PipeBarrier<PIPE_V>();
        } else {
            Muls(yLocalFp32, xLocalFp32, (float)1.0, this->numLastDim);
            PipeBarrier<PIPE_V>();
        }
        ScaleTensor(yLocalFp32, xLocalFp32, scalesLocalOut, idx);
        PipeBarrier<PIPE_V>();
        RoundFloat2IntQuant<T_Y>(yOutLocal, yLocalFp32, this->numLastDim);
        PipeBarrier<PIPE_V>();
        yQue.EnQue(yLocal);
    }

    __aicore__ inline void ScaleTensor(
        LocalTensor<float>& srcTensor, LocalTensor<float>& tmpTensor, LocalTensor<float>& scaleTensor, int32_t idx)
    {
        Abs(tmpTensor, srcTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(tmpTensor, this->numLastDim);
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float maxTemp = tmpTensor.GetValue(0);
        float scaleTemp = this->quantMaxVal / maxTemp;
        scaleTensor.SetValue(idx, 1 / scaleTemp);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(srcTensor, srcTensor, scaleTemp, this->numLastDim);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOut(int32_t gmOffset)
    {
        LocalTensor<T_Y> res12 = yQue.template DeQue<T_Y>();
        auto res1 = res12[0];
        DataCopyEx(this->yGm[gmOffset], res1, this->numLastDim);
        yQue.FreeTensor(res12);
    }

    __aicore__ inline void CopyOutScale(int32_t gmOffset, int32_t copyInNums)
    {
        LocalTensor<float> outScalesLocal = scalesQue.template DeQue<float>();
        DataCopyEx(this->outScaleGm[gmOffset], outScalesLocal, copyInNums);
        scalesQue.FreeTensor(outScalesLocal);
    }

    __aicore__ inline void CopyInXAndGamma(int32_t gmOffset)
    {
        LocalTensor<T> xLocalIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(xLocalIn[0], this->xGm[gmOffset], this->numLastDim);
        DataCopyEx(xLocalIn[this->numLastDimAligned], this->gammaGm, this->numLastDim);
        inRowsQue.EnQue(xLocalIn);
    }

    __aicore__ inline void CopyInBeta()
    {
        LocalTensor<T> betaCopyIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(betaCopyIn[0], this->betaGm, this->numLastDim);
        inRowsQue.EnQue(betaCopyIn);
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scalesQue;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;

    TBuf<TPosition::VECCALC> smoothBuf;
};

#endif // RMS_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_
