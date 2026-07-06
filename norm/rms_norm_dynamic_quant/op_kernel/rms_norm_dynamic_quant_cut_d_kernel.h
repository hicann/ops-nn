/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_DYNAMIC_QUANT_CUT_D_KERNEL_H_
#define RMS_NORM_DYNAMIC_QUANT_CUT_D_KERNEL_H_

#include "rms_norm_dynamic_quant_base.h"

template <typename T, typename T_Y, int TILING_KEY, int BUFFER_NUM = 1>
class KernelRmsNormDynamicQuantSliceD : public KernelRmsNormDynamicQuantBase<T, T_Y, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelRmsNormDynamicQuantSliceD(TPipe* pipe) { Ppipe = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR smooth, GM_ADDR beta, GM_ADDR y, GM_ADDR outScale,
                                GM_ADDR workspace, const RmsNormDynamicQuantTilingData* tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x, gamma, smooth, beta);
        this->InitOutGlobalTensors(y, outScale);

        workspaceGm.SetGlobalBuffer((__gm__ float*)(workspace) + this->blockIdx_ * this->numLastDim);

        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, 2 * this->lastDimSliceLen * sizeof(T));
        Ppipe->InitBuffer(outRowQue, BUFFER_NUM, this->lastDimSliceLen * sizeof(T));
        Ppipe->InitBuffer(tmpOutQue, BUFFER_NUM, this->lastDimSliceLen * sizeof(float));
        Ppipe->InitBuffer(xBufFp32, this->lastDimSliceLen * sizeof(float));
        Ppipe->InitBuffer(yBufFp32, this->lastDimSliceLen * sizeof(float));
        Ppipe->InitBuffer(zBufFp32, this->lastDimSliceLen * sizeof(float));
        Ppipe->InitBuffer(scalesQue, BUFFER_NUM, ELEM_PER_BLK_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t baseGmOffset = 0;
        uint32_t rowGmOffset = 0;
        for (int32_t rowIdx = 0; rowIdx < this->rowWork; ++rowIdx) {
            rowGmOffset = 0;
            this->localSum = ZERO;
            this->localMax1 = ZERO;
            for (int32_t colIdx = 0; colIdx < this->lastDimLoopNum; ++colIdx) {
                CopyInX(baseGmOffset, rowGmOffset, this->lastDimSliceLen);
                this->localSum += ReduceSquareSumSlice(this->lastDimSliceLen);
                PipeBarrier<PIPE_V>();
                rowGmOffset += this->lastDimSliceLen;
            }
            {
                CopyInX(baseGmOffset, rowGmOffset, this->lastDimSliceLenTail);
                this->localSum += ReduceSquareSumSlice(this->lastDimSliceLenTail);
                PipeBarrier<PIPE_V>();
            }
            float rstdLocalTemp = 1 / sqrt(this->localSum * this->aveNum + this->eps);
            PIPE_S_V();
            PIPE_MTE3_MTE2();

            rowGmOffset = 0;
            for (int32_t colIdx = 0; colIdx < this->lastDimLoopNum; ++colIdx) {
                ComputeRmsNormAndSmoothMax(baseGmOffset, rowGmOffset, this->lastDimSliceLen, rstdLocalTemp);
                rowGmOffset += this->lastDimSliceLen;
            }
            {
                ComputeRmsNormAndSmoothMax(baseGmOffset, rowGmOffset, this->lastDimSliceLenTail, rstdLocalTemp);
            }
            this->localMax1 = this->quantMaxVal / this->localMax1;
            PIPE_S_V();
            PIPE_MTE3_MTE2();

            rowGmOffset = 0;
            for (int32_t colIdx = 0; colIdx < this->lastDimLoopNum; ++colIdx) {
                ComputeDynamicQuant(rowGmOffset, this->lastDimSliceLen);
                CopyOutQuant(baseGmOffset, rowGmOffset, this->lastDimSliceLen);
                rowGmOffset += this->lastDimSliceLen;
            }
            {
                ComputeDynamicQuant(rowGmOffset, this->lastDimSliceLenTail);
                CopyOutQuant(baseGmOffset, rowGmOffset, this->lastDimSliceLenTail);
            }
            LocalTensor<float> scalesTensor = scalesQue.template AllocTensor<float>();
            scalesTensor.SetValue(0, 1 / this->localMax1);
            scalesQue.EnQue(scalesTensor);
            CopyOutScale(rowIdx);

            baseGmOffset += this->numLastDim;
        }
    }

private:
    __aicore__ inline void ComputeDynamicQuant(int32_t rowGmOffset, int32_t elementCount)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<T_Y> y12Local = outRowQue.template AllocTensor<T_Y>();
        CopyInSmoothNorm(xLocalFp32, rowGmOffset, elementCount, this->localMax1);
        auto y1Local = y12Local[0];
        RoundFloat2IntQuant<T_Y>(y1Local, xLocalFp32, elementCount);
        outRowQue.template EnQue<T_Y>(y12Local);
    }

    __aicore__ inline void CopyOutQuant(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        LocalTensor<T_Y> yOut = outRowQue.template DeQue<T_Y>();
        DataCopyEx(this->yGm[baseGmOffset + rowGmOffset], yOut, elementCount);
        outRowQue.FreeTensor(yOut);
    }

    __aicore__ inline void CopyOutScale(int32_t idx)
    {
        LocalTensor<float> scalesOut = scalesQue.template DeQue<float>();
        DataCopyEx(this->outScaleGm[idx], scalesOut[0], 1);
        scalesQue.FreeTensor(scalesOut);
    }

    __aicore__ inline void CopyInSmoothNorm(LocalTensor<float>& dstLocal, int32_t rowGmOffset, int32_t elementCount,
                                            float localMaxVal)
    {
        LocalTensor<T> smoothLocal = inRowsQue.template AllocTensor<T>();
        if (this->smoothExist) {
            DataCopyEx(smoothLocal[0], this->smoothGm[rowGmOffset], elementCount);
            Cast(dstLocal, smoothLocal, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Muls(dstLocal, dstLocal, localMaxVal, elementCount);
        } else {
            Muls(dstLocal, dstLocal, localMaxVal, elementCount);
        }
        PipeBarrier<PIPE_V>();
        inRowsQue.FreeTensor(smoothLocal);
    }

    __aicore__ inline void ComputeRmsNormAndSmoothMax(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount,
                                                      float rstdLocalTemp)
    {
        LocalTensor<T> xLocalIn = inRowsQue.template AllocTensor<T>();
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();

        DataCopyEx(xLocalIn[0], this->xGm[baseGmOffset + rowGmOffset], elementCount);
        DataCopyEx(xLocalIn[this->lastDimSliceLen], this->gammaGm[rowGmOffset], elementCount);

        Cast(xLocalFp32, xLocalIn[0], RoundMode::CAST_NONE, elementCount);
        Cast(yLocalFp32, xLocalIn[this->lastDimSliceLen], RoundMode::CAST_NONE, elementCount);

        Mul(xLocalFp32, xLocalFp32, yLocalFp32, elementCount); // xLocalFp32 <- x * gamma
        PipeBarrier<PIPE_V>();
        Muls(xLocalFp32, xLocalFp32, rstdLocalTemp, elementCount); // xLocalFp32 <- x * rstd * gamma
        PipeBarrier<PIPE_V>();

        if (this->betaFlag) {
            LocalTensor<T> betaLocal = inRowsQue.template AllocTensor<T>();
            DataCopyEx(betaLocal[0], this->betaGm[rowGmOffset], elementCount);
            Cast(yLocalFp32, betaLocal, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Add(xLocalFp32, xLocalFp32, yLocalFp32, elementCount);
            PipeBarrier<PIPE_V>();
            inRowsQue.FreeTensor(betaLocal);
        }

        if (this->smoothExist) {
            LocalTensor<T> smoothLocal = inRowsQue.template AllocTensor<T>();
            DataCopyEx(smoothLocal[0], this->smoothGm[rowGmOffset], elementCount);
            Cast(yLocalFp32, smoothLocal, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Mul(zLocalFp32, xLocalFp32, yLocalFp32, elementCount); // zLocalFp32 <- y * smooth
            inRowsQue.FreeTensor(smoothLocal);
        } else {
            Muls(zLocalFp32, xLocalFp32, (float)1.0, elementCount); // zLocalFp32 <- y
        }
        PipeBarrier<PIPE_V>();
        float tmpMax1 = FindSliceMax(zLocalFp32, xLocalFp32, elementCount);
        this->localMax1 = (tmpMax1 > this->localMax1) ? tmpMax1 : localMax1;
        inRowsQue.FreeTensor(xLocalIn);
    }

    __aicore__ inline float ReduceSquareSumSlice(int32_t elementCount)
    {
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();
        LocalTensor<T> xLocalIn = inRowsQue.template DeQue<T>();
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        Cast(xLocalFp32, xLocalIn, RoundMode::CAST_NONE, elementCount);
        PipeBarrier<PIPE_V>();
        Mul(zLocalFp32, xLocalFp32, xLocalFp32, elementCount);
        PipeBarrier<PIPE_V>();
        float sumTemp = ReduceSumHalfInterval(zLocalFp32, elementCount);
        inRowsQue.FreeTensor(xLocalIn);
        return sumTemp;
    }

    __aicore__ inline void CopyInX(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        LocalTensor<T> x32Local = inRowsQue.template AllocTensor<T>();
        DataCopyEx(x32Local[0], this->xGm[baseGmOffset + rowGmOffset], elementCount);
        inRowsQue.EnQue(x32Local);
    }

    __aicore__ inline float FindSliceMax(LocalTensor<float>& srcTensor, LocalTensor<float>& tmpTensor,
                                         int32_t elementCount)
    {
        Abs(tmpTensor, srcTensor, elementCount);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(tmpTensor, elementCount);
        PIPE_V_S();
        float maxTemp = tmpTensor.GetValue(0);
        return maxTemp;
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outRowQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> tmpOutQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scalesQue;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;
    TBuf<TPosition::VECCALC> zBufFp32;

    float localSum;
    float localMax1;

    __aicore__ inline void PIPE_MTE3_MTE2()
    {
        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
    }

    __aicore__ inline void PIPE_S_V()
    {
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
    }

    __aicore__ inline void PIPE_V_S()
    {
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    }

    GlobalTensor<float> workspaceGm;
};

#endif // RMS_NORM_DYNAMIC_QUANT_CUT_D_KERNEL_H_
