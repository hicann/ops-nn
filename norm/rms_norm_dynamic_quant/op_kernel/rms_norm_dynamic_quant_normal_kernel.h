/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_
#define RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_

#include "rms_norm_dynamic_quant_base.h"

template <typename T, typename T_Y, int TILING_KEY, int BUFFER_NUM = 1>
class KernelRmsNormDynamicQuantNormal : public KernelRmsNormDynamicQuantBase<T, T_Y, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelRmsNormDynamicQuantNormal(TPipe* pipe)
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
        this->numRowsAligned = (this->rowStep + ELEM_PER_BLK_FP32 - 1) / ELEM_PER_BLK_FP32 * ELEM_PER_BLK_FP32;
        this->ubAligned = static_cast<uint32_t>((this->numLastDimAligned - this->numLastDim) / ELEM_PER_BLK_FP16);

        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, 2 * this->rowStep * this->numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(outRowsQue, BUFFER_NUM, 2 * this->rowStep * this->numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(xBufFp32, this->rowStep * this->numLastDimAligned * sizeof(float));
        Ppipe->InitBuffer(yBufFp32, this->rowStep * this->numLastDimAligned * sizeof(float));
        Ppipe->InitBuffer(weightBuf01, this->numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(weightBuf02, this->numLastDimAligned * sizeof(T));
        if (this->betaFlag == 1) {
            Ppipe->InitBuffer(weightBuf03, this->numLastDimAligned * sizeof(T));
        }
        Ppipe->InitBuffer(scalesBuf, this->numRowsAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t rowMoveCnt = CEIL_DIV(this->rowWork, this->rowStep);
        CopyInWeights();

        LocalTensor<T> gammaLocal = weightBuf01.template Get<T>();

        int32_t gmOffset = 0;
        int32_t gmOffsetScale = 0;
        int32_t elementCount = this->numLastDimAligned * this->rowStep;

        for (int32_t rowIdx = 0; rowIdx < rowMoveCnt - 1; ++rowIdx) {
            CopyInX(gmOffset, this->rowStep, elementCount);
            ComputeRmsNorm(this->rowStep, elementCount, gammaLocal);
            ComputeDynamicQuant(this->rowStep, elementCount);
            CopyOut(gmOffset, gmOffsetScale, this->rowStep);
            gmOffset += this->rowStep * this->numLastDim;
            gmOffsetScale += this->rowStep;
        }
        {
            elementCount = this->numLastDimAligned * this->rowTail_;
            CopyInX(gmOffset, this->rowTail_, elementCount);
            ComputeRmsNorm(this->rowTail_, elementCount, gammaLocal);
            ComputeDynamicQuant(this->rowTail_, elementCount);
            CopyOut(gmOffset, gmOffsetScale, this->rowTail_);
        }
    }

private:
    __aicore__ inline void CopyInX(int32_t gmOffset, int32_t rowCount, int32_t elementCount)
    {
        LocalTensor<T> xLocalIn = inRowsQue.template AllocTensor<T>();
        DataCopyExStride(xLocalIn, this->xGm[gmOffset], this->numLastDim, rowCount, this->ubAligned);
        inRowsQue.EnQue(xLocalIn);
    }

    __aicore__ inline void CopyInWeights()
    {
        LocalTensor<T> gammaLocal = weightBuf01.template Get<T>();
        DataCopyEx(gammaLocal, this->gammaGm, this->numLastDim);
        if (this->smoothExist) {
            LocalTensor<T> smoothLocal = weightBuf02.template Get<T>();
            DataCopyEx(smoothLocal, this->smoothGm, this->numLastDim);
        }
        if (this->betaFlag == 1) {
            LocalTensor<T> betaLocal = weightBuf03.template Get<T>();
            DataCopyEx(betaLocal, this->betaGm, this->numLastDim);
        }
    }

    __aicore__ inline void ComputeRmsNorm(int32_t nums, int32_t elementCount, LocalTensor<T>& gammaLocal)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<T> xInputLocal = inRowsQue.template DeQue<T>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        Cast(xLocalFp32, xInputLocal, RoundMode::CAST_NONE, elementCount);

        Mul(yLocalFp32, xLocalFp32, xLocalFp32, elementCount);
        PipeBarrier<PIPE_V>();

        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * this->numLastDimAligned;
            float squareSumTemp = ReduceSumHalfInterval(yLocalFp32[roundOffset], this->numLastDim);
            float rstdLocalTemp = 1 / sqrt(squareSumTemp * this->aveNum + this->eps);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Muls(xLocalFp32[roundOffset], xLocalFp32[roundOffset], rstdLocalTemp, this->numLastDim);
        }
        PipeBarrier<PIPE_V>();

        Cast(yLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->numLastDim);
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * this->numLastDimAligned;
            Mul(xLocalFp32[roundOffset], xLocalFp32[roundOffset], yLocalFp32, this->numLastDim);
            PipeBarrier<PIPE_V>();
        }

        if (this->betaFlag == 1) {
            LocalTensor<T> betaLocal = weightBuf03.template Get<T>();
            Cast(yLocalFp32, betaLocal, RoundMode::CAST_NONE, this->numLastDim);
            for (int32_t rid = 0; rid < nums; ++rid) {
                auto roundOffset = rid * this->numLastDimAligned;
                PipeBarrier<PIPE_V>();
                Add(xLocalFp32[roundOffset], xLocalFp32[roundOffset], yLocalFp32, this->numLastDim);
                PipeBarrier<PIPE_V>();
            }
        }
        inRowsQue.FreeTensor(xInputLocal);
    }

    __aicore__ inline void ComputeDynamicQuant(int32_t nums, int32_t elementCount)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> scaleLocal = scalesBuf.Get<float>();
        LocalTensor<float> zLocalFp32 = outRowsQue.template AllocTensor<float>();
        LocalTensor<T_Y> outQuant01 = zLocalFp32.ReinterpretCast<T_Y>();

        LocalTensor<float> tmpFp32 = inRowsQue.template AllocTensor<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> scale1Local = scaleLocal[0];
        if (this->smoothExist) {
            LocalTensor<T> smoothLocal = weightBuf02.Get<T>();
            LocalTensor<float> smoothFp32 = yLocalFp32[(nums - 1) * this->numLastDimAligned];
            Cast(smoothFp32, smoothLocal, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            for (int32_t rid = 0; rid < nums; ++rid) {
                Mul(yLocalFp32[rid * this->numLastDimAligned], xLocalFp32[rid * this->numLastDimAligned], smoothFp32,
                    this->numLastDim);
            }
            PipeBarrier<PIPE_V>();
        } else {
            for (int32_t rid = 0; rid < nums; ++rid) {
                Muls(yLocalFp32[rid * this->numLastDimAligned], xLocalFp32[rid * this->numLastDimAligned], (float)(1.0),
                    this->numLastDim);
            }
            PipeBarrier<PIPE_V>();
        }
        ScaleTensor(yLocalFp32, tmpFp32, scale1Local, elementCount, nums);
        PipeBarrier<PIPE_V>();
        Cast(yLocalFp32.ReinterpretCast<int32_t>(), yLocalFp32, RoundMode::CAST_RINT, elementCount);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(yLocalFp32.ReinterpretCast<half>(), yLocalFp32.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, elementCount);
        PipeBarrier<PIPE_V>();
        Cast(outQuant01, yLocalFp32.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, elementCount);
        PipeBarrier<PIPE_V>();
        inRowsQue.FreeTensor(tmpFp32);

        outRowsQue.EnQue(zLocalFp32);
    }

    __aicore__ inline void CopyOut(int32_t gmOffset, int32_t gmOffsetScale, int32_t rowCount)
    {
        LocalTensor<T_Y> outY12 = outRowsQue.template DeQue<T_Y>();
        LocalTensor<float> scaleLocal = scalesBuf.Get<float>();
        LocalTensor<T_Y> outQuant01 = outY12[0];
        LocalTensor<float> scale1Local = scaleLocal[0];
        DataCopyEx(this->yGm[gmOffset], outQuant01, this->numLastDim, rowCount);
        DataCopyEx(this->outScaleGm[gmOffsetScale], scale1Local, rowCount);
        outRowsQue.FreeTensor(outY12);
    }

    __aicore__ inline void ScaleTensor(
        LocalTensor<float>& srcTensor, LocalTensor<float>& tmpTensor, LocalTensor<float>& scaleTensor, int32_t size,
        int32_t nums)
    {
        float maxTemp;
        float scaleTemp;
        event_t eventVS;
        event_t eventSV;
        Abs(tmpTensor, srcTensor, size);
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            ReduceMaxInplace(tmpTensor[rid * this->numLastDimAligned], this->numLastDim);
            eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            maxTemp = tmpTensor[rid * this->numLastDimAligned].GetValue(0);
            scaleTemp = this->quantMaxVal / maxTemp;
            scaleTensor.SetValue(rid, 1 / scaleTemp);
            eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            auto srcSlice = srcTensor[rid * this->numLastDimAligned];
            Muls(srcSlice, srcSlice, scaleTemp, this->numLastDim);
        }
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outRowsQue;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;

    TBuf<TPosition::VECCALC> weightBuf01;
    TBuf<TPosition::VECCALC> weightBuf02;
    TBuf<TPosition::VECCALC> weightBuf03;
    TBuf<TPosition::VECCALC> scalesBuf;

    uint32_t numRowsAligned;
    uint32_t ubAligned;
};

#endif // RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_
