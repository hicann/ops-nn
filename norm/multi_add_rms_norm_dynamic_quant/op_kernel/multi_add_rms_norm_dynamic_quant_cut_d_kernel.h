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
 * \file multi_add_rms_norm_dynamic_quant_cut_d_kernel.h
 * \brief
 */

#ifndef __multi_add_rms_norm_dynamic_quant_SLICE_D_H_
#define __multi_add_rms_norm_dynamic_quant_SLICE_D_H_

#include "multi_add_rms_norm_dynamic_quant_base.h"

namespace MultiAddRmsNormDynQnt {

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelMultiAddRmsNormDynamicQuantSliceD : public KernelMultiAddRmsNormDynamicQuantBase<T, TILING_KEY, BUFFER_NUM>
{
public:
    using xSrcGMList = GlobalTensor<T>[X1_LIST_MAX_SIZE];

    __aicore__ inline KernelMultiAddRmsNormDynamicQuantSliceD(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x,
        GM_ADDR y, GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace,
        const MultiAddRmsNormDynamicQuantTilingData* tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x1, x2, gamma, smooth1, smooth2);
        this->InitOutGlobalTensors(y1, y2, x, y, outScale1, outScale2);

        if (this->smooth2Exist) {
            workspaceGm.SetGlobalBuffer((__gm__ float*)(workspace) + 2 * this->blockIdx_ * this->numLastDim);
        } else {
            workspaceGm.SetGlobalBuffer((__gm__ float*)(workspace) + this->blockIdx_ * this->numLastDim);
        }

        /*
          colFactor = 8864
          UB = 3 * colFactor * sizeof(T) + 1 * colFactor * sizeof(float)
              + 3 * colFactor * sizeof(float)
              + 256B_for_reduce + 64B_for_scale
        */
        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, 2 * this->lastDimSliceLen * sizeof(T)); // 2 * D * 2
        Ppipe->InitBuffer(outRowQue, BUFFER_NUM, this->lastDimSliceLen * sizeof(T));     // D * 2
        Ppipe->InitBuffer(tmpOutQue, BUFFER_NUM, this->lastDimSliceLen * sizeof(float)); // D * 4
        Ppipe->InitBuffer(xBufFp32, this->lastDimSliceLen * sizeof(float));              // D * 4
        Ppipe->InitBuffer(yBufFp32, this->lastDimSliceLen * sizeof(float));              // D * 4
        Ppipe->InitBuffer(zBufFp32, this->lastDimSliceLen * sizeof(float));              // D * 4
        // 2 dynamic quant operator required 2 scale buffer.
        Ppipe->InitBuffer(scalesQue, BUFFER_NUM, 2 * MultiAddRNDQ::ELEM_PER_BLK_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t baseGmOffset = 0;
        uint32_t rowGmOffset = 0;
        for (int32_t rowIdx = 0; rowIdx < this->rowWork; ++rowIdx) {
            rowGmOffset = 0;
            this->localSum = MultiAddRNDQ::ZERO;
            this->localMax1 = MultiAddRNDQ::ZERO;
            this->localMax2 = MultiAddRNDQ::ZERO;
            for (int32_t colIdx = 0; colIdx < this->lastDimLoopNum; ++colIdx) {
                ComputeSliceAdd(baseGmOffset, rowGmOffset, this->lastDimSliceLen);
                this->localSum += ReduceSquareSumSlice(this->lastDimSliceLen);
                PipeBarrier<PIPE_V>();
                rowGmOffset += this->lastDimSliceLen;
            }
            {
                ComputeSliceAdd(baseGmOffset, rowGmOffset, this->lastDimSliceLenTail);
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

            this->localMax1 = DYNAMIC_QUANT_DIVIDEND / this->localMax1;
            if (this->smooth2Exist) {
                this->localMax2 = DYNAMIC_QUANT_DIVIDEND / this->localMax2;
            }

            PIPE_S_V();
            PIPE_MTE3_MTE2();

            rowGmOffset = 0;
            for (int32_t colIdx = 0; colIdx < this->lastDimLoopNum; ++colIdx) {
                ComputeDynamicQuant(baseGmOffset, rowGmOffset, this->lastDimSliceLen);
                CopyOutQuant(baseGmOffset, rowGmOffset, this->lastDimSliceLen);
                rowGmOffset += this->lastDimSliceLen;
            }
            {
                ComputeDynamicQuant(baseGmOffset, rowGmOffset, this->lastDimSliceLenTail);
                CopyOutQuant(baseGmOffset, rowGmOffset, this->lastDimSliceLenTail);
            }

            LocalTensor<float> scalesTensor = scalesQue.template AllocTensor<float>();
            scalesTensor.SetValue(0, 1 / this->localMax1);
            if (this->smooth2Exist) {
                scalesTensor.SetValue(MultiAddRNDQ::ELEM_PER_BLK_FP32, 1 / this->localMax2);
            }
            scalesQue.EnQue(scalesTensor);
            CopyOutScale(rowIdx);

            baseGmOffset += this->numLastDim;
        }
    }

private:
    __aicore__ inline void ComputeDynamicQuant(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        if (this->smooth2Exist) {
            CopyInSmoothNorm(xLocalFp32, 0, rowGmOffset, elementCount, this->localMax1);
            CopyInSmoothNorm(yLocalFp32, this->numLastDim, rowGmOffset, elementCount, this->localMax2);
            LocalTensor<int8_t> y12Local = outRowQue.template AllocTensor<int8_t>();
            auto y1Local = y12Local[0];
            auto y2Local = y12Local[this->lastDimSliceLen];
            RoundFloat2Int8(y1Local, xLocalFp32, elementCount);
            RoundFloat2Int8(y2Local, yLocalFp32, elementCount);
            outRowQue.template EnQue<int8_t>(y12Local);
        } else {
            CopyInSmoothNorm(xLocalFp32, 0, rowGmOffset, elementCount, this->localMax1);
            LocalTensor<int8_t> yLocal = outRowQue.template AllocTensor<int8_t>();
            RoundFloat2Int8(yLocal, xLocalFp32, elementCount);
            outRowQue.template EnQue<int8_t>(yLocal);
        }
    }

    __aicore__ inline void CopyOutQuant(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        LocalTensor<int8_t> yOut = outRowQue.template DeQue<int8_t>();
        DataCopyEx(this->y1Gm[baseGmOffset + rowGmOffset], yOut, elementCount);
        if (this->smooth2Exist) {
            DataCopyEx(this->y2Gm[baseGmOffset + rowGmOffset], yOut[this->lastDimSliceLen], elementCount);
        }
        outRowQue.FreeTensor(yOut);
    }

    __aicore__ inline void CopyOutScale(int32_t idx)
    {
        LocalTensor<float> scalesOut = scalesQue.template DeQue<float>();
        DataCopyEx(this->outScale1Gm[idx], scalesOut[0], 1);
        if (this->smooth2Exist) {
            DataCopyEx(this->outScale2Gm[idx], scalesOut[MultiAddRNDQ::ELEM_PER_BLK_FP32], 1);
        }
        scalesQue.FreeTensor(scalesOut);
    }

    __aicore__ inline void CopyInSmoothNorm(
        LocalTensor<float>& dstLocal, int32_t workspaceOffset, int32_t rowGmOffset, int32_t elementCount,
        float scaleNum)
    {
        LocalTensor<float> smoothYLocalIn = inRowsQue.template AllocTensor<float>();
        DataCopyEx(smoothYLocalIn, this->workspaceGm[workspaceOffset + rowGmOffset], elementCount);
        inRowsQue.EnQue(smoothYLocalIn);
        LocalTensor<float> smoothYLocal = inRowsQue.template DeQue<float>();
        Muls(dstLocal, smoothYLocal, scaleNum, elementCount);
        PipeBarrier<PIPE_V>();
        inRowsQue.FreeTensor(smoothYLocal);
    }

    __aicore__ inline void ComputeSliceAdd(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        CopySliceAddInX(baseGmOffset, rowGmOffset, elementCount);
        CopyOutX(baseGmOffset, rowGmOffset, elementCount);
    }

    __aicore__ inline void ComputeRmsNormAndSmoothMax(
        int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount, float rstdLocalTemp)
    {
        CopyInTmpX(baseGmOffset, rowGmOffset, elementCount, rstdLocalTemp);
        CopyInGamma(baseGmOffset, rowGmOffset, elementCount);
        CopyInSmooth(baseGmOffset, rowGmOffset, elementCount);
        ComputeNormAndSmooth(baseGmOffset, rowGmOffset, elementCount);
        // Copy out local RMS result as y
        CopyOutY(baseGmOffset, rowGmOffset, elementCount);
        UpdateLocalMax(rowGmOffset, rstdLocalTemp, elementCount);
    }

    __aicore__ inline void CopyInTmpX(
        int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount, float rstdLocalTemp)
    {
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> xLocalIn = inRowsQue.template AllocTensor<float>();
        DataCopyEx(xLocalIn, this->workspaceGm[rowGmOffset], elementCount);
        inRowsQue.EnQue(xLocalIn);
        LocalTensor<float> xLocal = inRowsQue.template DeQue<float>();
        Muls(yLocalFp32, xLocal, rstdLocalTemp, elementCount);
        PipeBarrier<PIPE_V>();
        inRowsQue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyInGamma(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();
        LocalTensor<T> gammaLocalIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(gammaLocalIn, this->gammaGm[rowGmOffset], elementCount);
        inRowsQue.EnQue(gammaLocalIn);
        LocalTensor<T> gammaLocal = inRowsQue.template DeQue<T>();
        Cast(zLocalFp32, gammaLocal, RoundMode::CAST_NONE, elementCount); // xLocalFp32 <- gammaFp32
        PipeBarrier<PIPE_V>();
        inRowsQue.FreeTensor(gammaLocal);
    }

    __aicore__ inline void CopyInSmooth(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        if (this->smooth1Exist) {
            LocalTensor<T> smooth12CopyIn = inRowsQue.template AllocTensor<T>();
            LocalTensor<T> smooth1In = smooth12CopyIn[0];
            DataCopyEx(smooth1In, this->smooth1Gm[rowGmOffset], elementCount);
            if (this->smooth2Exist) {
                LocalTensor<T> smooth2In = smooth12CopyIn[this->lastDimSliceLen];
                DataCopyEx(smooth2In, this->smooth2Gm[rowGmOffset], elementCount);
            }
            inRowsQue.EnQue(smooth1In);
        }
    }

    __aicore__ inline void ComputeNormAndSmooth(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();

        // Compute the RMS
        Mul(xLocalFp32, yLocalFp32, zLocalFp32, elementCount); // yLocalFp32 <- x * rstd * gamma
        PipeBarrier<PIPE_V>();

        if (this->smooth1Exist) {
            LocalTensor<T> smooth12Local = inRowsQue.template DeQue<T>();
            LocalTensor<T> smooth1Local = smooth12Local[0];
            Cast(yLocalFp32, smooth1Local, RoundMode::CAST_NONE, elementCount); // yLocalFp32 <- smooth1
            if (this->smooth2Exist) {
                LocalTensor<T> smooth2Local = smooth12Local[this->lastDimSliceLen];
                Cast(zLocalFp32, smooth2Local, RoundMode::CAST_NONE, elementCount); // zLocalFp32 <- smooth2
            }
            inRowsQue.FreeTensor(smooth12Local);
            PipeBarrier<PIPE_V>();
            Mul(yLocalFp32, xLocalFp32, yLocalFp32, elementCount); // yLocalFp32 <- norm * smooth1
            PipeBarrier<PIPE_V>();
            CopyOutSmoothNorm(yLocalFp32, 0, rowGmOffset, elementCount);
            if (this->smooth2Exist) {
                Mul(zLocalFp32, xLocalFp32, zLocalFp32, elementCount); // zLocalFp32 <- norm * smooth2
                PipeBarrier<PIPE_V>();
                CopyOutSmoothNorm(zLocalFp32, this->numLastDim, rowGmOffset, elementCount);
            }
            PipeBarrier<PIPE_V>();
        } else {
            CopyOutSmoothNorm(xLocalFp32, 0, rowGmOffset, elementCount);
        }
        // Preprocess RMS for output Y
        LocalTensor<T> yOut = outRowQue.template AllocTensor<T>();
        if constexpr (is_same<T, half>::value) {
            Cast(yOut, xLocalFp32, RoundMode::CAST_NONE, elementCount);
        } else { // BF16
            Cast(yOut, xLocalFp32, RoundMode::CAST_RINT, elementCount);
        }
        PipeBarrier<PIPE_V>();
        outRowQue.EnQue(yOut);
    }

    __aicore__ inline void UpdateLocalMax(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();
        if (this->smooth2Exist) {
            float tmpMax1 = FindSliceMax(yLocalFp32, xLocalFp32, elementCount);
            float tmpMax2 = FindSliceMax(zLocalFp32, xLocalFp32, elementCount);
            this->localMax1 = (tmpMax1 > this->localMax1) ? tmpMax1 : localMax1;
            this->localMax2 = (tmpMax2 > this->localMax2) ? tmpMax2 : localMax2;
        } else if (this->smooth1Exist) {
            float tmpMax = FindSliceMax(yLocalFp32, xLocalFp32, elementCount);
            this->localMax1 = (tmpMax > this->localMax1) ? tmpMax : localMax1;
        } else {
            float tmpMax = FindSliceMax(xLocalFp32, yLocalFp32, elementCount);
            this->localMax1 = (tmpMax > this->localMax1) ? tmpMax : localMax1;
        }
    }

    __aicore__ inline float FindSliceMax(
        LocalTensor<float>& srcTensor, LocalTensor<float>& tmpTensor, int32_t elementCount)
    {
        Abs(tmpTensor, srcTensor, elementCount); // tmpLocal <-- |y * smooth|
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(tmpTensor, elementCount);
        PIPE_V_S();
        float maxTemp = tmpTensor.GetValue(0);
        return maxTemp;
    }

    __aicore__ inline void CopyOutSmoothNorm(
        LocalTensor<float>& smoothNormTensor, int32_t workspaceOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        LocalTensor<float> ySmoothLocal = tmpOutQue.template AllocTensor<float>();
        Adds(ySmoothLocal, smoothNormTensor, MultiAddRNDQ::ZERO, elementCount);
        tmpOutQue.template EnQue<float>(ySmoothLocal);
        LocalTensor<float> ySmooth = tmpOutQue.template DeQue<float>();
        DataCopyEx(this->workspaceGm[workspaceOffset + rowGmOffset], ySmooth, elementCount);
        tmpOutQue.FreeTensor(ySmooth);
    }

    __aicore__ inline void DynamicXCopyAddIn(
        const xSrcGMList& xGmTsr, const uint64_t xLocalOffset, LocalTensor<float>& dstTmpTsr, int32_t baseGmOffset,
        int32_t rowGmOffset, int32_t elementCount, const int32_t tsrListLen)
    {
        // Setup xLocalIn
        LocalTensor<T> xLocalIn = inRowsQue.template AllocTensor<T>();
        xLocalIn = xLocalIn[xLocalOffset];
        // Init xLocalIn results
        DataCopyEx(xLocalIn, xGmTsr[0][baseGmOffset + rowGmOffset], elementCount);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        // Setup temp buffer used in list Add
        Cast(dstTmpTsr, xLocalIn, RoundMode::CAST_NONE, elementCount);
        if (tsrListLen > 1) {
            // Init copy-in temp buffer
            // 4Buf: using xLocalFp32 could block Add computation of 'x=y+z' due to resource computation.
            LocalTensor<float> wLocalCopyIn = xBufFp32.Get<float>();
            for (int32_t i = 1; i < tsrListLen; ++i) {
                // Next copy-in MUST wait for the previous VC flow one done
                SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                DataCopyEx(xLocalIn, xGmTsr[i][baseGmOffset + rowGmOffset], elementCount);
                // Converted read-in x form T into Fp32 to rerserve precision
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                Cast(wLocalCopyIn, xLocalIn, RoundMode::CAST_NONE, elementCount);
                PipeBarrier<PIPE_V>();
                Add(dstTmpTsr, dstTmpTsr, wLocalCopyIn, elementCount); // x += x_i
                PipeBarrier<PIPE_V>();
            }
        }
        inRowsQue.FreeTensor(xLocalIn);
    }

    __aicore__ inline void CopySliceAddInX(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2); // Protect yLocalFp32 and zLocalFp32 till Last add is done

        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();
        DynamicXCopyAddIn(
            this->x1GmList, 0, yLocalFp32, baseGmOffset, rowGmOffset, elementCount, this->x1Num); // x1->yLocalFp32
        DynamicXCopyAddIn(
            this->x2GmList, this->lastDimSliceLen, zLocalFp32, baseGmOffset, rowGmOffset, elementCount,
            this->x2Num); // x2->zLocalFp32

        // A synchronization barrier is needed to guarantee Flow order
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        // AddX1X2Slice:
        // Attends have been prepared in previous DynamicXCopyAddIn calls, while xLocalFp32 is NOT occupied by parallel
        // copy-in flows.
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        PipeBarrier<PIPE_V>();
        Add(xLocalFp32, yLocalFp32, zLocalFp32, elementCount);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> xLocal = outRowQue.template AllocTensor<T>();
        LocalTensor<float> xLocalToWorkSpace = tmpOutQue.template AllocTensor<float>();
        if constexpr (is_same<T, half>::value) {
            Cast(xLocal, xLocalFp32, RoundMode::CAST_NONE, elementCount);
        } else { // BF16
            Cast(xLocal, xLocalFp32, RoundMode::CAST_RINT, elementCount);
        }
        outRowQue.template EnQue<T>(xLocal);
        Adds(xLocalToWorkSpace, xLocalFp32, MultiAddRNDQ::ZERO, elementCount);
        tmpOutQue.template EnQue<float>(xLocalToWorkSpace);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutY(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        LocalTensor<T> y = outRowQue.template DeQue<T>();
        DataCopyEx(this->yGm[baseGmOffset + rowGmOffset], y, elementCount);
        outRowQue.FreeTensor(y);
    }

    __aicore__ inline void CopyOutX(int32_t baseGmOffset, int32_t rowGmOffset, int32_t elementCount)
    {
        LocalTensor<T> x = outRowQue.template DeQue<T>();
        DataCopyEx(this->xGm[baseGmOffset + rowGmOffset], x, elementCount);
        outRowQue.FreeTensor(x);
        LocalTensor<float> xFp32 = tmpOutQue.template DeQue<float>();
        DataCopyEx(this->workspaceGm[rowGmOffset], xFp32, elementCount);
        tmpOutQue.FreeTensor(xFp32);
    }

    __aicore__ inline float ReduceSquareSumSlice(int32_t elementCount)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        Mul(yLocalFp32, xLocalFp32, xLocalFp32, elementCount); // yLocalFp32 <- x ** 2
        PipeBarrier<PIPE_V>();
        return ReduceSumHalfInterval(yLocalFp32, elementCount); // aveLocalTemp <-- E(x**2)
    }

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

private:
    TPipe* Ppipe = nullptr;
    GlobalTensor<float> workspaceGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outRowQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> tmpOutQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scalesQue;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;
    TBuf<TPosition::VECCALC> zBufFp32;
    TBuf<TPosition::VECCALC> reduceBuf;

    float localMax1;
    float localMax2;
    float localSum;
};
} // namespace MultiAddRmsNormDynQnt
#endif // __multi_add_rms_norm_dynamic_quant_SLICE_D_H_