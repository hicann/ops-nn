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
 * \file dynamic_quant_update_scatter_v2.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_V2_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_V2_H

#include "dynamic_quant_update_scatter_v2_base.h"

namespace DynamicQuantUpdateScatterV2NDOpt {
using namespace AscendC;

template <typename xDtype, typename yDtype>
class DynamicQuantUpdateScatterV2 : public DynamicQuantUpdateScatterV2Base {
public:
    __aicore__ inline DynamicQuantUpdateScatterV2(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace,
        const DynamicQuantUpdateScatterV2TilingData* __restrict tilingData)
    {
        ParseTilingData(tilingData);
        InitParams();
        InitBaseBuffer();
        SetGlobalBuffer(x, indices, y, scale, offset);
        InitLocalBuffer();
    }

    __aicore__ inline void Process()
    {
        CopyIn(rowPerCore, 0);
        CopyIndicesIn();
        ComputAsymmetric();
    }

    __aicore__ inline void SetGlobalBuffer(GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR scale, GM_ADDR offset)
    {
        if (blockIdx < tilingData_.headCoreNum) {
            inGm.SetGlobalBuffer((__gm__ xDtype*)x + blockIdx * lenHead, lenHead);
        } else {
            inGm.SetGlobalBuffer(
                (__gm__ xDtype*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail,
                lenTail);
        }
        outGm.SetGlobalBuffer((__gm__ yDtype*)y, outLen);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale, outLenQuant);
        offsetGm.SetGlobalBuffer((__gm__ float*)offset, outLenQuant);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, tilingData_.batchSize);
    }

    __aicore__ inline void CopyIn(uint64_t multiRow, uint64_t loopNum)
    {
        LocalTensor<xDtype> inLocal = inQueue.AllocTensor<xDtype>();
        if (isPad) {
            DataCopyParams copyParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(xDtype)), 0, 0};
            DataCopyPadParams padParams{true, 0, rightPadding, 0};
            DataCopyPad(inLocal, inGm[loopNum * lenGMMultiRow], copyParams, padParams);
        } else {
            DataCopy(inLocal, inGm[loopNum * lenGMMultiRow], multiRow * tilingData_.rowLen);
        }
        inQueue.EnQue(inLocal);
    }

private:
    __aicore__ inline void InitLocalBuffer()
    {
        pPipe->InitBuffer(inQueue, BUFFER_NUM, lenMultiRow * sizeof(xDtype));
        pPipe->InitBuffer(outQueue, BUFFER_NUM, outAlignLen * sizeof(yDtype));
        pPipe->InitBuffer(scaleQueue, BUFFER_NUM, 1 * sizeof(float));
        pPipe->InitBuffer(offsetQueue, BUFFER_NUM, 1 * sizeof(float));
        pPipe->InitBuffer(indicesQueue, BUFFER_NUM, sizeIntLen * sizeof(int32_t));
    }

    __aicore__ inline void ComputAsymmetric()
    {
        float maxValue;
        float minValue;
        float scale;
        float offset;
        float backScale;
        LocalTensor<xDtype> inLocal = inQueue.DeQue<xDtype>();
        LocalTensor<int32_t> indicesLocal = indicesQueue.DeQue<int32_t>();
        LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
        LocalTensor<float> offsetLocal = offsetQueue.AllocTensor<float>();
        LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
        LocalTensor<yDtype> outLocal = outQueue.AllocTensor<yDtype>();
        LocalTensor<float> temp = fp32_buf_.Get<float>();
        LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();
        auto tempHalf = temp.ReinterpretCast<half>();
        for (uint64_t i = 0; i < rowPerCore; i++) {
            // x fp16->fp32
            Cast(tempFp32, inLocal[i * sizeHalfLen], RoundMode::CAST_NONE, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            ReduceMax(temp, tempFp32, temp, tilingData_.rowLen, false);
            PipeBarrier<PIPE_V>();
            maxValue = temp.GetValue(0);
            ReduceMin(temp, tempFp32, temp, tilingData_.rowLen, false);
            PipeBarrier<PIPE_V>();
            minValue = temp.GetValue(0);
            GetScaleAndOffset(maxValue, minValue, scale, offset);
            backScale = 1 / scale;
            scaleLocal.SetValue(0, scale);
            offsetLocal.SetValue(0, -offset);
            Muls(tempFp32, tempFp32, backScale, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Adds(tempFp32, tempFp32, offset, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            SetDeqScale(static_cast<half>(1.0));
            PipeBarrier<PIPE_V>();
            Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Cast(outLocal, tempHalf, RoundMode::CAST_TRUNC, tilingData_.rowLen);
            outQueue.EnQue<yDtype>(outLocal);
            scaleQueue.EnQue<float>(scaleLocal);
            offsetQueue.EnQue<float>(offsetLocal);
            CopyOut(i, indicesLocal);
        }
        inQueue.FreeTensor(inLocal);
        indicesQueue.FreeTensor(indicesLocal);
        outQueue.FreeTensor(outLocal);
        offsetQueue.FreeTensor(offsetLocal);
        scaleQueue.FreeTensor(scaleLocal);
    }

    __aicore__ inline void CopyOut(uint64_t loopNum, const LocalTensor<int32_t>& indicesLocal)
    {
        LocalTensor<yDtype> outLocal = outQueue.DeQue<yDtype>();
        LocalTensor<float> scaleLocal = scaleQueue.DeQue<float>();
        LocalTensor<float> offsetLocal = offsetQueue.DeQue<float>();

        uint64_t dstOffset = GetDstOffset(loopNum, 0, indicesLocal);
        event_t eventSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventSMTE3);

        DataCopyExtParams copyParams{1, (uint16_t)(tilingData_.rowLen * sizeof(int4b_t)), 0, 0, 0};
        copyParams.blockLen = copyParams.blockLen >> 1;
        DataCopyPad(outGm[dstOffset], outLocal, copyParams);

        DataCopyExtParams copyParams1{1, (uint16_t)(1 * sizeof(float)), 0, 0, 0};
        DataCopyPad(scaleGm[dstOffset / tilingData_.rowLen], scaleLocal, copyParams1);
        DataCopyPad(offsetGm[dstOffset / tilingData_.rowLen], offsetLocal, copyParams1);
    }

private:
    /* ascendc variable */
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scaleQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> offsetQueue;

    /* global memory address */
    GlobalTensor<xDtype> inGm;
    GlobalTensor<yDtype> outGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offsetGm;
};
} // namespace DynamicQuantUpdateScatterV2NDOpt
#endif // DYNAMIC_QUANT
