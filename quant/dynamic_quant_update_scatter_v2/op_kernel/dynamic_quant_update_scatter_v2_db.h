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
 * \file dynamic_quant_update_scatter_v2_db.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_V2_DB_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_V2_DB_H

#include "dynamic_quant_update_scatter_v2_base.h"

namespace DynamicQuantUpdateScatterV2NDOpt {
using namespace AscendC;

template <typename xDtype, typename yDtype>
class DynamicQuantUpdateScatterV2DbOpt : public DynamicQuantUpdateScatterV2Base {
public:
    __aicore__ inline DynamicQuantUpdateScatterV2DbOpt(TPipe* pipe)
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
        SetDbGlobalBuffer(x, indices, y, scale, offset);
        InitDbLocalBuffer();
    }

    __aicore__ inline void Process()
    {
        CopyIndicesIn();
        LocalTensor<int32_t> indicesLocal = indicesQueue.DeQue<int32_t>();

        CopyIn(multiRowNum, 0);
        for (uint32_t i = 1; i < loopCnt; i++) {
            CopyIn(multiRowNum, i);
            ProcessAsymmetric(multiRowNum, i - 1, indicesLocal);
        }

        if (remainRow > 0) {
            CopyIn(remainRow, loopCnt);
        }

        ProcessAsymmetric(multiRowNum, loopCnt - 1, indicesLocal);

        if (remainRow > 0) {
            ProcessAsymmetric(remainRow, loopCnt, indicesLocal);
        }

        indicesQueue.FreeTensor(indicesLocal);
    }

    __aicore__ inline void CopyIn(uint64_t multiRow, uint64_t loopNum)
    {
        LocalTensor<xDtype> inLocal = inQueue.AllocTensor<xDtype>();
        if (isPad) {
            DataCopyParams copyParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(xDtype)), 0, 0};
            DataCopyPadParams padParams{true, 0, rightPadding, 0};
            DataCopyPad(inLocal, InputGm[loopNum * lenGMMultiRow], copyParams, padParams);
        } else {
            DataCopy(inLocal, InputGm[loopNum * lenGMMultiRow], multiRow * tilingData_.rowLen);
        }
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void SetDbGlobalBuffer(GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR scale, GM_ADDR offset)
    {
        if (blockIdx < tilingData_.headCoreNum) {
            InputGm.SetGlobalBuffer((__gm__ xDtype*)x + blockIdx * lenHead, lenHead);
        } else {
            InputGm.SetGlobalBuffer(
                (__gm__ xDtype*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail,
                lenTail);
        }
        outputGm.SetGlobalBuffer((__gm__ yDtype*)y, outLen);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale, outLenQuant);
        offsetGm.SetGlobalBuffer((__gm__ float*)offset, outLenQuant);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, tilingData_.batchSize);
    }

private:
    __aicore__ inline void InitDbLocalBuffer()
    {
        pPipe->InitBuffer(inQueue, DOUBLE_BUFFER_NUM, lenMultiRow * sizeof(xDtype));
        pPipe->InitBuffer(outBuf, outAlignLen * sizeof(yDtype));
        pPipe->InitBuffer(scaleBuf, 1 * sizeof(float));
        pPipe->InitBuffer(offsetBuf, 1 * sizeof(float));
        pPipe->InitBuffer(indicesQueue, BUFFER_NUM, sizeIntLen * sizeof(int32_t));
    }

    __aicore__ inline void ProcessAsymmetric(
        uint64_t multiRow, uint64_t loopNum, const LocalTensor<int32_t>& indicesLocal)
    {
        uint64_t index = 0;
        float maxValue;
        float minValue;
        float backScale;
        float scale;
        float offset;

        LocalTensor<xDtype> inLocal = inQueue.DeQue<xDtype>();
        LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
        LocalTensor<float> offsetLocal = offsetBuf.Get<float>();
        LocalTensor<float> tempCast = tempCastUb.Get<float>();
        LocalTensor<yDtype> outLocal = outBuf.AllocTensor<yDtype>();
        LocalTensor<float> temp = fp32_buf_.Get<float>();
        LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();
        auto tempHalfCast = temp.ReinterpretCast<half>();

        event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3S);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
        for (uint64_t i = 0; i < multiRow; i++) {
            // cast to fp32
            Cast(tempCast, inLocal[i * sizeHalfLen], RoundMode::CAST_NONE, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            ReduceMax(temp, tempCast, temp, tilingData_.rowLen, false);
            event_t event_v_s_max = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event_v_s_max);
            WaitFlag<HardEvent::V_S>(event_v_s_max);
            maxValue = temp.GetValue(0);
            ReduceMin(temp, tempCast, temp, tilingData_.rowLen, false);
            event_t event_v_s_min = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event_v_s_min);
            WaitFlag<HardEvent::V_S>(event_v_s_min);
            minValue = temp.GetValue(0);
            scale = GetMax((maxValue - minValue) / DYNAMIC_QUANT_INT4_SCALE, DYNAMIC_QUANT_EPSINON);
            offset = DYNAMIC_QUANT_INT4_OFFSET - SafeDiv(maxValue, scale);
            backScale = SafeDiv(1.0, scale);
            if (unlikely(i == 0)) {
                WaitFlag<HardEvent::MTE3_S>(eventMTE3S);
            }
            scaleLocal.SetValue(0, scale);
            offsetLocal.SetValue(0, -offset);
            Muls(tempCast, tempCast, backScale, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Adds(tempCast, tempCast, offset, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Cast(tempInt32, tempCast, RoundMode::CAST_RINT, tilingData_.rowLen);
            SetDeqScale(static_cast<half>(1.0));
            PipeBarrier<PIPE_V>();
            Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            if (unlikely(i == 0)) {
                WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
            }
            Cast(outLocal, tempHalfCast, RoundMode::CAST_TRUNC, tilingData_.rowLen);

            event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(event_v_mte3);
            WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
            uint64_t dstOffset = GetDstOffset(i, loopNum, indicesLocal);
            event_t eventSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventSMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventSMTE3);

            // copy out var
            DataCopyExtParams copyParams{1, (uint16_t)(tilingData_.rowLen * sizeof(int4b_t)), 0, 0, 0};
            copyParams.blockLen = copyParams.blockLen >> 1;
            DataCopyPad(outputGm[dstOffset], outLocal, copyParams);
            // copy out scale offset
            DataCopyExtParams copyParams1{1, (uint16_t)(1 * sizeof(float)), 0, 0, 0};
            DataCopyPad(scaleGm[dstOffset / tilingData_.rowLen], scaleLocal, copyParams1);
            DataCopyPad(offsetGm[dstOffset / tilingData_.rowLen], offsetLocal, copyParams1);
        }
        inQueue.FreeTensor(inLocal);
    }

private:
    /* ascendc variable */
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueue;
    TBuf<> outBuf;
    TBuf<> scaleBuf;
    TBuf<> offsetBuf;

    /* global memory address */
    GlobalTensor<xDtype> InputGm;
    GlobalTensor<yDtype> outputGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offsetGm;
};
} // namespace DynamicQuantUpdateScatterV2NDOpt
#endif // DYNAMIC_QUANT_DB_H