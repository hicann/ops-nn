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
 * \file add_rms_norm_quant_single_n.h
 * \brief add rms norm quant single n file
 */
#ifndef _ADD_RMS_NORM_QUANT_SINGLE_N_H_
#define _ADD_RMS_NORM_QUANT_SINGLE_N_H_
#include "add_rms_norm_quant_base.h"

using namespace AscendC;
using namespace AddRmsNormQuantBase;

template <typename TX, typename TScale, typename TOffset, bool RN, bool A, bool PT>
class KernelAddRmsNormQuantSingleN {
public:
    __aicore__ inline KernelAddRmsNormQuantSingleN(TPipe* pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zero_points1,
        GM_ADDR zero_points2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR res_out,
        const AddRMSNormQuantTilingData* tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");

        this->numRow = tilingData->numRow;
        this->numCol = tilingData->numCol;
        this->blockFactor = tilingData->blockFactor;
        this->rowFactor = tilingData->rowFactor;
        this->ubFactor = tilingData->ubFactor;
        this->epsilon = tilingData->epsilon;
        this->avgFactor = (float)1.0 / numCol;
        this->hasZeroPoints1 = tilingData->hasZeroPoints1;
        this->hasBeta = tilingData->hasBeta;
        this->divMode = tilingData->divMode;
        this->hasScales2 = tilingData->hasScales2 && !PT;
        this->hasZeroPoints2 = tilingData->hasZeroPoints2 && !PT;

        blockIdx_ = GetBlockIdx();
        uint32_t blockNum = GetBlockNum() - 1;
        if (blockIdx_ < blockNum) {
            this->rowWork = blockFactor;
        } else {
            this->rowWork = numRow - blockNum * blockFactor;
        }
        // get start index for current core, core parallel
        uint64_t gmOffset = blockIdx_ * blockFactor * numCol;
        uint64_t calcNum = rowWork * numCol;
        x1Gm.SetGlobalBuffer((__gm__ TX*)x1 + gmOffset, calcNum);
        x2Gm.SetGlobalBuffer((__gm__ TX*)x2 + gmOffset, calcNum);
        gammaGm.SetGlobalBuffer((__gm__ TX*)gamma, numCol);
        scales1Gm.SetGlobalBuffer((__gm__ TScale*)scales1, numCol);
        y1Gm.SetGlobalBuffer((__gm__ int8_t*)y1 + gmOffset, calcNum);
        y2Gm.SetGlobalBuffer((__gm__ int8_t*)y2 + gmOffset, calcNum);
        xGm.SetGlobalBuffer((__gm__ TX*)x + gmOffset, calcNum);
        if(RN) {
            resOutGm.SetGlobalBuffer((__gm__ TX*)res_out + gmOffset, calcNum);
        }
        initSingleNUbSize(scales2, zero_points1, zero_points2, beta);
    }

    __aicore__ inline void initSingleNUbSize(GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2, GM_ADDR beta)
    {
        uint32_t optionNum = (hasScales2 + hasZeroPoints2) * sizeof(float) + hasBeta * sizeof(float);
        uint32_t ubSingleNSize = 191 * 1024;
        Ppipe->InitBuffer(unitBuf, ubSingleNSize - optionNum * ubFactor);

        if (hasBeta) {
            betaGm.SetGlobalBuffer((__gm__ TX*)beta, numCol);
        }
        if (hasZeroPoints1) {
            zeroPoints1Gm.SetGlobalBuffer((__gm__ TOffset*)zero_points1, numCol);
        }
        if (hasBeta || hasScales2) {
            Ppipe->InitBuffer(inQueueBeta, BUFFER_NUM, ubFactor * sizeof(float));
        }
        if (hasScales2) {
            scales2Gm.SetGlobalBuffer((__gm__ TScale*)scales2, numCol);
            Ppipe->InitBuffer(scales2Buf, ubFactor * sizeof(float));
            if (hasZeroPoints2) {
                zeroPoints2Gm.SetGlobalBuffer((__gm__ TOffset*)zero_points2, numCol);
                Ppipe->InitBuffer(zeroPoints2Buf, ubFactor * sizeof(int32_t));
            }
        }
    }

    __aicore__ inline void Process()
    {
        CopyInBeta();
        if constexpr (is_same<TX, half>::value && !PT) {
            ProcessFp16();
        } else if constexpr (is_same<TX, bfloat16_t>::value || PT) {
            ProcessBf16();
        } else {
        }
    }

private:
    __aicore__ inline void CopyInBeta()
    {
        if (hasBeta) {
            LocalTensor<TX> betaLocal = inQueueBeta.AllocTensor<TX>();
            DataCopyCustom<TX>(betaLocal, betaGm, numCol);
            inQueueBeta.EnQue(betaLocal);
        }

        if (hasScales2) {
            AddRmsNormQuantBase::CopyInScales<TScale>(scales2Buf, scales2Gm, numCol, ubFactor);
            AddRmsNormQuantBase::CopyInZeroPoints<TOffset>(
                zeroPoints2Buf, zeroPoints2Gm, numCol, ubFactor, hasZeroPoints2);
        }
    }

    __aicore__ inline void ProcessFp16()
    {
        LocalTensor<float> ubLocal = unitBuf.Get<float>();
        LocalTensor<TX> xLocal = ubLocal.template ReinterpretCast<TX>();
        LocalTensor<TX> x1Local = xLocal[0];
        LocalTensor<TX> x2Local = xLocal[ubFactor];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];
        // copy in x1, x2 and calc x1+x2
        DataCopyCustom<TX>(x1Local, x1Gm, numCol);
        event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V1);
        DataCopyCustom<TX>(x2Local, x2Gm, numCol);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Add(x1Local, x1Local, x2Local, numCol);
        PipeBarrier<PIPE_V>();
        // copy gamma
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);

        DataCopyCustom<TX>(x2Local, gammaGm, numCol); // gammaLocal use x2Local
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
        DataCopyCustom<TX>(xGm, x1Local, numCol);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
        DataCopyCustom<TScale>(tmpLocal, scales1Gm, numCol);
        event_t eventMTE2V3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V3);

        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();
        Muls(sqxLocal, sqxLocal, avgFactor, numCol);
        PipeBarrier<PIPE_V>();
        ReduceSumHalfInterval(sqxLocal, numCol);
        PipeBarrier<PIPE_V>();
        Adds(sqxLocal, sqxLocal, epsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(xFp32Local, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, xFp32Local, sqxLocal, 1);
        PipeBarrier<PIPE_V>();
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventSMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        SetFlag<HardEvent::S_MTE2>(eventSMTE2);
        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        Cast(x1Local, xFp32Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Mul(x1Local, x1Local, x2Local, numCol);
        PipeBarrier<PIPE_V>();

        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        if (hasBeta) {
            LocalTensor<TX> betaLocal = inQueueBeta.DeQue<TX>();
            LocalTensor<float> betaFp32 = ubLocal[0];
            Cast(betaFp32, betaLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, betaFp32, numCol);
            PipeBarrier<PIPE_V>();
            inQueueBeta.FreeTensor(betaLocal);
        }

        if (hasScales2) {
            LocalTensor<float> y1Local = inQueueBeta.AllocTensor<float>();
            AddRmsNormQuantBase::doScales(y1Local, xFp32Local, scales2Buf, divMode, numCol);
            AddRmsNormQuantBase::doZeroPoints(y1Local, zeroPoints2Buf, numCol, hasZeroPoints2);

            LocalTensor<int8_t> y2Local = scales2Buf.Get<int8_t>();
            RoundFloat2Int8(y2Local, y1Local, numCol);
            event_t event_V_MTE3_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(event_V_MTE3_1);
            WaitFlag<HardEvent::V_MTE3>(event_V_MTE3_1);
            DataCopyCustom<int8_t>(y2Gm, y2Local, numCol);
            inQueueBeta.FreeTensor(y1Local);
        }

        WaitFlag<HardEvent::MTE2_V>(eventMTE2V3);
        if (divMode) {
            Div(xFp32Local, xFp32Local, tmpLocal, numCol);
            PipeBarrier<PIPE_V>();
        } else {
            Mul(xFp32Local, xFp32Local, tmpLocal, numCol);
            PipeBarrier<PIPE_V>();
        }
        WaitFlag<HardEvent::S_MTE2>(eventSMTE2);
        if (hasZeroPoints1) {
            DataCopyCustom<TOffset>(sqxLocal.ReinterpretCast<TOffset>(), zeroPoints1Gm, numCol);
            event_t eventMTE2V4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V4);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V4);
            Cast(sqxLocal, sqxLocal.ReinterpretCast<TOffset>(), RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, sqxLocal, numCol);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<int8_t> y1Out = tmpLocal.ReinterpretCast<int8_t>();
        RoundFloat2Int8(y1Out, xFp32Local, numCol);
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<int8_t>(y1Gm, y1Out, numCol);
    }

    __aicore__ inline void ProcessBf16()
    {
        LocalTensor<float> ubLocal = unitBuf.Get<float>();
        LocalTensor<TX> xLocal = ubLocal.template ReinterpretCast<TX>();
        LocalTensor<TX> x1Local = xLocal[0];
        LocalTensor<TX> x2Local = xLocal[ubFactor];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];

        DataCopyCustom<TX>(x1Local, x1Gm, numCol);
        event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V1);
        DataCopyCustom<TX>(x2Local, x2Gm, numCol);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Add(xFp32Local, xFp32Local, sqxLocal, numCol);
        PipeBarrier<PIPE_V>();
        #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        Cast(x1Local, xFp32Local, RoundMode::CAST_NONE, numCol);
        #else
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
        #endif
        PipeBarrier<PIPE_V>();
        // copy gamma
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);

        DataCopyCustom<TX>(x2Local, gammaGm, numCol); // gammaLocal use x2Local
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

        // copy x out
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        if constexpr (A) {
            DataCopyCustom<TX>(xGm, x1Local, numCol);
        }
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);

        if constexpr (!PT) {
            Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        }
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();
        Muls(sqxLocal, sqxLocal, avgFactor, numCol);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol);
        PipeBarrier<PIPE_V>();
        Adds(sqxLocal, sqxLocal, epsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
        // copy in scales
        if constexpr (is_same<TScale, bfloat16_t>::value) {
            DataCopyCustom<TScale>(tmpLocal.template ReinterpretCast<TScale>()[ubFactor], scales1Gm, numCol);
        } else { // float
            DataCopyCustom<TScale>(tmpLocal.template ReinterpretCast<TScale>(), scales1Gm, numCol);
        }
        event_t eventMTE2V3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V3);

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Mul(xFp32Local, xFp32Local, sqxLocal, numCol);
        PipeBarrier<PIPE_V>();
        if constexpr (RN) {
            #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
            Cast(x1Local, xFp32Local, RoundMode::CAST_NONE, numCol);
            #else
            Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
            #endif
            SetFlag<HardEvent::V_MTE3>(eventVMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
            DataCopyCustom<TX>(resOutGm, x1Local, numCol);
        }

        if (hasBeta) {
            LocalTensor<TX> betaLocal = inQueueBeta.DeQue<TX>();
            PipeBarrier<PIPE_ALL>();
            Cast(sqxLocal, betaLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, sqxLocal, numCol);
            PipeBarrier<PIPE_V>();
            inQueueBeta.FreeTensor(betaLocal);
        }
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V3);

        if (hasScales2) {
            LocalTensor<float> y2Local = inQueueBeta.AllocTensor<float>();
            AddRmsNormQuantBase::doScales(y2Local, xFp32Local, scales2Buf, divMode, numCol);
            AddRmsNormQuantBase::doZeroPoints(y2Local, zeroPoints2Buf, numCol, hasZeroPoints2);
            LocalTensor<int8_t> y2Out = scales2Buf.Get<int8_t>();
            RoundFloat2Int8(y2Out, y2Local, numCol);
            event_t event_V_MTE3_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(event_V_MTE3_1);
            WaitFlag<HardEvent::V_MTE3>(event_V_MTE3_1);
            DataCopyCustom<int8_t>(y2Gm, y2Out, numCol);
            inQueueBeta.FreeTensor(y2Local);
        }

        if constexpr (is_same<TScale, bfloat16_t>::value) {
            Cast(tmpLocal, tmpLocal.template ReinterpretCast<TScale>()[ubFactor], RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        }

        Quant_Mul(scales1Gm, xFp32Local, tmpLocal, numCol);

        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
        if (hasZeroPoints1) {
            if constexpr (is_same<TOffset, bfloat16_t>::value) {
                DataCopyCustom<TOffset>(sqxLocal.ReinterpretCast<TOffset>()[ubFactor], zeroPoints1Gm, numCol);
            } else { // int32
                DataCopyCustom<TOffset>(sqxLocal.ReinterpretCast<TOffset>(), zeroPoints1Gm, numCol);
            }

            SetFlag<HardEvent::MTE2_V>(eventMTE2V3);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V3);
            if constexpr (is_same<TOffset, bfloat16_t>::value) {
                Cast(sqxLocal, sqxLocal.ReinterpretCast<TOffset>()[ubFactor], RoundMode::CAST_NONE, numCol);
            } else { // int32
                Cast(sqxLocal, sqxLocal.ReinterpretCast<TOffset>(), RoundMode::CAST_NONE, numCol);
            }
            PipeBarrier<PIPE_V>();
            if (PT) {
                int32_t eventIDVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventIDVToS);
                WaitFlag<HardEvent::V_S>(eventIDVToS);
                Adds(xFp32Local, xFp32Local, sqxLocal.GetValue(0), numCol);
            } else {
                Add(xFp32Local, xFp32Local, sqxLocal, numCol);
            }
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<int8_t> y1Out = tmpLocal.ReinterpretCast<int8_t>();
        RoundFloat2Int8(y1Out, xFp32Local, numCol);
        event_t event_V_MTE3_0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event_V_MTE3_0);
        WaitFlag<HardEvent::V_MTE3>(event_V_MTE3_0);
        DataCopyCustom<int8_t>(y1Gm, y1Out, numCol);
    }

    __aicore__ inline void Quant_Mul(
        GlobalTensor<TScale> scales1Gm, LocalTensor<float> xFp32Local,
                                    LocalTensor<float> tmpLocal, uint32_t numCol){
        if constexpr (PT) {
            float scale;
            if constexpr (is_same<TScale, bfloat16_t>::value) {
                scale = 1.0f / ToFloat(scales1Gm.GetValue(0));
            } else {
                scale = 1.0f / scales1Gm.GetValue(0);
            }
            Muls(xFp32Local, xFp32Local, scale, numCol);
        } else {
            if (divMode) {
                Div(xFp32Local, xFp32Local, tmpLocal, numCol);
                PipeBarrier<PIPE_V>();
            } else {
                Mul(xFp32Local, xFp32Local, tmpLocal, numCol);
                PipeBarrier<PIPE_V>();
            }
        }
    }

private:
    TPipe* Ppipe = nullptr;

    TBuf<TPosition::VECCALC> unitBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBeta;
    TBuf<TPosition::VECCALC> zeroPoints2Buf;
    TBuf<TPosition::VECCALC> scales2Buf;
    GlobalTensor<TX> x1Gm;
    GlobalTensor<TX> x2Gm;
    GlobalTensor<TX> gammaGm;
    GlobalTensor<TX> betaGm;
    GlobalTensor<TScale> scales1Gm;
    GlobalTensor<TScale> scales2Gm;
    GlobalTensor<TOffset> zeroPoints1Gm;
    GlobalTensor<TOffset> zeroPoints2Gm;
    GlobalTensor<int8_t> y1Gm;
    GlobalTensor<int8_t> y2Gm;
    GlobalTensor<TX> xGm;
    GlobalTensor<TX> resOutGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    uint32_t hasZeroPoints1 = 0;
    uint32_t hasBeta = 0;
    uint32_t divMode = 1;
    uint32_t hasScales2 = 0;
    uint32_t hasZeroPoints2 = 0;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
};
#endif // _ADD_RMS_NORM_QUANT_SINGLE_N_H_