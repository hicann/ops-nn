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
 * \file fused_add_rms_norm_single_n.h
 * \brief add rms norm single n file
 */
#ifndef FUSED_ADD_RMS_NORM_SINGLE_N_H_
#define FUSED_ADD_RMS_NORM_SINGLE_N_H_
#include "../../../../norm/rms_norm/op_kernel/rms_norm_base.h"
#include "fused_add_rms_norm_common.h"

using namespace AscendC;
using namespace FusedAddRmsNormKernel;
using namespace RmsNorm;

template <typename T>
class KernelFusedAddRmsNormSingleN {
    static constexpr int32_t MAXBUFFER = 195584;
public:
    __aicore__ inline KernelFusedAddRmsNormSingleN(TPipe* pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
        const FusedAddRMSNormTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");

        this->numCol = tiling->num_col;
        this->blockFactor = 1; // in this case, blockFactor = 1
        this->ubFactor = tiling->ub_factor;
        this->epsilon = tiling->epsilon;
        this->avgFactor = (numCol != 0) ? (float)1.0 / numCol : 0;

        this->rowWork = 1;
        blockIdx_ = GetBlockIdx();
        // get start index for current core, core parallel
        x1Gm.SetGlobalBuffer((__gm__ T*)x1 + blockIdx_ * numCol, numCol);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2 + blockIdx_ * numCol, numCol);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * numCol, numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_, 1);
        xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * numCol, numCol);

        Ppipe->InitBuffer(unitBuf, MAXBUFFER); // (192 - 1) * 1024 byte
    }

    __aicore__ inline void Process()
    {
        if constexpr (is_same<T, half>::value) {
            ProcessFp16();
        } else if constexpr (is_same<T, float>::value) {
            ProcessFp32();
        } else {
            ProcessBf16();
        }
    }

private:
    __aicore__ inline void ProcessFp16()
    {
        LocalTensor<float> ubLocal = unitBuf.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];

        event_t eventMTE2V2 = AddInputs(x1Local, x2Local);
        CopyFp16Gamma(x2Local, eventMTE2V2);
        event_t eventVMTE3;
        event_t eventMTE3V = CopyFp16ResidualOut(x1Local, eventVMTE3);
        float rstdValue = ComputeFp16Rstd(x1Local, xFp32Local, sqxLocal, tmpLocal, eventVMTE3);
        NormalizeAndCopyFp16Y(x1Local, x2Local, xFp32Local, rstdValue, eventMTE3V, eventMTE2V2);
    }

    __aicore__ inline event_t AddInputs(LocalTensor<T> x1Local, LocalTensor<T> x2Local)
    {
        DataCopyCustom<T>(x1Local, x1Gm, numCol);
        DataCopyCustom<T>(x2Local, x2Gm, numCol);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Add(x1Local, x1Local, x2Local, numCol);
        PipeBarrier<PIPE_V>();
        return eventMTE2V2;
    }

    __aicore__ inline void CopyFp16Gamma(LocalTensor<T> gammaLocal, event_t eventMTE2V2)
    {
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
    }

    __aicore__ inline event_t CopyFp16ResidualOut(LocalTensor<T> xLocal, event_t& eventVMTE3)
    {
        eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(xGm, xLocal, numCol);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
        return eventMTE3V;
    }

    __aicore__ inline float ComputeFp16Rstd(
        LocalTensor<T> xLocal, LocalTensor<float> xFp32Local, LocalTensor<float> sqxLocal,
        LocalTensor<float> tmpLocal, event_t eventVMTE3)
    {
        Cast(xFp32Local, xLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        BuildRstd(sqxLocal, xFp32Local, tmpLocal, avgFactor, epsilon, numCol);
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<float>(rstdGm, sqxLocal, 1);
#endif
        return ReadLocalFloat(sqxLocal);
    }

    __aicore__ inline void NormalizeAndCopyFp16Y(
        LocalTensor<T> yLocal, LocalTensor<T> gammaLocal, LocalTensor<float> xFp32Local, float rstdValue,
        event_t eventMTE3V, event_t eventMTE2V2)
    {
        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        Cast(yLocal, xFp32Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Mul(yLocal, yLocal, gammaLocal, numCol);
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(yGm, yLocal, numCol);
    }

    __aicore__ inline void ProcessFp32()
    {
        LocalTensor<float> ubLocal = unitBuf.Get<float>();
        LocalTensor<T> x1Local = ubLocal[0];
        LocalTensor<T> x2Local = ubLocal[ubFactor];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];

        event_t eventMTE2V2 = AddInputs(x1Local, x2Local);
        CopyFp32Gamma(x2Local, eventMTE2V2);
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventMTE3V = CopyFp32ResidualOut(x1Local, eventVMTE3);
        float rstdValue = ComputeFp32Rstd(x1Local, sqxLocal, tmpLocal, eventVMTE3);
        NormalizeAndCopyFp32Y(x1Local, x2Local, rstdValue, eventMTE3V, eventMTE2V2, eventVMTE3);
    }

    __aicore__ inline void CopyFp32Gamma(LocalTensor<T> gammaLocal, event_t eventMTE2V2)
    {
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
    }

    __aicore__ inline event_t CopyFp32ResidualOut(LocalTensor<T> xLocal, event_t& eventVMTE3)
    {
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(xGm, xLocal, numCol);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
        return eventMTE3V;
    }

    __aicore__ inline float ComputeFp32Rstd(
        LocalTensor<T> xLocal, LocalTensor<float> sqxLocal, LocalTensor<float> tmpLocal, event_t eventVMTE3)
    {
        BuildRstd(sqxLocal, xLocal, tmpLocal, avgFactor, epsilon, numCol);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<float>(rstdGm, sqxLocal, 1);
#endif
        return ReadLocalFloat(sqxLocal);
    }

    __aicore__ inline void NormalizeAndCopyFp32Y(
        LocalTensor<T> yLocal, LocalTensor<T> gammaLocal, float rstdValue, event_t eventMTE3V, event_t eventMTE2V2,
        event_t eventVMTE3)
    {
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        Muls(yLocal, yLocal, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Mul(yLocal, yLocal, gammaLocal, numCol);
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(yGm, yLocal, numCol);
    }

    __aicore__ inline void ProcessBf16()
    {
        LocalTensor<float> ubLocal = unitBuf.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];

        AddBf16Input(x1Local, x2Local, xFp32Local, sqxLocal);
        event_t eventMTE2V2BF16 = CopyBf16Gamma(x2Local);
        event_t eventMTE3VBF16 = CopyBf16ResidualOut(x1Local);
        event_t eventRstdMTE3VBF16;
        float rstdValue = ComputeBf16Rstd(x1Local, xFp32Local, sqxLocal, tmpLocal, eventRstdMTE3VBF16);
        NormalizeAndCopyBf16Y(
            x1Local, x2Local, xFp32Local, sqxLocal, rstdValue, eventMTE3VBF16, eventMTE2V2BF16,
            eventRstdMTE3VBF16);
    }

    __aicore__ inline void AddBf16Input(
        LocalTensor<T> x1Local, LocalTensor<T> x2Local, LocalTensor<float> xFp32Local, LocalTensor<float> x2Fp32Local)
    {
        DataCopyCustom<T>(x1Local, x1Gm, numCol);
        event_t eventMTE2V1_BF16_0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(eventMTE2V1_BF16_0);
        DataCopyCustom<T>(x2Local, x2Gm, numCol);
        event_t eventMTE2V2_BF16_0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2_BF16_0);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V1_BF16_0);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventMTE2V1_BF16_0);
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2_BF16_0);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventMTE2V2_BF16_0);
        Cast(x2Fp32Local, x2Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Add(xFp32Local, xFp32Local, x2Fp32Local, numCol);
        PipeBarrier<PIPE_V>();
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline event_t CopyBf16Gamma(LocalTensor<T> gammaLocal)
    {
        event_t eventVMTE2_BF16_0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2_BF16_0);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2_BF16_0);
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        event_t eventMTE2V2_BF16_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2_BF16_1);
        return eventMTE2V2_BF16_1;
    }

    __aicore__ inline event_t CopyBf16ResidualOut(LocalTensor<T> xLocal)
    {
        event_t eventVMTE3_BF16_0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_0);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_0);
        DataCopyCustom<T>(xGm, xLocal, numCol);
        event_t eventMTE3V_BF16_0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        SetFlag<HardEvent::MTE3_V>(eventMTE3V_BF16_0);
        return eventMTE3V_BF16_0;
    }

    __aicore__ inline float ComputeBf16Rstd(
        LocalTensor<T> xLocal, LocalTensor<float> xFp32Local, LocalTensor<float> sqxLocal,
        LocalTensor<float> tmpLocal, event_t& eventRstdMTE3V)
    {
        Cast(xFp32Local, xLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        BuildRstd(sqxLocal, xFp32Local, tmpLocal, avgFactor, epsilon, numCol);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        event_t eventVMTE3_BF16_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_1);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_1);
        DataCopyCustom<float>(rstdGm, sqxLocal, 1);
        eventRstdMTE3V = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        SetFlag<HardEvent::MTE3_V>(eventRstdMTE3V);
#endif
        return ReadLocalFloat(sqxLocal);
    }

    __aicore__ inline void NormalizeAndCopyBf16Y(
        LocalTensor<T> yLocal, LocalTensor<T> gammaLocal, LocalTensor<float> xFp32Local, LocalTensor<float> gammaFp32,
        float rstdValue, event_t eventMTE3VBF16, event_t eventMTE2V2BF16, event_t eventRstdMTE3VBF16)
    {
        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(eventMTE3VBF16);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventMTE3VBF16);
        Cast(yLocal, xFp32Local, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
        Cast(xFp32Local, yLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2BF16);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        WaitFlag<HardEvent::MTE3_V>(eventRstdMTE3VBF16);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventRstdMTE3VBF16);
#endif
        Cast(gammaFp32, gammaLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Mul(xFp32Local, xFp32Local, gammaFp32, numCol);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, xFp32Local, RoundMode::CAST_RINT, numCol);
        event_t eventVMTE3_BF16_2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_2);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_2);
        DataCopyCustom<T>(yGm, yLocal, numCol);
    }

private:
    TPipe* Ppipe = nullptr;

    TBuf<TPosition::VECCALC> unitBuf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T> xGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
};
#endif  // _FUSED_ADD_RMS_NORM_SINGLE_N_H_
