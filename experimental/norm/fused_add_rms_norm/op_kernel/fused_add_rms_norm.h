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
 * \file fused_add_rms_norm.h
 * \brief add rms norm file
 */
#ifndef FUSED_ADD_RMS_NORM_H_
#define FUSED_ADD_RMS_NORM_H_
#include "../../../../norm/rms_norm/op_kernel/rms_norm_base.h"
#include "fused_add_rms_norm_common.h"

using namespace AscendC;
using namespace FusedAddRmsNormKernel;
using namespace RmsNorm;

template <typename T>
class KernelFusedAddRmsNorm {
public:
    __aicore__ inline KernelFusedAddRmsNorm(TPipe* pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
        const FusedAddRMSNormTilingData* tiling)
    {
        FUSED_ADD_RMS_NORM_INIT_ROW_COMMON(tiling, x1, x2, gamma, y, rstd, x);
        this->scale = tiling->scale;

        // pipe alloc memory to queue, the unit is Bytes
        Ppipe->InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();

        uint32_t i_o_max = RmsNorm::CeilDiv(rowWork, rowFactor);
        uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;

        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, rowFactor, gammaLocal);
        }
        SubProcess(i_o_max - 1, row_tail, gammaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, LocalTensor<T>& gammaLocal)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            uint32_t gm_bias = (i_o * rowFactor + i_i) * numCol;
            CopyIn(gm_bias);
            Compute(i_i, gammaLocal, rstdLocal);
            CopyOutY(gm_bias);
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calc_row_num);
    }

private:
    __aicore__ inline void CopyIn(uint32_t gm_bias)
    {
        LocalTensor<T> x1Local_in = inQueueX.AllocTensor<T>();
        LocalTensor<T> x2Local = sqxBuf.Get<T>();
        LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();

        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            x2Local = x2Local[ubFactor];
        }

        DataCopyCustom<T>(x1Local_in, x1Gm[gm_bias], numCol);
        DataCopyCustom<T>(x2Local, x2Gm[gm_bias], numCol);
        inQueueX.EnQue(x1Local_in);
        auto x1Local = inQueueX.DeQue<T>();

        if constexpr (is_same<T, half>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Muls(x1_fp32, x1_fp32, scale, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1_fp32, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xLocal, xLocal, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            // x output is the norm input, so keep the rounded output value in fp32 for RMSNorm.
            Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else if constexpr (is_same<T, bfloat16_t>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            LocalTensor<float> x2_fp32 = sqxBuf.Get<float>();
            Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, numCol);
            Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Muls(x1_fp32, x1_fp32, scale, numCol);
            PipeBarrier<PIPE_V>();
            Add(x1_fp32, x1_fp32, x2_fp32, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();
        } else {
            Muls(x1Local, x1Local, scale, numCol);
            PipeBarrier<PIPE_V>();
            Add(x1Local, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Adds(xLocal, x1Local, (float)0, numCol);
        }
        inQueueX.FreeTensor(x1Local);

        // CopyOut x1 + x2
        outQueueY.EnQue(xLocal);
        auto x_out = outQueueY.DeQue<T>();
        DataCopyCustom<T>(xGm[gm_bias], x_out, numCol);
        outQueueY.FreeTensor(x_out);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<float> gammaLocal, LocalTensor<float> rstdLocal)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
        BuildRstd(sqx, xLocal, reduce_buf_local, avgFactor, epsilon, numCol);
        float rstdValue = ReadLocalFloat(sqx);
        rstdLocal.SetValue(inner_progress, rstdValue);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        Muls(yLocal, xLocal, rstdValue, numCol);
        inQueueX.FreeTensor(xLocal);
        PipeBarrier<PIPE_V>();
        Mul(yLocal, gammaLocal, yLocal, numCol);
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<float>(yLocal);
    }

    __aicore__ inline void Compute(
        uint32_t inner_progress, LocalTensor<bfloat16_t> gammaLocal, LocalTensor<float> rstdLocal)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

        ComputeTypedRstd(inner_progress, x_fp32, sqx, reduce_buf_local, rstdLocal);
        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
        Cast(x_fp32, yLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol); // gamma_fp32 reuse sqx
        PipeBarrier<PIPE_V>();
        Mul(x_fp32, x_fp32, sqx, numCol);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();

        event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event_v_mte);
        WaitFlag<HardEvent::V_MTE2>(event_v_mte);

        outQueueY.EnQue<bfloat16_t>(yLocal);
    }

    __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<half> gammaLocal, LocalTensor<float> rstdLocal)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

        ComputeTypedRstd(inner_progress, x_fp32, sqx, reduce_buf_local, rstdLocal);
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        Cast(yLocal, x_fp32, RoundMode::CAST_NONE, numCol);

        event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event_v_mte);
        WaitFlag<HardEvent::V_MTE2>(event_v_mte);

        PipeBarrier<PIPE_V>();
        Mul(yLocal, gammaLocal, yLocal, numCol);
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<half>(yLocal);
    }

    __aicore__ inline void ComputeTypedRstd(
        uint32_t inner_progress, LocalTensor<float> xLocal, LocalTensor<float> sqx, LocalTensor<float> reduceLocal,
        LocalTensor<float> rstdLocal)
    {
        BuildRstd(sqx, xLocal, reduceLocal, avgFactor, epsilon, numCol);
        float rstdValue = ReadLocalFloat(sqx);
        rstdLocal.SetValue(inner_progress, rstdValue);
        PipeBarrier<PIPE_V>();
        Muls(xLocal, xLocal, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutY(uint32_t progress)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(yGm[progress], yLocal, numCol);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        DataCopyCustom<float>(rstdGm[outer_progress * rowFactor], rstdLocal, num);
#endif
        outQueueRstd.FreeTensor(rstdLocal);
    }

private:
    FUSED_ADD_RMS_NORM_COMMON_MEMBERS;
    float scale;
};
#endif  // FUSED_ADD_RMS_NORM_H_
