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
 * \file add_rms_norm_cast.h
 * \brief add rms norm cast file
 */
#ifndef ADD_RMS_NORM_CAST_H_
#define ADD_RMS_NORM_CAST_H_
#include "../rms_norm/rms_norm_base.h"

using namespace AscendC;
using namespace RmsNorm;
constexpr uint32_t VL = 64;
constexpr uint32_t NUMTHREE = 3;

template <typename T>
class KernelAddRmsNormCast {
public:
    __aicore__ inline KernelAddRmsNormCast(TPipe* pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace,
        const AddRMSNormCastTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        this->numRow = tiling->num_row;
        this->numCol = tiling->num_col;
        this->blockFactor = tiling->block_factor;
        this->rowFactor = tiling->row_factor;
        this->ubFactor = tiling->ub_factor;
        this->epsilon = tiling->epsilon;
        this->avgFactor = (numCol != 0) ? (float)1.0 / numCol : 0;

        blockIdx_ = GetBlockIdx();
        if (blockIdx_ < GetBlockNum() - 1) {
            this->rowWork = blockFactor;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
        }
        // get start index for current core, core parallel
        uint64_t calcOffset = blockIdx_ * blockFactor * numCol;
        uint64_t calcNum = rowWork * numCol;
        x1Gm.SetGlobalBuffer((__gm__ T*)x1 + calcOffset, calcNum);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2 + calcOffset, calcNum);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        y1Gm.SetGlobalBuffer((__gm__ float*)y1 + calcOffset, calcNum);
        y2Gm.SetGlobalBuffer((__gm__ T*)y2 + calcOffset, calcNum);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_ * blockFactor, blockFactor);
        xGm.SetGlobalBuffer((__gm__ T*)x + calcOffset, calcNum);

        // pipe alloc memory to queue, the unit is Bytes
        Ppipe->InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));

        Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(tmpBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(tmpBuf2, ubFactor * NUMTHREE * sizeof(T));
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t i_o_max = RmsNorm::CeilDiv(rowWork, rowFactor);
        uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;

        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, rowFactor);
        }
        SubProcess(i_o_max - 1, row_tail);
    }

    __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num)
    {
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            uint32_t gm_bias = (i_o * rowFactor + i_i) * numCol;
            CopyIn(gm_bias);
            if constexpr (is_same<T, half>::value) {
                Computefp16(i_o, i_i, gm_bias);
            } else {
                Computebf16(i_o, i_i, gm_bias);
            }
            CopyOutY(gm_bias);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t gm_bias)
    {
        LocalTensor<T> x1Local_in = inQueueX.AllocTensor<T>();
        LocalTensor<T> x2Local = tmpBuf.Get<T>();
        x2Local = x2Local[ubFactor];

        DataCopyCustom<T>(x1Local_in, x1Gm[gm_bias], numCol);
        DataCopyCustom<T>(x2Local, x2Gm[gm_bias], numCol);
        inQueueX.EnQue(x1Local_in);
        auto x1Local = inQueueX.DeQue<T>();

        if constexpr (is_same<T, half>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            Add(x1Local_in, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Cast(x1_fp32, x1Local_in, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            LocalTensor<float> x2_fp32 = tmpBuf.Get<float>();
            Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, numCol);
            Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(x1_fp32, x1_fp32, x2_fp32, numCol);
            PipeBarrier<PIPE_V>();
            Cast(x1Local_in, x1_fp32, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();
        }
        event_t eventVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVMte3);
        DataCopyCustom<T>(xGm[gm_bias], x1Local_in, numCol);
        event_t eventMte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);
        inQueueX.FreeTensor(x1Local);
    }

    __aicore__ inline void Computebf16(uint32_t outer_progress, uint32_t inner_progress, uint32_t progress)
    {
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduceBufLocal = reduceFp32Buf.Get<float>();
        LocalTensor<float> resFp32 = tmpBuf2.Get<float>();
        LocalTensor<T> gammaLocal = resFp32.template ReinterpretCast<T>()[ubFactor * 2];
        if (inner_progress == 0) {
            DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        }
        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);

        Mul(sqx, xFp32, xFp32, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqx, sqx, reduceBufLocal, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        Duplicate(reduceBufLocal, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(reduceBufLocal, reduceBufLocal, sqx, 1);
        PipeBarrier<PIPE_V>();
        event_t eventVMte31 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMte31);
        WaitFlag<HardEvent::V_MTE3>(eventVMte31);
        Brcb(sqx, reduceBufLocal, 1, {1, 8});
        DataCopyCustom<float>(rstdGm[outer_progress * rowFactor + inner_progress], reduceBufLocal, 1);
        event_t eventMte3V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMte3V1);

        PipeBarrier<PIPE_V>();
        uint32_t repeat = (numCol + VL - 1) / VL;
        uint64_t mask = VL;
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
        Mul(xFp32, xFp32, sqx, mask, repeat, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
        Mul(xFp32, xFp32, sqx, numCol);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<bfloat16_t>(yLocal);
        Cast(resFp32, yLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        event_t eventVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVMte3);
        DataCopyCustom<float>(y1Gm[progress], resFp32, numCol);
        WaitFlag<HardEvent::MTE3_V>(eventMte3V1);
    }

    __aicore__ inline void Computefp16(uint32_t outer_progress, uint32_t inner_progress, uint32_t progress)
    {
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduceBufLocal = reduceFp32Buf.Get<float>();
        LocalTensor<float> resFp32 = tmpBuf2.Get<float>();
        LocalTensor<T> gammaLocal = resFp32.template ReinterpretCast<T>()[ubFactor * 2];
        if (inner_progress == 0) {
            DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        }
        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);

        Mul(sqx, xFp32, xFp32, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();

        ReduceSumCustom(sqx, sqx, reduceBufLocal, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        Duplicate(reduceBufLocal, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, reduceBufLocal, sqx, 1);
        PipeBarrier<PIPE_V>();
        event_t eventVMte31 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMte31);
        WaitFlag<HardEvent::V_MTE3>(eventVMte31);
        Brcb(reduceBufLocal, sqx, 1, {1, 8});
        DataCopyCustom<float>(rstdGm[outer_progress * rowFactor + inner_progress], sqx, 1);
        event_t eventMte3V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMte3V1);
        PipeBarrier<PIPE_V>();
        uint32_t repeat = (numCol + VL - 1) / VL;
        uint64_t mask = VL;
        Mul(xFp32, xFp32, reduceBufLocal, mask, repeat, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();

        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        Cast(yLocal, xFp32, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);

        Mul(yLocal, gammaLocal, yLocal, numCol);
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<half>(yLocal);
        Cast(resFp32, yLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event_v_mte3);
        WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
        DataCopyCustom<float>(y1Gm[progress], resFp32, numCol);
        WaitFlag<HardEvent::MTE3_V>(eventMte3V1);
    }

    __aicore__ inline void CopyOutY(uint32_t progress)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(y2Gm[progress], yLocal, numCol);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220
        DataCopyCustom<float>(rstdGm[outer_progress * rowFactor], rstdLocal, num);
#endif
        outQueueRstd.FreeTensor(rstdLocal);
    }

private:
    TPipe* Ppipe = nullptr;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;

    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> tmpBuf2;
    TBuf<TPosition::VECCALC> reduceFp32Buf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<float> y1Gm;
    GlobalTensor<T> y2Gm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T> xGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
};
#endif // _ADD_RMS_NORM_CAST_H_