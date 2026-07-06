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
 * \file fused_add_rms_norm_common.h
 * \brief common init helpers for fused add rms norm kernels
 */
#ifndef FUSED_ADD_RMS_NORM_COMMON_H_
#define FUSED_ADD_RMS_NORM_COMMON_H_

#define FUSED_ADD_RMS_NORM_BIND_GM(TILING_PTR, X1_PTR, X2_PTR, GAMMA_PTR, Y_PTR, RSTD_PTR, X_PTR) \
    do { \
        this->x1Gm.SetGlobalBuffer((__gm__ T*)(X1_PTR) + this->blockIdx_ * this->blockFactor * this->numCol, \
            this->rowWork * this->numCol); \
        this->x2Gm.SetGlobalBuffer((__gm__ T*)(X2_PTR) + this->blockIdx_ * this->blockFactor * this->numCol, \
            this->rowWork * this->numCol); \
        this->gammaGm.SetGlobalBuffer((__gm__ T*)(GAMMA_PTR), this->numCol); \
        this->yGm.SetGlobalBuffer((__gm__ T*)(Y_PTR) + this->blockIdx_ * this->blockFactor * this->numCol, \
            this->rowWork * this->numCol); \
        this->rstdGm.SetGlobalBuffer((__gm__ float*)(RSTD_PTR) + this->blockIdx_ * this->blockFactor, \
            this->blockFactor); \
        this->xGm.SetGlobalBuffer((__gm__ T*)(X_PTR) + this->blockIdx_ * this->blockFactor * this->numCol, \
            this->rowWork * this->numCol); \
    } while (0)

#define FUSED_ADD_RMS_NORM_INIT_ROW_COMMON(TILING_PTR, X1_PTR, X2_PTR, GAMMA_PTR, Y_PTR, RSTD_PTR, X_PTR) \
    do { \
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!"); \
        this->numRow = (TILING_PTR)->num_row; \
        this->numCol = (TILING_PTR)->num_col; \
        this->blockFactor = (TILING_PTR)->block_factor; \
        this->rowFactor = (TILING_PTR)->row_factor; \
        this->ubFactor = (TILING_PTR)->ub_factor; \
        this->epsilon = (TILING_PTR)->epsilon; \
        this->avgFactor = (this->numCol != 0) ? (float)1.0 / this->numCol : 0; \
        this->blockIdx_ = GetBlockIdx(); \
        this->rowWork = (this->blockIdx_ < GetBlockNum() - 1) ? \
            this->blockFactor : \
            this->numRow - (GetBlockNum() - 1) * this->blockFactor; \
        FUSED_ADD_RMS_NORM_BIND_GM(TILING_PTR, X1_PTR, X2_PTR, GAMMA_PTR, Y_PTR, RSTD_PTR, X_PTR); \
    } while (0)

#define FUSED_ADD_RMS_NORM_INIT_MULTI_ROW_COMMON(TILING_PTR, X1_PTR, X2_PTR, GAMMA_PTR, Y_PTR, RSTD_PTR, X_PTR) \
    do { \
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!"); \
        this->numRow = (TILING_PTR)->num_row; \
        this->numCol = (TILING_PTR)->num_col; \
        this->numColAlign = (TILING_PTR)->num_col_align; \
        this->blockFactor = (TILING_PTR)->block_factor; \
        this->rowFactor = (TILING_PTR)->row_factor; \
        this->ubFactor = (TILING_PTR)->ub_factor; \
        this->epsilon = (TILING_PTR)->epsilon; \
        this->avgFactor = (TILING_PTR)->avg_factor; \
        this->blockIdx_ = GetBlockIdx(); \
        if (this->blockIdx_ < GetBlockNum() - 1) { \
            this->rowWork = this->blockFactor; \
            this->rowLoop = (TILING_PTR)->row_loop; \
            this->rowTail = (TILING_PTR)->row_tail; \
        } else if (this->blockIdx_ == GetBlockNum() - 1) { \
            this->rowWork = (TILING_PTR)->last_block_factor; \
            this->rowLoop = (TILING_PTR)->last_block_row_loop; \
            this->rowTail = (TILING_PTR)->last_block_row_tail; \
        } \
        FUSED_ADD_RMS_NORM_BIND_GM(TILING_PTR, X1_PTR, X2_PTR, GAMMA_PTR, Y_PTR, RSTD_PTR, X_PTR); \
    } while (0)

namespace FusedAddRmsNormKernel {
using namespace AscendC;
using namespace RmsNorm;

__aicore__ inline float ReadLocalFloat(LocalTensor<float> srcLocal, uint32_t index = 0)
{
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);
    float value = srcLocal.GetValue(index);
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventSV);
    WaitFlag<HardEvent::S_V>(eventSV);
    return value;
}

__aicore__ inline void BuildRstd(
    LocalTensor<float> sqx, LocalTensor<float> xLocal, LocalTensor<float> reduceLocal, float avgFactor, float epsilon,
    uint32_t count)
{
    Mul(sqx, xLocal, xLocal, count);
    PipeBarrier<PIPE_V>();
    Muls(sqx, sqx, avgFactor, count);
    PipeBarrier<PIPE_V>();
    ReduceSumCustom(sqx, sqx, reduceLocal, count);
    PipeBarrier<PIPE_V>();
    Adds(sqx, sqx, epsilon, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(sqx, sqx, 1);
    Duplicate(reduceLocal, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(sqx, reduceLocal, sqx, 1);
    PipeBarrier<PIPE_V>();
}
}

#define FUSED_ADD_RMS_NORM_COMMON_MEMBERS \
    TPipe* Ppipe = nullptr; \
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX; \
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma; \
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY; \
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd; \
    TBuf<TPosition::VECCALC> xFp32Buf; \
    TBuf<TPosition::VECCALC> sqxBuf; \
    TBuf<TPosition::VECCALC> reduceFp32Buf; \
    GlobalTensor<T> x1Gm; \
    GlobalTensor<T> x2Gm; \
    GlobalTensor<T> gammaGm; \
    GlobalTensor<T> yGm; \
    GlobalTensor<float> rstdGm; \
    GlobalTensor<T> xGm; \
    uint32_t numRow; \
    uint32_t numCol; \
    uint32_t blockFactor; \
    uint32_t rowFactor; \
    uint32_t ubFactor; \
    float epsilon; \
    float avgFactor; \
    int32_t blockIdx_; \
    uint32_t rowWork = 1

#endif  // FUSED_ADD_RMS_NORM_COMMON_H_
