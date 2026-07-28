/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been technically reviewed for functional accuracy.
 */

/*!
 * \file l2_normalize_grad_regbase_dx_split_d.h
 * \brief L2NormalizeGrad DX split-D kernel (TilingKey 7010).
 *
 * Applies when inner == 1 but D (reduced-axis length) exceeds UB. The row is streamed in chunks
 * of UB_FACTOR_DX_SPLIT_D. Pass 1 (FormerProcess) reduces each chunk (sum(x*x), sum(y*dy)) into a
 * per-chunk accumulator buffer, then reduces the accumulators to per-row scalars sq/s. Pass 2
 * (LatterProcess) re-streams the row and writes dx = (dy - y*s) / max(sqrt(sq), eps).
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H
#define L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "l2_normalize_grad_regbase_common.h"

namespace L2NormalizeGrad {
using namespace AscendC;

constexpr uint32_t SPLIT_D_MAX_CHUNKS = 256; // supports D up to 256 * UB_FACTOR_DX_SPLIT_D = 1M

template <typename T_X>
class RegbaseDxSplitD {
public:
    __aicore__ inline RegbaseDxSplitD(TPipe* pipe, const L2NormalizeGradTilingData* tilingData)
        : Ppipe_(pipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* dy, __gm__ uint8_t* dx)
    {
        usedCoreNum_ = tiling_->usedCoreNum;
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        rows_ = tiling_->outer;
        cols_ = tiling_->dimLen; // D
        blockFactor_ = tiling_->blockFactor;
        eps_ = tiling_->eps;
        ubFactorD_ = UB_FACTOR_DX_SPLIT_D; // 2VL aligned chunk size
        numChunks_ = DivCeil(cols_, ubFactorD_);

        xGm_.SetGlobalBuffer((__gm__ T_X*)x + coreIdx * blockFactor_ * cols_);
        yGm_.SetGlobalBuffer((__gm__ T_X*)y + coreIdx * blockFactor_ * cols_);
        dyGm_.SetGlobalBuffer((__gm__ T_X*)dy + coreIdx * blockFactor_ * cols_);
        dxGm_.SetGlobalBuffer((__gm__ T_X*)dx + coreIdx * blockFactor_ * cols_);

        Ppipe_->InitBuffer(inQueueX_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueY_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueDy_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(outQueueDx_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(reduceBufSq_, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(reduceBufS_, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(accumBufSq_, SPLIT_D_MAX_CHUNKS * sizeof(float));
        Ppipe_->InitBuffer(accumBufS_, SPLIT_D_MAX_CHUNKS * sizeof(float));
        Ppipe_->InitBuffer(tmpSumSqBuf_, V_LENGTH * sizeof(float));
        Ppipe_->InitBuffer(tmpSumSBuf_, V_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        int64_t blockTail = rows_ - (usedCoreNum_ - 1) * blockFactor_;
        int64_t calcRowNum = coreIdx == usedCoreNum_ - 1 ? blockTail : blockFactor_;
        for (int64_t rowIdx = 0; rowIdx < calcRowNum; rowIdx++) {
            FormerProcess(rowIdx);
            LatterProcess(rowIdx);
        }
    }

    // Reduce the whole row into per-row scalars sq (sum x*x) and s (sum y*dy).
    __aicore__ inline void FormerProcess(int64_t rowIdx)
    {
        LocalTensor<float> accumSqLocal = accumBufSq_.Get<float>();
        LocalTensor<float> accumSLocal = accumBufS_.Get<float>();
        int64_t accumAlign = AlignUp(numChunks_, static_cast<int64_t>(FLOAT_NUM_2VL));
        Duplicate(accumSqLocal, 0.0f, static_cast<int32_t>(accumAlign));
        Duplicate(accumSLocal, 0.0f, static_cast<int32_t>(accumAlign));

        int64_t chunkIdx = 0;
        for (int64_t colIdx = 0; colIdx < cols_; colIdx += ubFactorD_, chunkIdx++) {
            int64_t cnt = Min(ubFactorD_, cols_ - colIdx);
            ReduceChunk(rowIdx, colIdx, cnt, accumSqLocal, accumSLocal, chunkIdx);
        }

        LocalTensor<float> tmpSumSqLocal = tmpSumSqBuf_.Get<float>();
        LocalTensor<float> tmpSumSLocal = tmpSumSBuf_.Get<float>();
        uint32_t accShape[2] = {1U, static_cast<uint32_t>(accumAlign)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(tmpSumSqLocal, accumSqLocal, accShape, false);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(tmpSumSLocal, accumSLocal, accShape, false);
    }

    // Load one chunk, compute sum(x*x) and sum(y*dy) over it, store into accum[chunkIdx].
    __aicore__ inline void ReduceChunk(int64_t rowIdx, int64_t colIdx, int64_t cnt, LocalTensor<float>& accumSqLocal,
                                       LocalTensor<float>& accumSLocal, int64_t chunkIdx)
    {
        CopyIn(inQueueX_, xGm_, rowIdx, colIdx, cnt);
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        CopyIn(inQueueY_, yGm_, rowIdx, colIdx, cnt);
        LocalTensor<float> yLocal = inQueueY_.DeQue<float>();
        CopyIn(inQueueDy_, dyGm_, rowIdx, colIdx, cnt);
        LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();

        LocalTensor<float> reduceSqLocal = reduceBufSq_.Get<float>();
        LocalTensor<float> reduceSLocal = reduceBufS_.Get<float>();
        int64_t cntAlign2VL = AlignUp(cnt, static_cast<int64_t>(FLOAT_NUM_2VL));
        Duplicate(reduceSqLocal, 0.0f, static_cast<int32_t>(cntAlign2VL));
        Duplicate(reduceSLocal, 0.0f, static_cast<int32_t>(cntAlign2VL));

        constexpr uint32_t oneRepeat = V_LENGTH;
        uint16_t repeatCount = static_cast<uint16_t>(DivCeil(cnt, static_cast<int64_t>(oneRepeat)));
        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_X* yAddr = (__ubuf__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ T_X* dyAddr = (__ubuf__ T_X*)dyLocal.GetPhyAddr();
        __local_mem__ float* reduceSqAddr = (__ubuf__ float*)reduceSqLocal.GetPhyAddr();
        __local_mem__ float* reduceSAddr = (__ubuf__ float*)reduceSLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, yReg, dyReg, sqReg, sReg;
            uint32_t sreg = static_cast<uint32_t>(cnt);
            MaskReg maskReg;
            for (uint16_t i = 0; i < repeatCount; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAndCast(xReg, xAddr, maskReg, i * oneRepeat);
                Mul(sqReg, xReg, xReg, maskReg);
                DataCopy(reduceSqAddr + static_cast<uint32_t>(i * oneRepeat), sqReg, maskReg);
                LoadAndCast(yReg, yAddr, maskReg, i * oneRepeat);
                LoadAndCast(dyReg, dyAddr, maskReg, i * oneRepeat);
                Mul(sReg, yReg, dyReg, maskReg);
                DataCopy(reduceSAddr + static_cast<uint32_t>(i * oneRepeat), sReg, maskReg);
            }
        }
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
        inQueueDy_.FreeTensor(dyLocal);

        uint32_t chunkShape[2] = {1U, static_cast<uint32_t>(cntAlign2VL)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(accumSqLocal[chunkIdx], reduceSqLocal, chunkShape,
                                                                      false);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(accumSLocal[chunkIdx], reduceSLocal, chunkShape,
                                                                      false);
    }

    // Re-stream the row and write dx = (dy - y*s) / max(sqrt(sq), eps).
    __aicore__ inline void LatterProcess(int64_t rowIdx)
    {
        LocalTensor<float> tmpSumSqLocal = tmpSumSqBuf_.Get<float>();
        LocalTensor<float> tmpSumSLocal = tmpSumSBuf_.Get<float>();
        __local_mem__ float* sqSumAddr = (__ubuf__ float*)tmpSumSqLocal.GetPhyAddr();
        __local_mem__ float* sSumAddr = (__ubuf__ float*)tmpSumSLocal.GetPhyAddr();

        for (int64_t colIdx = 0; colIdx < cols_; colIdx += ubFactorD_) {
            int64_t cnt = Min(ubFactorD_, cols_ - colIdx);
            CopyIn(inQueueY_, yGm_, rowIdx, colIdx, cnt);
            LocalTensor<float> yLocal = inQueueY_.DeQue<float>();
            CopyIn(inQueueDy_, dyGm_, rowIdx, colIdx, cnt);
            LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();
            LocalTensor<float> dxLocal = outQueueDx_.AllocTensor<float>();

            constexpr uint32_t oneRepeat = V_LENGTH;
            uint16_t repeatCount = static_cast<uint16_t>(DivCeil(cnt, static_cast<int64_t>(oneRepeat)));
            __local_mem__ T_X* yAddr = (__ubuf__ T_X*)yLocal.GetPhyAddr();
            __local_mem__ T_X* dyAddr = (__ubuf__ T_X*)dyLocal.GetPhyAddr();
            __local_mem__ T_X* dxAddr = (__ubuf__ T_X*)dxLocal.GetPhyAddr();
            __VEC_SCOPE__
            {
                RegTensor<float> yReg, dyReg, sqReg, sReg, nReg, ysReg, subReg, dxReg;
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                DataCopy<float, LoadDist::DIST_BRC_B32>(sqReg, sqSumAddr);
                DataCopy<float, LoadDist::DIST_BRC_B32>(sReg, sSumAddr);
                Sqrt(nReg, sqReg, maskAll);
                Maxs(nReg, nReg, eps_, maskAll);
                uint32_t sreg = static_cast<uint32_t>(cnt);
                MaskReg maskReg;
                for (uint16_t i = 0; i < repeatCount; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAndCast(yReg, yAddr, maskReg, i * oneRepeat);
                    LoadAndCast(dyReg, dyAddr, maskReg, i * oneRepeat);
                    Mul(ysReg, yReg, sReg, maskReg);
                    Sub(subReg, dyReg, ysReg, maskReg);
                    Div(dxReg, subReg, nReg, maskReg);
                    StoreDx<T_X>(dxAddr, static_cast<uint32_t>(i * oneRepeat), dxReg, maskReg);
                }
            }
            inQueueY_.FreeTensor(yLocal);
            inQueueDy_.FreeTensor(dyLocal);
            outQueueDx_.EnQue(dxLocal);
            CopyOutDx(rowIdx, colIdx, cnt);
        }
    }

    __aicore__ inline void CopyIn(TQue<QuePosition::VECIN, DEPTH_TWO>& que, GlobalTensor<T_X>& gm, int64_t rowIdx,
                                  int64_t colIdx, int64_t cnt)
    {
        LocalTensor<T_X> local = que.AllocTensor<T_X>();
        DataCopyExtParams copyParams{
            1,                                        // blockCount
            static_cast<uint32_t>(cnt * sizeof(T_X)), // blockLen (bytes)
            0,                                        // srcStride
            0,                                        // dstStride
            0                                         // rsv
        };
        DataCopyPad(local, gm[rowIdx * cols_ + colIdx], copyParams, {true, 0, 0, 0});
        que.EnQue(local);
    }

    __aicore__ inline void CopyOutDx(int64_t rowIdx, int64_t colIdx, int64_t cnt)
    {
        LocalTensor<T_X> dxLocal = outQueueDx_.DeQue<T_X>();
        DataCopyExtParams copyParams{
            1,                                        // blockCount
            static_cast<uint32_t>(cnt * sizeof(T_X)), // blockLen (bytes)
            0,                                        // srcStride
            0,                                        // dstStride
            0                                         // rsv
        };
        DataCopyPad(dxGm_[rowIdx * cols_ + colIdx], dxLocal, copyParams);
        outQueueDx_.FreeTensor(dxLocal);
    }

private:
    TPipe* Ppipe_;
    const L2NormalizeGradTilingData* tiling_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_X> yGm_;
    GlobalTensor<T_X> dyGm_;
    GlobalTensor<T_X> dxGm_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueX_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueY_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueDy_;
    TQue<QuePosition::VECOUT, DEPTH_TWO> outQueueDx_;
    TBuf<TPosition::VECCALC> reduceBufSq_;
    TBuf<TPosition::VECCALC> reduceBufS_;
    TBuf<TPosition::VECCALC> accumBufSq_;
    TBuf<TPosition::VECCALC> accumBufS_;
    TBuf<TPosition::VECCALC> tmpSumSqBuf_;
    TBuf<TPosition::VECCALC> tmpSumSBuf_;

    uint32_t usedCoreNum_;
    int64_t rows_;
    int64_t cols_;
    int64_t blockFactor_;
    int64_t ubFactorD_;
    int64_t numChunks_;
    float eps_;
};
} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H
