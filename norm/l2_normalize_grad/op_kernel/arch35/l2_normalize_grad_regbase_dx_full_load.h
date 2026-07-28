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
 * \file l2_normalize_grad_regbase_dx_full_load.h
 * \brief L2NormalizeGrad DX full-load kernel (TilingKey 7000).
 *
 * Applies when inner == 1 (reduced axis is the innermost/last axis, e.g. 2D [N, C] with dim=1)
 * and D (= reduced-axis length) fits UB. Each reduced group is one contiguous row of D elements;
 * outer groups are split across cores. Two reductions per row:
 *   sq = sum(x*x)     -> denom  n = max(sqrt(sq), eps)
 *   s  = sum(y*dy)    -> dx = (dy - y*s) / n   (broadcast n, s back over the row)
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_DX_FULL_LOAD_H
#define L2_NORMALIZE_GRAD_REGBASE_DX_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "l2_normalize_grad_regbase_common.h"

namespace L2NormalizeGrad {
using namespace AscendC;

template <typename T_X>
class RegbaseDxFullLoad {
public:
    __aicore__ inline RegbaseDxFullLoad(TPipe* pipe, const L2NormalizeGradTilingData* tilingData)
        : Ppipe_(pipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* dy, __gm__ uint8_t* dx)
    {
        uint32_t coreIdx = GetBlockIdx();
        usedCoreNum_ = tiling_->usedCoreNum;
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        cols_ = tiling_->dimLen; // D (reduced-axis length)
        rows_ = tiling_->outer;  // number of reduced groups
        blockFactor_ = tiling_->blockFactor;
        eps_ = tiling_->eps;

        colsAlignBlock_ = IsSameType<T_X, float>::value ? AlignUp(cols_, static_cast<int64_t>(FLOAT_NUM_BLOCK)) :
                                                          AlignUp(cols_, static_cast<int64_t>(HALF_NUM_BLOCK));
        colsAlign2VL_ = AlignUp(cols_, static_cast<int64_t>(FLOAT_NUM_2VL));

        ubFactor_ = UB_FACTOR_DX_FULL_LOAD;
        ubFactorD_ = colsAlign2VL_;
        ubFactorN_ = ubFactor_ / ubFactorD_; // rows processed per UB batch (>= 1 by tiling)

        // inner == 1 -> row stride in GM is exactly cols_.
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + coreIdx * blockFactor_ * cols_);
        yGm_.SetGlobalBuffer((__gm__ T_X*)y + coreIdx * blockFactor_ * cols_);
        dyGm_.SetGlobalBuffer((__gm__ T_X*)dy + coreIdx * blockFactor_ * cols_);
        dxGm_.SetGlobalBuffer((__gm__ T_X*)dx + coreIdx * blockFactor_ * cols_);

        Ppipe_->InitBuffer(inQueueX_, DB_NUM, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueY_, DB_NUM, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueDy_, DB_NUM, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(outQueueDx_, DB_NUM, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(reduceBufSq_, ubFactorN_ * colsAlign2VL_ * sizeof(float));
        Ppipe_->InitBuffer(reduceBufS_, ubFactorN_ * colsAlign2VL_ * sizeof(float));
        Ppipe_->InitBuffer(tmpSumSqBuf_, AlignUp(ubFactorN_, static_cast<int64_t>(V_LENGTH)) * sizeof(float));
        Ppipe_->InitBuffer(tmpSumSBuf_, AlignUp(ubFactorN_, static_cast<int64_t>(V_LENGTH)) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        int64_t blockTail = rows_ - (usedCoreNum_ - 1) * blockFactor_;
        int64_t calcRowNum = coreIdx == usedCoreNum_ - 1 ? blockTail : blockFactor_;
        int64_t calcRowNumRemain = calcRowNum;
        for (int64_t rowIdx = 0; rowIdx < calcRowNum; rowIdx += ubFactorN_) {
            int64_t calcRowNumSub = Min(ubFactorN_, calcRowNumRemain);
            SubProcess(rowIdx, calcRowNumSub);
            calcRowNumRemain -= ubFactorN_;
        }
    }

    __aicore__ inline void SubProcess(int64_t rowIdx, int64_t calcRowNumSub)
    {
        CopyIn(inQueueX_, xGm_, rowIdx, calcRowNumSub);
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        CopyIn(inQueueY_, yGm_, rowIdx, calcRowNumSub);
        LocalTensor<float> yLocal = inQueueY_.DeQue<float>();
        CopyIn(inQueueDy_, dyGm_, rowIdx, calcRowNumSub);
        LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();

        LocalTensor<float> reduceSqLocal = reduceBufSq_.Get<float>();
        LocalTensor<float> reduceSLocal = reduceBufS_.Get<float>();
        LocalTensor<float> tmpSumSqLocal = tmpSumSqBuf_.Get<float>();
        LocalTensor<float> tmpSumSLocal = tmpSumSBuf_.Get<float>();

        // Zero the reduce buffers so masked stores below leave a clean [D, colsAlign2VL) tail
        // (correct ReduceSum for arbitrary, non-VL-aligned D).
        Duplicate(reduceSqLocal, 0.0f, static_cast<int32_t>(calcRowNumSub * colsAlign2VL_));
        Duplicate(reduceSLocal, 0.0f, static_cast<int32_t>(calcRowNumSub * colsAlign2VL_));

        uint16_t loopRow = static_cast<uint16_t>(calcRowNumSub);
        constexpr uint32_t oneRepeat = V_LENGTH;
        int64_t colsBlk = colsAlignBlock_;
        uint16_t repeatCount = static_cast<uint16_t>(DivCeil(cols_, static_cast<int64_t>(oneRepeat)));

        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_X* yAddr = (__ubuf__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ T_X* dyAddr = (__ubuf__ T_X*)dyLocal.GetPhyAddr();
        __local_mem__ float* reduceSqAddr = (__ubuf__ float*)reduceSqLocal.GetPhyAddr();
        __local_mem__ float* reduceSAddr = (__ubuf__ float*)reduceSLocal.GetPhyAddr();

        // Pass 1: two reductions. Masked (ZEROING) so the last partial VL only touches valid lanes.
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, yReg, dyReg, sqReg, sReg;
            for (uint16_t r = 0; r < loopRow; r++) {
                uint32_t sreg = static_cast<uint32_t>(cols_);
                MaskReg maskReg;
                for (uint16_t i = 0; i < repeatCount; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAndCast(xReg, xAddr, maskReg, r * colsBlk + i * oneRepeat);
                    Mul(sqReg, xReg, xReg, maskReg);
                    DataCopy(reduceSqAddr + static_cast<uint32_t>(r * colsAlign2VL_ + i * oneRepeat), sqReg, maskReg);
                    LoadAndCast(yReg, yAddr, maskReg, r * colsBlk + i * oneRepeat);
                    LoadAndCast(dyReg, dyAddr, maskReg, r * colsBlk + i * oneRepeat);
                    Mul(sReg, yReg, dyReg, maskReg);
                    DataCopy(reduceSAddr + static_cast<uint32_t>(r * colsAlign2VL_ + i * oneRepeat), sReg, maskReg);
                }
            }
        }

        uint32_t srcShape[2] = {static_cast<uint32_t>(calcRowNumSub), static_cast<uint32_t>(colsAlign2VL_)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(tmpSumSqLocal, reduceSqLocal, srcShape, false);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(tmpSumSLocal, reduceSLocal, srcShape, false);

        LocalTensor<float> dxLocal = outQueueDx_.AllocTensor<float>();
        __local_mem__ float* sqSumAddr = (__ubuf__ float*)tmpSumSqLocal.GetPhyAddr();
        __local_mem__ float* sSumAddr = (__ubuf__ float*)tmpSumSLocal.GetPhyAddr();
        __local_mem__ T_X* dxAddr = (__ubuf__ T_X*)dxLocal.GetPhyAddr();

        // Pass 2: dx = (dy - y*s) / max(sqrt(sq), eps), broadcasting the per-row scalars.
        __VEC_SCOPE__
        {
            RegTensor<float> yReg, dyReg, sqReg, sReg, nReg, ysReg, subReg, dxReg;
            for (uint16_t r = 0; r < loopRow; r++) {
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                DataCopy<float, LoadDist::DIST_BRC_B32>(sqReg, sqSumAddr + static_cast<uint32_t>(r));
                DataCopy<float, LoadDist::DIST_BRC_B32>(sReg, sSumAddr + static_cast<uint32_t>(r));
                Sqrt(nReg, sqReg, maskAll);
                Maxs(nReg, nReg, eps_, maskAll);
                uint32_t sreg = static_cast<uint32_t>(cols_);
                MaskReg maskReg;
                for (uint16_t i = 0; i < repeatCount; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAndCast(yReg, yAddr, maskReg, r * colsBlk + i * oneRepeat);
                    LoadAndCast(dyReg, dyAddr, maskReg, r * colsBlk + i * oneRepeat);
                    Mul(ysReg, yReg, sReg, maskReg);
                    Sub(subReg, dyReg, ysReg, maskReg);
                    Div(dxReg, subReg, nReg, maskReg);
                    StoreDx<T_X>(dxAddr, static_cast<uint32_t>(r * colsBlk + i * oneRepeat), dxReg, maskReg);
                }
            }
        }

        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
        inQueueDy_.FreeTensor(dyLocal);
        outQueueDx_.EnQue(dxLocal);
        CopyOutDx(rowIdx, calcRowNumSub);
    }

    __aicore__ inline void CopyIn(TQue<QuePosition::VECIN, DEPTH_TWO>& que, GlobalTensor<T_X>& gm, int64_t rowIdx,
                                  int64_t calcRow)
    {
        LocalTensor<T_X> local = que.AllocTensor<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(calcRow),             // blockCount
            static_cast<uint32_t>(cols_ * sizeof(T_X)), // blockLen (bytes)
            0,                                          // srcStride
            0,                                          // dstStride
            0                                           // rsv
        };
        // rightPadding true -> pad each row to 32B block alignment with zeros (avoids VEC_ERROR on odd D).
        DataCopyPad(local, gm[rowIdx * cols_], copyParams, {true, 0, 0, 0});
        que.EnQue(local);
    }

    __aicore__ inline void CopyOutDx(int64_t rowIdx, int64_t calcRow)
    {
        LocalTensor<T_X> dxLocal = outQueueDx_.DeQue<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(calcRow),             // blockCount
            static_cast<uint32_t>(cols_ * sizeof(T_X)), // blockLen (bytes)
            0,                                          // srcStride
            0,                                          // dstStride
            0                                           // rsv
        };
        DataCopyPad(dxGm_[rowIdx * cols_], dxLocal, copyParams);
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
    TBuf<TPosition::VECCALC> tmpSumSqBuf_;
    TBuf<TPosition::VECCALC> tmpSumSBuf_;

    uint32_t usedCoreNum_;
    int64_t rows_;
    int64_t cols_;
    int64_t colsAlignBlock_;
    int64_t colsAlign2VL_;
    int64_t blockFactor_;
    int64_t ubFactor_;
    int64_t ubFactorN_;
    int64_t ubFactorD_;
    float eps_;
};
} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_DX_FULL_LOAD_H
