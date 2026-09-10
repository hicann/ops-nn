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
#include "l2_normalize_grad_regbase_base.h"

namespace L2NormalizeGrad {
using namespace AscendC;

template <typename T_X>
class RegbaseDxFullLoad : public RegbaseDxBase<T_X> {
    using Base = RegbaseDxBase<T_X>;
    // 模板基类的成员不在非依赖名字查找里,逐个引入(比满篇 this-> 可读)
    using Base::blockFactor_;
    using Base::CopyIn2D;
    using Base::CopyOut2D;
    using Base::coreIdx_;
    using Base::dxGm_;
    using Base::dyGm_;
    using Base::eps_;
    using Base::InitCommon;
    using Base::InitQueues;
    using Base::Ppipe_;
    using Base::tiling_;
    using Base::usedCoreNum_;
    using Base::xGm_;
    using Base::yGm_;

public:
    __aicore__ inline RegbaseDxFullLoad(TPipe* pipe, const L2NormalizeGradTilingData* tilingData)
        : Base(pipe, tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* dy, __gm__ uint8_t* dx)
    {
        cols_ = tiling_->dimLen; // D (reduced-axis length)
        rows_ = tiling_->outer;  // number of reduced groups
        if (!InitCommon(x, y, dy, dx, cols_)) {
            return; // 本核无任务
        }

        // 对齐值与批次数一律取 host 下发,内核不再自行 AlignUp/除法(避免两套算法)
        colsAlignBlock_ = tiling_->colsAlignBlock;
        colsAlignVL_ = tiling_->colsAlignVL;
        ubFactor_ = tiling_->ubFactorElems;
        ubFactorD_ = tiling_->colsAlignVL;
        ubFactorN_ = tiling_->ubFactorN;

        // UB 尺寸一律透传 host 下发值,内核不自行推算(见 tiling_data 注释)
        InitQueues(inQueueX_, inQueueY_, inQueueDy_, outQueueDx_);
        Ppipe_->InitBuffer(reduceBufSq_, tiling_->reduceBufBytes);
        Ppipe_->InitBuffer(reduceBufS_, tiling_->reduceBufBytes);
        Ppipe_->InitBuffer(tmpSumSqBuf_, tiling_->tmpBufBytes);
        Ppipe_->InitBuffer(tmpSumSBuf_, tiling_->tmpBufBytes);
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

        // 不需要 Duplicate 清零: 行宽按 1VL 对齐,下面的 VF 循环恰好铺满 [0, colsAlignVL);
        // Mul 的默认 MaskMergeMode 是 ZEROING,末轮超出 D 的 lane 在寄存器里已是 0,
        // 用全掩码整 VL 写出即可把尾部一并写成 0 —— 清零是计算的副产品,不占额外一趟。
        uint16_t loopRow = static_cast<uint16_t>(calcRowNumSub);
        constexpr uint32_t oneRepeat = V_LENGTH;
        int64_t colsBlk = colsAlignBlock_;
        uint16_t repeatCount = static_cast<uint16_t>(DivCeil(cols_, static_cast<int64_t>(oneRepeat)));

        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_X* yAddr = (__ubuf__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ T_X* dyAddr = (__ubuf__ T_X*)dyLocal.GetPhyAddr();
        __local_mem__ float* reduceSqAddr = (__ubuf__ float*)reduceSqLocal.GetPhyAddr();
        __local_mem__ float* reduceSAddr = (__ubuf__ float*)reduceSLocal.GetPhyAddr();

        ReducePass(xAddr, yAddr, dyAddr, reduceSqAddr, reduceSAddr, loopRow, repeatCount, colsBlk);

        uint32_t srcShape[2] = {static_cast<uint32_t>(calcRowNumSub), static_cast<uint32_t>(colsAlignVL_)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(tmpSumSqLocal, reduceSqLocal, srcShape, false);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(tmpSumSLocal, reduceSLocal, srcShape, false);

        LocalTensor<float> dxLocal = outQueueDx_.AllocTensor<float>();
        __local_mem__ float* sqSumAddr = (__ubuf__ float*)tmpSumSqLocal.GetPhyAddr();
        __local_mem__ float* sSumAddr = (__ubuf__ float*)tmpSumSLocal.GetPhyAddr();
        __local_mem__ T_X* dxAddr = (__ubuf__ T_X*)dxLocal.GetPhyAddr();

        ComputePass(yAddr, dyAddr, sqSumAddr, sSumAddr, dxAddr, loopRow, repeatCount, colsBlk);

        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
        inQueueDy_.FreeTensor(dyLocal);
        outQueueDx_.EnQue(dxLocal);
        CopyOutDx(rowIdx, calcRowNumSub);
    }

    // Pass 1:两路归约展开(x² 与 y·dy)。Masked(ZEROING),末轮只触碰有效 lane;
    // 整 VL 写出让尾部 slack 顺带写 0,省一趟 Duplicate。
    __aicore__ inline void ReducePass(__local_mem__ T_X* xAddr, __local_mem__ T_X* yAddr, __local_mem__ T_X* dyAddr,
                                      __local_mem__ float* reduceSqAddr, __local_mem__ float* reduceSAddr,
                                      uint16_t loopRow, uint16_t repeatCount, int64_t colsBlk)
    {
        constexpr uint32_t oneRepeat = V_LENGTH;
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, yReg, dyReg, sqReg, sReg;
            MaskReg fullMask = CreateMask<float>(); // MaskPattern::ALL, 整 VL 写出
            for (uint16_t r = 0; r < loopRow; r++) {
                uint32_t sreg = static_cast<uint32_t>(cols_);
                MaskReg maskReg;
                for (uint16_t i = 0; i < repeatCount; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAndCast(xReg, xAddr, maskReg, r * colsBlk + i * oneRepeat);
                    Mul(sqReg, xReg, xReg, maskReg);
                    DataCopy(reduceSqAddr + static_cast<uint32_t>(r * colsAlignVL_ + i * oneRepeat), sqReg, fullMask);
                    LoadAndCast(yReg, yAddr, maskReg, r * colsBlk + i * oneRepeat);
                    LoadAndCast(dyReg, dyAddr, maskReg, r * colsBlk + i * oneRepeat);
                    Mul(sReg, yReg, dyReg, maskReg);
                    DataCopy(reduceSAddr + static_cast<uint32_t>(r * colsAlignVL_ + i * oneRepeat), sReg, fullMask);
                }
            }
        }
    }

    // Pass 2:dx = (dy - y*s) / max(sqrt(sq), eps),按行广播两个标量。
    __aicore__ inline void ComputePass(__local_mem__ T_X* yAddr, __local_mem__ T_X* dyAddr,
                                       __local_mem__ float* sqSumAddr, __local_mem__ float* sSumAddr,
                                       __local_mem__ T_X* dxAddr, uint16_t loopRow, uint16_t repeatCount,
                                       int64_t colsBlk)
    {
        constexpr uint32_t oneRepeat = V_LENGTH;
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
    }

    __aicore__ inline void CopyIn(TQue<QuePosition::VECIN, DEPTH_TWO>& que, GlobalTensor<T_X>& gm, int64_t rowIdx,
                                  int64_t calcRow)
    {
        CopyIn2D(que, gm, rowIdx * cols_, calcRow, cols_, 0); // 行在 GM 上连续,无空隙
    }

    __aicore__ inline void CopyOutDx(int64_t rowIdx, int64_t calcRow)
    {
        CopyOut2D(outQueueDx_, rowIdx * cols_, calcRow, cols_, 0);
    }

private:
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueX_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueY_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueDy_;
    TQue<QuePosition::VECOUT, DEPTH_TWO> outQueueDx_;

    TBuf<TPosition::VECCALC> reduceBufSq_;
    TBuf<TPosition::VECCALC> reduceBufS_;
    TBuf<TPosition::VECCALC> tmpSumSqBuf_;
    TBuf<TPosition::VECCALC> tmpSumSBuf_;

    int64_t rows_;
    int64_t cols_;
    int64_t colsAlignBlock_;
    int64_t colsAlignVL_;
    int64_t ubFactor_;
    int64_t ubFactorN_;
    int64_t ubFactorD_;
};
} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_DX_FULL_LOAD_H
