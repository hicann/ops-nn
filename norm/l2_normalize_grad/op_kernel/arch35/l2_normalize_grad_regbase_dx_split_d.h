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
 * of tilingData.ubFactorElems (host 反推). Pass 1 (FormerProcess) reduces each chunk (sum(x*x), sum(y*dy)) into a
 * per-chunk accumulator buffer, then reduces the accumulators to per-row scalars sq/s. Pass 2
 * (LatterProcess) re-streams the row and writes dx = (dy - y*s) / max(sqrt(sq), eps).
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H
#define L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "l2_normalize_grad_regbase_base.h"

namespace L2NormalizeGrad {
using namespace AscendC;

constexpr uint32_t SPLIT_D_MAX_CHUNKS = 256; // 累加槽数; chunk 数超过它时按组累加(D 无上限)

template <typename T_X>
class RegbaseDxSplitD : public RegbaseDxBase<T_X> {
    using Base = RegbaseDxBase<T_X>;
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
    __aicore__ inline RegbaseDxSplitD(TPipe* pipe, const L2NormalizeGradTilingData* tilingData) : Base(pipe, tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* dy, __gm__ uint8_t* dx)
    {
        rows_ = tiling_->outer;
        cols_ = tiling_->dimLen; // D
        if (!InitCommon(x, y, dy, dx, cols_)) {
            return; // 本核无任务
        }
        ubFactorD_ = tiling_->ubFactorElems; // 2VL 对齐的分块长度,由 host 从 ubSize 反推下发
        numChunks_ = tiling_->numChunks;     // host 下发,内核不再 DivCeil

        // UB 尺寸一律透传 host 下发值,内核不自行推算(见 tiling_data 注释)
        InitQueues(inQueueX_, inQueueY_, inQueueDy_, outQueueDx_);
        Ppipe_->InitBuffer(reduceBufSq_, tiling_->reduceBufBytes);
        Ppipe_->InitBuffer(reduceBufS_, tiling_->reduceBufBytes);
        Ppipe_->InitBuffer(accumBufSq_, tiling_->accumBufBytes);
        Ppipe_->InitBuffer(accumBufS_, tiling_->accumBufBytes);
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
        LocalTensor<float> tmpSumSqLocal = tmpSumSqBuf_.Get<float>();
        LocalTensor<float> tmpSumSLocal = tmpSumSBuf_.Get<float>();
        // accum 缓冲固定 SPLIT_D_MAX_CHUNKS(=256) 个槽。chunk 数超过它时必须**按组**累加
        // (组内向量规约 + 组间标量累加), 不能按 numChunks_ 直接 Duplicate/写槽:
        // issue #31 的 1 维 D=1.51e8 需要约 3.7 万个槽, 原实现直接 Duplicate(accum, 0, 36928) 就把
        // 256 槽的缓冲写穿 → errcode 341(VEC 访问 UB 越界)。分组后 D 不再有上限。
        const int64_t maxSlots = static_cast<int64_t>(SPLIT_D_MAX_CHUNKS);
        float totalSq = 0.0f;
        float totalS = 0.0f;
        int64_t colIdx = 0;
        while (colIdx < cols_) {
            int64_t remainChunks = DivCeil(cols_ - colIdx, ubFactorD_);
            int64_t groupSlots = Min(remainChunks, maxSlots);
            // 不清零、不对齐: 归约范围直接取真实槽数 groupSlots,ReduceSum<AR> 对非 32B 对齐的
            // 末轴走 ReduceARReuseSourceUnAligned 分支,pad 槽根本不进入归约,自然无需 Duplicate。
            for (int64_t slot = 0; slot < groupSlots; slot++, colIdx += ubFactorD_) {
                int64_t cnt = Min(ubFactorD_, cols_ - colIdx);
                ReduceChunk(rowIdx, colIdx, cnt, accumSqLocal, accumSLocal, slot);
            }
            uint32_t accShape[2] = {1U, static_cast<uint32_t>(groupSlots)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(tmpSumSqLocal, accumSqLocal, accShape, false);
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(tmpSumSLocal, accumSLocal, accShape, false);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            totalSq += tmpSumSqLocal.GetValue(0);
            totalS += tmpSumSLocal.GetValue(0);
        }
        // 组间总和写回 slot0, 供 LatterProcess 以 DIST_BRC_B32 广播读取
        tmpSumSqLocal.SetValue(0, totalSq);
        tmpSumSLocal.SetValue(0, totalS);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
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
        // 1VL 对齐 => VF 循环恰好铺满 [0, cntAlignVL); Mul 的 ZEROING 已把末轮多余 lane 置 0,
        // 全掩码整 VL 写出即完成尾部清零,无需 Duplicate。
        int64_t cntAlignVL = AlignUp(cnt, static_cast<int64_t>(V_LENGTH));

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
            MaskReg fullMask = CreateMask<float>(); // MaskPattern::ALL, 整 VL 写出
            uint32_t sreg = static_cast<uint32_t>(cnt);
            MaskReg maskReg;
            for (uint16_t i = 0; i < repeatCount; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAndCast(xReg, xAddr, maskReg, i * oneRepeat);
                Mul(sqReg, xReg, xReg, maskReg);
                DataCopy(reduceSqAddr + static_cast<uint32_t>(i * oneRepeat), sqReg, fullMask);
                LoadAndCast(yReg, yAddr, maskReg, i * oneRepeat);
                LoadAndCast(dyReg, dyAddr, maskReg, i * oneRepeat);
                Mul(sReg, yReg, dyReg, maskReg);
                DataCopy(reduceSAddr + static_cast<uint32_t>(i * oneRepeat), sReg, fullMask);
            }
        }
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
        inQueueDy_.FreeTensor(dyLocal);

        uint32_t chunkShape[2] = {1U, static_cast<uint32_t>(cntAlignVL)};
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
        CopyIn2D(que, gm, rowIdx * cols_ + colIdx, 1, cnt, 0); // 单行分块,GM 上连续
    }

    __aicore__ inline void CopyOutDx(int64_t rowIdx, int64_t colIdx, int64_t cnt)
    {
        CopyOut2D(outQueueDx_, rowIdx * cols_ + colIdx, 1, cnt, 0);
    }

private:
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

    int64_t rows_;
    int64_t cols_;
    int64_t ubFactorD_;
    int64_t numChunks_;
};
} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H
