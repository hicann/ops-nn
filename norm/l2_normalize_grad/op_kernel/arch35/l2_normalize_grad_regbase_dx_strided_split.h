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
 * \file l2_normalize_grad_regbase_dx_strided_split.h
 * \brief L2NormalizeGrad DX strided-split kernel (TilingKey 7030) - inner > 1 且整段 D 放不下 UB。
 *
 * 与 7020 的差别只在“D 是否整段常驻”：7020 一次把 [D, colTile] 全部载入 UB，两趟都在 UB 内完成；
 * 本模板沿 D 分块（每块 dFactor 行），因此累加器必须**常驻 UB**（寄存器活不过分块间的 DataCopy）：
 *   Pass 1  逐块载入 x/y/dy，逐元素求 x^2 / y*dy 落进 fp32 中间量 tile，用平台
 *           ReduceSum<Pattern::Reduce::RA> 按列规约出本块列和，再累进跨分块常驻的每列和；
 *   Pass 2  逐块重载 y/dy，用 n=max(sqrt(accSq),eps)、s=accS 合成 dx 写回。
 * 代价是 y/dy 各多读一遍 GM（3 读 1 写 -> 5 读 1 写），只发生在这一档；结构与 split_d(7010)
 * 的 FormerProcess/LatterProcess 同构，区别是归约方向按 inner 列而非行内规约。
 *
 * UB 预算与所有对齐值均由 host 算准下发（qBufBytes/accumBufBytes/colFactorAlign/tailColAlign/accElems/accVLs），
 * 且 dFactor <= 65535（DataCopyPad 的 blockCount 为 uint16，不可截断）。内核不自行放大。
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_DX_STRIDED_SPLIT_H
#define L2_NORMALIZE_GRAD_REGBASE_DX_STRIDED_SPLIT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "l2_normalize_grad_regbase_base.h"

namespace L2NormalizeGrad {
using namespace AscendC;

template <typename T_X>
class RegbaseDxStridedSplit : public RegbaseDxBase<T_X> {
    using Base = RegbaseDxBase<T_X>;
    using Base::blockFactor_;
    using Base::CalcOuterNum;
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
    __aicore__ inline RegbaseDxStridedSplit(TPipe* pipe, const L2NormalizeGradTilingData* tilingData)
        : Base(pipe, tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* dy, __gm__ uint8_t* dx)
    {
        outer_ = tiling_->outer;
        D_ = tiling_->dimLen;
        inner_ = tiling_->inner;
        colFactor_ = tiling_->colFactor;
        dFactor_ = tiling_->dFactor;

        colFactorAlign_ = tiling_->colFactorAlign; // host 下发,内核不再 AlignUp
        tailColAlign_ = tiling_->tailColAlign;
        colFactorAlignF32_ = tiling_->colFactorAlignF32;
        tailColAlignF32_ = tiling_->tailColAlignF32;
        if (!InitCommon(x, y, dy, dx, D_ * inner_)) {
            return; // 本核无任务
        }

        // tile = dFactor_ * AlignUp(colFactor, block) + V_LENGTH slack；
        // 每列和缓冲 = AlignUp(colFactor, block) + V_LENGTH slack。
        // 两者均由 host 一次算准下发（qBufBytes / accumBufBytes），内核不自行推算。
        InitQueues(inQueueX_, inQueueY_, inQueueDy_, outQueueDx_);

        accElems_ = tiling_->accElems;
        accVLs_ = static_cast<uint16_t>(tiling_->accVLs);
        // 归约用平台 ReduceSum<RA>:需两块 fp32 中间量 tile + 每分块规约结果 + 跨分块常驻的每列和。
        // 归约全部走平台接口,内核不再手写累加与补偿。
        Ppipe_->InitBuffer(sqBuf_, tiling_->midBufBytes);
        Ppipe_->InitBuffer(prodBuf_, tiling_->midBufBytes);
        Ppipe_->InitBuffer(chunkSqBuf_, tiling_->sumBufBytes);
        Ppipe_->InitBuffer(chunkSBuf_, tiling_->sumBufBytes);
        Ppipe_->InitBuffer(sumSqBuf_, tiling_->sumBufBytes);
        Ppipe_->InitBuffer(sumSBuf_, tiling_->sumBufBytes);
        // 跨分块结果先落槽位数组,满组后由 ReduceSum<RA> 沿槽位轴树形规约再累进每列和。
        // s=Σ(y·dy) 会对消,顺序累加链越长残差越大;槽位数由 host 按剩余 UB 解出,=1 时退化为顺序累加。
        chunkSlots_ = tiling_->chunkSlots;
        slotStride_ = tiling_->slotStride;
        Ppipe_->InitBuffer(slotSqBuf_, tiling_->slotBufBytes);
        Ppipe_->InitBuffer(slotSBuf_, tiling_->slotBufBytes);
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        int64_t blockTail = outer_ - (usedCoreNum_ - 1) * blockFactor_;
        int64_t calcOuterNum = coreIdx == usedCoreNum_ - 1 ? blockTail : blockFactor_;
        for (int64_t o = 0; o < calcOuterNum; o++) {
            for (int64_t colStart = 0; colStart < inner_; colStart += colFactor_) {
                int64_t colTile = Min(colFactor_, inner_ - colStart);
                ZeroSums();
                int64_t slotIdx = 0;
                for (int64_t dStart = 0; dStart < D_; dStart += dFactor_) {
                    AccumChunk(o, dStart, Min(dFactor_, D_ - dStart), colStart, colTile, slotIdx);
                    if (++slotIdx == chunkSlots_) {
                        FlushSlots(slotIdx, colTile);
                        slotIdx = 0;
                    }
                }
                if (slotIdx > 0) {
                    FlushSlots(slotIdx, colTile);
                }
                for (int64_t dStart = 0; dStart < D_; dStart += dFactor_) {
                    WriteChunk(o, dStart, Min(dFactor_, D_ - dStart), colStart, colTile);
                }
            }
        }
    }

private:
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueX_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueY_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueDy_;
    TQue<QuePosition::VECOUT, DEPTH_TWO> outQueueDx_;

    // 跨分块常驻的每列和清零(含 VL slack:整 VL 读会把 slack 读进寄存器)
    __aicore__ inline void ZeroSums()
    {
        LocalTensor<float> sumSq = sumSqBuf_.Get<float>();
        LocalTensor<float> sumS = sumSBuf_.Get<float>();
        __local_mem__ float* a0 = (__ubuf__ float*)sumSq.GetPhyAddr();
        __local_mem__ float* a1 = (__ubuf__ float*)sumS.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> zeroReg;
            uint32_t sreg = static_cast<uint32_t>(accElems_);
            for (uint16_t i = 0; i < accVLs_; i++) {
                MaskReg maskReg = UpdateMask<float>(sreg);
                Duplicate(zeroReg, 0.0f, maskReg);
                uint32_t off = static_cast<uint32_t>(i * V_LENGTH);
                DataCopy(a0 + off, zeroReg, maskReg);
                DataCopy(a1 + off, zeroReg, maskReg);
            }
        }
    }

    // Pass 1:载入 [dTile, colTile] 一块 -> 逐元素 x^2 / y*dy(fp32 中间量) -> 平台 ReduceSum<RA>
    //         按列规约出本块列和 -> 累进跨分块常驻的每列和(普通 Add,不再有手写补偿量)。
    __aicore__ inline void AccumChunk(int64_t o, int64_t dStart, int64_t dTile, int64_t colStart, int64_t colTile,
                                      int64_t slotIdx)
    {
        CopyInTile(inQueueX_, xGm_, o, dStart, dTile, colStart, colTile);
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        CopyInTile(inQueueY_, yGm_, o, dStart, dTile, colStart, colTile);
        LocalTensor<float> yLocal = inQueueY_.DeQue<float>();
        CopyInTile(inQueueDy_, dyGm_, o, dStart, dTile, colStart, colTile);
        LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();

        constexpr uint32_t oneRepeat = V_LENGTH;
        uint16_t repeatCount = static_cast<uint16_t>(DivCeil(colTile, static_cast<int64_t>(oneRepeat)));
        const bool isTail = (colTile != colFactor_);
        const int64_t rowStride = isTail ? tailColAlign_ : colFactorAlign_;
        const int64_t rowStrideF32 = isTail ? tailColAlignF32_ : colFactorAlignF32_;
        uint16_t dLoop = static_cast<uint16_t>(dTile);

        LocalTensor<float> sqLocal = sqBuf_.Get<float>();
        LocalTensor<float> prodLocal = prodBuf_.Get<float>();
        LocalTensor<float> slotSq = slotSqBuf_.Get<float>()[slotIdx * slotStride_];
        LocalTensor<float> slotS = slotSBuf_.Get<float>()[slotIdx * slotStride_];
        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_X* yAddr = (__ubuf__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ T_X* dyAddr = (__ubuf__ T_X*)dyLocal.GetPhyAddr();
        __local_mem__ float* sqAddr = (__ubuf__ float*)sqLocal.GetPhyAddr();
        __local_mem__ float* prodAddr = (__ubuf__ float*)prodLocal.GetPhyAddr();

        // 不需要对 pad 车道清零:RA 是**按列独立**规约,pad 列的残值只会污染 pad 列的输出,
        // 而后续计算被 mask 限制在 colTile 个车道内,那些输出不被读取。
        // (full_load/split_d 用的是 AR——pad 落在被规约轴内,那里才必须先清零。)
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, yReg, dyReg, tmpReg;
            uint32_t sregOuter = static_cast<uint32_t>(colTile);
            for (uint16_t i = 0; i < repeatCount; i++) {
                MaskReg maskReg = UpdateMask<float>(sregOuter);
                for (uint16_t d = 0; d < dLoop; d++) {
                    uint32_t srcOff = static_cast<uint32_t>(d * rowStride + i * oneRepeat);
                    uint32_t dstOff = static_cast<uint32_t>(d * rowStrideF32 + i * oneRepeat);
                    LoadAndCast(xReg, xAddr, maskReg, srcOff);
                    Mul(tmpReg, xReg, xReg, maskReg);
                    DataCopy(sqAddr + dstOff, tmpReg, maskReg);
                    LoadAndCast(yReg, yAddr, maskReg, srcOff);
                    LoadAndCast(dyReg, dyAddr, maskReg, srcOff);
                    Mul(tmpReg, yReg, dyReg, maskReg);
                    DataCopy(prodAddr + dstOff, tmpReg, maskReg);
                }
            }
        }

        // A 维传**对齐后的宽度**、srcInnerPad=false —— 与 full_load / confusion_softmax_grad 同款用法:
        // 缓冲已整块清零,pad 车道贡献 0;输出前 colTile 个即所需列和。
        uint32_t srcShape[2] = {static_cast<uint32_t>(dTile), static_cast<uint32_t>(rowStrideF32)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(slotSq, sqLocal, srcShape, false);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(slotS, prodLocal, srcShape, false);

        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
        inQueueDy_.FreeTensor(dyLocal);
    }

    // 一组槽位满(或 D 走完)时:沿**槽位轴**再做一次 ReduceSum<RA> 树形规约,组结果累进每列和。
    // 组数 = ceil(numChunks / chunkSlots),顺序链由此缩短 chunkSlots 倍。
    __aicore__ inline void FlushSlots(int64_t nSlots, int64_t colTile)
    {
        LocalTensor<float> chunkSq = chunkSqBuf_.Get<float>();
        LocalTensor<float> chunkS = chunkSBuf_.Get<float>();
        uint32_t slotShape[2] = {static_cast<uint32_t>(nSlots), static_cast<uint32_t>(slotStride_)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(chunkSq, slotSqBuf_.Get<float>(), slotShape,
                                                                      false);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(chunkS, slotSBuf_.Get<float>(), slotShape, false);

        constexpr uint32_t oneRepeat = V_LENGTH;
        uint16_t repeatCount = static_cast<uint16_t>(DivCeil(colTile, static_cast<int64_t>(oneRepeat)));

        LocalTensor<float> sumSq = sumSqBuf_.Get<float>();
        LocalTensor<float> sumS = sumSBuf_.Get<float>();
        __local_mem__ float* sumSqAddr = (__ubuf__ float*)sumSq.GetPhyAddr();
        __local_mem__ float* sumSAddr = (__ubuf__ float*)sumS.GetPhyAddr();
        __local_mem__ float* chunkSqAddr = (__ubuf__ float*)chunkSq.GetPhyAddr();
        __local_mem__ float* chunkSAddr = (__ubuf__ float*)chunkS.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> accReg, curReg;
            uint32_t sregOuter = static_cast<uint32_t>(colTile);
            for (uint16_t i = 0; i < repeatCount; i++) {
                MaskReg maskReg = UpdateMask<float>(sregOuter);
                uint32_t off = static_cast<uint32_t>(i * oneRepeat);
                DataCopy(accReg, sumSqAddr + off);
                DataCopy(curReg, chunkSqAddr + off);
                Add(accReg, accReg, curReg, maskReg);
                DataCopy(sumSqAddr + off, accReg, maskReg);
                DataCopy(accReg, sumSAddr + off);
                DataCopy(curReg, chunkSAddr + off);
                Add(accReg, accReg, curReg, maskReg);
                DataCopy(sumSAddr + off, accReg, maskReg);
            }
        }
    }

    // Pass 2:重载 y/dy 一块,用常驻的每列和合成 dx = (dy - y*s) / max(sqrt(sumSq), eps)。
    __aicore__ inline void WriteChunk(int64_t o, int64_t dStart, int64_t dTile, int64_t colStart, int64_t colTile)
    {
        CopyInTile(inQueueY_, yGm_, o, dStart, dTile, colStart, colTile);
        LocalTensor<float> yLocal = inQueueY_.DeQue<float>();
        CopyInTile(inQueueDy_, dyGm_, o, dStart, dTile, colStart, colTile);
        LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();
        LocalTensor<float> dxLocal = outQueueDx_.AllocTensor<float>();

        constexpr uint32_t oneRepeat = V_LENGTH;
        uint16_t repeatCount = static_cast<uint16_t>(DivCeil(colTile, static_cast<int64_t>(oneRepeat)));
        const bool isTail = (colTile != colFactor_);
        const int64_t rowStride = isTail ? tailColAlign_ : colFactorAlign_;
        uint16_t dLoop = static_cast<uint16_t>(dTile);
        __local_mem__ T_X* yAddr = (__ubuf__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ T_X* dyAddr = (__ubuf__ T_X*)dyLocal.GetPhyAddr();
        __local_mem__ T_X* dxAddr = (__ubuf__ T_X*)dxLocal.GetPhyAddr();
        __local_mem__ float* sumSqAddr = (__ubuf__ float*)sumSqBuf_.Get<float>().GetPhyAddr();
        __local_mem__ float* sumSAddr = (__ubuf__ float*)sumSBuf_.Get<float>().GetPhyAddr();

        __VEC_SCOPE__
        {
            RegTensor<float> yReg, dyReg, sqReg, sReg, nReg, ysReg, subReg, dxReg;
            uint32_t sregOuter = static_cast<uint32_t>(colTile);
            for (uint16_t i = 0; i < repeatCount; i++) {
                MaskReg maskReg = UpdateMask<float>(sregOuter);
                uint32_t sumOff = static_cast<uint32_t>(i * oneRepeat);
                DataCopy(sqReg, sumSqAddr + sumOff);
                DataCopy(sReg, sumSAddr + sumOff);
                Sqrt(nReg, sqReg, maskReg);
                Maxs(nReg, nReg, eps_, maskReg);
                for (uint16_t d = 0; d < dLoop; d++) {
                    uint32_t off = static_cast<uint32_t>(d * rowStride + i * oneRepeat);
                    LoadAndCast(yReg, yAddr, maskReg, off);
                    LoadAndCast(dyReg, dyAddr, maskReg, off);
                    Mul(ysReg, yReg, sReg, maskReg);
                    Sub(subReg, dyReg, ysReg, maskReg);
                    Div(dxReg, subReg, nReg, maskReg);
                    StoreDx<T_X>(dxAddr, off, dxReg, maskReg);
                }
            }
        }

        inQueueY_.FreeTensor(yLocal);
        inQueueDy_.FreeTensor(dyLocal);
        outQueueDx_.EnQue(dxLocal);
        CopyOutTile(o, dStart, dTile, colStart, colTile);
    }

    // UB 行距:满块用 colFactorAlign_、尾块用 tailColAlign_,两者均由 host 算好下发(内核只做选择)。
    __aicore__ inline int64_t RowStride(int64_t colTile)
    {
        return (colTile == colFactor_) ? colFactorAlign_ : tailColAlign_;
    }

    // 载入组 o 的 [dStart, dStart+dTile) 行 × [colStart, colStart+colTile) 列子块（行间跨 inner_）。
    __aicore__ inline void CopyInTile(TQue<QuePosition::VECIN, DEPTH_TWO>& que, GlobalTensor<T_X>& gm, int64_t o,
                                      int64_t dStart, int64_t dTile, int64_t colStart, int64_t colTile)
    {
        // 沿 D 分块:本块 dTile 行 x colTile 列,行间在 GM 上隔 (inner - colTile) 个元素
        CopyIn2D(que, gm, o * D_ * inner_ + dStart * inner_ + colStart, dTile, colTile, inner_ - colTile);
    }

    __aicore__ inline void CopyOutTile(int64_t o, int64_t dStart, int64_t dTile, int64_t colStart, int64_t colTile)
    {
        CopyOut2D(outQueueDx_, o * D_ * inner_ + dStart * inner_ + colStart, dTile, colTile, inner_ - colTile);
    }

private:
    TBuf<TPosition::VECCALC> sqBuf_;
    TBuf<TPosition::VECCALC> prodBuf_;
    TBuf<TPosition::VECCALC> chunkSqBuf_;
    TBuf<TPosition::VECCALC> chunkSBuf_;
    TBuf<TPosition::VECCALC> sumSqBuf_;
    TBuf<TPosition::VECCALC> sumSBuf_;
    TBuf<TPosition::VECCALC> slotSqBuf_;
    TBuf<TPosition::VECCALC> slotSBuf_;

    int64_t outer_;
    int64_t D_;
    int64_t inner_;
    int64_t colFactor_;
    int64_t colFactorAlign_;
    int64_t tailColAlign_;
    int64_t colFactorAlignF32_;
    int64_t tailColAlignF32_;
    int64_t dFactor_;
    int64_t accElems_;
    uint16_t accVLs_;
    int64_t chunkSlots_;
    int64_t slotStride_;
};
} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_DX_STRIDED_SPLIT_H
