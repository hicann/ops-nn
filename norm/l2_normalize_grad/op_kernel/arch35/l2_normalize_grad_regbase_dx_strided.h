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
 * \file l2_normalize_grad_regbase_dx_strided.h
 * \brief L2NormalizeGrad DX strided kernel (TilingKey 7020) - general path when inner > 1.
 *
 * When the reduced axis dim is not the innermost axis (e.g. 4D NCHW with dim=1) the reduced-group
 * elements are strided by inner. 归约用平台 ReduceSum<Pattern::Reduce::RA>(沿首轴 D 按列规约),
 * for each outer group o and each inner column c, sum over d of x^2 and y*dy. The [D, inner] slice
 * of a group is contiguous in GM (D rows of inner). We load [D, colTile] tiles, accumulate the two
 * sums per inner column across d (fused multiply-add), clamp the denom, then write dx per d. No
 * cross-lane reduction (the accumulators are indexed by inner column, same layout as the data).
 * This is the correctness-first general path (data streamed twice over D), not the fast path.
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_DX_STRIDED_H
#define L2_NORMALIZE_GRAD_REGBASE_DX_STRIDED_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "l2_normalize_grad_regbase_base.h"

namespace L2NormalizeGrad {
using namespace AscendC;

template <typename T_X>
class RegbaseDxStrided : public RegbaseDxBase<T_X> {
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
    __aicore__ inline RegbaseDxStrided(TPipe* pipe, const L2NormalizeGradTilingData* tilingData)
        : Base(pipe, tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* dy, __gm__ uint8_t* dx)
    {
        outer_ = tiling_->outer;
        D_ = tiling_->dimLen;
        inner_ = tiling_->inner;
        colFactor_ = tiling_->colFactor; // inner columns processed per tile

        colFactorAlign_ = tiling_->colFactorAlign; // host 下发的对齐列宽,内核不再 AlignUp
        tailColTile_ = tiling_->tailColTile;
        tailColAlign_ = tiling_->tailColAlign;
        colFactorAlignF32_ = tiling_->colFactorAlignF32; // fp32 中间量 tile 的行距(按 fp32 block 对齐)
        tailColAlignF32_ = tiling_->tailColAlignF32;
        if (!InitCommon(x, y, dy, dx, D_ * inner_)) {
            return; // 本核无任务
        }

        // tile = D_ * AlignUp(colFactor, block) + V_LENGTH slack(末行整 VL 载入不越界);
        // 该尺寸由 host 一次算准下发(qBufBytes),内核不自行推算(见 tiling_data 注释)。
        InitQueues(inQueueX_, inQueueY_, inQueueDy_, outQueueDx_);
        // 平台 ReduceSum 的输入需为 fp32 LocalTensor,故另开两块 fp32 中间量 tile(x^2 / y*dy)
        // 与两块每列规约结果;尺寸同样由 host 算好下发。
        Ppipe_->InitBuffer(sqBuf_, tiling_->midBufBytes);
        Ppipe_->InitBuffer(prodBuf_, tiling_->midBufBytes);
        Ppipe_->InitBuffer(sumSqBuf_, tiling_->sumBufBytes);
        Ppipe_->InitBuffer(sumSBuf_, tiling_->sumBufBytes);
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
                SubProcess(o, colStart, colTile);
            }
        }
    }

    __aicore__ inline void SubProcess(int64_t o, int64_t colStart, int64_t colTile)
    {
        CopyInTile(inQueueX_, xGm_, o, colStart, colTile);
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        CopyInTile(inQueueY_, yGm_, o, colStart, colTile);
        LocalTensor<float> yLocal = inQueueY_.DeQue<float>();
        CopyInTile(inQueueDy_, dyGm_, o, colStart, colTile);
        LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();
        LocalTensor<float> dxLocal = outQueueDx_.AllocTensor<float>();

        constexpr uint32_t oneRepeat = V_LENGTH;
        uint16_t repeatCount = static_cast<uint16_t>(DivCeil(colTile, static_cast<int64_t>(oneRepeat)));
        const bool isTail = (colTile != colFactor_);
        // 两种行距均由 host 下发:载入 tile 按 T_X 的 block 对齐,fp32 中间量 tile 按 fp32 block 对齐
        // (后者是 ReduceSum 的 srcInnerPad 语义要求)。
        const int64_t rowStride = isTail ? tailColAlign_ : colFactorAlign_;
        const int64_t rowStrideF32 = isTail ? tailColAlignF32_ : colFactorAlignF32_;
        uint16_t dLoop = static_cast<uint16_t>(D_);

        LocalTensor<float> sqLocal = sqBuf_.Get<float>();
        LocalTensor<float> prodLocal = prodBuf_.Get<float>();
        LocalTensor<float> sumSqLocal = sumSqBuf_.Get<float>();
        LocalTensor<float> sumSLocal = sumSBuf_.Get<float>();
        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_X* yAddr = (__ubuf__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ T_X* dyAddr = (__ubuf__ T_X*)dyLocal.GetPhyAddr();
        __local_mem__ T_X* dxAddr = (__ubuf__ T_X*)dxLocal.GetPhyAddr();
        __local_mem__ float* sqAddr = (__ubuf__ float*)sqLocal.GetPhyAddr();
        __local_mem__ float* prodAddr = (__ubuf__ float*)prodLocal.GetPhyAddr();
        __local_mem__ float* sumSqAddr = (__ubuf__ float*)sumSqLocal.GetPhyAddr();
        __local_mem__ float* sumSAddr = (__ubuf__ float*)sumSLocal.GetPhyAddr();

        BuildMidTiles(xAddr, yAddr, dyAddr, sqAddr, prodAddr, colTile, repeatCount, dLoop, rowStride, rowStrideF32);

        // ── ② 沿首轴(D)按列规约:用平台接口,不自行手写累加/补偿 ──
        // Pattern::Reduce::RA = 首轴规约保留末轴,与 full_load/split_d 用的 AR 同族。
        // A 维传**对齐后的宽度**、srcInnerPad=false —— 与 full_load / confusion_softmax_grad 同款用法:
        // 缓冲已整块清零,pad 车道贡献 0;输出前 colTile 个即所需列和。
        uint32_t srcShape[2] = {static_cast<uint32_t>(D_), static_cast<uint32_t>(rowStrideF32)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumSqLocal, sqLocal, srcShape, false);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumSLocal, prodLocal, srcShape, false);

        SynthesizeDx(yAddr, dyAddr, dxAddr, sumSqAddr, sumSAddr, colTile, repeatCount, dLoop, rowStride);

        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
        inQueueDy_.FreeTensor(dyLocal);
        outQueueDx_.EnQue(dxLocal);
        CopyOutTile(o, colStart, colTile);
    }

    // ① 逐元素求 x^2 与 y*dy,以 fp32 落进中间量 tile(fp16 载入时顺带 cast)。
    // 不需要对 pad 车道清零:RA 是**按列独立**规约,pad 列的残值只会污染 pad 列的输出,
    // 而后续计算被 mask 限制在 colTile 个车道内,那些输出不被读取。
    // (full_load/split_d 用的是 AR——pad 落在被规约轴内,那里才必须先清零。)
    __aicore__ inline void BuildMidTiles(__local_mem__ T_X* xAddr, __local_mem__ T_X* yAddr, __local_mem__ T_X* dyAddr,
                                         __local_mem__ float* sqAddr, __local_mem__ float* prodAddr, int64_t colTile,
                                         uint16_t repeatCount, uint16_t dLoop, int64_t rowStride, int64_t rowStrideF32)
    {
        constexpr uint32_t oneRepeat = V_LENGTH;
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
    }

    // ③ dx = (dy - y*s) / max(sqrt(sq), eps),列和已由 ReduceSum<RA> 备好。
    __aicore__ inline void SynthesizeDx(__local_mem__ T_X* yAddr, __local_mem__ T_X* dyAddr, __local_mem__ T_X* dxAddr,
                                        __local_mem__ float* sumSqAddr, __local_mem__ float* sumSAddr, int64_t colTile,
                                        uint16_t repeatCount, uint16_t dLoop, int64_t rowStride)
    {
        constexpr uint32_t oneRepeat = V_LENGTH;
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
    }

    // Load a [D, colTile] sub-block of group o: D rows of colTile inner columns, strided by inner_.
    __aicore__ inline void CopyInTile(TQue<QuePosition::VECIN, DEPTH_TWO>& que, GlobalTensor<T_X>& gm, int64_t o,
                                      int64_t colStart, int64_t colTile)
    {
        // 整段 D 常驻:D 行 x colTile 列,行间在 GM 上隔 (inner - colTile) 个元素
        CopyIn2D(que, gm, o * D_ * inner_ + colStart, D_, colTile, inner_ - colTile);
    }

    __aicore__ inline void CopyOutTile(int64_t o, int64_t colStart, int64_t colTile)
    {
        CopyOut2D(outQueueDx_, o * D_ * inner_ + colStart, D_, colTile, inner_ - colTile);
    }

private:
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueX_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueY_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueDy_;
    TQue<QuePosition::VECOUT, DEPTH_TWO> outQueueDx_;

    TBuf<TPosition::VECCALC> sqBuf_;
    TBuf<TPosition::VECCALC> prodBuf_;
    TBuf<TPosition::VECCALC> sumSqBuf_;
    TBuf<TPosition::VECCALC> sumSBuf_;

    int64_t outer_;
    int64_t D_;
    int64_t inner_;
    int64_t colFactor_;
    int64_t colFactorAlign_;
    int64_t tailColTile_;
    int64_t tailColAlign_;
    int64_t colFactorAlignF32_;
    int64_t tailColAlignF32_;
};
} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_DX_STRIDED_H
