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
 * elements are strided by inner. Reduction is turned into a per-column vector accumulation over D:
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
#include "l2_normalize_grad_regbase_common.h"

namespace L2NormalizeGrad {
using namespace AscendC;

template <typename T_X>
class RegbaseDxStrided {
public:
    __aicore__ inline RegbaseDxStrided(TPipe* pipe, const L2NormalizeGradTilingData* tilingData)
        : Ppipe_(pipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* dy, __gm__ uint8_t* dx)
    {
        usedCoreNum_ = tiling_->usedCoreNum;
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        outer_ = tiling_->outer;
        D_ = tiling_->dimLen;
        inner_ = tiling_->inner;
        blockFactor_ = tiling_->blockFactor;
        colFactor_ = tiling_->colFactor; // inner columns processed per tile
        eps_ = tiling_->eps;

        colFactorAlign_ = IsSameType<T_X, float>::value ? AlignUp(colFactor_, static_cast<int64_t>(FLOAT_NUM_BLOCK)) :
                                                          AlignUp(colFactor_, static_cast<int64_t>(HALF_NUM_BLOCK));
        int64_t groupNum = D_ * inner_;

        xGm_.SetGlobalBuffer((__gm__ T_X*)x + coreIdx * blockFactor_ * groupNum);
        yGm_.SetGlobalBuffer((__gm__ T_X*)y + coreIdx * blockFactor_ * groupNum);
        dyGm_.SetGlobalBuffer((__gm__ T_X*)dy + coreIdx * blockFactor_ * groupNum);
        dxGm_.SetGlobalBuffer((__gm__ T_X*)dx + coreIdx * blockFactor_ * groupNum);

        // +V_LENGTH slack so the last d-row's full-VL load never reads past the buffer (masked out anyway).
        int64_t tileElems = D_ * colFactorAlign_ + V_LENGTH;
        Ppipe_->InitBuffer(inQueueX_, DB_NUM, tileElems * sizeof(float));
        Ppipe_->InitBuffer(inQueueY_, DB_NUM, tileElems * sizeof(float));
        Ppipe_->InitBuffer(inQueueDy_, DB_NUM, tileElems * sizeof(float));
        Ppipe_->InitBuffer(outQueueDx_, DB_NUM, tileElems * sizeof(float));
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
        // UB row stride between consecutive d: DataCopyPad (dstStride=0) lays each d-row block-aligned,
        // so the stride is AlignUp(colTile, block) for THIS tile (colTile may be < colFactor_ on the last tile).
        int64_t rowStride = IsSameType<T_X, float>::value ? AlignUp(colTile, static_cast<int64_t>(FLOAT_NUM_BLOCK)) :
                                                            AlignUp(colTile, static_cast<int64_t>(HALF_NUM_BLOCK));
        uint16_t dLoop = static_cast<uint16_t>(D_);
        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_X* yAddr = (__ubuf__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ T_X* dyAddr = (__ubuf__ T_X*)dyLocal.GetPhyAddr();
        __local_mem__ T_X* dxAddr = (__ubuf__ T_X*)dxLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            RegTensor<float> xReg, yReg, dyReg, accSq, accS, nReg, ysReg, subReg, dxReg;
            uint32_t sregOuter = static_cast<uint32_t>(colTile);
            for (uint16_t i = 0; i < repeatCount; i++) {
                MaskReg maskReg = UpdateMask<float>(sregOuter);
                Duplicate(accSq, 0.0f, maskReg);
                Duplicate(accS, 0.0f, maskReg);
                // Pass 1: accumulate over d for this inner-column VL chunk.
                for (uint16_t d = 0; d < dLoop; d++) {
                    uint32_t off = static_cast<uint32_t>(d * rowStride + i * oneRepeat);
                    LoadAndCast(xReg, xAddr, maskReg, off);
                    Mula(accSq, xReg, xReg, maskReg);
                    LoadAndCast(yReg, yAddr, maskReg, off);
                    LoadAndCast(dyReg, dyAddr, maskReg, off);
                    Mula(accS, yReg, dyReg, maskReg);
                }
                Sqrt(nReg, accSq, maskReg);
                Maxs(nReg, nReg, eps_, maskReg);
                // Pass 2: dx = (dy - y*s) / n over d (accS / nReg indexed by inner column).
                for (uint16_t d = 0; d < dLoop; d++) {
                    uint32_t off = static_cast<uint32_t>(d * rowStride + i * oneRepeat);
                    LoadAndCast(yReg, yAddr, maskReg, off);
                    LoadAndCast(dyReg, dyAddr, maskReg, off);
                    Mul(ysReg, yReg, accS, maskReg);
                    Sub(subReg, dyReg, ysReg, maskReg);
                    Div(dxReg, subReg, nReg, maskReg);
                    StoreDx<T_X>(dxAddr, off, dxReg, maskReg);
                }
            }
        }

        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
        inQueueDy_.FreeTensor(dyLocal);
        outQueueDx_.EnQue(dxLocal);
        CopyOutTile(o, colStart, colTile);
    }

    // Load a [D, colTile] sub-block of group o: D rows of colTile inner columns, strided by inner_.
    __aicore__ inline void CopyInTile(TQue<QuePosition::VECIN, DEPTH_TWO>& que, GlobalTensor<T_X>& gm, int64_t o,
                                      int64_t colStart, int64_t colTile)
    {
        LocalTensor<T_X> local = que.AllocTensor<T_X>();
        int64_t base = o * D_ * inner_ + colStart;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(D_),                               // blockCount (D rows)
            static_cast<uint32_t>(colTile * sizeof(T_X)),            // blockLen (bytes)
            static_cast<uint32_t>((inner_ - colTile) * sizeof(T_X)), // srcStride (gap between d rows)
            0,                                                       // dstStride
            0                                                        // rsv
        };
        DataCopyPad(local, gm[base], copyParams, {true, 0, 0, 0});
        que.EnQue(local);
    }

    __aicore__ inline void CopyOutTile(int64_t o, int64_t colStart, int64_t colTile)
    {
        LocalTensor<T_X> dxLocal = outQueueDx_.DeQue<T_X>();
        int64_t base = o * D_ * inner_ + colStart;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(D_),                               // blockCount
            static_cast<uint32_t>(colTile * sizeof(T_X)),            // blockLen (bytes)
            0,                                                       // srcStride
            static_cast<uint32_t>((inner_ - colTile) * sizeof(T_X)), // dstStride (gap between d rows)
            0                                                        // rsv
        };
        DataCopyPad(dxGm_[base], dxLocal, copyParams);
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

    uint32_t usedCoreNum_;
    int64_t outer_;
    int64_t D_;
    int64_t inner_;
    int64_t blockFactor_;
    int64_t colFactor_;
    int64_t colFactorAlign_;
    float eps_;
};
} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_DX_STRIDED_H
