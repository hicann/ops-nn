/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_quant_regbase.h
 * \brief
 */
#ifndef ADD_RMS_NORM_QUANT_REBASE_H_
#define ADD_RMS_NORM_QUANT_REBASE_H_

#include "add_rms_norm_quant_regbase_common.h"

namespace AddRmsNormQuant {

template <typename T_X, typename T_Y, typename T_SCALES, typename T_ZEROPOINTS, uint64_t TILING_KEY>
class KernelAddRmsNormQuantRegbase {
#define DIV_MODE ((TILING_KEY / 100) == 1)
#define INPUT_KEY ((TILING_KEY % 100) / 10)
#define HAS_ZEROPOINTS1 ((INPUT_KEY >> 2) % 2 == 1)
#define HAS_SCALE2 ((INPUT_KEY >> 1) % 2 == 1)
#define HAS_ZEROPOINTS2 (INPUT_KEY % 2 == 1)
#define HAS_Y2 (HAS_SCALE2 || HAS_ZEROPOINTS2)
public:
    __aicore__ inline KernelAddRmsNormQuantRegbase(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroPoints1,
        GM_ADDR zeroPoints2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, const AddRmsNormQuantRegbaseTilingData* tilingData)
    {
        numM_ = tilingData->numM;
        numN_ = tilingData->numN;
        baseM_ = tilingData->baseM;
        baseN_ = tilingData->baseN;
        baseNDtypeAlign_ = tilingData->baseNDtypeAlign;
        baseNReduceAlign_ = tilingData->baseNReduceAlign;
        powerSplit_ = tilingData->powerSplit;
        mPerCore_ = tilingData->mPerCore;
        mLastCore_ = tilingData->mLastCore;
        epsilon_ = tilingData->epsilon;
        avgFactor_ = tilingData->avgFactor;

        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        CalBlockTail();
        InitBuffer(x1, x2, gamma, scales1, scales2, zeroPoints1, zeroPoints2, y1, y2, x);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore_ = blockIdx_ == (blockNum_ - 1) ? mLastCore_ : mPerCore_;
        mCnt_ = CeilDiv(mCore_, baseM_);
        tailM_ = mCore_ - (mCnt_ - 1) * baseM_;

        baseNB8Align_ = CeilAlign(baseN_, B8_BLOCK_NUM);
        uint64_t reduceBufLen = baseNReduceAlign_ / (2 * V_LENGTH);
        reduceBufAlign_ = CeilAlign(reduceBufLen, B32_BLOCK_NUM);
    }

    __aicore__ inline void InitBuffer(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroPoints1,
        GM_ADDR zeroPoints2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x)
    {
        uint64_t gmOffset = blockIdx_ * mPerCore_ * numN_;
        uint64_t gmLen = mCore_ * numN_;
        x1Gm_.SetGlobalBuffer((__gm__ T_X*)x1 + gmOffset, gmLen);
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
        gammaGm_.SetGlobalBuffer((__gm__ T_X*)gamma, numN_);
        scales1Gm_.SetGlobalBuffer((__gm__ T_SCALES*)scales1, numN_);
        y1Gm_.SetGlobalBuffer((__gm__ T_Y*)y1 + gmOffset, gmLen);
        y2Gm_.SetGlobalBuffer((__gm__ T_Y*)y2 + gmOffset, gmLen);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
        if constexpr (HAS_SCALE2) {
            scales2Gm_.SetGlobalBuffer((__gm__ T_SCALES*)scales2, numN_);
        }
        if constexpr (HAS_ZEROPOINTS1) {
            zeroPoints1Gm_.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints1, numN_);
        }
        if constexpr (HAS_ZEROPOINTS2) {
            zeroPoints2Gm_.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints2, numN_);
        }

        uint64_t ubFactorQuant = CeilAlign(numN_, BLOCK_SIZE / sizeof(T_SCALES));
        uint64_t ubFactorRstd = CeilAlign(baseM_, B32_BLOCK_NUM);
        pipe_->InitBuffer(inQueueX1_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueX2_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueX_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueGamma_, 1, baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueScales1_, 1, ubFactorQuant * sizeof(T_SCALES));
        pipe_->InitBuffer(outQueueY1_, 1, baseNB8Align_ * sizeof(T_Y));
        if constexpr (HAS_SCALE2) {
            pipe_->InitBuffer(inQueueScales2_, 1, ubFactorQuant * sizeof(T_SCALES));
        }
        if constexpr (HAS_ZEROPOINTS1) {
            pipe_->InitBuffer(inQueueZeroPoints1_, 1, ubFactorQuant * sizeof(T_ZEROPOINTS));
        }
        if constexpr (HAS_ZEROPOINTS2) {
            pipe_->InitBuffer(inQueueZeroPoints2_, 1, ubFactorQuant * sizeof(T_ZEROPOINTS));
        }
        if constexpr (HAS_Y2) {
            pipe_->InitBuffer(outQueueY2_, 1, baseNB8Align_ * sizeof(T_Y));
        }

        pipe_->InitBuffer(rstdBuf_, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, reduceBufAlign_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        CopyInQuant();
        LocalTensor<T_X> gammaLocal = inQueueGamma_.DeQue<T_X>();
        LocalTensor<T_SCALES> scales1Local = inQueueScales1_.DeQue<T_SCALES>();
        LocalTensor<T_SCALES> scales2Local;
        LocalTensor<T_ZEROPOINTS> zeroPoints1Local, zeroPoints2Local;
        if constexpr (HAS_SCALE2) {
            scales2Local = inQueueScales2_.DeQue<T_SCALES>();
        }
        if constexpr (HAS_ZEROPOINTS1) {
            zeroPoints1Local = inQueueZeroPoints1_.DeQue<T_ZEROPOINTS>();
        }
        if constexpr (HAS_ZEROPOINTS2) {
            zeroPoints2Local = inQueueZeroPoints2_.DeQue<T_ZEROPOINTS>();
        }
        for (uint64_t mOuterIdx = 0; mOuterIdx < mCnt_; mOuterIdx++) {
            uint64_t realM = mOuterIdx == (mCnt_ - 1) ? tailM_ : baseM_;

            for (uint64_t mInnerIdx = 0; mInnerIdx < realM; mInnerIdx++) {
                uint64_t gmOffset = (mOuterIdx * baseM_ + mInnerIdx) * numN_;

                CopyInX(inQueueX1_, x1Gm_, gmOffset, numN_, 0, baseNDtypeAlign_ - numN_);
                CopyInX(inQueueX2_, x2Gm_, gmOffset, numN_, 0, baseNDtypeAlign_ - numN_);
                Compute(
                    mInnerIdx, gmOffset, gammaLocal, scales1Local, zeroPoints1Local, scales2Local, zeroPoints2Local);
                CopyOutY(y1Gm_, outQueueY1_, gmOffset, numN_);
                if constexpr (HAS_Y2) {
                    CopyOutY(y2Gm_, outQueueY2_, gmOffset, numN_);
                }
            }
        }
        inQueueGamma_.FreeTensor(gammaLocal);
        inQueueScales1_.FreeTensor(scales1Local);
        if constexpr (HAS_SCALE2) {
            inQueueScales2_.FreeTensor(scales2Local);
        }
        if constexpr (HAS_ZEROPOINTS1) {
            inQueueZeroPoints1_.FreeTensor(zeroPoints1Local);
        }
        if constexpr (HAS_ZEROPOINTS2) {
            inQueueZeroPoints2_.FreeTensor(zeroPoints2Local);
        }
    }

private:
    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T_X> gammaLocal = inQueueGamma_.AllocTensor<T_X>();
        RmsNorm::DataCopyImpl<T_X>(gammaLocal, gammaGm_, 1, numN_);
        inQueueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInQuant()
    {
        LocalTensor<T_SCALES> scales1Local = inQueueScales1_.AllocTensor<T_SCALES>();
        RmsNorm::DataCopyImpl<T_SCALES>(scales1Local, scales1Gm_, 1, numN_);
        inQueueScales1_.EnQue(scales1Local);
        if constexpr (HAS_SCALE2) {
            LocalTensor<T_SCALES> scales2Local = inQueueScales2_.AllocTensor<T_SCALES>();
            RmsNorm::DataCopyImpl<T_SCALES>(scales2Local, scales2Gm_, 1, numN_);
            inQueueScales2_.EnQue(scales2Local);
        }
        if constexpr (HAS_ZEROPOINTS1) {
            LocalTensor<T_ZEROPOINTS> zeroPoints1Local = inQueueZeroPoints1_.AllocTensor<T_ZEROPOINTS>();
            RmsNorm::DataCopyImpl<T_ZEROPOINTS>(zeroPoints1Local, zeroPoints1Gm_, 1, numN_);
            inQueueZeroPoints1_.EnQue(zeroPoints1Local);
        }
        if constexpr (HAS_ZEROPOINTS2) {
            LocalTensor<T_ZEROPOINTS> zeroPoints2Local = inQueueZeroPoints2_.AllocTensor<T_ZEROPOINTS>();
            RmsNorm::DataCopyImpl<T_ZEROPOINTS>(zeroPoints2Local, zeroPoints2Gm_, 1, numN_);
            inQueueZeroPoints2_.EnQue(zeroPoints2Local);
        }
    }

    __aicore__ inline void Compute(
        uint64_t mInnerIdx, uint64_t gmOffset, LocalTensor<T_X>& gammaLocal, LocalTensor<T_SCALES>& scales1Local,
        LocalTensor<T_ZEROPOINTS>& zeroPoints1Local, LocalTensor<T_SCALES>& scales2Local,
        LocalTensor<T_ZEROPOINTS>& zeroPoints2Local)
    {
        LocalTensor<T_X> x1Local = inQueueX1_.DeQue<T_X>();
        LocalTensor<T_X> x2Local = inQueueX2_.DeQue<T_X>();
        LocalTensor<float> reduceLocal = reduceBuf_.Get<float>();
        LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
        uint64_t dupLen = baseNReduceAlign_ - baseNDtypeAlign_;
        if (dupLen > 0) {
            Duplicate(x1Local[baseNDtypeAlign_], (T_X)0.0, dupLen);
            Duplicate(x2Local[baseNDtypeAlign_], (T_X)0.0, dupLen);
        }
        Duplicate(reduceLocal, (float)0.0, reduceBufAlign_);
        PipeBarrier<PIPE_V>();

        LocalTensor<T_X> xOutLocal = outQueueX_.AllocTensor<T_X>();
        NormCommon::ReduceSumRstd<T_X, true, false, true>(
            rstdLocal, xOutLocal, reduceLocal, x1Local, x2Local, reduceLocal, mInnerIdx, baseNReduceAlign_, powerSplit_,
            avgFactor_, epsilon_);
        inQueueX1_.FreeTensor(x1Local);
        inQueueX2_.FreeTensor(x2Local);
        outQueueX_.EnQue<T_X>(xOutLocal);

        LocalTensor<T_Y> y1Local = outQueueY1_.AllocTensor<T_Y>();
        LocalTensor<T_Y> y2Local;
        if constexpr (HAS_Y2) {
            y2Local = outQueueY2_.AllocTensor<T_Y>();
        }

        ComputeY<T_X, T_Y, T_SCALES, T_ZEROPOINTS, true, HAS_ZEROPOINTS1, DIV_MODE>(
            y1Local, xOutLocal, rstdLocal, gammaLocal, scales1Local, zeroPoints1Local, mInnerIdx, baseNDtypeAlign_);
        if constexpr (HAS_Y2) {
            ComputeY<T_X, T_Y, T_SCALES, T_ZEROPOINTS, HAS_SCALE2, HAS_ZEROPOINTS2, DIV_MODE>(
                y2Local, xOutLocal, rstdLocal, gammaLocal, scales2Local, zeroPoints2Local, mInnerIdx, baseNDtypeAlign_);
        }
        CopyOutX(xGm_, outQueueX_, gmOffset, numN_);

        outQueueY1_.EnQue<T_Y>(y1Local);
        if constexpr (HAS_Y2) {
            outQueueY2_.EnQue<T_Y>(y2Local);
        }
    }

private:
    TPipe* pipe_ = nullptr;
    // GM Buffer
    GlobalTensor<T_X> x1Gm_;
    GlobalTensor<T_X> x2Gm_;
    GlobalTensor<T_X> gammaGm_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_SCALES> scales1Gm_, scales2Gm_;
    GlobalTensor<T_ZEROPOINTS> zeroPoints1Gm_, zeroPoints2Gm_;
    GlobalTensor<T_Y> y1Gm_;
    GlobalTensor<T_Y> y2Gm_;
    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1_, inQueueX2_, inQueueGamma_;
    TQue<QuePosition::VECIN, 1> inQueueScales1_, inQueueScales2_, inQueueZeroPoints1_, inQueueZeroPoints2_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_, outQueueY2_, outQueueX_;
    TBuf<TPosition::VECCALC> rstdBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;

    // Tiling data
    uint64_t numN_{0};
    uint64_t numM_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseNDtypeAlign_{0};
    uint64_t baseNReduceAlign_{0};
    uint64_t reduceBufAlign_{0};
    uint64_t powerSplit_{0};
    uint64_t mPerCore_{0};
    uint64_t mLastCore_{0};
    float epsilon_{0};
    float avgFactor_{0};
    // Platform
    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    // Cal params
    uint64_t mCore_;
    uint64_t mCnt_;
    uint64_t tailM_;
    uint64_t baseNB8Align_;
    // Other
    uint32_t vfLength_{0};
};
} // namespace AddRmsNormQuant
#endif // ADD_RMS_NORM_QUANT_REGBASE_H_
