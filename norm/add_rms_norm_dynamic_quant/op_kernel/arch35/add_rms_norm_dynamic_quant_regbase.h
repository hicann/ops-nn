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
 * \file add_rms_norm_dynamic_quant_regbase.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_H_

#include "add_rms_norm_dynamic_quant_regbase_common.h"

namespace AddRmsNormDynamicQuant {

template <typename T_X, typename T_Y, uint64_t TILING_KEY>
class KernelAddRmsNormDynamicQuantRegbase {
#define INPUT_KEY ((TILING_KEY % 100) / 10)
#define HAS_SMOOTH_SCALE1 ((INPUT_KEY >> 1) % 2 == 1)
#define HAS_SMOOTH_SCALE2 (INPUT_KEY % 2 == 1)
#define HAS_Y2_SCALE2 HAS_SMOOTH_SCALE2
#define T_SMOOTH_SCALE T_X
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantRegbase(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1, GM_ADDR smooathScale2, GM_ADDR y1, GM_ADDR y2,
        GM_ADDR x, GM_ADDR scale1, GM_ADDR scale2, const AddRmsNormDynamicQuantRegbaseTilingData* tilingData)
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
        InitBuffer(x1, x2, gamma, smooathScale1, smooathScale2, y1, y2, x, scale1, scale2);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore_ = blockIdx_ == (blockNum_ - 1) ? mLastCore_ : mPerCore_;
        mOuterCnt_ = CeilDiv(mCore_, baseM_);
        tailMOuter_ = mCore_ - (mOuterCnt_ - 1) * baseM_;

        baseNB8Align_ = CeilAlign(baseN_, B8_BLOCK_NUM);
        uint64_t reduceSumBufLen = baseNReduceAlign_ / (2 * V_LENGTH);
        reduceSumBufAlign_ = CeilAlign(reduceSumBufLen, B32_BLOCK_NUM);
        baseNB32Align_ = CeilAlign(baseN_, B32_BLOCK_NUM);
    }

    __aicore__ inline void InitBuffer(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1, GM_ADDR smooathScale2, GM_ADDR y1, GM_ADDR y2,
        GM_ADDR x, GM_ADDR scale1, GM_ADDR scale2)
    {
        uint64_t gmOffset = blockIdx_ * mPerCore_ * numN_;
        uint64_t gmLen = mCore_ * numN_;
        uint64_t scalesGmOffset = blockIdx_ * mPerCore_;
        x1Gm_.SetGlobalBuffer((__gm__ T_X*)x1 + gmOffset, gmLen);
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
        gammaGm_.SetGlobalBuffer((__gm__ T_X*)gamma, numN_);
        y1Gm_.SetGlobalBuffer((__gm__ T_Y*)y1 + gmOffset, gmLen);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
        scale1Gm_.SetGlobalBuffer((__gm__ float*)scale1 + scalesGmOffset, mCore_);
        if constexpr (HAS_SMOOTH_SCALE1) {
            smoothScale1Gm_.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smooathScale1, numN_);
        }
        if constexpr (HAS_SMOOTH_SCALE2) {
            smoothScale2Gm_.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smooathScale2, numN_);
        }
        if constexpr (HAS_Y2_SCALE2) {
            y2Gm_.SetGlobalBuffer((__gm__ T_Y*)y2 + gmOffset, gmLen);
            scale2Gm_.SetGlobalBuffer((__gm__ float*)scale2 + scalesGmOffset, mCore_);
        }

        InitUBBuffer();
    }

    __aicore__ inline void InitUBBuffer()
    {
        uint64_t ubFactorQuant = CeilAlign(numN_, BLOCK_SIZE / sizeof(T_SMOOTH_SCALE));
        uint64_t ubFactorRstd = CeilAlign(baseM_, B32_BLOCK_NUM);
        pipe_->InitBuffer(inQueueX1_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueX2_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueX_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueGamma_, 1, baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueY1_, 1, baseNB8Align_ * sizeof(T_Y));
        pipe_->InitBuffer(outQueueScale1_, 1, ubFactorRstd * sizeof(float));
        if constexpr (HAS_SMOOTH_SCALE1) {
            pipe_->InitBuffer(inQueueSmoothScale1_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if constexpr (HAS_SMOOTH_SCALE2) {
            pipe_->InitBuffer(inQueueSmoothScale2_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if constexpr (HAS_Y2_SCALE2) {
            pipe_->InitBuffer(outQueueY2_, 1, baseNB8Align_ * sizeof(T_Y));
            pipe_->InitBuffer(outQueueScale2_, 1, ubFactorRstd * sizeof(float));
        }

        pipe_->InitBuffer(xOutTmpBuf_, baseNReduceAlign_ * sizeof(float));
        pipe_->InitBuffer(y1TmpBuf_, baseNB32Align_ * sizeof(float));
        if constexpr (HAS_Y2_SCALE2) {
            pipe_->InitBuffer(y2TmpBuf_, baseNB32Align_ * sizeof(float));
        }
        pipe_->InitBuffer(rstdBuf_, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(reduceSumBuf_, reduceSumBufAlign_ * sizeof(float));
    }

    __aicore__ inline void SubProcess(
        LocalTensor<float>& scale1Local, LocalTensor<float>& scale2Local, LocalTensor<T_X>& gammaLocal,
        LocalTensor<T_SMOOTH_SCALE>& smoothScale1Local, LocalTensor<T_SMOOTH_SCALE>& smoothScale2Local,
        uint64_t& mInnerCnt, uint64_t& mOuterOffset)
    {
        for (uint64_t mInnerIdx = 0; mInnerIdx < mInnerCnt; mInnerIdx++) {
            uint64_t gmOffsetXY = (mOuterOffset + mInnerIdx) * numN_;

            CopyInX(inQueueX1_, x1Gm_, gmOffsetXY, numN_, 0, baseNDtypeAlign_ - numN_);
            CopyInX(inQueueX2_, x2Gm_, gmOffsetXY, numN_, 0, baseNDtypeAlign_ - numN_);
            Compute(scale1Local, scale2Local, gammaLocal, smoothScale1Local, smoothScale2Local, mInnerIdx, gmOffsetXY);
            CopyOutY(y1Gm_, outQueueY1_, gmOffsetXY, numN_);
            if constexpr (HAS_Y2_SCALE2) {
                CopyOutY(y2Gm_, outQueueY2_, gmOffsetXY, numN_);
            }
        }
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        CopyInDynamicQuant();
        LocalTensor<T_X> gammaLocal = inQueueGamma_.DeQue<T_X>();
        LocalTensor<T_SMOOTH_SCALE> smoothScale1Local;
        LocalTensor<T_SMOOTH_SCALE> smoothScale2Local;
        if constexpr (HAS_SMOOTH_SCALE1) {
            smoothScale1Local = inQueueSmoothScale1_.DeQue<T_SMOOTH_SCALE>();
        }
        if constexpr (HAS_SMOOTH_SCALE2) {
            smoothScale2Local = inQueueSmoothScale2_.DeQue<T_SMOOTH_SCALE>();
        }

        for (uint64_t mOuterIdx = 0; mOuterIdx < mOuterCnt_; mOuterIdx++) {
            uint64_t realM = mOuterIdx == (mOuterCnt_ - 1) ? tailMOuter_ : baseM_;
            uint64_t mOuterOffset = mOuterIdx * baseM_;

            LocalTensor<float> scale1Local = outQueueScale1_.AllocTensor<float>();
            LocalTensor<float> scale2Local;
            if constexpr (HAS_Y2_SCALE2) {
                scale2Local = outQueueScale2_.AllocTensor<float>();
            }
            SubProcess(scale1Local, scale2Local, gammaLocal, smoothScale1Local, smoothScale2Local, realM, mOuterOffset);
            outQueueScale1_.EnQue<float>(scale1Local);
            if constexpr (HAS_Y2_SCALE2) {
                outQueueScale2_.EnQue<float>(scale2Local);
            }
            CopyOutScale(scale1Gm_, outQueueScale1_, mOuterOffset, realM);
            if constexpr (HAS_Y2_SCALE2) {
                CopyOutScale(scale2Gm_, outQueueScale2_, mOuterOffset, realM);
            }
        }
        inQueueGamma_.FreeTensor(gammaLocal);
        if constexpr (HAS_SMOOTH_SCALE1) {
            inQueueSmoothScale1_.FreeTensor(smoothScale1Local);
        }
        if constexpr (HAS_SMOOTH_SCALE2) {
            inQueueSmoothScale2_.FreeTensor(smoothScale2Local);
        }
    }

private:
    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T_X> gammaLocal = inQueueGamma_.AllocTensor<T_X>();
        RmsNorm::DataCopyImpl<T_X>(gammaLocal, gammaGm_, 1, numN_);
        inQueueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInDynamicQuant()
    {
        if constexpr (HAS_SMOOTH_SCALE1) {
            LocalTensor<T_SMOOTH_SCALE> smoothScale1Local = inQueueSmoothScale1_.AllocTensor<T_SMOOTH_SCALE>();
            RmsNorm::DataCopyImpl<T_SMOOTH_SCALE>(smoothScale1Local, smoothScale1Gm_, 1, numN_);
            inQueueSmoothScale1_.EnQue(smoothScale1Local);
        }
        if constexpr (HAS_SMOOTH_SCALE2) {
            LocalTensor<T_SMOOTH_SCALE> smoothScale2Local = inQueueSmoothScale2_.AllocTensor<T_SMOOTH_SCALE>();
            RmsNorm::DataCopyImpl<T_SMOOTH_SCALE>(smoothScale2Local, smoothScale2Gm_, 1, numN_);
            inQueueSmoothScale2_.EnQue(smoothScale2Local);
        }
    }

    __aicore__ inline void Compute(
        LocalTensor<float>& scale1Local, LocalTensor<float>& scale2Local, LocalTensor<T_X>& gammaLocal,
        LocalTensor<T_SMOOTH_SCALE>& smoothScale1Local, LocalTensor<T_SMOOTH_SCALE>& smoothScale2Local,
        uint64_t mInnerIdx, uint64_t gmOffset)
    {
        LocalTensor<T_X> x1Local = inQueueX1_.DeQue<T_X>();
        LocalTensor<T_X> x2Local = inQueueX2_.DeQue<T_X>();
        LocalTensor<float> reduceLocal = reduceSumBuf_.Get<float>();
        LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
        LocalTensor<float> xOutTmpLocal = xOutTmpBuf_.Get<float>();
        LocalTensor<float> y1TmpLocal = y1TmpBuf_.Get<float>();
        LocalTensor<float> y2TmpLocal;
        if constexpr (HAS_Y2_SCALE2) {
            y2TmpLocal = y2TmpBuf_.Get<float>();
        }
        uint64_t dupLen = baseNReduceAlign_ - baseNDtypeAlign_;
        if (dupLen > 0) {
            Duplicate(x1Local[baseNDtypeAlign_], (T_X)0.0, dupLen);
            Duplicate(x2Local[baseNDtypeAlign_], (T_X)0.0, dupLen);
        }
        Duplicate(reduceLocal, (float)0.0, reduceSumBufAlign_);
        PipeBarrier<PIPE_V>();

        // 1. Calc xOut
        LocalTensor<T_X> xOutLocal = outQueueX_.AllocTensor<T_X>();
        NormCommon::ReduceSumRstd<T_X, true, true, true>(
            rstdLocal, xOutLocal, xOutTmpLocal, x1Local, x2Local, reduceLocal, mInnerIdx, baseNReduceAlign_,
            powerSplit_, avgFactor_, epsilon_);
        inQueueX1_.FreeTensor(x1Local);
        inQueueX2_.FreeTensor(x2Local);
        outQueueX_.EnQue<T_X>(xOutLocal);
        CopyOutX(xGm_, outQueueX_, gmOffset, numN_);

        // 2. Calc Scale and Y
        LocalTensor<T_Y> y1Local = outQueueY1_.AllocTensor<T_Y>();
        LocalTensor<T_Y> y2Local;
        if constexpr (HAS_Y2_SCALE2) {
            y2Local = outQueueY2_.AllocTensor<T_Y>();
        }

        ComputeYScale<float, T_X, T_SMOOTH_SCALE, HAS_SMOOTH_SCALE1, T_Y>(
            y1Local, scale1Local, xOutTmpLocal, rstdLocal, gammaLocal, smoothScale1Local, y1TmpLocal, mInnerIdx,
            baseNDtypeAlign_);
        if constexpr (HAS_SMOOTH_SCALE2) {
            ComputeYScale<float, T_X, T_SMOOTH_SCALE, HAS_SMOOTH_SCALE2, T_Y>(
                y2Local, scale2Local, xOutTmpLocal, rstdLocal, gammaLocal, smoothScale2Local, y2TmpLocal, mInnerIdx,
                baseNDtypeAlign_);
        }

        outQueueY1_.EnQue<T_Y>(y1Local);
        if constexpr (HAS_Y2_SCALE2) {
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
    GlobalTensor<T_SMOOTH_SCALE> smoothScale1Gm_;
    GlobalTensor<T_SMOOTH_SCALE> smoothScale2Gm_;
    GlobalTensor<T_Y> y1Gm_;
    GlobalTensor<T_Y> y2Gm_;
    GlobalTensor<float> scale1Gm_;
    GlobalTensor<float> scale2Gm_;
    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1_;
    TQue<QuePosition::VECIN, 1> inQueueX2_;
    TQue<QuePosition::VECIN, 1> inQueueGamma_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale1_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale2_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_;
    TQue<QuePosition::VECOUT, 1> outQueueY2_;
    TQue<QuePosition::VECOUT, 1> outQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueScale1_;
    TQue<QuePosition::VECOUT, 1> outQueueScale2_;
    TBuf<TPosition::VECCALC> rstdBuf_;
    TBuf<TPosition::VECCALC> y1TmpBuf_;
    TBuf<TPosition::VECCALC> y2TmpBuf_;
    TBuf<TPosition::VECCALC> xOutTmpBuf_;
    TBuf<TPosition::VECCALC> reduceSumBuf_;

    // Tiling data
    uint64_t numN_{0};
    uint64_t numM_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseNDtypeAlign_{0};
    uint64_t baseNReduceAlign_{0};
    uint64_t baseNB32Align_{0};
    uint64_t reduceSumBufAlign_{0};
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
    uint64_t mOuterCnt_;
    uint64_t tailMOuter_;
    uint64_t baseNB8Align_;
    // Other
    uint32_t vfLength_{0};
};
} // namespace AddRmsNormDynamicQuant
#endif // _ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_H_
