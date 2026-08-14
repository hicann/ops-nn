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

template <typename T_X, typename T_Y, bool Y3_MODE, bool Y4_MODE>
class KernelAddRmsNormDynamicQuantRegbase {
    using T_SMOOTH_SCALE = T_X;
    using yCopyDtype = YCopyDtype<T_Y>;

public:
    __aicore__ inline KernelAddRmsNormDynamicQuantRegbase(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1, GM_ADDR smooathScale2,
                                GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3, GM_ADDR y4, GM_ADDR x, GM_ADDR scale1,
                                GM_ADDR scale2, const AddRmsNormDynamicQuantRegbaseTilingData* tilingData)
    {
        InitTiling(numM_, numN_, baseM_, baseN_, baseNDtypeAlign_, baseNReduceAlign_, powerSplit_, mPerCore_,
                   mLastCore_, epsilon_, avgFactor_, tilingData);

        hasSmoothScale1_ = tilingData->hasSmoothScale1;
        hasSmoothScale2_ = tilingData->hasSmoothScale2;
        hasBeta_ = tilingData->hasBeta;
        outQuant1Flag_ = tilingData->outQuant1Flag;
        outQuant2Flag_ = tilingData->outQuant2Flag;

        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();
        oriOverflowMode_ = GetOverflowMode<T_Y>();

        CalBlockTail();
        InitBuffer(x1, x2, gamma, smooathScale1, smooathScale2, beta, y1, y2, y3, y4, x, scale1, scale2);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore_ = blockIdx_ == (blockNum_ - 1) ? mLastCore_ : mPerCore_;
        mOuterCnt_ = CeilDiv(mCore_, baseM_);
        tailMOuter_ = mCore_ - (mOuterCnt_ - 1) * baseM_;

        baseNB8Align_ = CeilAlign(baseN_, B8_BLOCK_NUM);
        uint64_t reduceSumBufLen = baseNReduceAlign_ / (REDUCE_VREG_PER_REPEAT * V_LENGTH);
        reduceSumBufAlign_ = CeilAlign(reduceSumBufLen, B32_BLOCK_NUM);
        baseNB32Align_ = CeilAlign(baseN_, B32_BLOCK_NUM);
    }

    __aicore__ inline void InitBuffer(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1,
                                      GM_ADDR smooathScale2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3,
                                      GM_ADDR y4, GM_ADDR x, GM_ADDR scale1, GM_ADDR scale2)
    {
        uint64_t gmOffset = blockIdx_ * mPerCore_ * numN_;
        uint64_t gmLen = mCore_ * numN_;
        uint64_t scalesGmOffset = blockIdx_ * mPerCore_;
        x1Gm_.SetGlobalBuffer((__gm__ T_X*)x1 + gmOffset, gmLen);
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
        gammaGm_.SetGlobalBuffer((__gm__ T_X*)gamma, numN_);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
        uint64_t yGmOffset = gmOffset;
        uint64_t yGmLen = gmLen;
        if constexpr (IsSameType<T_Y, int4b_t>::value) {
            yGmOffset = yGmOffset >> 1;
            yGmLen = yGmLen >> 1;
        }
        if (outQuant1Flag_) {
            y1Gm_.SetGlobalBuffer((__gm__ yCopyDtype*)y1 + yGmOffset, yGmLen);
            scale1Gm_.SetGlobalBuffer((__gm__ float*)scale1 + scalesGmOffset, mCore_);
        }
        if (hasSmoothScale1_) {
            smoothScale1Gm_.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smooathScale1, numN_);
        }
        if (hasSmoothScale2_) {
            smoothScale2Gm_.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smooathScale2, numN_);
        }
        if (outQuant2Flag_) {
            y2Gm_.SetGlobalBuffer((__gm__ yCopyDtype*)y2 + yGmOffset, yGmLen);
            scale2Gm_.SetGlobalBuffer((__gm__ float*)scale2 + scalesGmOffset, mCore_);
        }
        if (hasBeta_) {
            betaGm_.SetGlobalBuffer((__gm__ T_X*)beta, numN_);
        }
        if constexpr (Y3_MODE) {
            y3Gm_.SetGlobalBuffer((__gm__ float*)y3 + gmOffset, gmLen);
        }
        if constexpr (Y4_MODE) {
            y4Gm_.SetGlobalBuffer((__gm__ T_X*)y4 + gmOffset, gmLen);
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
        if (outQuant1Flag_) {
            pipe_->InitBuffer(outQueueY1_, 1, baseNB8Align_ * sizeof(yCopyDtype));
            pipe_->InitBuffer(outQueueScale1_, 1, ubFactorRstd * sizeof(float));
        }
        if (hasSmoothScale1_) {
            pipe_->InitBuffer(inQueueSmoothScale1_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if (hasSmoothScale2_) {
            pipe_->InitBuffer(inQueueSmoothScale2_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if (outQuant2Flag_) {
            pipe_->InitBuffer(outQueueY2_, 1, baseNB8Align_ * sizeof(yCopyDtype));
            pipe_->InitBuffer(outQueueScale2_, 1, ubFactorRstd * sizeof(float));
        }
        if (hasBeta_) {
            pipe_->InitBuffer(inQueueBeta_, 1, baseNDtypeAlign_ * sizeof(T_X));
        }
        pipe_->InitBuffer(xOutTmpBuf_, baseNReduceAlign_ * sizeof(float));
        if (outQuant1Flag_) {
            pipe_->InitBuffer(y1TmpBuf_, baseNB32Align_ * sizeof(float));
        }
        if (outQuant2Flag_) {
            pipe_->InitBuffer(y2TmpBuf_, baseNB32Align_ * sizeof(float));
        }
        pipe_->InitBuffer(rstdBuf_, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(reduceSumBuf_, reduceSumBufAlign_ * sizeof(float));
        if constexpr (Y3_MODE) {
            pipe_->InitBuffer(outQueueY3_, 1, baseNB32Align_ * sizeof(float));
        }
        if constexpr (Y4_MODE) {
            pipe_->InitBuffer(outQueueY4_, 1, baseNDtypeAlign_ * sizeof(T_X));
        }
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        CopyInDynamicQuant();
        LocalTensor<T_X> gammaLocal = inQueueGamma_.DeQue<T_X>();
        LocalTensor<T_SMOOTH_SCALE> smoothScale1Local;
        LocalTensor<T_SMOOTH_SCALE> smoothScale2Local;
        LocalTensor<T_X> betaLocal;
        if (hasSmoothScale1_) {
            smoothScale1Local = inQueueSmoothScale1_.DeQue<T_SMOOTH_SCALE>();
        }
        if (hasSmoothScale2_) {
            smoothScale2Local = inQueueSmoothScale2_.DeQue<T_SMOOTH_SCALE>();
        }
        if (hasBeta_) {
            CopyInBeta();
            betaLocal = inQueueBeta_.DeQue<T_X>();
        }

        for (uint64_t mOuterIdx = 0; mOuterIdx < mOuterCnt_; mOuterIdx++) {
            uint64_t realM = mOuterIdx == (mOuterCnt_ - 1) ? tailMOuter_ : baseM_;
            uint64_t mOuterOffset = mOuterIdx * baseM_;

            LocalTensor<float> scale1Local;
            LocalTensor<float> scale2Local;
            if (outQuant1Flag_) {
                scale1Local = outQueueScale1_.AllocTensor<float>();
            }
            if (outQuant2Flag_) {
                scale2Local = outQueueScale2_.AllocTensor<float>();
            }

            for (uint64_t mInnerIdx = 0; mInnerIdx < realM; mInnerIdx++) {
                uint64_t gmOffsetXY = (mOuterOffset + mInnerIdx) * numN_;

                CopyInX(inQueueX1_, x1Gm_, gmOffsetXY, numN_, 0, baseNDtypeAlign_ - numN_);
                CopyInX(inQueueX2_, x2Gm_, gmOffsetXY, numN_, 0, baseNDtypeAlign_ - numN_);
                Compute(scale1Local, scale2Local, gammaLocal, smoothScale1Local, smoothScale2Local, betaLocal,
                        mInnerIdx, gmOffsetXY);
                uint64_t yGmOffset = gmOffsetXY;
                uint32_t yCopyLen = static_cast<uint32_t>(numN_);
                if constexpr (IsSameType<T_Y, int4b_t>::value) {
                    yGmOffset = yGmOffset >> 1;
                    yCopyLen = yCopyLen >> 1;
                }
                if (outQuant1Flag_) {
                    CopyOutY(y1Gm_, outQueueY1_, yGmOffset, yCopyLen);
                }
                if (outQuant2Flag_) {
                    CopyOutY(y2Gm_, outQueueY2_, yGmOffset, yCopyLen);
                }
                if constexpr (Y3_MODE) {
                    CopyOutY<float>(y3Gm_, outQueueY3_, gmOffsetXY, numN_);
                }
                if constexpr (Y4_MODE) {
                    CopyOutY<T_X>(y4Gm_, outQueueY4_, gmOffsetXY, numN_);
                }
            }

            if (outQuant1Flag_) {
                outQueueScale1_.EnQue<float>(scale1Local);
                CopyOutScale(scale1Gm_, outQueueScale1_, mOuterOffset, realM);
            }
            if (outQuant2Flag_) {
                outQueueScale2_.EnQue<float>(scale2Local);
                CopyOutScale(scale2Gm_, outQueueScale2_, mOuterOffset, realM);
            }
        }
        inQueueGamma_.FreeTensor(gammaLocal);
        if (hasSmoothScale1_) {
            inQueueSmoothScale1_.FreeTensor(smoothScale1Local);
        }
        if (hasSmoothScale2_) {
            inQueueSmoothScale2_.FreeTensor(smoothScale2Local);
        }
        if (hasBeta_) {
            inQueueBeta_.FreeTensor(betaLocal);
        }
    }

    template <typename T_GAMMA, typename T_SMOOTH>
    __aicore__ inline void DispatchYScale(LocalTensor<yCopyDtype>& yLocal, LocalTensor<float>& scaleLocal,
                                          LocalTensor<float>& xLocal, LocalTensor<float>& rstdLocal,
                                          LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<T_GAMMA>& betaLocal,
                                          LocalTensor<T_SMOOTH>& smoothScaleLocal, LocalTensor<float>& yTmpLocal,
                                          LocalTensor<float>& y3Local, LocalTensor<T_X>& y4Local,
                                          uint32_t rstdScaleOffset, uint32_t calCount, bool hasSmoothScale,
                                          bool hasBeta)
    {
        if (hasSmoothScale) {
            if (hasBeta) {
                ComputeYScale<float, T_GAMMA, T_SMOOTH, true, true, T_Y, Y3_MODE, Y4_MODE>(
                    yLocal, scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, y3Local,
                    y4Local, rstdScaleOffset, calCount);
            } else {
                ComputeYScale<float, T_GAMMA, T_SMOOTH, true, false, T_Y, Y3_MODE, Y4_MODE>(
                    yLocal, scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, y3Local,
                    y4Local, rstdScaleOffset, calCount);
            }
        } else {
            if (hasBeta) {
                ComputeYScale<float, T_GAMMA, T_SMOOTH, false, true, T_Y, Y3_MODE, Y4_MODE>(
                    yLocal, scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, y3Local,
                    y4Local, rstdScaleOffset, calCount);
            } else {
                ComputeYScale<float, T_GAMMA, T_SMOOTH, false, false, T_Y, Y3_MODE, Y4_MODE>(
                    yLocal, scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, y3Local,
                    y4Local, rstdScaleOffset, calCount);
            }
        }
    }

private:
    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T_X> gammaLocal = inQueueGamma_.AllocTensor<T_X>();
        RmsNorm::DataCopyImpl<T_X>(gammaLocal, gammaGm_, 1, numN_);
        inQueueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInBeta()
    {
        LocalTensor<T_X> betaLocal = inQueueBeta_.AllocTensor<T_X>();
        RmsNorm::DataCopyImpl<T_X>(betaLocal, betaGm_, 1, numN_);
        inQueueBeta_.EnQue(betaLocal);
    }

    __aicore__ inline void CopyInDynamicQuant()
    {
        if (hasSmoothScale1_) {
            LocalTensor<T_SMOOTH_SCALE> smoothScale1Local = inQueueSmoothScale1_.AllocTensor<T_SMOOTH_SCALE>();
            RmsNorm::DataCopyImpl<T_SMOOTH_SCALE>(smoothScale1Local, smoothScale1Gm_, 1, numN_);
            inQueueSmoothScale1_.EnQue(smoothScale1Local);
        }
        if (hasSmoothScale2_) {
            LocalTensor<T_SMOOTH_SCALE> smoothScale2Local = inQueueSmoothScale2_.AllocTensor<T_SMOOTH_SCALE>();
            RmsNorm::DataCopyImpl<T_SMOOTH_SCALE>(smoothScale2Local, smoothScale2Gm_, 1, numN_);
            inQueueSmoothScale2_.EnQue(smoothScale2Local);
        }
    }

    __aicore__ inline void Compute(LocalTensor<float>& scale1Local, LocalTensor<float>& scale2Local,
                                   LocalTensor<T_X>& gammaLocal, LocalTensor<T_SMOOTH_SCALE>& smoothScale1Local,
                                   LocalTensor<T_SMOOTH_SCALE>& smoothScale2Local, LocalTensor<T_X>& betaLocal,
                                   uint64_t mInnerIdx, uint64_t gmOffset)
    {
        LocalTensor<T_X> x1Local = inQueueX1_.DeQue<T_X>();
        LocalTensor<T_X> x2Local = inQueueX2_.DeQue<T_X>();
        LocalTensor<float> reduceLocal = reduceSumBuf_.Get<float>();
        LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
        LocalTensor<float> xOutTmpLocal = xOutTmpBuf_.Get<float>();
        LocalTensor<float> y1TmpLocal;
        if (outQuant1Flag_) {
            y1TmpLocal = y1TmpBuf_.Get<float>();
        }
        LocalTensor<float> y2TmpLocal;
        if (outQuant2Flag_) {
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
        NormCommon::ReduceSumRstd<T_X, true, true, true>(rstdLocal, xOutLocal, xOutTmpLocal, x1Local, x2Local,
                                                         reduceLocal, mInnerIdx, baseNReduceAlign_, powerSplit_,
                                                         avgFactor_, epsilon_);
        inQueueX1_.FreeTensor(x1Local);
        inQueueX2_.FreeTensor(x2Local);
        outQueueX_.EnQue<T_X>(xOutLocal);
        CopyOutX(xGm_, outQueueX_, gmOffset, numN_);

        // 2. Calc Scale and Y
        LocalTensor<yCopyDtype> y1Local;
        LocalTensor<yCopyDtype> y2Local;
        if (outQuant1Flag_) {
            y1Local = outQueueY1_.AllocTensor<yCopyDtype>();
        }
        if (outQuant2Flag_) {
            y2Local = outQueueY2_.AllocTensor<yCopyDtype>();
        }

        LocalTensor<float> y3Local;
        LocalTensor<T_X> y4Local;
        if constexpr (Y3_MODE) {
            y3Local = outQueueY3_.AllocTensor<float>();
        }
        if constexpr (Y4_MODE) {
            y4Local = outQueueY4_.AllocTensor<T_X>();
        }

        SetOverflowMode<T_Y>(0);
        if (outQuant1Flag_) {
            DispatchYScale<T_X, T_SMOOTH_SCALE>(y1Local, scale1Local, xOutTmpLocal, rstdLocal, gammaLocal, betaLocal,
                                                smoothScale1Local, y1TmpLocal, y3Local, y4Local, mInnerIdx, baseN_,
                                                hasSmoothScale1_, hasBeta_);
        }
        if (outQuant2Flag_) {
            DispatchYScale<T_X, T_SMOOTH_SCALE>(y2Local, scale2Local, xOutTmpLocal, rstdLocal, gammaLocal, betaLocal,
                                                smoothScale2Local, y2TmpLocal, y3Local, y4Local, mInnerIdx, baseN_,
                                                hasSmoothScale2_, hasBeta_);
        }
        SetOverflowMode<T_Y>(oriOverflowMode_);

        if (outQuant1Flag_) {
            outQueueY1_.EnQue<yCopyDtype>(y1Local);
        }
        if (outQuant2Flag_) {
            outQueueY2_.EnQue<yCopyDtype>(y2Local);
        }
        if constexpr (Y3_MODE) {
            outQueueY3_.EnQue<float>(y3Local);
        }
        if constexpr (Y4_MODE) {
            outQueueY4_.EnQue<T_X>(y4Local);
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
    GlobalTensor<T_X> betaGm_;
    GlobalTensor<yCopyDtype> y1Gm_;
    GlobalTensor<yCopyDtype> y2Gm_;
    GlobalTensor<float> y3Gm_;
    GlobalTensor<T_X> y4Gm_;
    GlobalTensor<float> scale1Gm_;
    GlobalTensor<float> scale2Gm_;
    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1_;
    TQue<QuePosition::VECIN, 1> inQueueX2_;
    TQue<QuePosition::VECIN, 1> inQueueGamma_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale1_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale2_;
    TQue<QuePosition::VECIN, 1> inQueueBeta_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_;
    TQue<QuePosition::VECOUT, 1> outQueueY2_;
    TQue<QuePosition::VECOUT, 1> outQueueY3_;
    TQue<QuePosition::VECOUT, 1> outQueueY4_;
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
    float epsilon_{0};
    float avgFactor_{0};
    uint64_t powerSplit_{0};
    uint64_t mPerCore_{0};
    uint64_t mLastCore_{0};
    bool hasSmoothScale1_{false};
    bool hasSmoothScale2_{false};
    bool hasBeta_{false};
    uint32_t outQuant1Flag_{1};
    uint32_t outQuant2Flag_{0};
    // Platform
    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    // Cal params
    uint64_t mCore_;
    uint64_t mOuterCnt_;
    uint64_t tailMOuter_;
    uint64_t baseNB8Align_;
    // Other
    int64_t oriOverflowMode_{0};
};
} // namespace AddRmsNormDynamicQuant
#endif // _ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_H_
