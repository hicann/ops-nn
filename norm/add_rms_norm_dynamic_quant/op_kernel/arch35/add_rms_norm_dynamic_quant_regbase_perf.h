/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_quant_regbase_perf.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_PERF_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_PERF_H_

#include "add_rms_norm_dynamic_quant_regbase_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace AddRmsNormDynamicQuant {
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

template <typename T_X, typename T_Y, bool Y3_MODE, bool Y4_MODE>
class KernelAddRmsNormDynamicQuantRegbasePerf {
    using T_SMOOTH_SCALE = T_X;
    using yCopyDtype = YCopyDtype<T_Y>;

public:
    __aicore__ inline KernelAddRmsNormDynamicQuantRegbasePerf(TPipe* pipe) { pipe_ = pipe; }

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

        oriOverflowMode_ = GetOverflowMode<T_Y>();
        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        CalBlockTail();
        InitBuffer(x1, x2, gamma, smooathScale1, smooathScale2, beta, y1, y2, y3, y4, x, scale1, scale2);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore_ = blockIdx_ == (blockNum_ - 1) ? mLastCore_ : mPerCore_;
        mOuterCnt_ = CeilDiv(mCore_, baseM_);
        tailMOuter_ = mCore_ - (mOuterCnt_ - 1) * baseM_;
        baseNB8Align_ = CeilAlign(baseN_, B8_BLOCK_NUM);
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
        if (outQuant1Flag_) {
            uint64_t y1GmOffset = gmOffset;
            uint64_t y1GmLen = gmLen;
            if constexpr (IsSameType<T_Y, int4b_t>::value) {
                y1GmOffset = y1GmOffset >> 1;
                y1GmLen = y1GmLen >> 1;
            }
            y1Gm_.SetGlobalBuffer((__gm__ yCopyDtype*)y1 + y1GmOffset, y1GmLen);
            scale1Gm_.SetGlobalBuffer((__gm__ float*)scale1 + scalesGmOffset, mCore_);
        }
        InitOptionalGmBuffers<T_X, T_Y, T_SMOOTH_SCALE, Y3_MODE, Y4_MODE>(
            smoothScale1Gm_, smoothScale2Gm_, y2Gm_, scale2Gm_, y3Gm_, y4Gm_, betaGm_, smooathScale1, smooathScale2, y2,
            scale2, y3, y4, beta, gmOffset, gmLen, scalesGmOffset, mCore_, numN_, hasSmoothScale1_, hasSmoothScale2_,
            outQuant2Flag_, hasBeta_);

        InitUBBuffer();
    }

    __aicore__ inline void InitUBBuffer()
    {
        uint64_t ubFactorQuant = CeilAlign(numN_, BLOCK_SIZE / sizeof(T_SMOOTH_SCALE));
        uint64_t ubFactorRstd = CeilAlign(baseM_, B32_BLOCK_NUM);
        uint64_t firstVcaddResult = baseM_ * (((powerSplit_ + V_LENGTH - 1) / V_LENGTH + B32_BLOCK_NUM - 1) /
                                              B32_BLOCK_NUM * B32_BLOCK_NUM);
        pipe_->InitBuffer(inQueueX1_, 1, baseM_ * baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueX2_, 1, baseM_ * baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueX_, 1, baseM_ * baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueGamma_, 1, baseNDtypeAlign_ * sizeof(T_X));
        if (outQuant1Flag_) {
            pipe_->InitBuffer(outQueueY1_, DOUBLE_BUFFER, baseM_ * baseNB8Align_ * sizeof(yCopyDtype));
            pipe_->InitBuffer(outQueueScale1_, DOUBLE_BUFFER, ubFactorRstd * sizeof(float));
        }

        pipe_->InitBuffer(xOutTmpBuf_, baseM_ * baseNDtypeAlign_ * sizeof(float));
        pipe_->InitBuffer(y1TmpBuf_, baseM_ * baseNB32Align_ * sizeof(float));

        pipe_->InitBuffer(rstdBuf_, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(xReduceTmpBuf_, ubFactorRstd * sizeof(float));
        // CalculateSquareReduceSum函数中会额外开辟ub空间存放中间值
        pipe_->InitBuffer(xTmpBuf_, firstVcaddResult * sizeof(float));
        if (hasSmoothScale1_) {
            pipe_->InitBuffer(inQueueSmoothScale1_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if (hasSmoothScale2_) {
            pipe_->InitBuffer(inQueueSmoothScale2_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if (outQuant2Flag_) {
            pipe_->InitBuffer(outQueueY2_, DOUBLE_BUFFER, baseM_ * baseNB8Align_ * sizeof(yCopyDtype));
            pipe_->InitBuffer(outQueueScale2_, DOUBLE_BUFFER, ubFactorRstd * sizeof(float));
        }
        if (hasBeta_) {
            pipe_->InitBuffer(inQueueBeta_, 1, baseNDtypeAlign_ * sizeof(T_X));
        }
        if constexpr (Y3_MODE) {
            pipe_->InitBuffer(outQueueY3_, DOUBLE_BUFFER, baseM_ * baseNB32Align_ * sizeof(float));
        }
        if constexpr (Y4_MODE) {
            pipe_->InitBuffer(outQueueY4_, DOUBLE_BUFFER, baseM_ * baseNDtypeAlign_ * sizeof(T_X));
        }
    }
    __aicore__ inline void Process()
    {
        CopyInParamToQueue(inQueueGamma_, gammaGm_, numN_);
        CopyInDynamicQuantCommon(inQueueSmoothScale1_, inQueueSmoothScale2_, smoothScale1Gm_, smoothScale2Gm_, numN_,
                                 hasSmoothScale1_, hasSmoothScale2_);
        LocalTensor<T_X> gammaLocal = inQueueGamma_.DeQue<T_X>();
        LocalTensor<T_SMOOTH_SCALE> smoothScale1Local;
        LocalTensor<T_SMOOTH_SCALE> smoothScale2Local;
        LocalTensor<T_X> betaLocal;
        PrepareOptionalParamLocals(inQueueSmoothScale1_, inQueueSmoothScale2_, inQueueBeta_, betaGm_, smoothScale1Local,
                                   smoothScale2Local, betaLocal, numN_, hasSmoothScale1_, hasSmoothScale2_, hasBeta_);

        for (uint64_t mOuterIdx = 0; mOuterIdx < mOuterCnt_; mOuterIdx++) {
            uint64_t realM = mOuterIdx == (mOuterCnt_ - 1) ? tailMOuter_ : baseM_;
            uint64_t mOuterOffset = mOuterIdx * baseM_;
            uint64_t gmOffset = mOuterOffset * baseN_;
            LocalTensor<float> scale1Local;
            LocalTensor<float> scale2Local;
            if (outQuant1Flag_) {
                scale1Local = outQueueScale1_.AllocTensor<float>();
            }
            if (outQuant2Flag_) {
                scale2Local = outQueueScale2_.AllocTensor<float>();
            }
            // 1.x1 + x2
            CopyInXMutiMoveAlign(gmOffset, realM);
            LocalTensor<T_X> xLocal1 = inQueueX1_.DeQue<T_X>();
            LocalTensor<T_X> xLocal2 = inQueueX2_.DeQue<T_X>();
            LocalTensor<T_X> xOutLocal = outQueueX_.AllocTensor<T_X>();
            LocalTensor<float> xOutTmpLocal = xOutTmpBuf_.Get<float>();
            CalculateXAdd(xLocal1, xLocal2, xOutLocal, xOutTmpLocal, realM);
            inQueueX1_.FreeTensor(xLocal1);
            inQueueX2_.FreeTensor(xLocal2);
            outQueueX_.EnQue<T_X>(xOutLocal);
            CopyOutX(gmOffset, realM);

            // 2.二分累加计算SquareReduceSum、Rstd
            LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
            LocalTensor<float> xReduceLocal = xReduceTmpBuf_.Get<float>();
            NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(
                xOutTmpLocal, xReduceLocal, xReduceTmpBuf_, static_cast<uint16_t>(realM),
                static_cast<uint32_t>(baseNDtypeAlign_), static_cast<uint32_t>(baseN_),
                static_cast<uint32_t>(powerSplit_), static_cast<uint32_t>(B32_BLOCK_NUM));
            NormCommon::ComputeRstdNewtonRaphson<true, true>(xReduceLocal, rstdLocal, realM, epsilon_, avgFactor_,
                                                             V_LENGTH);

            LocalTensor<yCopyDtype> y1Local;
            LocalTensor<yCopyDtype> y2Local;
            if (outQuant1Flag_) {
                y1Local = outQueueY1_.AllocTensor<yCopyDtype>();
            }
            LocalTensor<float> y1TmpLocal = y1TmpBuf_.Get<float>();
            LocalTensor<float> y3Local;
            LocalTensor<T_X> y4Local;
            if (outQuant2Flag_) {
                y2Local = outQueueY2_.AllocTensor<yCopyDtype>();
            }
            if constexpr (Y3_MODE) {
                y3Local = outQueueY3_.AllocTensor<float>();
            }
            if constexpr (Y4_MODE) {
                y4Local = outQueueY4_.AllocTensor<T_X>();
            }
            SetOverflowMode<T_Y>(0);
            if (outQuant1Flag_) {
                DispatchMutlScale<T_X, T_SMOOTH_SCALE>(scale1Local, xOutTmpLocal, rstdLocal, gammaLocal, betaLocal,
                                                       smoothScale1Local, y1TmpLocal, y3Local, y4Local, baseN_, realM,
                                                       baseN_, baseNDtypeAlign_, hasSmoothScale1_, hasBeta_);
                PipeBarrier<PIPE_V>();
                ComputeMutlY(y1Local, scale1Local, y1TmpLocal, baseN_, realM);
            }
            if (outQuant2Flag_) {
                DispatchMutlScale<T_X, T_SMOOTH_SCALE>(scale2Local, xOutTmpLocal, rstdLocal, gammaLocal, betaLocal,
                                                       smoothScale2Local, y1TmpLocal, y3Local, y4Local, baseN_, realM,
                                                       baseN_, baseNDtypeAlign_, hasSmoothScale2_, hasBeta_);
                PipeBarrier<PIPE_V>();
                ComputeMutlY(y2Local, scale2Local, y1TmpLocal, baseN_, realM);
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
            CopyOutY(gmOffset, realM);

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

private:
    __aicore__ inline void CopyInXMutiMoveAlign(uint64_t gmOffset, uint32_t realM)
    {
        LocalTensor<T_X> xLocal1 = inQueueX1_.AllocTensor<T_X>();
        LocalTensor<T_X> xLocal2 = inQueueX2_.AllocTensor<T_X>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(realM),                // blockCount
            static_cast<uint32_t>(baseN_ * sizeof(T_X)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPadExtParams<T_X> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T_X>(0.0)    // paddingValue
        };
        DataCopyPad(xLocal1, x1Gm_[gmOffset], extParams, padParams);
        DataCopyPad(xLocal2, x2Gm_[gmOffset], extParams, padParams);
        inQueueX1_.EnQue(xLocal1);
        inQueueX2_.EnQue(xLocal2);
    }

    __aicore__ inline void CopyOutX(uint64_t offset, uint32_t realM)
    {
        LocalTensor<T_X> xLocal = outQueueX_.DeQue<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(realM),                // blockCount
            static_cast<uint32_t>(baseN_ * sizeof(T_X)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPad(xGm_[offset], xLocal, copyParams);
        outQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint32_t realM)
    {
        uint64_t gmOffset = offset;
        uint32_t yBlockLen = static_cast<uint32_t>(baseN_ * sizeof(yCopyDtype));
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(realM), // blockCount
            yBlockLen,                    // blockLen
            static_cast<uint32_t>(0),     // srcStride
            static_cast<uint32_t>(0),     // dstStride
            0                             // rsv
        };
        if constexpr (IsSameType<T_Y, int4b_t>::value) {
            gmOffset = gmOffset >> 1;
            copyParams.blockLen = yBlockLen >> 1;
            copyParams.srcStride = (baseNB8Align_ - (baseN_ / 2)) * sizeof(yCopyDtype) / BLOCK_SIZE;
        }
        if (outQuant1Flag_) {
            LocalTensor<yCopyDtype> y1Local = outQueueY1_.DeQue<yCopyDtype>();
            DataCopyPad(y1Gm_[gmOffset], y1Local, copyParams);
            outQueueY1_.FreeTensor(y1Local);
        }
        if (outQuant2Flag_) {
            LocalTensor<yCopyDtype> y2Local = outQueueY2_.DeQue<yCopyDtype>();
            DataCopyPad(y2Gm_[gmOffset], y2Local, copyParams);
            outQueueY2_.FreeTensor(y2Local);
        }
        if constexpr (Y3_MODE) {
            LocalTensor<float> y3Local = outQueueY3_.DeQue<float>();
            copyParams = DataCopyExtParams{
                static_cast<uint16_t>(realM),                  // blockCount
                static_cast<uint32_t>(baseN_ * sizeof(float)), // blockLen
                static_cast<uint32_t>(0),                      // srcStride
                static_cast<uint32_t>(0),                      // dstStride
                0                                              // rsv
            };
            DataCopyPad(y3Gm_[offset], y3Local, copyParams);
            outQueueY3_.FreeTensor(y3Local);
        }
        if constexpr (Y4_MODE) {
            LocalTensor<T_X> y4Local = outQueueY4_.DeQue<T_X>();
            copyParams = DataCopyExtParams{
                static_cast<uint16_t>(realM),                // blockCount
                static_cast<uint32_t>(baseN_ * sizeof(T_X)), // blockLen
                static_cast<uint32_t>(0),                    // srcStride
                static_cast<uint32_t>(0),                    // dstStride
                0                                            // rsv
            };
            DataCopyPad(y4Gm_[offset], y4Local, copyParams);
            outQueueY4_.FreeTensor(y4Local);
        }
    }

    __aicore__ inline void CalculateXAdd(LocalTensor<T_X>& xLocal1, LocalTensor<T_X>& xLocal2,
                                         LocalTensor<T_X>& xOutLocal, LocalTensor<float>& xOutTmpLocal, uint32_t realM)
    {
        __ubuf__ T_X* x1InUb = (__ubuf__ T_X*)xLocal1.GetPhyAddr();
        __ubuf__ T_X* x2InUb = (__ubuf__ T_X*)xLocal2.GetPhyAddr();
        __ubuf__ T_X* xOutInUb = (__ubuf__ T_X*)xOutLocal.GetPhyAddr();
        __ubuf__ float* xOutTmp = (__ubuf__ float*)xOutTmpLocal.GetPhyAddr();

        uint32_t sreg = realM * baseNDtypeAlign_;
        uint16_t loopCount = (sreg + V_LENGTH - 1) / V_LENGTH;

        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> xSum;
            RegTensor<float> x2;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * V_LENGTH;
                pregLoop = UpdateMask<float>(sreg);
                LoadRegForDtype<T_X>(x1InUb, x1, pregLoop, offset);
                LoadRegForDtype<T_X>(x2InUb, x2, pregLoop, offset);
                Add(xSum, x1, x2, pregLoop);
                StoreRegForDtype<T_X>(xOutInUb, xSum, pregLoop, offset);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(xOutTmp + offset, xSum, pregLoop);
            }
        }
    }

    template <typename T_XPF32, typename T_GAMMA, typename T_SMOOTHSCALE = float, bool HAS_SMOOTH_SCALE = true,
              bool HAS_BETA = false, typename T_YB8>
    __aicore__ inline void ComputeMutlScale(LocalTensor<float>& scaleLocal, LocalTensor<T_XPF32>& xLocal,
                                            LocalTensor<float>& rstdLocal, LocalTensor<T_GAMMA>& gammaLocal,
                                            LocalTensor<T_GAMMA>& betaLocal,
                                            LocalTensor<T_SMOOTHSCALE>& smoothScaleLocal, LocalTensor<float>& yTmpLocal,
                                            LocalTensor<float>& y3Local, LocalTensor<T_X>& y4Local, uint32_t calCount,
                                            uint32_t realM, uint64_t baseN, uint64_t baseNDtypeAlign)
    {
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, V_LENGTH));
        uint32_t remainderM = realM / NUM_TWO * NUM_TWO;
        uint16_t remainderLoop = static_cast<uint16_t>(realM - remainderM);
        uint16_t headLoops = static_cast<uint16_t>(remainderM / NUM_TWO);
        uint32_t mStride = static_cast<uint32_t>(baseNDtypeAlign);
        uint32_t m32Stride = static_cast<uint32_t>(baseNB32Align_);
        __ubuf__ T_XPF32* xAddr = (__ubuf__ T_XPF32*)xLocal.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __ubuf__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __ubuf__ T_GAMMA* betaAddr;
        if constexpr (HAS_BETA) {
            betaAddr = (__ubuf__ T_GAMMA*)betaLocal.GetPhyAddr();
        }
        __ubuf__ T_SMOOTHSCALE* smoothScaleAddr;
        if constexpr (HAS_SMOOTH_SCALE) {
            smoothScaleAddr = (__ubuf__ T_SMOOTHSCALE*)smoothScaleLocal.GetPhyAddr();
        }
        __ubuf__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
        __ubuf__ float* yTmpAddr = (__ubuf__ float*)yTmpLocal.GetPhyAddr();
        __ubuf__ float* y3Addr;
        if constexpr (Y3_MODE) {
            y3Addr = (__ubuf__ float*)y3Local.GetPhyAddr();
        }
        __ubuf__ T_X* y4Addr;
        if constexpr (Y4_MODE) {
            y4Addr = (__ubuf__ T_X*)y4Local.GetPhyAddr();
        }

        __VEC_SCOPE__
        {
            // VF0. Calc scale
            RegTensor<float> rstdReg, scaleReg, rstdReg1, scaleReg1;
            RegTensor<float> xRegFp32, yRegFp32, gammaRegFp32, betaRegFp32, smoothScaleRegFp32;
            RegTensor<float> xRegFp32One, yRegFp32One;
            MaskReg maskRegFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg maskRegOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg maskReg;

            for (uint16_t curA = 0; curA < headLoops; curA++) {
                Duplicate(scaleReg, static_cast<float>(-INFINITY), maskRegFull);  // Abs before reducemax, scaleReg >= 0
                Duplicate(scaleReg1, static_cast<float>(-INFINITY), maskRegFull); // Abs before reducemax, scaleReg >= 0
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + curA * NUM_TWO);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg1, rstdAddr + NUM_ONE + curA * NUM_TWO);
                uint32_t sregElewiseNum = calCount;
                for (uint16_t idx = 0; idx < repeatTimes; idx++) {
                    maskReg = UpdateMask<float>(sregElewiseNum);
                    NormCommon::LoadCastRegVF(xRegFp32, xAddr + curA * NUM_TWO * mStride, idx, maskReg);
                    NormCommon::LoadCastRegVF(xRegFp32One, xAddr + (curA * NUM_TWO + NUM_ONE) * mStride, idx, maskReg);
                    NormCommon::LoadCastRegVF(gammaRegFp32, gammaAddr, idx, maskReg);
                    if constexpr (HAS_BETA) {
                        NormCommon::LoadCastRegVF(betaRegFp32, betaAddr, idx, maskReg);
                    }
                    if constexpr (HAS_SMOOTH_SCALE) {
                        NormCommon::LoadCastRegVF(smoothScaleRegFp32, smoothScaleAddr, idx, maskReg);
                    }
                    Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
                    Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
                    Mul(xRegFp32One, xRegFp32One, rstdReg1, maskReg);
                    Mul(xRegFp32One, xRegFp32One, gammaRegFp32, maskReg);
                    if constexpr (Y3_MODE) {
                        StoreRegForDtype<float>(y3Addr + curA * NUM_TWO * m32Stride, xRegFp32, maskReg, idx * V_LENGTH);
                        StoreRegForDtype<float>(y3Addr + (curA * NUM_TWO + NUM_ONE) * m32Stride, xRegFp32One, maskReg,
                                                idx * V_LENGTH);
                    }
                    if constexpr (Y4_MODE) {
                        StoreRegForDtype<T_X>(y4Addr + curA * NUM_TWO * mStride, xRegFp32, maskReg, idx * V_LENGTH);
                        StoreRegForDtype<T_X>(y4Addr + (curA * NUM_TWO + NUM_ONE) * mStride, xRegFp32One, maskReg,
                                              idx * V_LENGTH);
                    }
                    if constexpr (HAS_BETA) {
                        Add(xRegFp32, xRegFp32, betaRegFp32, maskReg);
                        Add(xRegFp32One, xRegFp32One, betaRegFp32, maskReg);
                    }
                    if constexpr (HAS_SMOOTH_SCALE) {
                        Mul(yRegFp32, xRegFp32, smoothScaleRegFp32, maskReg);
                        Mul(yRegFp32One, xRegFp32One, smoothScaleRegFp32, maskReg);
                        StoreAlign<float>(yTmpAddr + curA * NUM_TWO * m32Stride + idx * V_LENGTH, yRegFp32, maskReg);
                        StoreAlign<float>(yTmpAddr + (curA * NUM_TWO + NUM_ONE) * m32Stride + idx * V_LENGTH,
                                          yRegFp32One, maskReg);
                        Abs(yRegFp32, yRegFp32, maskReg);                    // VF abs is zeroing mode
                        Abs(yRegFp32One, yRegFp32One, maskReg);              // VF abs is zeroing mode
                        Max(scaleReg, scaleReg, yRegFp32, maskRegFull);      // Using full mask
                        Max(scaleReg1, scaleReg1, yRegFp32One, maskRegFull); // Using full mask
                    } else {
                        StoreAlign<float>(yTmpAddr + curA * NUM_TWO * m32Stride + idx * V_LENGTH, xRegFp32, maskReg);
                        StoreAlign<float>(yTmpAddr + (curA * NUM_TWO + NUM_ONE) * m32Stride + idx * V_LENGTH,
                                          xRegFp32One, maskReg);
                        Abs(yRegFp32, xRegFp32, maskReg);                    // VF abs is zeroing mode
                        Abs(yRegFp32One, xRegFp32One, maskReg);              // VF abs is zeroing mode
                        Max(scaleReg, scaleReg, yRegFp32, maskRegFull);      // Using full mask
                        Max(scaleReg1, scaleReg1, yRegFp32One, maskRegFull); // Using full mask
                    }
                }
                Reduce<AscendC::Reg::ReduceType::MAX>(scaleReg, scaleReg, maskRegFull);
                Reduce<AscendC::Reg::ReduceType::MAX>(scaleReg1, scaleReg1, maskRegFull);
                if constexpr (IsSameType<T_YB8, int8_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_INT8, maskRegOne);
                    Muls(scaleReg1, scaleReg1, DIV_FACTOR_INT8, maskRegOne);
                } else if constexpr (IsSameType<T_YB8, fp8_e4m3fn_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E4M3FN, maskRegOne);
                    Muls(scaleReg1, scaleReg1, DIV_FACTOR_FP8E4M3FN, maskRegOne);
                } else if constexpr (IsSameType<T_YB8, fp8_e5m2_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E5M2, maskRegOne);
                    Muls(scaleReg1, scaleReg1, DIV_FACTOR_FP8E5M2, maskRegOne);
                } else if constexpr (IsSameType<T_YB8, hifloat8_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_HIFP8, maskRegOne);
                    Muls(scaleReg1, scaleReg1, DIV_FACTOR_HIFP8, maskRegOne);
                } else if constexpr (IsSameType<T_YB8, int4b_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_INT4, maskRegOne);
                    Muls(scaleReg1, scaleReg1, DIV_FACTOR_INT4, maskRegOne);
                }
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + curA * NUM_TWO, scaleReg, maskRegOne);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + curA * NUM_TWO + NUM_ONE, scaleReg1,
                                                                     maskRegOne);
            }
            for (uint16_t curA = 0; curA < remainderLoop; curA++) {
                Duplicate(scaleReg, static_cast<float>(-INFINITY), maskRegFull); // Abs before reducemax, scaleReg >= 0
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + remainderM);
                uint32_t sregElewiseNum = calCount;
                for (uint16_t idx = 0; idx < repeatTimes; idx++) {
                    maskReg = UpdateMask<float>(sregElewiseNum);
                    NormCommon::LoadCastRegVF(xRegFp32, xAddr + remainderM * mStride, idx, maskReg);
                    NormCommon::LoadCastRegVF(gammaRegFp32, gammaAddr, idx, maskReg);
                    if constexpr (HAS_BETA) {
                        NormCommon::LoadCastRegVF(betaRegFp32, betaAddr, idx, maskReg);
                    }
                    if constexpr (HAS_SMOOTH_SCALE) {
                        NormCommon::LoadCastRegVF(smoothScaleRegFp32, smoothScaleAddr, idx, maskReg);
                    }
                    Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
                    Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
                    if constexpr (Y3_MODE) {
                        StoreRegForDtype<float>(y3Addr + remainderM * m32Stride, xRegFp32, maskReg, idx * V_LENGTH);
                    }
                    if constexpr (Y4_MODE) {
                        StoreRegForDtype<T_X>(y4Addr + remainderM * mStride, xRegFp32, maskReg, idx * V_LENGTH);
                    }
                    if constexpr (HAS_BETA) {
                        Add(xRegFp32, xRegFp32, betaRegFp32, maskReg);
                    }
                    if constexpr (HAS_SMOOTH_SCALE) {
                        Mul(yRegFp32, xRegFp32, smoothScaleRegFp32, maskReg);
                        StoreAlign<float>(yTmpAddr + remainderM * m32Stride + idx * V_LENGTH, yRegFp32, maskReg);
                        Abs(yRegFp32, yRegFp32, maskReg);               // VF abs is zeroing mode
                        Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
                    } else {
                        StoreAlign<float>(yTmpAddr + remainderM * m32Stride + idx * V_LENGTH, xRegFp32, maskReg);
                        Abs(yRegFp32, xRegFp32, maskReg);               // VF abs is zeroing mode
                        Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
                    }
                }
                Reduce<AscendC::Reg::ReduceType::MAX>(scaleReg, scaleReg, maskRegFull);
                if constexpr (IsSameType<T_YB8, int8_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_INT8, maskRegOne);
                } else if constexpr (IsSameType<T_YB8, fp8_e4m3fn_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E4M3FN, maskRegOne);
                } else if constexpr (IsSameType<T_YB8, fp8_e5m2_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E5M2, maskRegOne);
                } else if constexpr (IsSameType<T_YB8, hifloat8_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_HIFP8, maskRegOne);
                } else if constexpr (IsSameType<T_YB8, int4b_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_INT4, maskRegOne);
                }
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + remainderM, scaleReg, maskRegOne);
            }
        }
    }

    __aicore__ inline void ComputeMutlY(LocalTensor<yCopyDtype>& yLocal, LocalTensor<float>& scaleLocal,
                                        LocalTensor<float>& yTmpLocal, uint32_t calCount, uint32_t realM)
    {
        uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);
        uint16_t curAloops = static_cast<uint16_t>(realM);

        __ubuf__ yCopyDtype* yAddr = (__ubuf__ yCopyDtype*)yLocal.GetPhyAddr();
        __ubuf__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
        __ubuf__ float* yTmpAddr = (__ubuf__ float*)yTmpLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            // VF1. Calc y
            RegTensor<float> yRegFp32, yRegFp32Tmp, scaleReg;
            RegTensor<half> yRegHalf;
            RegTensor<yCopyDtype> yReg;
            MaskReg maskReg;
            for (uint16_t curA = 0; curA < curAloops; curA++) {
                uint32_t sregElewiseNum = calCount;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + curA);
                for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
                    maskReg = UpdateMask<float>(sregElewiseNum);
                    LoadAlign<float>(yRegFp32, yTmpAddr + curA * baseNB32Align_ + idx * V_LENGTH);
                    Div(yRegFp32, yRegFp32, scaleReg, maskReg);
                    if constexpr (IsSameType<T_Y, int8_t>::value) {
                        Truncate<float, RoundMode::CAST_RINT>(yRegFp32Tmp, yRegFp32, maskReg);
                        Cast<half, float, castTraitFp322Fp16>(yRegHalf, yRegFp32Tmp, maskReg);
                        Cast<T_Y, half, castTraitFp162Int8>(yReg, yRegHalf, maskReg);
                        StoreAlign<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + curA * baseNB8Align_ + idx * V_LENGTH, yReg,
                                                                   maskReg);
                    } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value) {
                        Cast<T_Y, float, castTraitFp322Fp8>(yReg, yRegFp32, maskReg);
                        StoreAlign<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + curA * baseNB8Align_ + idx * V_LENGTH, yReg,
                                                                   maskReg);
                    } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
                        Cast<T_Y, float, castTraitFp322Hifp8>(yReg, yRegFp32, maskReg);
                        StoreAlign<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + curA * baseNB8Align_ + idx * V_LENGTH, yReg,
                                                                   maskReg);
                    } else if constexpr (IsSameType<T_Y, int4b_t>::value) {
                        RegTensor<int16_t> vregInt16Y;
                        RegTensor<uint16_t> vregTmp1Y;
                        RegTensor<uint8_t> vregTmp2Y;
                        MaskReg mask4Int4 = CreateMask<float, MaskPattern::H>();
                        Cast<int16_t, float, castTraitFp322Int16>(vregInt16Y, yRegFp32, maskReg);
                        Cast<half, int16_t, castTraitInt162Half>(yRegHalf, vregInt16Y, maskReg);
                        Pack(vregTmp1Y, (RegTensor<uint32_t>&)yRegHalf);
                        Cast<int4x2_t, half, castTraitF162I8>((RegTensor<int4x2_t>&)vregTmp2Y,
                                                              (RegTensor<half>&)vregTmp1Y, maskReg);
                        StoreAlign<uint8_t, StoreDist::DIST_PACK4_B32>(
                            yAddr + curA * baseNB8Align_ + idx * V_LENGTH / 2, vregTmp2Y, mask4Int4);
                    }
                }
            }
        }
    }

    template <typename T_GAMMA, typename T_SMOOTHSCALE>
    __aicore__ inline void DispatchMutlScale(
        LocalTensor<float>& scaleLocal, LocalTensor<float>& xLocal, LocalTensor<float>& rstdLocal,
        LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<T_GAMMA>& betaLocal, LocalTensor<T_SMOOTHSCALE>& smoothScaleLocal,
        LocalTensor<float>& yTmpLocal, LocalTensor<float>& y3Local, LocalTensor<T_X>& y4Local, uint32_t calCount,
        uint32_t realM, uint64_t baseN, uint64_t baseNDtypeAlign, bool hasSmoothScale, bool hasBeta)
    {
        if (hasSmoothScale) {
            if (hasBeta) {
                ComputeMutlScale<float, T_GAMMA, T_SMOOTHSCALE, true, true, T_Y>(
                    scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, y3Local, y4Local,
                    calCount, realM, baseN, baseNDtypeAlign);
            } else {
                ComputeMutlScale<float, T_GAMMA, T_SMOOTHSCALE, true, false, T_Y>(
                    scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, y3Local, y4Local,
                    calCount, realM, baseN, baseNDtypeAlign);
            }
        } else {
            if (hasBeta) {
                ComputeMutlScale<float, T_GAMMA, T_SMOOTHSCALE, false, true, T_Y>(
                    scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, y3Local, y4Local,
                    calCount, realM, baseN, baseNDtypeAlign);
            } else {
                ComputeMutlScale<float, T_GAMMA, T_SMOOTHSCALE, false, false, T_Y>(
                    scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, y3Local, y4Local,
                    calCount, realM, baseN, baseNDtypeAlign);
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    // GM Buffer
    GlobalTensor<T_X> x1Gm_;
    GlobalTensor<T_X> x2Gm_;
    GlobalTensor<T_X> gammaGm_;
    GlobalTensor<T_X> betaGm_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_SMOOTH_SCALE> smoothScale1Gm_;
    GlobalTensor<T_SMOOTH_SCALE> smoothScale2Gm_;
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
    TQue<QuePosition::VECIN, 1> inQueueBeta_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale1_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale2_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_;
    TQue<QuePosition::VECOUT, 1> outQueueY2_;
    TQue<QuePosition::VECOUT, 1> outQueueY3_;
    TQue<QuePosition::VECOUT, 1> outQueueY4_;
    TQue<QuePosition::VECOUT, 1> outQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueScale1_;
    TQue<QuePosition::VECOUT, 1> outQueueScale2_;
    TBuf<TPosition::VECCALC> rstdBuf_;
    TBuf<TPosition::VECCALC> y1TmpBuf_;
    TBuf<TPosition::VECCALC> xOutTmpBuf_;
    TBuf<TPosition::VECCALC> xReduceTmpBuf_;
    TBuf<TPosition::VECCALC> xTmpBuf_;

    // Tiling data
    uint64_t numN_{0};
    uint64_t numM_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseNDtypeAlign_{0};
    uint64_t baseNReduceAlign_{0};
    uint64_t baseNB32Align_{0};
    uint64_t powerSplit_{0};
    uint64_t mPerCore_{0};
    uint64_t mLastCore_{0};
    float epsilon_{0};
    float avgFactor_{0};
    bool hasSmoothScale1_{false};
    bool hasSmoothScale2_{false};
    bool hasBeta_{false};
    uint32_t outQuant1Flag_{1};
    uint32_t outQuant2Flag_{0};
    // Platform
    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    int64_t oriOverflowMode_{0};
    uint64_t mCore_;
    uint64_t mOuterCnt_;
    uint64_t tailMOuter_;
    uint64_t baseNB8Align_;
    // Other
};
} // namespace AddRmsNormDynamicQuant
#endif // ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_PERF_H_
