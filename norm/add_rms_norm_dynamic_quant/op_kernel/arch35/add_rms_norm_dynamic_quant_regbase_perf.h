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

namespace AddRmsNormDynamicQuant {

template <typename T_X, typename T_Y, uint64_t TILING_KEY>
class KernelAddRmsNormDynamicQuantRegbasePerf {
#define INPUT_KEY ((TILING_KEY % 100) / 10)
#define HAS_SMOOTH_SCALE1 ((INPUT_KEY >> 1) % 2 == 1)
#define HAS_SMOOTH_SCALE2 (INPUT_KEY % 2 == 1)
#define HAS_Y2_SCALE2 HAS_SMOOTH_SCALE2
#define T_SMOOTH_SCALE T_X
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantRegbasePerf(TPipe* pipe)
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
        uint64_t firstVcaddResult =
            baseM_ * (((powerSplit_ + V_LENGTH - 1) / V_LENGTH + B32_BLOCK_NUM - 1) / B32_BLOCK_NUM * B32_BLOCK_NUM);
        pipe_->InitBuffer(inQueueX1_, 1, baseM_ * baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueX2_, 1, baseM_ * baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueX_, 1, baseM_ * baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueGamma_, 1, baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueY1_, DOUBLE_BUFFER, baseM_ * baseNB8Align_ * sizeof(T_Y));
        pipe_->InitBuffer(outQueueScale1_, DOUBLE_BUFFER, ubFactorRstd * sizeof(float));

        pipe_->InitBuffer(xOutTmpBuf_, baseM_ * baseNDtypeAlign_ * sizeof(float));
        pipe_->InitBuffer(y1TmpBuf_, baseM_ * baseNB32Align_ * sizeof(float));

        pipe_->InitBuffer(rstdBuf_, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(xReduceTmpBuf_, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(xTmpBuf_, firstVcaddResult * sizeof(float));
        if constexpr (HAS_SMOOTH_SCALE1) {
            pipe_->InitBuffer(inQueueSmoothScale1_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if constexpr (HAS_SMOOTH_SCALE2) {
            pipe_->InitBuffer(inQueueSmoothScale2_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if constexpr (HAS_Y2_SCALE2) {
            pipe_->InitBuffer(outQueueY2_, DOUBLE_BUFFER, baseM_ * baseNB8Align_ * sizeof(T_Y));
            pipe_->InitBuffer(outQueueScale2_, DOUBLE_BUFFER, ubFactorRstd * sizeof(float));
            pipe_->InitBuffer(y2TmpBuf_, baseM_ * baseNB32Align_ * sizeof(float));
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
            uint64_t gmOffset = mOuterOffset * baseN_;
            LocalTensor<float> scale1Local = outQueueScale1_.AllocTensor<float>();
            LocalTensor<float> scale2Local;
            if constexpr (HAS_Y2_SCALE2) {
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
            CalculateSquareReduceSum(xOutTmpLocal, xReduceLocal, realM);
            CalculateRstd(xReduceLocal, rstdLocal, realM);

            LocalTensor<T_Y> y1Local = outQueueY1_.AllocTensor<T_Y>();
            LocalTensor<float> y1TmpLocal = y1TmpBuf_.Get<float>();
            LocalTensor<T_Y> y2Local;
            LocalTensor<float> y2TmpLocal;
            if constexpr (HAS_Y2_SCALE2) {
                y2Local = outQueueY2_.AllocTensor<T_Y>();
                y2TmpLocal = y2TmpBuf_.Get<float>();
            }
            ComputeMutlScale<float, T_X, T_SMOOTH_SCALE, HAS_SMOOTH_SCALE1, T_Y>(
                scale1Local, xOutTmpLocal, rstdLocal, gammaLocal, smoothScale1Local, y1TmpLocal, baseN_, realM, baseN_, baseNDtypeAlign_);
            PipeBarrier<PIPE_V>();
            ComputeMutlY<T_Y>(y1Local, scale1Local, y1TmpLocal, baseN_, realM);
            if constexpr (HAS_SMOOTH_SCALE2) {
                ComputeMutlScale<float, T_X, T_SMOOTH_SCALE, HAS_SMOOTH_SCALE2, T_Y>(
                    scale2Local, xOutTmpLocal, rstdLocal, gammaLocal, smoothScale2Local, y2TmpLocal, baseN_, realM, baseN_, baseNDtypeAlign_);
                PipeBarrier<PIPE_V>();
                ComputeMutlY<T_Y>(y2Local, scale2Local, y2TmpLocal, baseN_, realM);
            }

            outQueueY1_.EnQue<T_Y>(y1Local);
            if constexpr (HAS_Y2_SCALE2) {
                outQueueY2_.EnQue<T_Y>(y2Local);
            }
            CopyOutY(gmOffset, realM);

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

    template <typename T_IN>
    __aicore__ inline void LoadTensorForDtypeTIn(
        __local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_IN, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
        } else {
            RegTensor<T_IN> xIn;
            DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
            Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
        }
    }

    template <typename T_OUT>
    __aicore__ inline void StoreTensorForDtypeTOut(
        __local_mem__ T_OUT* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_OUT, float>::value) {
            DataCopy<T_OUT, StoreDist::DIST_NORM>(dst + offset, src, preg);
        } else {
            RegTensor<T_OUT> xOut;
            Cast<T_OUT, float, castTraitB322B16>(xOut, src, preg);
            DataCopy<T_OUT, StoreDist::DIST_PACK_B32>(dst + offset, xOut, preg);
        }
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
        LocalTensor<T_Y> y1Local = outQueueY1_.DeQue<T_Y>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(realM),                // blockCount
            static_cast<uint32_t>(baseN_ * sizeof(T_Y)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPad(y1Gm_[offset], y1Local, copyParams);
        outQueueY1_.FreeTensor(y1Local);
        if constexpr (HAS_Y2_SCALE2) {
            LocalTensor<T_Y> y2Local = outQueueY2_.DeQue<T_Y>();
            DataCopyPad(y2Gm_[offset], y2Local, copyParams);
            outQueueY2_.FreeTensor(y2Local);
        }
    }

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

    __aicore__ inline void CalculateSquareReduceSum(
        LocalTensor<float>& xOutTmpLocal, LocalTensor<float>& xReduceLocal, uint32_t realM)
    {
        LocalTensor<float> binaryAddBuffTmp = xTmpBuf_.Get<float>();
        __local_mem__ float* xOutTmp = (__local_mem__ float*)xOutTmpLocal.GetPhyAddr();
        __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        __local_mem__ float* tmpUb = (__local_mem__ float*)binaryAddBuffTmp.GetPhyAddr();

        if (baseN_ <= V_LENGTH) {
            CalculateSquareReduceSumLessThanVL(xOutTmp, xReduceUb, realM, baseN_, baseNDtypeAlign_);
        } else if (baseN_ <= V_LENGTH + V_LENGTH) {
            CalculateSquareReduceSumLessThanTwoVL(xOutTmp, xReduceUb, realM, baseN_, baseNDtypeAlign_);
        } else if (baseN_ <= V_LENGTH * V_LENGTH * NUM_TWO) {
            CalculateSquareReduceSumCommon<NUM_ONE>(xOutTmp, xReduceUb, tmpUb, realM, baseN_, baseNDtypeAlign_);
        } else {
            CalculateSquareReduceSumCommon<NUM_TWO>(xOutTmp, xReduceUb, tmpUb, realM, baseN_, baseNDtypeAlign_);
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanVL(
        __local_mem__ float* xOutTmp, __local_mem__ float* xReduceUb, uint16_t realM, uint64_t baseN, uint64_t baseNDtypeAlign)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> vMean;
            RegTensor<float> rstdReg;
            RegTensor<float> onesReg;

            uint32_t sreg0 = baseN;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < realM; i++) {
                LoadTensorForDtypeTIn<float>(xOutTmp, x, pregLoop, i * baseNDtypeAlign);
                Mul(x, x, x, pregLoop);
                ReduceSum(vMean, x, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(
        __local_mem__ float* xOutTmp, __local_mem__ float* xReduceUb, uint16_t realM, uint64_t baseN, uint64_t baseNDtypeAlign)
    {
        uint32_t tailLen = baseN - V_LENGTH;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> rstdReg;
            RegTensor<float> onesReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregTail = UpdateMask<float>(tailLen);
            Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < realM; ++i) {
                LoadTensorForDtypeTIn<float>(xOutTmp, x, pregFull, i * baseNDtypeAlign);
                LoadTensorForDtypeTIn<float>(xOutTmp + V_LENGTH, xFold, pregTail, i * baseNDtypeAlign);
                Mul(x, x, x, pregFull);
                Mul(xFold, xFold, xFold, pregTail);
                ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregTail);
                Add(sumReg, x, xFold, pregFull);
                ReduceSum(vMean, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateXAdd(
        LocalTensor<T_X>& xLocal1, LocalTensor<T_X>& xLocal2, LocalTensor<T_X>& xOutLocal,
        LocalTensor<float>& xOutTmpLocal, uint32_t realM)
    {
        __local_mem__ T_X* x1InUb = (__local_mem__ T_X*)xLocal1.GetPhyAddr();
        __local_mem__ T_X* x2InUb = (__local_mem__ T_X*)xLocal2.GetPhyAddr();
        __local_mem__ T_X* xOutInUb = (__local_mem__ T_X*)xOutLocal.GetPhyAddr();
        __local_mem__ float* xOutTmp = (__local_mem__ float*)xOutTmpLocal.GetPhyAddr();

        uint32_t sreg = realM * baseNDtypeAlign_;
        uint16_t loopCount = (sreg + V_LENGTH - 1) / V_LENGTH;

        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> xSum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * V_LENGTH;
                pregLoop = UpdateMask<float>(sreg);
                LoadTensorForDtypeTIn<T_X>(x1InUb, x1, pregLoop, offset);
                LoadTensorForDtypeTIn<T_X>(x2InUb, x2, pregLoop, offset);
                Add(xSum, x1, x2, pregLoop);
                StoreTensorForDtypeTOut<T_X>(xOutInUb, xSum, pregLoop, offset);
                DataCopy<float, StoreDist::DIST_NORM_B32>(xOutTmp + offset, xSum, pregLoop);
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateSquareReduceSumCommon(
        __local_mem__ float* xOutTmp, __local_mem__ float* xReduceUb, __local_mem__ float* tmpUb, uint16_t realM, uint64_t baseN, uint64_t baseNDtypeAlign)
    {
        uint32_t binaryAddQuotient = powerSplit_;
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + V_LENGTH - 1) / V_LENGTH;

        uint32_t lastBinaryAddNum = binaryAddQuotient / V_LENGTH;
        uint32_t lastBinaryAddNumAlign = (binaryAddQuotientLoop + B32_BLOCK_NUM - 1) / B32_BLOCK_NUM * B32_BLOCK_NUM;

        uint32_t binaryAddRemainder = baseN - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + V_LENGTH - 1) / V_LENGTH;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / V_LENGTH;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> rstdReg;
            RegTensor<float> onesReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;
            Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < realM; ++i) {
                uint32_t baseOffset = i * baseNDtypeAlign;
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; ++r) {
                    uint32_t offset = r * V_LENGTH + baseOffset;
                    LoadTensorForDtypeTIn<float>(xOutTmp, x, pregFull, offset);
                    LoadTensorForDtypeTIn<float>(xOutTmp + binaryAddQuotient, xFold, pregFull, offset);
                    Mul(x, x, x, pregFull);
                    Mul(xFold, xFold, xFold, pregFull);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + r), vMean, pregOne);
                }
                uint32_t sregRemainder = binaryAddRemainder - binaryAddRemainderFloorLoop * V_LENGTH;
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    uint16_t offset = baseOffset;
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn<float>(xOutTmp + binaryAddRemainderFloorLoop * V_LENGTH, x, pregFull, offset);
                    LoadTensorForDtypeTIn<float>(
                        xOutTmp + binaryAddRemainderFloorLoop * V_LENGTH + binaryAddQuotient, xFold, pregLoop, offset);
                    Mul(x, x, x, pregFull);
                    Mul(xFold, xFold, xFold, pregLoop);
                    ShiftLefts(
                        (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLoop);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), vMean,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    uint32_t offset = r * V_LENGTH + baseOffset;
                    LoadTensorForDtypeTIn<float>(xOutTmp + binaryAddRemainderCeilLoop * V_LENGTH, x, pregFull, offset);
                    Mul(x, x, x, pregFull);
                    ReduceSum(vMean, x, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r),
                        vMean, pregOne);
                }
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < realM; ++i) {
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    ReduceSum(vMean, x, pregLast);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                lastBinaryAddNum -= V_LENGTH;
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < realM; ++i) {
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    DataCopy(xFold, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + V_LENGTH));
                    ShiftLefts(
                        (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLast);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
                }
            }
        }
    }

    __aicore__ inline void CalculateRstd(
        LocalTensor<float>& xReduceLocal, LocalTensor<float>& rstdLocal, uint32_t realM)
    {
        static constexpr float POS_INF = 3.40282366920938E+38;
        static constexpr float SCALAR1 = -0.5;
        static constexpr float SCALAR2 = 1.5;
        static constexpr float SCALAR3 = 0.5;
        static constexpr float SCALAR0 = -99.99;

        __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        uint16_t loopRows = static_cast<uint16_t>((realM + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> var;
            RegTensor<float> rstd;
            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> one;
            RegTensor<float> scalar1;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;

            uint32_t sreg = static_cast<uint32_t>(realM);
            for (uint16_t i = 0; i < loopRows; ++i) {
                pregLoop = UpdateMask<float>(sreg);
                Duplicate(scalarInf, POS_INF, pregLoop);
                Duplicate(scalarZero, float(0.0), pregLoop);
                Duplicate(one, float(1.0), pregLoop);
                Duplicate(scalar1, SCALAR3, pregLoop);
                Duplicate(t1, SCALAR2, pregLoop);
                Duplicate(s, float(1.0), pregLoop);
                // rstd
                DataCopy(var, xReduceUb + i * V_LENGTH);
                Muls(var, var, avgFactor_, pregLoop);
                Adds(var, var, epsilon_, pregLoop);
                Maxs(var, var, SCALAR0, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, SCALAR1, pregLoop);
                Mul(t, t, y, pregLoop);                // -0.5 * x * y
                Mula(t1, t, y, pregLoop);              // 1.5 + (-0.5 * x * y) * y
                Mul(rstd, y, t1, pregLoop);            // y = y * (1.5 - 0.5 * x * y)
                Muls(t3, var, float(-1.0), pregLoop);  // -1 * x
                Mula(s, t3, r, pregLoop);              // 1 + (-1) * x * r
                Muls(t4, rstd, float(-1.0), pregLoop); // (-1) * y
                Mula(r, t4, rstd, pregLoop);           // r + (-1) * y * y
                Mula(s, var, r, pregLoop);             // s + x * t
                Mul(s, s, rstd, pregLoop);             // e * y
                Mula(rstd, s, scalar1, pregLoop);      // y + y * e * 0.5
                CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
                Select(rstd, scalarZero, rstd, cmpRegZero);
                CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
                Select(rstd, scalarInf, rstd, cmpRegInf);
                DataCopy(rstdInUb + i * V_LENGTH, rstd, pregLoop);
            }
        }
    }

    template <
        typename T_XPF32, typename T_GAMMA, typename T_SMOOTHSCALE = float, bool HAS_SMOOTH_SCALE = true,
        typename T_YB8>
    __aicore__ inline void ComputeMutlScale(
        LocalTensor<float>& scaleLocal, LocalTensor<T_XPF32>& xLocal, LocalTensor<float>& rstdLocal,
        LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<T_SMOOTHSCALE>& smoothScaleLocal, LocalTensor<float>& yTmpLocal,
        uint32_t calCount, uint32_t realM, uint64_t baseN, uint64_t baseNDtypeAlign)
    {
        uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);
        uint16_t curAloops = static_cast<uint16_t>(realM);
        __local_mem__ T_XPF32* xAddr = (__ubuf__ T_XPF32*)xLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_SMOOTHSCALE* smoothScaleAddr;
        if constexpr (HAS_SMOOTH_SCALE) {
            smoothScaleAddr = (__ubuf__ T_SMOOTHSCALE*)smoothScaleLocal.GetPhyAddr();
        }
        __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
        __local_mem__ float* yTmpAddr = (__ubuf__ float*)yTmpLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            // VF0. Calc scale
            RegTensor<float> rstdReg, scaleReg;
            RegTensor<float> xRegFp32, yRegFp32, gammaRegFp32, smoothScaleRegFp32;
            MaskReg maskRegFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg maskRegOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg maskReg;

            for (uint16_t curA = 0; curA < curAloops; curA++) {
                Duplicate(scaleReg, static_cast<float>(-INFINITY), maskRegFull); // Abs before reducemax, scaleReg >= 0
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + static_cast<uint32_t>(curA));
                uint32_t sregElewiseNum = calCount;
                for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
                    maskReg = UpdateMask<float>(sregElewiseNum);
                    NormCommon::LoadCastRegVF(xRegFp32, xAddr + curA * baseNDtypeAlign, idx, maskReg);
                    NormCommon::LoadCastRegVF(gammaRegFp32, gammaAddr, idx, maskReg);
                    if constexpr (HAS_SMOOTH_SCALE) {
                        NormCommon::LoadCastRegVF(smoothScaleRegFp32, smoothScaleAddr, idx, maskReg);
                    }
                    Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
                    Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
                    if constexpr (HAS_SMOOTH_SCALE) {
                        Mul(yRegFp32, xRegFp32, smoothScaleRegFp32, maskReg);
                        DataCopy<float>(yTmpAddr + curA * baseNB32Align_ + idx * V_LENGTH, yRegFp32, maskReg);
                        Abs(yRegFp32, yRegFp32, maskReg);               // VF abs is zeroing mode
                        Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
                    } else {
                        DataCopy<float>(yTmpAddr + curA * baseNB32Align_ + idx * V_LENGTH, xRegFp32, maskReg);
                        Abs(yRegFp32, xRegFp32, maskReg);               // VF abs is zeroing mode
                        Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
                    }
                }
                ReduceMax(scaleReg, scaleReg, maskRegFull);
                if constexpr (IsSameType<T_YB8, int8_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_INT8, maskRegOne);
                } else if constexpr (IsSameType<T_YB8, fp8_e4m3fn_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E4M3FN, maskRegOne);
                } else if constexpr (IsSameType<T_YB8, fp8_e5m2_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E5M2, maskRegOne);
                } else if constexpr (IsSameType<T_YB8, hifloat8_t>::value) {
                    Muls(scaleReg, scaleReg, DIV_FACTOR_HIFP8, maskRegOne);
                }
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + curA, scaleReg, maskRegOne);
            }
        }
    }

    template <typename T_YB8>
    __aicore__ inline void ComputeMutlY(
        LocalTensor<T_YB8>& yLocal, LocalTensor<float>& scaleLocal, LocalTensor<float>& yTmpLocal, uint32_t calCount,
        uint32_t realM)
    {
        uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);
        uint16_t curAloops = static_cast<uint16_t>(realM);

        __local_mem__ T_YB8* yAddr = (__ubuf__ T_YB8*)yLocal.GetPhyAddr();
        __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
        __local_mem__ float* yTmpAddr = (__ubuf__ float*)yTmpLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            // VF1. Calc y
            RegTensor<float> yRegFp32, yRegFp32Tmp, scaleReg;
            RegTensor<int32_t> yRegInt32;
            RegTensor<half> yRegFp16;
            RegTensor<T_YB8> yReg;
            MaskReg maskReg;
            for (uint16_t curA = 0; curA < curAloops; curA++) {
                uint32_t sregElewiseNum = calCount;
                DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + curA);
                for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
                    maskReg = UpdateMask<float>(sregElewiseNum);
                    DataCopy<float>(yRegFp32, yTmpAddr + curA * baseNB32Align_ + idx * V_LENGTH);
                    Div(yRegFp32, yRegFp32, scaleReg, maskReg);
                    if constexpr (IsSameType<T_YB8, int8_t>::value) {
                        Truncate<float, RoundMode::CAST_RINT>(yRegFp32Tmp, yRegFp32, maskReg);
                        Cast<half, float, castTraitFp322Fp16>(yRegFp16, yRegFp32Tmp, maskReg);
                        Cast<T_YB8, half, castTraitFp162Int8>(yReg, yRegFp16, maskReg);
                    } else if constexpr (
                        IsSameType<T_YB8, fp8_e4m3fn_t>::value || IsSameType<T_YB8, fp8_e5m2_t>::value) {
                        Cast<T_YB8, float, castTraitFp322Fp8>(yReg, yRegFp32, maskReg);
                    } else if constexpr (IsSameType<T_YB8, hifloat8_t>::value) {
                        Cast<T_YB8, float, castTraitFp322Hifp8>(yReg, yRegFp32, maskReg);
                    }
                    DataCopy<T_YB8, StoreDist::DIST_PACK4_B32>(
                        yAddr + curA * baseNB8Align_ + idx * V_LENGTH, yReg, maskReg);
                }
            }
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
