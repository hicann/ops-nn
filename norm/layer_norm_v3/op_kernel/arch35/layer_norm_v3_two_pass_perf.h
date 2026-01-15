/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_v3_two_pass_perf.h
 * \brief
 */

#ifndef LAYER_NORM_V3_TWO_PASS_PERF_H
#define LAYER_NORM_V3_TWO_PASS_PERF_H

#include "layer_norm_v3_common.h"

namespace LayerNormV3 {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

template <typename T, typename U, typename M>
class LayerNormV3RegbaseTwoPassPerf {
public:
    __aicore__ inline LayerNormV3RegbaseTwoPassPerf(
        const LayerNormV3TilingDataRegBaseTwoPassPerf* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        tl_ = tilingData;
        hasGamma_ = (tilingData->nullptrGamma == 0);
        hasBeta_ = (tilingData->nullptrBeta == 0);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd)
    {
        // copy in gamma and beta
        gammaGm_.SetGlobalBuffer((__gm__ U*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ U*)beta);
        pipe_->InitBuffer(gammaBetaQueue_, 1, tl_->rAlign * NUM_TWO * sizeof(U));

        int64_t meanBlockOffset = GetBlockIdx() * tl_->aBlockFactor;
        int64_t xBlockOffset = meanBlockOffset * tl_->r;
        xGm_.SetGlobalBuffer((__gm__ T*)x + xBlockOffset);
        yGm_.SetGlobalBuffer((__gm__ T*)y + xBlockOffset);
        meanGm_.SetGlobalBuffer((__gm__ M*)mean + meanBlockOffset);
        rstdGm_.SetGlobalBuffer((__gm__ M*)rstd + meanBlockOffset);

        pipe_->InitBuffer(meanQueue_, DOUBLE_BUFFER, tl_->aUbFactorAlignB32 * sizeof(float));
        pipe_->InitBuffer(rstdQueue_, DOUBLE_BUFFER, tl_->aUbFactorAlignB32 * sizeof(float));

        elemNum_ = tl_->aUbFactor * tl_->rAlign;
        pipe_->InitBuffer(xQueue_, DOUBLE_BUFFER, elemNum_ * sizeof(T));
        pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, elemNum_ * sizeof(T));
        pipe_->InitBuffer(tmpBuf, elemNum_ * sizeof(float) + NUM_TWO * GetVecLen() * NUM_TWO);
    }

    __aicore__ inline void Process()
    {
        CopyInGammaBeta();
        gammaBetaInUb_ = gammaBetaQueue_.template DeQue<U>();
        int64_t ubLoopNum = GetBlockIdx() == GetBlockNum() - 1 ? tl_->tailBlockUbLoops : tl_->formerBlockUbLoops;
        int64_t tailA =
            GetBlockIdx() == GetBlockNum() - 1 ?
                tl_->a - tl_->aBlockFactor * (GetBlockNum() - 1) - (tl_->tailBlockUbLoops - 1) * tl_->aUbFactor :
                tl_->aBlockFactor - (tl_->formerBlockUbLoops - 1) * tl_->aUbFactor;
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx++) {
            int64_t currentA = ubLoopIdx == ubLoopNum - 1 ? tailA : tl_->aUbFactor;
            int64_t aOffset = ubLoopIdx * tl_->aUbFactor;
            int64_t offset = aOffset * tl_->r;
            CopyInX(offset, currentA);
            Caculate(currentA);
            CopyOutY(offset, currentA);
            CopyOutMeanRstd(aOffset, currentA);
        }

        gammaBetaQueue_.FreeTensor(gammaBetaInUb_);
    }

private:
    __aicore__ inline void CopyInGammaBeta()
    {
        DataCopyPadExtParams<U> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = tl_->r * sizeof(U);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        gammaBetaInUb_ = gammaBetaQueue_.AllocTensor<U>();
        if (hasGamma_) {
            DataCopyPad(gammaBetaInUb_, gammaGm_, copyInParams, dataCopyPadExtParams);
        }
        if (hasBeta_) {
            DataCopyPad(gammaBetaInUb_[tl_->rAlign], betaGm_, copyInParams, dataCopyPadExtParams);
        }
        gammaBetaQueue_.EnQue(gammaBetaInUb_);
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T> xInUb = xQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = (tl_->r != tl_->rAlign);
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentANum;
        copyInParams.blockLen = tl_->r * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(xInUb, xGm_[offset], copyInParams, dataCopyPadExtParams);
        xQueue_.EnQue(xInUb);
    }

    __aicore__ inline void Caculate(int64_t currentANum)
    {
        LocalTensor<T> xInUb = xQueue_.template DeQue<T>();
        meanOutUb_ = meanQueue_.AllocTensor<float>();
        rstdOutUb_ = rstdQueue_.AllocTensor<float>();
        LocalTensor<float> tmpTensor = tmpBuf.Get<float>();

        __local_mem__ T* xInUbAddr = (__local_mem__ T*)xInUb.GetPhyAddr();
        __local_mem__ float* meanOutUbAddr = (__local_mem__ float*)meanOutUb_.GetPhyAddr();
        __local_mem__ float* rstdOutUbAddr = (__local_mem__ float*)rstdOutUb_.GetPhyAddr();
        __local_mem__ float* xSubMeanUbAddr = (__local_mem__ float*)tmpTensor.GetPhyAddr();
        __local_mem__ float* tmpUbAddr = (__local_mem__ float*)tmpTensor.GetPhyAddr() + elemNum_;

        if (tl_->rAlign <= VL_B32) {
            CalculateMeanVarRLessThanVL(xInUbAddr, meanOutUbAddr, rstdOutUbAddr, xSubMeanUbAddr, currentANum);
        } else if (tl_->rAlign <= VL_B32 + VL_B32) {
            CalculateMeanVarRLessThanTwoVL(xInUbAddr, meanOutUbAddr, rstdOutUbAddr, xSubMeanUbAddr, currentANum);
        } else if (tl_->rAlign <= VL_B32 * VL_B32 * NUM_TWO) {
            CalculateMeanVarRCommon<1>(xInUbAddr, meanOutUbAddr, rstdOutUbAddr, xSubMeanUbAddr, tmpUbAddr, currentANum);
        } else {
            CalculateMeanVarRCommon<NUM_TWO>(
                xInUbAddr, meanOutUbAddr, rstdOutUbAddr, xSubMeanUbAddr, tmpUbAddr, currentANum);
        }
        CalculateRstdVF(rstdOutUbAddr, currentANum);

        LocalTensor<T> yOutUb = yQueue_.AllocTensor<T>();
        __local_mem__ U* gammaInUbAddr = (__local_mem__ U*)gammaBetaInUb_.GetPhyAddr();
        __local_mem__ U* betaInUbAddr = (__local_mem__ U*)gammaBetaInUb_.GetPhyAddr() + tl_->rAlign;
        __local_mem__ T* yOutUbAddr = (__local_mem__ T*)yOutUb.GetPhyAddr();
        if (hasGamma_ && hasBeta_) {
            CalculateNormalizeVF<true, true>(
                xSubMeanUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, rstdOutUbAddr, currentANum);
        } else if (!hasGamma_ && hasBeta_) {
            CalculateNormalizeVF<false, true>(
                xSubMeanUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, rstdOutUbAddr, currentANum);
        } else if (hasGamma_ && !hasBeta_) {
            CalculateNormalizeVF<true, false>(
                xSubMeanUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, rstdOutUbAddr, currentANum);
        } else {
            CalculateNormalizeVF<false, false>(
                xSubMeanUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, rstdOutUbAddr, currentANum);
        }
        xQueue_.FreeTensor(xInUb);
        yQueue_.EnQue(yOutUb);
    }

    __aicore__ inline void CalculateMeanVarRLessThanVL(
        __local_mem__ T* xInUb, __local_mem__ float* meanInUb, __local_mem__ float* rstdInUb,
        __local_mem__ float* xSubMeanUb, uint16_t currentANum)
    {
        uint32_t reduceNum = static_cast<uint32_t>(tl_->r);
        float n = static_cast<float>(1.0) / static_cast<float>(tl_->powerOfTwoForR);
        float nCorrectionFactor = static_cast<float>(tl_->powerOfTwoForR) / static_cast<float>(reduceNum);
        uint32_t aStride = tl_->rAlign;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> meanSum;
            RegTensor<float> mean;
            RegTensor<float> meanDup;
            RegTensor<float> xMeanSub;
            RegTensor<float> square;
            RegTensor<float> varSum;
            RegTensor<float> var;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            uint32_t sreg0 = reduceNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t a = 0; a < currentANum; a++) {
                LoadTensorForDtypeTIn(xInUb, x, pregLoop, (a * aStride));
                Muls(meanSum, x, n, pregLoop);
                ReduceSum(mean, meanSum, pregLoop);
                Muls(mean, mean, nCorrectionFactor, pregOne);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanInUb + a, mean, pregOne);

                Duplicate(meanDup, mean, pregFull);
                Sub(xMeanSub, x, meanDup, pregLoop);
                StoreTensorForDtypeTOut(xSubMeanUb, xMeanSub, pregLoop, (a * aStride));
                Mul(square, xMeanSub, xMeanSub, pregLoop);
                Muls(varSum, square, n, pregLoop);
                ReduceSum(var, varSum, pregLoop);
                Muls(var, var, nCorrectionFactor, pregOne);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdInUb + a, var, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateMeanVarRLessThanTwoVL(
        __local_mem__ T* xInUb, __local_mem__ float* meanInUb, __local_mem__ float* rstdInUb,
        __local_mem__ float* xSubMeanUb, uint16_t currentANum)
    {
        uint32_t reduceNum = static_cast<uint32_t>(tl_->r);
        float n = static_cast<float>(1.0) / static_cast<float>(tl_->powerOfTwoForR);
        float nCorrectionFactor = static_cast<float>(tl_->powerOfTwoForR) / static_cast<float>(reduceNum);
        uint32_t aStride = tl_->rAlign;
        uint32_t aTail = reduceNum - VL_B32;

        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> meanSum1;
            RegTensor<float> meanSum2;
            RegTensor<float> meanSum;
            RegTensor<float> mean;
            RegTensor<float> meanDup;
            RegTensor<float> xMeanSub1;
            RegTensor<float> xMeanSub2;
            RegTensor<float> square1;
            RegTensor<float> square2;
            RegTensor<float> varSum1;
            RegTensor<float> varSum2;
            RegTensor<float> varSum;
            RegTensor<float> var;

            MaskReg pregTail = UpdateMask<float>(aTail);
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            for (uint16_t a = 0; a < currentANum; a++) {
                LoadTensorForDtypeTIn(xInUb, x1, pregFull, (a * aStride));
                LoadTensorForDtypeTIn(xInUb + VL_B32, x2, pregTail, (a * aStride));
                Muls(meanSum1, x1, n, pregFull);
                Muls(meanSum2, x2, n, pregTail);
                Add(meanSum, meanSum1, meanSum2, pregFull);
                ReduceSum(mean, meanSum, pregFull);
                Muls(mean, mean, nCorrectionFactor, pregOne);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanInUb + a, mean, pregOne);

                Duplicate(meanDup, mean, pregFull);
                Sub(xMeanSub1, x1, meanDup, pregFull);
                Sub(xMeanSub2, x2, meanDup, pregTail);
                StoreTensorForDtypeTOut(xSubMeanUb, xMeanSub1, pregFull, (a * aStride));
                StoreTensorForDtypeTOut(xSubMeanUb, xMeanSub2, pregTail, (a * aStride + VL_B32));
                Mul(square1, xMeanSub1, xMeanSub1, pregFull);
                Mul(square2, xMeanSub2, xMeanSub2, pregTail);
                Muls(varSum1, square1, n, pregFull);
                Muls(varSum2, square2, n, pregTail);
                Add(varSum, varSum1, varSum2, pregFull);
                ReduceSum(var, varSum, pregFull);
                Muls(var, var, nCorrectionFactor, pregOne);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdInUb + a, var, pregOne);
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateMeanVarRCommon(
        __local_mem__ T* xInUb, __local_mem__ float* meanInUb, __local_mem__ float* rstdInUb,
        __local_mem__ float* xSubMeanUb, __local_mem__ float* tmpUb, uint16_t currentANum)
    {
        uint32_t reduceNum = static_cast<uint32_t>(tl_->r);
        float n = static_cast<float>(1.0) / static_cast<float>(tl_->powerOfTwoForR);
        float nCorrectionFactor = static_cast<float>(tl_->powerOfTwoForR) / static_cast<float>(reduceNum);
        uint32_t aStride = tl_->rAlign;

        uint32_t binaryAddQuotient =
            tl_->powerOfTwoForR >= tl_->r ? tl_->powerOfTwoForR / NUM_TWO : tl_->powerOfTwoForR;
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + VL_B32 - 1) / VL_B32;

        uint32_t lastBinaryAddNum = binaryAddQuotient / VL_B32;
        uint32_t lastBinaryAddNumTmp = lastBinaryAddNum;
        uint32_t lastBinaryAddNumAlign = (binaryAddQuotient / VL_B32 + BLK_B32 - 1) / BLK_B32 * BLK_B32;

        uint32_t binaryAddRemainder = reduceNum - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_B32 - 1) / VL_B32;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_B32;

        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> meanSum;
            RegTensor<float> mean;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t a = 0; a < currentANum; a++) {
                uint32_t sregRemainder = binaryAddRemainder;
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; r++) {
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn(xInUb, x1, pregFull, (r * VL_B32 + a * aStride));
                    LoadTensorForDtypeTIn(xInUb + binaryAddQuotient, x2, pregFull, (r * VL_B32 + a * aStride));
                    Muls(x1, x1, n, pregFull);
                    Muls(x2, x2, n, pregFull);
                    Add(meanSum, x1, x2, pregFull);
                    ReduceSum(mean, meanSum, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign + r), mean, pregOne);
                }
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn(
                        xInUb + binaryAddRemainderFloorLoop * VL_B32, x1, pregFull, (r * VL_B32 + a * aStride));
                    LoadTensorForDtypeTIn(
                        xInUb + binaryAddRemainderFloorLoop * VL_B32 + binaryAddQuotient, x2, pregLoop,
                        (r * VL_B32 + a * aStride));
                    Muls(x1, x1, n, pregFull);
                    Muls(x2, x2, n, pregLoop);
                    Add(meanSum, x1, x2, pregFull);
                    ReduceSum(mean, meanSum, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), mean,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    LoadTensorForDtypeTIn(
                        xInUb + binaryAddRemainderCeilLoop * VL_B32, x1, pregFull, (r * VL_B32 + a * aStride));
                    Muls(x1, x1, n, pregFull);
                    ReduceSum(mean, x1, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r), mean,
                        pregOne);
                }
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t a = 0; a < currentANum; a++) {
                    DataCopy(x1, tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign));
                    ReduceSum(mean, x1, pregLast);
                    Muls(mean, mean, nCorrectionFactor, pregOne);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanInUb + a, mean, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                uint32_t lastTailNum = lastBinaryAddNum - VL_B32;
                MaskReg pregLast = UpdateMask<float>(lastTailNum);
                RegTensor<float> shlReg;
                for (uint16_t a = 0; a < currentANum; a++) {
                    DataCopy(x1, tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign));
                    DataCopy(x2, tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign + VL_B32));
                    ShiftLefts(
                        (RegTensor<uint32_t>&)shlReg, (RegTensor<uint32_t>&)x2, static_cast<int16_t>(0), pregLast);
                    Add(x1, x1, shlReg, pregFull);
                    ReduceSum(mean, x1, pregFull);
                    Muls(mean, mean, nCorrectionFactor, pregOne);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanInUb + a, mean, pregOne);
                }
            }
        }
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> mean;
            RegTensor<float> xMeanSub1;
            RegTensor<float> xMeanSub2;
            RegTensor<float> square1;
            RegTensor<float> square2;
            RegTensor<float> varSum;
            RegTensor<float> var;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t a = 0; a < currentANum; a++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(mean, meanInUb + a);
                uint32_t sregRemainder = binaryAddRemainder;
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; r++) {
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn(xInUb, x1, pregFull, (r * VL_B32 + a * aStride));
                    LoadTensorForDtypeTIn(xInUb + binaryAddQuotient, x2, pregFull, (r * VL_B32 + a * aStride));
                    Sub(xMeanSub1, x1, mean, pregFull);
                    Sub(xMeanSub2, x2, mean, pregFull);
                    StoreTensorForDtypeTOut(xSubMeanUb, xMeanSub1, pregFull, (r * VL_B32 + a * aStride));
                    StoreTensorForDtypeTOut(
                        xSubMeanUb + binaryAddQuotient, xMeanSub2, pregFull, (r * VL_B32 + a * aStride));
                    Mul(square1, xMeanSub1, xMeanSub1, pregFull);
                    Mul(square2, xMeanSub2, xMeanSub2, pregFull);
                    Muls(square1, square1, n, pregFull);
                    Muls(square2, square2, n, pregFull);
                    Add(varSum, square1, square2, pregFull);
                    ReduceSum(var, varSum, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign + r), var, pregOne);
                }
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn(
                        xInUb + binaryAddRemainderFloorLoop * VL_B32, x1, pregFull, (r * VL_B32 + a * aStride));
                    LoadTensorForDtypeTIn(
                        xInUb + binaryAddRemainderFloorLoop * VL_B32 + binaryAddQuotient, x2, pregLoop,
                        (r * VL_B32 + a * aStride));
                    Sub(xMeanSub1, x1, mean, pregFull);
                    Sub(xMeanSub2, x2, mean, pregLoop);
                    StoreTensorForDtypeTOut(
                        xSubMeanUb + binaryAddRemainderFloorLoop * VL_B32, xMeanSub1, pregFull,
                        (r * VL_B32 + a * aStride));
                    StoreTensorForDtypeTOut(
                        xSubMeanUb + +binaryAddRemainderFloorLoop * VL_B32 + binaryAddQuotient, xMeanSub2, pregLoop,
                        (r * VL_B32 + a * aStride));
                    Mul(square1, xMeanSub1, xMeanSub1, pregFull);
                    Mul(square2, xMeanSub2, xMeanSub2, pregLoop);
                    Muls(square1, square1, n, pregFull);
                    Muls(square2, square2, n, pregLoop);
                    Add(varSum, square1, square2, pregFull);
                    ReduceSum(var, varSum, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), var,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    LoadTensorForDtypeTIn(
                        xInUb + binaryAddRemainderCeilLoop * VL_B32, x1, pregFull, (r * VL_B32 + a * aStride));
                    Sub(xMeanSub1, x1, mean, pregFull);
                    StoreTensorForDtypeTOut(
                        xSubMeanUb + binaryAddRemainderCeilLoop * VL_B32, xMeanSub1, pregFull,
                        (r * VL_B32 + a * aStride));
                    Mul(square1, xMeanSub1, xMeanSub1, pregFull);
                    Muls(square1, square1, n, pregFull);
                    ReduceSum(var, square1, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r), var,
                        pregOne);
                }
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNumTmp);
                for (uint16_t a = 0; a < currentANum; a++) {
                    DataCopy(x1, tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign));
                    ReduceSum(var, x1, pregLast);
                    Muls(var, var, nCorrectionFactor, pregOne);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdInUb + a, var, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                uint32_t lastTailNum = lastBinaryAddNum - VL_B32;
                MaskReg pregLast = UpdateMask<float>(lastTailNum);
                RegTensor<float> shlReg;
                for (uint16_t a = 0; a < currentANum; a++) {
                    DataCopy(x1, tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign));
                    DataCopy(x2, tmpUb + static_cast<uint32_t>(a * lastBinaryAddNumAlign + VL_B32));
                    ShiftLefts(
                        (RegTensor<uint32_t>&)shlReg, (RegTensor<uint32_t>&)x2, static_cast<int16_t>(0), pregLast);
                    Add(x1, x1, shlReg, pregFull);
                    ReduceSum(var, x1, pregFull);
                    Muls(var, var, nCorrectionFactor, pregOne);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdInUb + a, var, pregOne);
                }
            }
        }
    }

    __aicore__ inline void CalculateRstdVF(__local_mem__ float* rstdOutUb, uint16_t currentANum)
    {
        uint16_t aLoop = (currentANum + VL_B32 - 1) / VL_B32;
        float epsilonLocal = tl_->epsilon;

        __VEC_SCOPE__
        {
            RegTensor<float> mean;
            RegTensor<float> var;

            RegTensor<float> sqrtVar;
            RegTensor<float> one;
            RegTensor<float> rsqrtVar;

            RegTensor<float> runningMean;
            RegTensor<float> saveMean;
            RegTensor<float> runningVar;
            RegTensor<float> saveVar;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();

            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> e;
            RegTensor<float> scalar1;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> rstd;

            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregLoop;

            Duplicate(one, 1.0, pregMain);
            uint32_t sreg0 = currentANum;
            for (uint16_t a = 0; a < aLoop; a++) {
                pregLoop = UpdateMask<float>(sreg0);
                Duplicate(scalar1, float(0.5), pregLoop);
                Duplicate(scalarInf, POS_INF, pregLoop);
                Duplicate(scalarZero, zero, pregLoop);
                Duplicate(t1, float(1.5), pregLoop);
                Duplicate(s, float(1.0), pregLoop);

                // rstd
                DataCopy(var, rstdOutUb + a * VL_B32);
                Adds(var, var, epsilonLocal, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, float(-0.5), pregLoop);
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
                CompareScalar(cmpRegInf, var, zero, pregLoop);
                Select(rstd, scalarInf, rstd, cmpRegInf);
                DataCopy(rstdOutUb + a * VL_B32, rstd, pregLoop);
            }
        }
    }

    template <bool hasGammaFlag, bool hasBetaFlag>
    __aicore__ inline void CalculateNormalizeVF(
        __local_mem__ float* xSubMeanUb, __local_mem__ U* betaInUb, __local_mem__ U* gammaInUb, __local_mem__ T* yOutUb,
        __local_mem__ float* rstdOutUb, uint16_t currentANum)
    {
        uint32_t reduceNum = tl_->r;
        uint32_t aStride = tl_->rAlign;
        uint16_t loopCount = (reduceNum + VL_B32 - 1) / VL_B32;
        uint32_t remainderA = currentANum / NUM_TWO * NUM_TWO;
        uint16_t remainderLoop = currentANum - remainderA;
        __local_mem__ float* rstdOutUbPair = rstdOutUb + 1;
        __local_mem__ float* rstdOutUbRemainder = rstdOutUb + remainderA;

        __VEC_SCOPE__
        {
            RegTensor<float> beta;
            RegTensor<float> gamma;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> rsqrt1;
            RegTensor<float> x2;
            RegTensor<float> y2;
            RegTensor<float> rsqrt2;
            RegTensor<float> xRemainder;
            RegTensor<float> yRemainder;
            RegTensor<float> rsqrtRemainder;

            MaskReg pregLoop;

            for (uint16_t a = 0; a < static_cast<uint16_t>(currentANum / static_cast<uint16_t>(NUM_TWO)); a++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrt1, rstdOutUb + a * NUM_TWO);
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrt2, rstdOutUbPair + a * NUM_TWO);
                uint32_t sreg0 = reduceNum;
                for (uint16_t r = 0; r < loopCount; r++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadTensorForDtypeTIn(xSubMeanUb, x1, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    LoadTensorForDtypeTIn(xSubMeanUb + aStride, x2, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    Mul(y1, x1, rsqrt1, pregLoop);
                    Mul(y2, x2, rsqrt2, pregLoop);
                    if constexpr (hasGammaFlag) {
                        LoadTensorForDtypeTIn(gammaInUb, gamma, pregLoop, (r * VL_B32));
                    }
                    if constexpr (hasBetaFlag) {
                        LoadTensorForDtypeTIn(betaInUb, beta, pregLoop, (r * VL_B32));
                    }
                    if constexpr (hasGammaFlag && hasBetaFlag) {
                        FusedMulDstAdd(y1, gamma, beta, pregLoop);
                        FusedMulDstAdd(y2, gamma, beta, pregLoop);
                    } else {
                        if constexpr (hasGammaFlag) {
                            Mul(y1, y1, gamma, pregLoop);
                            Mul(y2, y2, gamma, pregLoop);
                        }
                        if constexpr (hasBetaFlag) {
                            Add(y1, y1, beta, pregLoop);
                            Add(y2, y2, beta, pregLoop);
                        }
                    }
                    StoreTensorForDtypeTOut(yOutUb, y1, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    StoreTensorForDtypeTOut(yOutUb + aStride, y2, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                }
            }
            for (uint16_t a = 0; a < remainderLoop; a++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrtRemainder, rstdOutUbRemainder);
                uint32_t sreg1 = reduceNum;
                for (uint16_t r = 0; r < loopCount; r++) {
                    pregLoop = UpdateMask<float>(sreg1);
                    LoadTensorForDtypeTIn(xSubMeanUb + aStride * remainderA, xRemainder, pregLoop, (r * VL_B32));
                    Mul(yRemainder, xRemainder, rsqrtRemainder, pregLoop);
                    if constexpr (hasGammaFlag) {
                        LoadTensorForDtypeTIn(gammaInUb, gamma, pregLoop, (r * VL_B32));
                    }
                    if constexpr (hasBetaFlag) {
                        LoadTensorForDtypeTIn(betaInUb, beta, pregLoop, (r * VL_B32));
                    }
                    if constexpr (hasGammaFlag && hasBetaFlag) {
                        FusedMulDstAdd(yRemainder, gamma, beta, pregLoop);
                    } else {
                        if constexpr (hasGammaFlag) {
                            Mul(yRemainder, yRemainder, gamma, pregLoop);
                        }
                        if constexpr (hasBetaFlag) {
                            Add(yRemainder, yRemainder, beta, pregLoop);
                        }
                    }
                    StoreTensorForDtypeTOut(yOutUb + aStride * remainderA, yRemainder, pregLoop, (r * VL_B32));
                }
            }
        }
    }

    __aicore__ inline void CastMeanRstd(int64_t currentANum)
    {
        __local_mem__ float* meanInAddr = (__local_mem__ float*)meanOutUb_.GetPhyAddr();
        __local_mem__ float* rstdInAddr = (__local_mem__ float*)rstdOutUb_.GetPhyAddr();
        __local_mem__ M* meanOutAddr = (__local_mem__ M*)meanOutUb_.GetPhyAddr();
        __local_mem__ M* rstdOutAddr = (__local_mem__ M*)rstdOutUb_.GetPhyAddr();

        uint32_t castCount = static_cast<uint32_t>(currentANum);
        uint16_t castLoops = static_cast<uint32_t>((castCount + VL_B32 - 1) / VL_B32);
        __VEC_SCOPE__
        {
            RegTensor<float> input_mean;
            RegTensor<float> input_rstd;
            RegTensor<M> output_mean;
            RegTensor<M> output_rstd;
            MicroAPI::MaskReg pregLoop;
            for (uint16_t i = 0; i < castLoops; i++) {
                pregLoop = MicroAPI::UpdateMask<float>(castCount);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_mean, meanInAddr + VL_B32 * i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_rstd, rstdInAddr + VL_B32 * i);
                Cast<M, float, castTraitB322B16>(output_mean, input_mean, pregLoop);
                Cast<M, float, castTraitB322B16>(output_rstd, input_rstd, pregLoop);
                DataCopy<M, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ M*)meanOutAddr + i * VL_B16), output_mean, pregLoop);
                DataCopy<M, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ M*)rstdOutAddr + i * VL_B16), output_rstd, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyOutMeanRstd(int64_t offset, int64_t currentANum)
    {
        if constexpr (!IsSameType<M, float>::value) {
            // float to bfloat16 or float16, input continue and output each repeat have only half value
            CastMeanRstd(currentANum);
            meanQueue_.EnQue(meanOutUb_);
            rstdQueue_.EnQue(rstdOutUb_);
            LocalTensor<M> meanInUb = meanQueue_.template DeQue<M>();
            LocalTensor<M> rstdInUb = rstdQueue_.template DeQue<M>();

            uint32_t castDmaCount = static_cast<uint32_t>(currentANum);
            uint32_t castDmaLoops = static_cast<uint32_t>(castDmaCount / VL_B32);
            if (castDmaLoops > 0) {
                DataCopyExtParams copyInParams;
                copyInParams.blockCount = castDmaLoops;
                copyInParams.blockLen = VL_B32 * sizeof(M);
                copyInParams.srcStride = (VECTOR_REG_WIDTH - VL_B32 * sizeof(M)) / BLOCK_SIZE;
                copyInParams.dstStride = 0;
                DataCopyPad(meanGm_[offset], meanInUb, copyInParams);
                DataCopyPad(rstdGm_[offset], rstdInUb, copyInParams);
            }
            // tail
            uint32_t tailSize = static_cast<uint32_t>(castDmaCount % VL_B32);
            if (tailSize > 0) {
                DataCopyExtParams copyInParamsTail;
                copyInParamsTail.blockCount = 1;
                copyInParamsTail.blockLen = tailSize * sizeof(M);
                copyInParamsTail.srcStride = 0;
                copyInParamsTail.dstStride = 0;
                DataCopyPad(meanGm_[offset + castDmaLoops * VL_B32], meanInUb[castDmaLoops * VL_B16], copyInParamsTail);
                DataCopyPad(rstdGm_[offset + castDmaLoops * VL_B32], rstdInUb[castDmaLoops * VL_B16], copyInParamsTail);
            }
            meanQueue_.FreeTensor(meanInUb);
            rstdQueue_.FreeTensor(rstdInUb);
        } else {
            meanQueue_.EnQue(meanOutUb_);
            rstdQueue_.EnQue(rstdOutUb_);
            LocalTensor<float> meanInUb = meanQueue_.template DeQue<float>();
            LocalTensor<float> rstdInUb = rstdQueue_.template DeQue<float>();
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = currentANum * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            DataCopyPad(meanGm_[offset], meanInUb, copyInParams);
            DataCopyPad(rstdGm_[offset], rstdInUb, copyInParams);
            meanQueue_.FreeTensor(meanInUb);
            rstdQueue_.FreeTensor(rstdInUb);
        }
    }

    __aicore__ inline void CopyOutY(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T> yOutUb = yQueue_.template DeQue<T>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentANum;
        copyInParams.blockLen = tl_->r * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(yGm_[offset], yOutUb, copyInParams);
        yQueue_.FreeTensor(yOutUb);
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

    /* global memory address */
    GlobalTensor<T> xGm_;
    GlobalTensor<U> betaGm_;
    GlobalTensor<U> gammaGm_;

    GlobalTensor<T> yGm_;
    GlobalTensor<M> meanGm_;
    GlobalTensor<M> rstdGm_;

    /* variable */
    bool hasGamma_ = true;
    bool hasBeta_ = true;
    int64_t elemNum_ = -1;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    const LayerNormV3TilingDataRegBaseTwoPassPerf* tl_ = nullptr;

    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> gammaBetaQueue_;

    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECOUT, 1> meanQueue_;
    TQue<QuePosition::VECOUT, 1> rstdQueue_;

    TBuf<TPosition::VECCALC> tmpBuf;

    LocalTensor<U> gammaBetaInUb_;
    LocalTensor<float> meanOutUb_;
    LocalTensor<float> rstdOutUb_;
};
} // namespace LayerNormV3

#endif // LAYER_NORM_V3_TWO_PASS_PERF_H