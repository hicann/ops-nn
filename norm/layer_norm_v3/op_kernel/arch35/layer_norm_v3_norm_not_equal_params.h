/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_v3_norm_not_equal_params.h
 * \brief
 */

#ifndef LAYER_NORM_V3_NORM_NOT_EQUAL_PARAMS_H
#define LAYER_NORM_V3_NORM_NOT_EQUAL_PARAMS_H

#include "layer_norm_v3_common.h"
#include "../../norm_common/reduce_common_regbase.h"

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
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

template <typename T, typename U, typename M, bool IsOutRstd>
class LayerNormV3RegbaseNormNotEqualParams {
public:
    __aicore__ inline LayerNormV3RegbaseNormNotEqualParams(
        const LayerNormV3TilingDataRegBaseNormNotEqualParams* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        tl_ = tilingData;
        hasGamma_ = (tilingData->nullptrGamma == 0);
        hasBeta_ = (tilingData->nullptrBeta == 0);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd)
    {
        int64_t meanBlockOffset = GetBlockIdx() * tl_->blockFactor;
        int64_t xBlockOffset = meanBlockOffset * tl_->r;
        xGm_.SetGlobalBuffer((__gm__ T*)x + xBlockOffset);
        yGm_.SetGlobalBuffer((__gm__ T*)y + xBlockOffset);
        meanGm_.SetGlobalBuffer((__gm__ M*)mean + meanBlockOffset);
        rstdGm_.SetGlobalBuffer((__gm__ M*)rstd + meanBlockOffset);

        if (hasGamma_) {
            gammaGm_.SetGlobalBuffer((__gm__ U*)gamma);
        }
        if (hasBeta_) {
            betaGm_.SetGlobalBuffer((__gm__ U*)beta);
        }

        if (tl_->r1 != 1) {
            // r1 != 1: broadcast gamma[R2] to [R=R1*R2], single buffer
            pipe_->InitBuffer(gammaBetaQueue_, 1, tl_->rAlign * NUM_TWO * sizeof(U));
        } else if (tl_->isFullB) {
            // isFullB: load gamma[B, R] all at once, single buffer
            pipe_->InitBuffer(gammaBetaQueue_, 1, tl_->b * tl_->rAlign * NUM_TWO * sizeof(U));
        } else {
            // !isFullB: load gamma per UB loop, double buffer
            pipe_->InitBuffer(gammaBetaQueue_, DOUBLE_BUFFER, tl_->rAxisCount * tl_->rAlign * NUM_TWO * sizeof(U));
        }

        elemNum_ = tl_->ubFactor * tl_->rAlign;
        pipe_->InitBuffer(meanQueue_, DOUBLE_BUFFER, tl_->ubFactorAlignB32 * sizeof(float));
        pipe_->InitBuffer(rstdQueue_, DOUBLE_BUFFER, tl_->ubFactorAlignB32 * sizeof(float));
        pipe_->InitBuffer(xQueue_, DOUBLE_BUFFER, elemNum_ * sizeof(T));
        pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, elemNum_ * sizeof(T));
        pipe_->InitBuffer(tmpBuf, elemNum_ * sizeof(float) + NUM_TWO * GetVecLen() * NUM_TWO);
        pipe_->InitBuffer(rstdTmpBuf_, tl_->ubFactorAlignB32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t ubLoopNum = GetBlockIdx() == GetBlockNum() - 1 ?
            tl_->tailBlockUbLoops : tl_->formerBlockUbLoops;
        int64_t tailA =
            GetBlockIdx() == GetBlockNum() - 1 ?
                tl_->a - tl_->blockFactor * (GetBlockNum() - 1) -
                    (tl_->tailBlockUbLoops - 1) * tl_->ubFactor :
                tl_->blockFactor - (tl_->formerBlockUbLoops - 1) * tl_->ubFactor;

        if (tl_->r1 != 1) {
            ProcessR1NotOne(ubLoopNum, tailA);
        } else if (tl_->isFullB) {
            ProcessFullB(ubLoopNum, tailA);
        } else {
            ProcessNotFullB(ubLoopNum, tailA);
        }
    }

private:
    __aicore__ inline void ProcessR1NotOne(int64_t ubLoopNum, int64_t tailA)
    {
        CopyInGammaBetaNddma();
        gammaBetaInUb_ = gammaBetaQueue_.template DeQue<U>();

        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx++) {
            int64_t currentA = ubLoopIdx == ubLoopNum - 1 ? tailA : tl_->ubFactor;
            int64_t aOffset = ubLoopIdx * tl_->ubFactor;
            int64_t offset = aOffset * tl_->r;
            CopyInX(offset, currentA);
            Caculate(aOffset, currentA);
            CopyOutY(offset, currentA);
            CopyOutMeanRstd(aOffset, currentA);
        }

        gammaBetaQueue_.FreeTensor(gammaBetaInUb_);
    }

    __aicore__ inline void ProcessFullB(int64_t ubLoopNum, int64_t tailA)
    {
        CopyInGammaBetaFull();
        gammaBetaInUb_ = gammaBetaQueue_.template DeQue<U>();

        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx++) {
            int64_t currentA = ubLoopIdx == ubLoopNum - 1 ? tailA : tl_->ubFactor;
            int64_t aOffset = ubLoopIdx * tl_->ubFactor;
            int64_t offset = aOffset * tl_->r;
            CopyInX(offset, currentA);
            CaculateWithBOffset(currentA, aOffset);
            CopyOutY(offset, currentA);
            CopyOutMeanRstd(aOffset, currentA);
        }

        gammaBetaQueue_.FreeTensor(gammaBetaInUb_);
    }

    __aicore__ inline void ProcessNotFullB(int64_t ubLoopNum, int64_t tailA)
    {
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx++) {
            int64_t currentA = ubLoopIdx == ubLoopNum - 1 ? tailA : tl_->ubFactor;
            int64_t aOffset = ubLoopIdx * tl_->ubFactor;
            int64_t offset = aOffset * tl_->r;
            int64_t globalRowStart = GetBlockIdx() * tl_->blockFactor + aOffset;

            CopyInGammaBetaPartial(currentA, globalRowStart);
            gammaBetaInUb_ = gammaBetaQueue_.template DeQue<U>();

            CopyInX(offset, currentA);
            CaculateWithBOffsetPartial(currentA, aOffset);
            CopyOutY(offset, currentA);
            CopyOutMeanRstd(aOffset, currentA);

            gammaBetaQueue_.FreeTensor(gammaBetaInUb_);
        }
    }

    __aicore__ inline void CopyInGammaBetaNddma()
    {
        gammaBetaInUb_ = gammaBetaQueue_.template AllocTensor<U>();
        int64_t r2 = tl_->r / tl_->r1;
        int64_t r1 = tl_->r1;

        static constexpr AscendC::MultiCopyConfig copyConfig = {false, 0, 0, false};
        constexpr int64_t MULTI_COPY_DIM = 2;
        MultiCopyLoopInfo<MULTI_COPY_DIM> multiCopyParams;
        multiCopyParams.loopSrcStride[0] = 1;
        multiCopyParams.loopSrcStride[1] = 0;

        multiCopyParams.loopDstStride[0] = 1;
        multiCopyParams.loopDstStride[1] = r2;

        multiCopyParams.loopSize[0] = r2;
        multiCopyParams.loopSize[1] = r1;

        U constValue = 0;
        AscendC::MultiCopyParams<U, MULTI_COPY_DIM> copyParams = {multiCopyParams, constValue};

        if (hasGamma_) {
            AscendC::DataCopy<U, MULTI_COPY_DIM, copyConfig>(gammaBetaInUb_, gammaGm_, copyParams);
        }
        if (hasBeta_) {
            AscendC::DataCopy<U, MULTI_COPY_DIM, copyConfig>(
                gammaBetaInUb_[tl_->rAlign], betaGm_, copyParams);
        }
        gammaBetaQueue_.EnQue(gammaBetaInUb_);
    }

    __aicore__ inline void CopyInGammaBetaFull()
    {
        gammaBetaInUb_ = gammaBetaQueue_.template AllocTensor<U>();
        DataCopyExtParams copyParams;
        DataCopyPadExtParams<U> padParams;
        padParams.isPad = false;

        if (hasGamma_) {
            copyParams.blockCount = tl_->b;
            copyParams.blockLen = tl_->r * sizeof(U);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(gammaBetaInUb_, gammaGm_, copyParams, padParams);
        }
        if (hasBeta_) {
            copyParams.blockCount = tl_->b;
            copyParams.blockLen = tl_->r * sizeof(U);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(gammaBetaInUb_[tl_->b * tl_->rAlign], betaGm_, copyParams, padParams);
        }
        gammaBetaQueue_.EnQue(gammaBetaInUb_);
    }
    __aicore__ inline void CopyInGammaBetaPartial(int64_t currentA, int64_t globalRowStart)
    {
        gammaBetaInUb_ = gammaBetaQueue_.template AllocTensor<U>();
        DataCopyExtParams copyParams;
        DataCopyPadExtParams<U> padParams;
        padParams.isPad = false;

        int64_t startB = globalRowStart % tl_->b;
        int64_t copyRows = currentA < tl_->rAxisCount ? currentA : tl_->rAxisCount;
        int64_t firstPart = tl_->b - startB;
        if (firstPart >= copyRows) {
            firstPart = copyRows;
        }
        int64_t secondPart = copyRows - firstPart;

        if (hasGamma_) {
            copyParams.blockCount = firstPart;
            copyParams.blockLen = tl_->r * sizeof(U);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(gammaBetaInUb_, gammaGm_[startB * tl_->r], copyParams, padParams);
            if (secondPart > 0) {
                copyParams.blockCount = secondPart;
                DataCopyPad(gammaBetaInUb_[firstPart * tl_->gammaBetaRAlign], gammaGm_, copyParams, padParams);
            }
        }
        if (hasBeta_) {
            int64_t betaBase = tl_->rAxisCount * tl_->rAlign;
            copyParams.blockCount = firstPart;
            copyParams.blockLen = tl_->r * sizeof(U);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(gammaBetaInUb_[betaBase], betaGm_[startB * tl_->r], copyParams, padParams);
            if (secondPart > 0) {
                copyParams.blockCount = secondPart;
                DataCopyPad(gammaBetaInUb_[betaBase + firstPart * tl_->gammaBetaRAlign], betaGm_, copyParams, padParams);
            }
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

    __aicore__ inline void CopyOutMeanRstd(int64_t offset, int64_t currentANum)
    {
        if constexpr (!IsSameType<M, float>::value) {
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

    // for r1 != 1, gamma is broadcast to full R, shared across all rows
    __aicore__ inline void Caculate(int64_t offset, int64_t currentANum)
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

        LocalTensor<float> rstdTmpTensor = rstdTmpBuf_.Get<float>();
        __local_mem__ float* rstdTmpUbAddr = (__local_mem__ float*)rstdTmpTensor.GetPhyAddr();
        CalculateRstdVF(rstdOutUbAddr, rstdTmpUbAddr, currentANum);
        __local_mem__ float* rstdForNorm;
        if constexpr (IsOutRstd) {
            rstdForNorm = rstdOutUbAddr;
        } else {
            rstdForNorm = rstdTmpUbAddr;
        }

        LocalTensor<T> yOutUb = yQueue_.AllocTensor<T>();
        __local_mem__ U* gammaInUbAddr = (__local_mem__ U*)gammaBetaInUb_.GetPhyAddr();
        __local_mem__ U* betaInUbAddr = (__local_mem__ U*)gammaBetaInUb_.GetPhyAddr() + tl_->rAlign;
        __local_mem__ T* yOutUbAddr = (__local_mem__ T*)yOutUb.GetPhyAddr();
        if (hasGamma_ && hasBeta_) {
            CalculateNormalizeVF<true, true>(
                xSubMeanUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, rstdForNorm, currentANum);
        } else if (!hasGamma_ && hasBeta_) {
            CalculateNormalizeVF<false, true>(
                xSubMeanUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, rstdForNorm, currentANum);
        } else if (hasGamma_ && !hasBeta_) {
            CalculateNormalizeVF<true, false>(
                xSubMeanUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, rstdForNorm, currentANum);
        } else {
            CalculateNormalizeVF<false, false>(
                xSubMeanUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, rstdForNorm, currentANum);
        }
        xQueue_.FreeTensor(xInUb);
        yQueue_.EnQue(yOutUb);
    }

    // for isFullB, gamma[B, R] fully loaded
    __aicore__ inline void CaculateWithBOffset(int64_t currentANum, int64_t aOffset)
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

        LocalTensor<float> rstdTmpTensor = rstdTmpBuf_.Get<float>();
        __local_mem__ float* rstdTmpUbAddr = (__local_mem__ float*)rstdTmpTensor.GetPhyAddr();
        CalculateRstdVF(rstdOutUbAddr, rstdTmpUbAddr, currentANum);
        __local_mem__ float* rstdForNorm;
        if constexpr (IsOutRstd) {
            rstdForNorm = rstdOutUbAddr;
        } else {
            rstdForNorm = rstdTmpUbAddr;
        }

        LocalTensor<T> yOutUb = yQueue_.AllocTensor<T>();
        __local_mem__ T* yOutUbAddr = (__local_mem__ T*)yOutUb.GetPhyAddr();
        __local_mem__ U* gammaBaseAddr = (__local_mem__ U*)gammaBetaInUb_.GetPhyAddr();
        __local_mem__ U* betaBaseAddr = (__local_mem__ U*)gammaBetaInUb_.GetPhyAddr() + tl_->b * tl_->rAlign;
        if (hasGamma_ && hasBeta_) {
            CalculateNormalizeVFFullB<true, true>(
                xSubMeanUbAddr, betaBaseAddr, gammaBaseAddr, yOutUbAddr, rstdForNorm, currentANum, aOffset);
        } else if (!hasGamma_ && hasBeta_) {
            CalculateNormalizeVFFullB<false, true>(
                xSubMeanUbAddr, betaBaseAddr, gammaBaseAddr, yOutUbAddr, rstdForNorm, currentANum, aOffset);
        } else if (hasGamma_ && !hasBeta_) {
            CalculateNormalizeVFFullB<true, false>(
                xSubMeanUbAddr, betaBaseAddr, gammaBaseAddr, yOutUbAddr, rstdForNorm, currentANum, aOffset);
        } else {
            CalculateNormalizeVFFullB<false, false>(
                xSubMeanUbAddr, betaBaseAddr, gammaBaseAddr, yOutUbAddr, rstdForNorm, currentANum, aOffset);
        }
        xQueue_.FreeTensor(xInUb);
        yQueue_.EnQue(yOutUb);
    }

    // for !isFullB, gamma loaded per UB loop
    __aicore__ inline void CaculateWithBOffsetPartial(int64_t currentANum, int64_t aOffset)
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

        LocalTensor<float> rstdTmpTensor = rstdTmpBuf_.Get<float>();
        __local_mem__ float* rstdTmpUbAddr = (__local_mem__ float*)rstdTmpTensor.GetPhyAddr();
        CalculateRstdVF(rstdOutUbAddr, rstdTmpUbAddr, currentANum);
        __local_mem__ float* rstdForNorm;
        if constexpr (IsOutRstd) {
            rstdForNorm = rstdOutUbAddr;
        } else {
            rstdForNorm = rstdTmpUbAddr;
        }

        LocalTensor<T> yOutUb = yQueue_.AllocTensor<T>();
        __local_mem__ T* yOutUbAddr = (__local_mem__ T*)yOutUb.GetPhyAddr();
        __local_mem__ U* gammaBaseAddr = (__local_mem__ U*)gammaBetaInUb_.GetPhyAddr();
        __local_mem__ U* betaBaseAddr = (__local_mem__ U*)gammaBetaInUb_.GetPhyAddr() + tl_->rAxisCount * tl_->rAlign;
        if (hasGamma_ && hasBeta_) {
            CalculateNormalizeVFNotFullB<true, true>(
                xSubMeanUbAddr, betaBaseAddr, gammaBaseAddr, yOutUbAddr, rstdForNorm, currentANum);
        } else if (!hasGamma_ && hasBeta_) {
            CalculateNormalizeVFNotFullB<false, true>(
                xSubMeanUbAddr, betaBaseAddr, gammaBaseAddr, yOutUbAddr, rstdForNorm, currentANum);
        } else if (hasGamma_ && !hasBeta_) {
            CalculateNormalizeVFNotFullB<true, false>(
                xSubMeanUbAddr, betaBaseAddr, gammaBaseAddr, yOutUbAddr, rstdForNorm, currentANum);
        } else {
            CalculateNormalizeVFNotFullB<false, false>(
                xSubMeanUbAddr, betaBaseAddr, gammaBaseAddr, yOutUbAddr, rstdForNorm, currentANum);
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
                LoadRegForDtype(xInUb, x, pregLoop, (a * aStride));
                Muls(meanSum, x, n, pregLoop);
                ReduceSum(mean, meanSum, pregLoop);
                Muls(mean, mean, nCorrectionFactor, pregOne);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanInUb + a, mean, pregOne);

                Duplicate(meanDup, mean, pregFull);
                Sub(xMeanSub, x, meanDup, pregLoop);
                StoreRegForDtype(xSubMeanUb, xMeanSub, pregLoop, (a * aStride));
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
                LoadRegForDtype(xInUb, x1, pregFull, (a * aStride));
                LoadRegForDtype(xInUb + VL_B32, x2, pregTail, (a * aStride));
                Muls(meanSum1, x1, n, pregFull);
                Muls(meanSum2, x2, n, pregTail);
                Add(meanSum, meanSum1, meanSum2, pregFull);
                ReduceSum(mean, meanSum, pregFull);
                Muls(mean, mean, nCorrectionFactor, pregOne);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanInUb + a, mean, pregOne);

                Duplicate(meanDup, mean, pregFull);
                Sub(xMeanSub1, x1, meanDup, pregFull);
                Sub(xMeanSub2, x2, meanDup, pregTail);
                StoreRegForDtype(xSubMeanUb, xMeanSub1, pregFull, (a * aStride));
                StoreRegForDtype(xSubMeanUb, xMeanSub2, pregTail, (a * aStride + VL_B32));
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
                    LoadRegForDtype(xInUb, x1, pregFull, (r * VL_B32 + a * aStride));
                    LoadRegForDtype(xInUb + binaryAddQuotient, x2, pregFull, (r * VL_B32 + a * aStride));
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
                    LoadRegForDtype(
                        xInUb + binaryAddRemainderFloorLoop * VL_B32, x1, pregFull, (r * VL_B32 + a * aStride));
                    LoadRegForDtype(
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
                    LoadRegForDtype(
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
                    LoadRegForDtype(xInUb, x1, pregFull, (r * VL_B32 + a * aStride));
                    LoadRegForDtype(xInUb + binaryAddQuotient, x2, pregFull, (r * VL_B32 + a * aStride));
                    Sub(xMeanSub1, x1, mean, pregFull);
                    Sub(xMeanSub2, x2, mean, pregFull);
                    StoreRegForDtype(xSubMeanUb, xMeanSub1, pregFull, (r * VL_B32 + a * aStride));
                    StoreRegForDtype(
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
                    LoadRegForDtype(
                        xInUb + binaryAddRemainderFloorLoop * VL_B32, x1, pregFull, (r * VL_B32 + a * aStride));
                    LoadRegForDtype(
                        xInUb + binaryAddRemainderFloorLoop * VL_B32 + binaryAddQuotient, x2, pregLoop,
                        (r * VL_B32 + a * aStride));
                    Sub(xMeanSub1, x1, mean, pregFull);
                    Sub(xMeanSub2, x2, mean, pregLoop);
                    StoreRegForDtype(
                        xSubMeanUb + binaryAddRemainderFloorLoop * VL_B32, xMeanSub1, pregFull,
                        (r * VL_B32 + a * aStride));
                    StoreRegForDtype(
                        xSubMeanUb + binaryAddRemainderFloorLoop * VL_B32 + binaryAddQuotient, xMeanSub2, pregLoop,
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
                    LoadRegForDtype(
                        xInUb + binaryAddRemainderCeilLoop * VL_B32, x1, pregFull, (r * VL_B32 + a * aStride));
                    Sub(xMeanSub1, x1, mean, pregFull);
                    StoreRegForDtype(
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

    __aicore__ inline void CalculateRstdVF(
        __local_mem__ float* rstdOutUb, __local_mem__ float* tmpUb, uint16_t currentANum)
    {
        float epsilonLocal = tl_->epsilon;
        uint16_t aLoop = static_cast<uint16_t>((currentANum + VL_B32 - 1) / VL_B32);
        uint32_t sreg = static_cast<uint32_t>(currentANum);
        __VEC_SCOPE__
        {
            RegTensor<float> varReg;
            RegTensor<float> rstdReg;
            MaskReg pregLoop;

            for (uint16_t a = 0; a < aLoop; a++) {
                DataCopy<float, LoadDist::DIST_NORM>(varReg, rstdOutUb + a * VL_B32);
                pregLoop = UpdateMask<float>(sreg);
                NormCommon::ComputeRstdNewtonRaphsonReg<false>(varReg, rstdReg, pregLoop, epsilonLocal);
                if constexpr (!IsOutRstd) {
                    // variance stays in rstdOutUb for output, compute rstd to tmpUb
                    DataCopy<float, StoreDist::DIST_NORM>(tmpUb + a * VL_B32, rstdReg, pregLoop);
                } else {
                    // compute rstd in-place to rstdOutUb for output
                    DataCopy<float, StoreDist::DIST_NORM>(rstdOutUb + a * VL_B32, rstdReg, pregLoop);
                }
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
                    LoadRegForDtype(xSubMeanUb, x1, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    LoadRegForDtype(xSubMeanUb + aStride, x2, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    Mul(y1, x1, rsqrt1, pregLoop);
                    Mul(y2, x2, rsqrt2, pregLoop);
                    if constexpr (hasGammaFlag) {
                        LoadRegForDtype(gammaInUb, gamma, pregLoop, (r * VL_B32));
                    }
                    if constexpr (hasBetaFlag) {
                        LoadRegForDtype(betaInUb, beta, pregLoop, (r * VL_B32));
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
                    StoreRegForDtype(yOutUb, y1, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    StoreRegForDtype(yOutUb + aStride, y2, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                }
            }
            for (uint16_t a = 0; a < remainderLoop; a++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrtRemainder, rstdOutUbRemainder);
                uint32_t sreg1 = reduceNum;
                for (uint16_t r = 0; r < loopCount; r++) {
                    pregLoop = UpdateMask<float>(sreg1);
                    LoadRegForDtype(xSubMeanUb + aStride * remainderA, xRemainder, pregLoop, (r * VL_B32));
                    Mul(yRemainder, xRemainder, rsqrtRemainder, pregLoop);
                    if constexpr (hasGammaFlag) {
                        LoadRegForDtype(gammaInUb, gamma, pregLoop, (r * VL_B32));
                    }
                    if constexpr (hasBetaFlag) {
                        LoadRegForDtype(betaInUb, beta, pregLoop, (r * VL_B32));
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
                    StoreRegForDtype(yOutUb + aStride * remainderA, yRemainder, pregLoop, (r * VL_B32));
                }
            }
        }
    }

    template <bool hasGammaFlag, bool hasBetaFlag>
    __aicore__ inline void CalculateNormalizeVFFullB(
        __local_mem__ float* xSubMeanUb, __local_mem__ U* betaInUb, __local_mem__ U* gammaInUb, __local_mem__ T* yOutUb,
        __local_mem__ float* rstdOutUb, uint16_t currentANum, int64_t aOffset)
    {
        uint32_t reduceNum = tl_->r;
        uint32_t aStride = tl_->rAlign;
        uint32_t gammaBetaStride = tl_->gammaBetaRAlign;
        uint32_t b = tl_->b;
        uint16_t loopCount = (reduceNum + VL_B32 - 1) / VL_B32;

        uint32_t baseOffset = GetBlockIdx() * tl_->blockFactor + aOffset;
        uint32_t firstStart = baseOffset % b;
        uint32_t firstEnd = b - firstStart < currentANum ? b - firstStart : currentANum;
        uint32_t firstRemainderA = firstEnd / NUM_TWO * NUM_TWO;
        uint16_t firstRemainderLoop = firstEnd - firstRemainderA;
        uint16_t secondLoopNum = (currentANum - firstEnd) / b;
        uint32_t secondRemainderA = b / NUM_TWO * NUM_TWO;
        uint16_t secondRemainderLoop = b - secondRemainderA;
        uint32_t thirdEnd = (currentANum - firstEnd) % b;
        uint32_t thirdRemainderA = thirdEnd / NUM_TWO * NUM_TWO;
        uint16_t thirdRemainderLoop = thirdEnd - thirdRemainderA;

        __local_mem__ float* rstdOutUbPair = rstdOutUb + 1;
        __local_mem__ float* rstdOutUbRemainder = rstdOutUb + firstRemainderA;
        __local_mem__ float* rstdOutUbSecondRemainder = rstdOutUb + firstEnd;
        __local_mem__ float* rstdOutUbThirdRemainder = rstdOutUb + firstEnd + b * secondLoopNum + thirdRemainderA;


        __VEC_SCOPE__
        {
            RegTensor<float> beta1;
            RegTensor<float> gamma1;
            RegTensor<float> beta2;
            RegTensor<float> gamma2;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> rsqrt1;
            RegTensor<float> x2;
            RegTensor<float> y2;
            RegTensor<float> rsqrt2;
            RegTensor<float> xRemainder;
            RegTensor<float> yRemainder;
            RegTensor<float> gammaRemainder;
            RegTensor<float> betaRemainder;
            RegTensor<float> rsqrtRemainder;

            MaskReg pregLoop;

            for (uint16_t a = 0; a < static_cast<uint16_t>(firstEnd / static_cast<uint16_t>(NUM_TWO)); a++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrt1, rstdOutUb + a * NUM_TWO);
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrt2, rstdOutUbPair + a * NUM_TWO);
                uint32_t sreg0 = reduceNum;
                for (uint16_t r = 0; r < loopCount; r++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadRegForDtype(xSubMeanUb, x1, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    LoadRegForDtype(xSubMeanUb + aStride, x2, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    Mul(y1, x1, rsqrt1, pregLoop);
                    Mul(y2, x2, rsqrt2, pregLoop);
                    if constexpr (hasGammaFlag) {
                        LoadRegForDtype(gammaInUb, gamma1, pregLoop, (r * VL_B32 + (firstStart + a * NUM_TWO) * gammaBetaStride));
                        LoadRegForDtype(gammaInUb, gamma2, pregLoop, (r * VL_B32 + (firstStart + a * NUM_TWO + 1) * gammaBetaStride));
                    }
                    if constexpr (hasBetaFlag) {
                        LoadRegForDtype(betaInUb, beta1, pregLoop, (r * VL_B32 + (firstStart + a * NUM_TWO) * gammaBetaStride));
                        LoadRegForDtype(betaInUb, beta2, pregLoop, (r * VL_B32 + (firstStart + a * NUM_TWO + 1) * gammaBetaStride));
                    }
                    if constexpr (hasGammaFlag && hasBetaFlag) {
                        FusedMulDstAdd(y1, gamma1, beta1, pregLoop);
                        FusedMulDstAdd(y2, gamma2, beta2, pregLoop);
                    } else {
                        if constexpr (hasGammaFlag) {
                            Mul(y1, y1, gamma1, pregLoop);
                            Mul(y2, y2, gamma2, pregLoop);
                        }
                        if constexpr (hasBetaFlag) {
                            Add(y1, y1, beta1, pregLoop);
                            Add(y2, y2, beta2, pregLoop);
                        }
                    }
                    StoreRegForDtype(yOutUb, y1, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    StoreRegForDtype(yOutUb + aStride, y2, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                }
            }
            for (uint16_t a = 0; a < firstRemainderLoop; a++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrtRemainder, rstdOutUbRemainder);
                uint32_t sreg0 = reduceNum;
                for (uint16_t r = 0; r < loopCount; r++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadRegForDtype(xSubMeanUb + firstRemainderA * aStride, xRemainder, pregLoop, (r * VL_B32));
                    Mul(yRemainder, xRemainder, rsqrtRemainder, pregLoop);
                    if constexpr (hasGammaFlag) {
                        LoadRegForDtype(gammaInUb, gammaRemainder, pregLoop, (r * VL_B32 + (firstStart + firstRemainderA) * gammaBetaStride));
                    }
                    if constexpr (hasBetaFlag) {
                        LoadRegForDtype(betaInUb, betaRemainder, pregLoop, (r * VL_B32 + (firstStart + firstRemainderA) * gammaBetaStride));
                    }
                    if constexpr (hasGammaFlag && hasBetaFlag) {
                        FusedMulDstAdd(yRemainder, gammaRemainder, betaRemainder, pregLoop);
                    } else {
                        if constexpr (hasGammaFlag) {
                            Mul(yRemainder, yRemainder, gammaRemainder, pregLoop);
                        }
                        if constexpr (hasBetaFlag) {
                            Add(yRemainder, yRemainder, betaRemainder, pregLoop);
                        }
                    }
                    StoreRegForDtype(yOutUb + aStride * firstRemainderA, yRemainder, pregLoop, (r * VL_B32));
                }
            }
        }

        __VEC_SCOPE__
        {
            RegTensor<float> beta1;
            RegTensor<float> gamma1;
            RegTensor<float> beta2;
            RegTensor<float> gamma2;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> rsqrt1;
            RegTensor<float> x2;
            RegTensor<float> y2;
            RegTensor<float> rsqrt2;
            RegTensor<float> xRemainder;
            RegTensor<float> yRemainder;
            RegTensor<float> gammaRemainder;
            RegTensor<float> betaRemainder;
            RegTensor<float> rsqrtRemainder;

            MaskReg pregLoop;

            for (uint16_t loop = 0; loop < secondLoopNum; loop++) {
                for (uint16_t a = 0; a < static_cast<uint16_t>(b / static_cast<uint16_t>(NUM_TWO)); a++) {
                    DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrt1, rstdOutUb + firstEnd + b * loop + a * NUM_TWO);
                    DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrt2, rstdOutUbPair + firstEnd + b * loop + a * NUM_TWO);
                    uint32_t sreg0 = reduceNum;
                    for (uint16_t r = 0; r < loopCount; r++) {
                        pregLoop = UpdateMask<float>(sreg0);
                        LoadRegForDtype(xSubMeanUb, x1, pregLoop, (r * VL_B32 + (firstEnd + b * loop + a * NUM_TWO) * aStride));
                        LoadRegForDtype(xSubMeanUb + aStride, x2, pregLoop, (r * VL_B32 + (firstEnd + b * loop + a * NUM_TWO) * aStride));
                        Mul(y1, x1, rsqrt1, pregLoop);
                        Mul(y2, x2, rsqrt2, pregLoop);
                        if constexpr (hasGammaFlag) {
                            LoadRegForDtype(gammaInUb, gamma1, pregLoop, (r * VL_B32 + a * NUM_TWO * gammaBetaStride));
                            LoadRegForDtype(gammaInUb, gamma2, pregLoop, (r * VL_B32 + (a * NUM_TWO + 1) * gammaBetaStride));
                        }
                        if constexpr (hasBetaFlag) {
                            LoadRegForDtype(betaInUb, beta1, pregLoop, (r * VL_B32 + a * NUM_TWO * gammaBetaStride));
                            LoadRegForDtype(betaInUb, beta2, pregLoop, (r * VL_B32 + (a * NUM_TWO + 1) * gammaBetaStride));
                        }
                        if constexpr (hasGammaFlag && hasBetaFlag) {
                            FusedMulDstAdd(y1, gamma1, beta1, pregLoop);
                            FusedMulDstAdd(y2, gamma2, beta2, pregLoop);
                        } else {
                            if constexpr (hasGammaFlag) {
                                Mul(y1, y1, gamma1, pregLoop);
                                Mul(y2, y2, gamma2, pregLoop);
                            }
                            if constexpr (hasBetaFlag) {
                                Add(y1, y1, beta1, pregLoop);
                                Add(y2, y2, beta2, pregLoop);
                            }
                        }
                        StoreRegForDtype(yOutUb, y1, pregLoop, (r * VL_B32 + (firstEnd + b * loop + a * NUM_TWO) * aStride));
                        StoreRegForDtype(yOutUb + aStride, y2, pregLoop, (r * VL_B32 + (firstEnd + b * loop + a * NUM_TWO) * aStride));
                    }
                }
                for (uint16_t a = 0; a < secondRemainderLoop; a++) {
                    DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrtRemainder, rstdOutUbSecondRemainder + b * loop + secondRemainderA);
                    uint32_t sreg0 = reduceNum;
                    for (uint16_t r = 0; r < loopCount; r++) {
                        pregLoop = UpdateMask<float>(sreg0);
                        LoadRegForDtype(xSubMeanUb + (firstEnd + b * loop + secondRemainderA) * aStride, xRemainder, pregLoop, (r * VL_B32));
                        Mul(yRemainder, xRemainder, rsqrtRemainder, pregLoop);
                        if constexpr (hasGammaFlag) {
                            LoadRegForDtype(gammaInUb, gammaRemainder, pregLoop, (r * VL_B32 + secondRemainderA * gammaBetaStride));
                        }
                        if constexpr (hasBetaFlag) {
                            LoadRegForDtype(betaInUb, betaRemainder, pregLoop, (r * VL_B32 + secondRemainderA * gammaBetaStride));
                        }
                        if constexpr (hasGammaFlag && hasBetaFlag) {
                            FusedMulDstAdd(yRemainder, gammaRemainder, betaRemainder, pregLoop);
                        } else {
                            if constexpr (hasGammaFlag) {
                                Mul(yRemainder, yRemainder, gammaRemainder, pregLoop);
                            }
                            if constexpr (hasBetaFlag) {
                                Add(yRemainder, yRemainder, betaRemainder, pregLoop);
                            }
                        }
                        StoreRegForDtype(yOutUb + (firstEnd + b * loop + secondRemainderA) * aStride, yRemainder, pregLoop, (r * VL_B32));
                    }
                }   
            }
        }


        __VEC_SCOPE__
        {
            RegTensor<float> beta1;
            RegTensor<float> gamma1;
            RegTensor<float> beta2;
            RegTensor<float> gamma2;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> rsqrt1;
            RegTensor<float> x2;
            RegTensor<float> y2;
            RegTensor<float> rsqrt2;
            RegTensor<float> xRemainder;
            RegTensor<float> yRemainder;
            RegTensor<float> gammaRemainder;
            RegTensor<float> betaRemainder;
            RegTensor<float> rsqrtRemainder;

            MaskReg pregLoop;

            for (uint16_t a = 0; a < static_cast<uint16_t>(thirdEnd / static_cast<uint16_t>(NUM_TWO)); a++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrt1, rstdOutUb + firstEnd + b * secondLoopNum + a * NUM_TWO);
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrt2, rstdOutUbPair + firstEnd + b * secondLoopNum + a * NUM_TWO);
                uint32_t sreg0 = reduceNum;
                for (uint16_t r = 0; r < loopCount; r++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadRegForDtype(xSubMeanUb, x1, pregLoop, (r * VL_B32 + (firstEnd + b * secondLoopNum + a * NUM_TWO) * aStride));
                    LoadRegForDtype(xSubMeanUb + aStride, x2, pregLoop, (r * VL_B32 + (firstEnd + b * secondLoopNum + a * NUM_TWO) * aStride));
                    Mul(y1, x1, rsqrt1, pregLoop);
                    Mul(y2, x2, rsqrt2, pregLoop);
                    if constexpr (hasGammaFlag) {
                        LoadRegForDtype(gammaInUb, gamma1, pregLoop, (r * VL_B32 + a * NUM_TWO * gammaBetaStride));
                        LoadRegForDtype(gammaInUb, gamma2, pregLoop, (r * VL_B32 + (a * NUM_TWO + 1) * gammaBetaStride));
                    }
                    if constexpr (hasBetaFlag) {
                        LoadRegForDtype(betaInUb, beta1, pregLoop, (r * VL_B32 + a * NUM_TWO * gammaBetaStride));
                        LoadRegForDtype(betaInUb, beta2, pregLoop, (r * VL_B32 + (a * NUM_TWO + 1) * gammaBetaStride));
                    }
                    if constexpr (hasGammaFlag && hasBetaFlag) {
                        FusedMulDstAdd(y1, gamma1, beta1, pregLoop);
                        FusedMulDstAdd(y2, gamma2, beta2, pregLoop);
                    } else {
                        if constexpr (hasGammaFlag) {
                            Mul(y1, y1, gamma1, pregLoop);
                            Mul(y2, y2, gamma2, pregLoop);
                        }
                        if constexpr (hasBetaFlag) {
                            Add(y1, y1, beta1, pregLoop);
                            Add(y2, y2, beta2, pregLoop);
                        }
                    }
                    StoreRegForDtype(yOutUb, y1, pregLoop, (r * VL_B32 + (firstEnd + b * secondLoopNum + a * NUM_TWO) * aStride));
                    StoreRegForDtype(yOutUb + aStride, y2, pregLoop, (r * VL_B32 + (firstEnd + b * secondLoopNum + a * NUM_TWO) * aStride));
                }
            }
            for (uint16_t a = 0; a < thirdRemainderLoop; a++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrtRemainder, rstdOutUbThirdRemainder);
                uint32_t sreg0 = reduceNum;
                for (uint16_t r = 0; r < loopCount; r++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadRegForDtype(xSubMeanUb + (firstEnd + b * secondLoopNum + thirdRemainderA) * aStride, xRemainder, pregLoop, (r * VL_B32));
                    Mul(yRemainder, xRemainder, rsqrtRemainder, pregLoop);
                    if constexpr (hasGammaFlag) {
                        LoadRegForDtype(gammaInUb, gammaRemainder, pregLoop, (r * VL_B32 + thirdRemainderA * gammaBetaStride));
                    }
                    if constexpr (hasBetaFlag) {
                        LoadRegForDtype(betaInUb, betaRemainder, pregLoop, (r * VL_B32 + thirdRemainderA * gammaBetaStride));
                    }
                    if constexpr (hasGammaFlag && hasBetaFlag) {
                        FusedMulDstAdd(yRemainder, gammaRemainder, betaRemainder, pregLoop);
                    } else {
                        if constexpr (hasGammaFlag) {
                            Mul(yRemainder, yRemainder, gammaRemainder, pregLoop);
                        }
                        if constexpr (hasBetaFlag) {
                            Add(yRemainder, yRemainder, betaRemainder, pregLoop);
                        }
                    }
                    StoreRegForDtype(yOutUb + (firstEnd + b * secondLoopNum + thirdRemainderA) * aStride, yRemainder, pregLoop, (r * VL_B32));
                }
            }
        }
    }

    template <bool hasGammaFlag, bool hasBetaFlag>
    __aicore__ inline void CalculateNormalizeVFNotFullB(
        __local_mem__ float* xSubMeanUb, __local_mem__ U* betaInUb, __local_mem__ U* gammaInUb, __local_mem__ T* yOutUb,
        __local_mem__ float* rstdOutUb, uint16_t currentANum)
    {
        uint32_t reduceNum = tl_->r;
        uint32_t aStride = tl_->rAlign;
        uint32_t gammaBetaStride = tl_->gammaBetaRAlign;
        uint32_t b = tl_->b;
        uint16_t loopCount = (reduceNum + VL_B32 - 1) / VL_B32;
        uint32_t remainderA = currentANum / NUM_TWO * NUM_TWO;
        uint16_t remainderLoop = currentANum - remainderA;
        __local_mem__ float* rstdOutUbPair = rstdOutUb + 1;
        __local_mem__ float* rstdOutUbRemainder = rstdOutUb + remainderA;
        

        __VEC_SCOPE__
        {
            RegTensor<float> beta1;
            RegTensor<float> gamma1;
            RegTensor<float> beta2;
            RegTensor<float> gamma2;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> rsqrt1;
            RegTensor<float> x2;
            RegTensor<float> y2;
            RegTensor<float> rsqrt2;
            RegTensor<float> xRemainder;
            RegTensor<float> yRemainder;
            RegTensor<float> gammaRemainder;
            RegTensor<float> betaRemainder;
            RegTensor<float> rsqrtRemainder;

            MaskReg pregLoop;

            for (uint16_t a = 0; a < static_cast<uint16_t>(currentANum / static_cast<uint16_t>(NUM_TWO)); a++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrt1, rstdOutUb + a * NUM_TWO);
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrt2, rstdOutUbPair + a * NUM_TWO);
                uint32_t sreg0 = reduceNum;
                for (uint16_t r = 0; r < loopCount; r++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadRegForDtype(xSubMeanUb, x1, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    LoadRegForDtype(xSubMeanUb + aStride, x2, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    Mul(y1, x1, rsqrt1, pregLoop);
                    Mul(y2, x2, rsqrt2, pregLoop);
                    if constexpr (hasGammaFlag) {
                        LoadRegForDtype(gammaInUb, gamma1, pregLoop, (r * VL_B32 + (a * NUM_TWO) * gammaBetaStride));
                        LoadRegForDtype(gammaInUb, gamma2, pregLoop, (r * VL_B32 + (a * NUM_TWO + 1) * gammaBetaStride));
                    }
                    if constexpr (hasBetaFlag) {
                        LoadRegForDtype(betaInUb, beta1, pregLoop, (r * VL_B32 + (a * NUM_TWO) * gammaBetaStride));
                        LoadRegForDtype(betaInUb, beta2, pregLoop, (r * VL_B32 + (a * NUM_TWO + 1) * gammaBetaStride));
                    }
                    if constexpr (hasGammaFlag && hasBetaFlag) {
                        FusedMulDstAdd(y1, gamma1, beta1, pregLoop);
                        FusedMulDstAdd(y2, gamma2, beta2, pregLoop);
                    } else {
                        if constexpr (hasGammaFlag) {
                            Mul(y1, y1, gamma1, pregLoop);
                            Mul(y2, y2, gamma2, pregLoop);
                        }
                        if constexpr (hasBetaFlag) {
                            Add(y1, y1, beta1, pregLoop);
                            Add(y2, y2, beta2, pregLoop);
                        }
                    }
                    StoreRegForDtype(yOutUb, y1, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                    StoreRegForDtype(yOutUb + aStride, y2, pregLoop, (r * VL_B32 + a * NUM_TWO * aStride));
                }
            }
            for (uint16_t a = 0; a < remainderLoop; a++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrtRemainder, rstdOutUbRemainder);
                uint32_t sreg0 = reduceNum;
                for (uint16_t r = 0; r < loopCount; r++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadRegForDtype(xSubMeanUb + remainderA * aStride, xRemainder, pregLoop, (r * VL_B32));
                    Mul(yRemainder, xRemainder, rsqrtRemainder, pregLoop);
                    if constexpr (hasGammaFlag) {
                        LoadRegForDtype(gammaInUb, gammaRemainder, pregLoop, (r * VL_B32 + remainderA * gammaBetaStride));
                    }
                    if constexpr (hasBetaFlag) {
                        LoadRegForDtype(betaInUb, betaRemainder, pregLoop, (r * VL_B32 + remainderA * gammaBetaStride));
                    }
                    if constexpr (hasGammaFlag && hasBetaFlag) {
                        FusedMulDstAdd(yRemainder, gammaRemainder, betaRemainder, pregLoop);
                    } else {
                        if constexpr (hasGammaFlag) {
                            Mul(yRemainder, yRemainder, gammaRemainder, pregLoop);
                        }
                        if constexpr (hasBetaFlag) {
                            Add(yRemainder, yRemainder, betaRemainder, pregLoop);
                        }
                    }
                    StoreRegForDtype(yOutUb + aStride * remainderA, yRemainder, pregLoop, (r * VL_B32));
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
    const LayerNormV3TilingDataRegBaseNormNotEqualParams* tl_ = nullptr;

    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> gammaBetaQueue_;

    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECOUT, 1> meanQueue_;
    TQue<QuePosition::VECOUT, 1> rstdQueue_;

    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> rstdTmpBuf_;

    LocalTensor<U> gammaBetaInUb_;
    LocalTensor<float> meanOutUb_;
    LocalTensor<float> rstdOutUb_;
};
} // namespace LayerNormV3

#endif // LAYER_NORM_V3_NORM_NOT_EQUAL_PARAMS_H
