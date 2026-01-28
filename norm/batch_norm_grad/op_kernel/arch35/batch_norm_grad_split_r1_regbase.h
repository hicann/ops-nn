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
 * \file batch_norm_grad_split_r1.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_SPLIT_R1_REGBASE_H
#define BATCH_NORM_GRAD_SPLIT_R1_REGBASE_H

#include "batch_norm_grad_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace BatchNormGrad {
using namespace AscendC;


template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1> class BatchNormGradRARSplitR1 {
public:
    __aicore__ inline BatchNormGradRARSplitR1(TPipe *pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline uint32_t CalcBinaryAddLastNum(uint32_t binaryAddQuotient)
    {
        if (binaryAddQuotient <= VL_FP32 * VL_FP32) {
            return binaryAddQuotient / VL_FP32;
        }
        return VL_FP32;
    }

    __aicore__ inline void Init(__gm__ uint8_t *dy, __gm__ uint8_t *x, __gm__ uint8_t *mean, __gm__ uint8_t *rstd,
        __gm__ uint8_t *gamma, __gm__ uint8_t *dx, __gm__ uint8_t *dgamma, __gm__ uint8_t *dbeta,
        __gm__ uint8_t *workspace, const BatchNormGradRARRecomputeTilingData *tilingData)
    {
        const BatchNormGradBaseTilingData &baseTilingData = tilingData->baseTilingData;
        r1Dim_ = baseTilingData.r1Dim;
        aDim_ = baseTilingData.aDim;
        r0Dim_ = baseTilingData.r0Dim;
        blockNum_ = baseTilingData.blockNum;
        rAlign_ = tilingData->ubRDimFactor;

        binaryAddParam_.binaryAddQuotient = tilingData->generalBinAddTilingData.binaryAddQuotient;
        binaryAddParam_.binaryAddk = tilingData->generalBinAddTilingData.binaryAddk;
        binaryAddParam_.binaryAddLastNum = tilingData->generalBinAddTilingData.binaryAddLastNum;
        aTailCoreNum_ = baseTilingData.tailBlockNum;
        ubRDimLoopNum_ = tilingData->ubRDimLoopNum;

        nFactor_ = tilingData->ubRDimFactor / r0Dim_;
        tailNFactor_ = r1Dim_ % nFactor_ == 0 ? nFactor_ : r1Dim_ % nFactor_;
        ubRDimTailLoopNum_ = CeilDiv(r1Dim_, nFactor_) - ubRDimLoopNum_;

        gmOffset_ = GetBlockIdx() < baseTilingData.tailBlockNum ?
            GetBlockIdx() * baseTilingData.tailBlockDim :
            baseTilingData.tailBlockNum * baseTilingData.tailBlockDim +
            (GetBlockIdx() - baseTilingData.tailBlockNum) * baseTilingData.formerBlockDim;

        aDimLoopNum_ =
            GetBlockIdx() < baseTilingData.tailBlockNum ? baseTilingData.tailBlockDim : baseTilingData.formerBlockDim;

        tailBinaryAddParam_.binaryAddQuotient = binaryAddParam_.binaryAddQuotient >> 1;
        tailBinaryAddParam_.binaryAddk = binaryAddParam_.binaryAddk > 1 ? binaryAddParam_.binaryAddk - 1 : 0;
        tailBinaryAddParam_.binaryAddLastNum = CalcBinaryAddLastNum(tailBinaryAddParam_.binaryAddQuotient);

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE *)(dy) + gmOffset_ * r0Dim_);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE *)(x) + gmOffset_ * r0Dim_);
        meanGm_.SetGlobalBuffer((__gm__ float *)(mean) + gmOffset_);
        rstdGm_.SetGlobalBuffer((__gm__ float *)(rstd) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE *)(gamma) + gmOffset_);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE *)(dx) + gmOffset_ * r0Dim_);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE *)(dgamma) + gmOffset_);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE *)(dbeta) + gmOffset_);

        dyBufSize_ = RoundUpTwoBlock(rAlign_ * sizeof(DY_TYPE));
        xBufSize_ = RoundUpTwoBlock(rAlign_ * sizeof(DY_TYPE));
        xBufElemNum_ = xBufSize_ / sizeof(DY_TYPE);
        halfXBufOffset_ = xBufElemNum_ / DIGIT_TWO;
        meanBufSize_ = ONE_BLK_SIZE;
        meanBufElemNum_ = meanBufSize_ / sizeof(float);
        gammaBufSize_ = ONE_BLK_SIZE;
        gammaBufElemNum_ = gammaBufSize_ / sizeof(WEIGHT_TYPE);
        binaryAddBufSize_ = RoundUpOneBlock(binaryAddParam_.binaryAddQuotient / VL_FP32 * sizeof(float));
        binaryAddBufElemNum_ = binaryAddBufSize_ / sizeof(float);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize_);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, xBufSize_);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, gammaBufSize_);
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize_ * BUFFER_NUM);
        pipe_->InitBuffer(cacheBuf_, CACHE_BUFF_SIZE);

        V_MTE3_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        MTE3_MTE2_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void PreProcess(uint64_t ni)
    {
        // PreProcess
        dBetaLocal_ = dbetaOutQue_.template AllocTensor<float>();
        dGammaLocal_ = dgammaOutQue_.template AllocTensor<float>();
        binaryAddTensor_ = binaryAddBuf_.Get<float>();
        cacheLocal_ = cacheBuf_.Get<float>();
        Duplicate(cacheLocal_, 0.0f, CACHE_BUFF_SIZE / sizeof(float));
        dBetaCacheLocal_ = cacheLocal_[0];
        dGammaCacheLocal_ = cacheLocal_[DGAMA_CACHE_INDEX];
        dBetaFoldCacheLocal_ = cacheLocal_[DBETA_FOLD_CACHE_INDEX];
        dGammaFoldCacheLocal_ = cacheLocal_[DGAMA_FOLD_CACHE_INDEX];

        meanLocal_ = meanInQue_.template AllocTensor<float>();
        PrepareMean(meanLocal_, ni, 1);
        meanInQue_.EnQue(meanLocal_);

        rstdLocal_ = rstdInQue_.template AllocTensor<float>();
        PrepareRstd(rstdLocal_, ni, 1);
        rstdInQue_.EnQue(rstdLocal_);

        gammaLocal_ = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
        PrepareInGamma(gammaLocal_, ni, 1);
        gammaInQue_.EnQue(gammaLocal_);

        meanLocal_ = meanInQue_.template DeQue<float>();
        rstdLocal_ = rstdInQue_.template DeQue<float>();
        gammaLocal_ = gammaInQue_.template DeQue<WEIGHT_TYPE>();
    }

    __aicore__ inline void PostProcess(uint64_t ni)
    {
        // PostProcess
        ReSaveDGammaDBeta(dGammaLocal_, dBetaLocal_);
        dbetaOutQue_.EnQue(dBetaLocal_);
        LocalTensor<WEIGHT_TYPE> dBetaLocal = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
        CopyOutDbeta(dBetaLocal, ni, 1);
        dbetaOutQue_.FreeTensor(dBetaLocal);

        dgammaOutQue_.EnQue(dGammaLocal_);
        LocalTensor<WEIGHT_TYPE> dGammaLocal = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
        CopyOutDgamma(dGammaLocal, ni, 1);
        dgammaOutQue_.FreeTensor(dGammaLocal);

        meanInQue_.FreeTensor(meanLocal_);
        rstdInQue_.FreeTensor(rstdLocal_);
        gammaInQue_.FreeTensor(gammaLocal_);
    }

    __aicore__ inline bool IsNeedUpdateLevel1Cache(uint64_t basiBlockIdx)
    {
        return ((basiBlockIdx + 1) & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1)) == 0;
    }

    __aicore__ inline bool IsNeedUpdateLevel2Cache(uint64_t basiBlockIdx)
    {
        return ((basiBlockIdx + 1) & 0xffff) == 0;
    }

    __aicore__ inline void UpdateCache(uint64_t basiBlockIdx)
    {
        if (IsNeedUpdateLevel1Cache(basiBlockIdx)) {
            uint64_t updateIdx = ((basiBlockIdx & 0xff00) >> 8);
            CustomReduceSum(dBetaCacheLocal_[CACHE_LEVEL1_INDEX], dBetaCacheLocal_[CACHE_LEVEL0_INDEX], updateIdx);
            CustomReduceSum(dGammaCacheLocal_[CACHE_LEVEL1_INDEX], dGammaCacheLocal_[CACHE_LEVEL0_INDEX], updateIdx);
        }
        if (IsNeedUpdateLevel2Cache(basiBlockIdx)) {
            uint64_t updateIdx = (basiBlockIdx >> 16);
            CustomReduceSum(dBetaCacheLocal_[CACHE_LEVEL2_INDEX], dBetaCacheLocal_[CACHE_LEVEL1_INDEX], updateIdx);
            CustomReduceSum(dGammaCacheLocal_[CACHE_LEVEL2_INDEX], dGammaCacheLocal_[CACHE_LEVEL1_INDEX], updateIdx);
        }
    }

    __aicore__ inline bool IsNeedUpdateFoldCache(uint64_t basiBlockIdx)
    {
        return ((basiBlockIdx + 1) % FOLD_CACHE_CAPACITY) == 0;
    }

    __aicore__ inline void UpdateFoldCache(uint64_t basiBlockIdx, uint64_t FoldLoopNum)
    {
        if (IsNeedUpdateFoldCache(basiBlockIdx)) {
            uint64_t updateIdx =
                ((basiBlockIdx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1)) + 1) - FOLD_CACHE_CAPACITY;
            Add(dBetaCacheLocal_[updateIdx], dBetaCacheLocal_[updateIdx], dBetaFoldCacheLocal_, FOLD_CACHE_CAPACITY);
            Add(dGammaCacheLocal_[updateIdx], dGammaCacheLocal_[updateIdx], dGammaFoldCacheLocal_, FOLD_CACHE_CAPACITY);
        } else if (basiBlockIdx + 1 == FoldLoopNum) {
            uint32_t updateNum = static_cast<uint32_t>((basiBlockIdx + 1) & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1));
            uint32_t updateIdx = CeilDiv(updateNum, FOLD_CACHE_CAPACITY) * FOLD_CACHE_CAPACITY - FOLD_CACHE_CAPACITY;
            updateNum = updateNum & (FOLD_CACHE_CAPACITY - 1);
            Add(dBetaCacheLocal_[updateIdx], dBetaCacheLocal_[updateIdx], dBetaFoldCacheLocal_, updateNum);
            Add(dGammaCacheLocal_[updateIdx], dGammaCacheLocal_[updateIdx], dGammaFoldCacheLocal_, updateNum);
        }
    }

    __aicore__ inline void ProcessSummation(uint64_t basicBlockNum)
    {
        if (basicBlockNum < ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY) {
            CustomReduceSum(dBetaLocal_, dBetaCacheLocal_[CACHE_LEVEL0_INDEX], 0);
            CustomReduceSum(dGammaLocal_, dGammaCacheLocal_[CACHE_LEVEL0_INDEX], 0);
        } else if (basicBlockNum < ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY * ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY) {
            CustomReduceSum(dBetaLocal_, dBetaCacheLocal_[CACHE_LEVEL1_INDEX], 0);
            CustomReduceSum(dGammaLocal_, dGammaCacheLocal_[CACHE_LEVEL1_INDEX], 0);
        } else {
            CustomReduceSum(dBetaLocal_, dBetaCacheLocal_[CACHE_LEVEL2_INDEX], 0);
            CustomReduceSum(dGammaLocal_, dGammaCacheLocal_[CACHE_LEVEL2_INDEX], 0);
        }
    }

    __aicore__ inline void CustomReduceSum(const LocalTensor<float> &dstTensor, const LocalTensor<float> &srcTensor,
        uint32_t idx)
    {
        __ubuf__ float *dst = (__ubuf__ float *)dstTensor.GetPhyAddr();
        __ubuf__ float *src = (__ubuf__ float *)srcTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> x1;
            MicroAPI::RegTensor<float> x2;
            MicroAPI::RegTensor<float> x3;
            MicroAPI::RegTensor<float> x4;
            MicroAPI::RegTensor<float> sum1;
            MicroAPI::RegTensor<float> sum2;
            MicroAPI::RegTensor<float> sum12;
            MicroAPI::RegTensor<float> vlSum;

            MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::DataCopy(x1, (__local_mem__ float *)(src));
            MicroAPI::DataCopy(x2, (__local_mem__ float *)(src) + 1 * VL_FP32);
            MicroAPI::DataCopy(x3, (__local_mem__ float *)(src) + DIGIT_TWO * VL_FP32);
            MicroAPI::DataCopy(x4, (__local_mem__ float *)(src) + DIGIT_THREE * VL_FP32);
            MicroAPI::Add(sum1, x1, x3, pregAll);
            MicroAPI::Add(sum2, x2, x4, pregAll);
            MicroAPI::Add(sum12, sum1, sum2, pregAll);
            MicroAPI::ReduceSum(vlSum, sum12, pregAll);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dst + idx, vlSum, pregAll);
        }
    }

    __aicore__ inline void ProcessMainBlock(uint64_t ni, uint64_t basicBlockIdx, uint64_t r1Factor)
    {
        uint64_t offset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_;
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, offset, r1Factor);
        dyInQue_.EnQue(dyLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        CalcDbeta(dyLocal, dBetaCacheLocal_, binaryAddTensor_, basicBlockIdx, r1Factor * r0Dim_, binaryAddParam_);
        if (ni > 0 && basicBlockIdx < BUFFER_NUM) {
            uint32_t bufferIdx = basicBlockIdx % BUFFER_NUM;
            WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
        }
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        PrepareInX(xLocal, 0, offset, r1Factor);
        xInQue_.EnQue(xLocal);
        xLocal = xInQue_.template DeQue<DY_TYPE>();
        CalcDgamma(dyLocal, xLocal, meanLocal_, rstdLocal_, dGammaCacheLocal_, binaryAddTensor_, basicBlockIdx,
            r1Factor * r0Dim_, binaryAddParam_);

        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessFoldBlock(uint64_t ni, uint64_t basicBlockIdx, uint64_t r1Factor,
        uint64_t r1TailFactor, uint64_t r1TailTailFactor)
    {
        uint64_t offset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_;
        uint32_t foldCacheIdx = static_cast<uint32_t>(basicBlockIdx & (FOLD_CACHE_CAPACITY - 1));
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, offset, r1Factor);
        uint64_t tailOffset = offset + ubRDimLoopNum_ * nFactor_ * r0Dim_ * aDim_;
        PrepareInDy(dyLocal, halfXBufOffset_, tailOffset, r1TailFactor);
        dyInQue_.EnQue(dyLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        CalcDbetaFold(dyLocal, dBetaCacheLocal_, binaryAddTensor_, basicBlockIdx, r1Factor * r0Dim_,
            r1TailFactor * r0Dim_, tailBinaryAddParam_);
        if (ni > 0 && basicBlockIdx == 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + 0);
            WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + 1);
        }
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        PrepareInX(xLocal, 0, offset, r1Factor);
        PrepareInX(xLocal, halfXBufOffset_, tailOffset, r1TailFactor);
        xInQue_.EnQue(xLocal);
        xLocal = xInQue_.template DeQue<DY_TYPE>();
        CalcDgammaFold(dyLocal, xLocal, meanLocal_, rstdLocal_, dGammaCacheLocal_, binaryAddTensor_, basicBlockIdx,
            r1Factor * r0Dim_, r1TailFactor * r0Dim_, tailBinaryAddParam_);

        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);

        uint64_t foldOffset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_ + r1Factor * r0Dim_ * aDim_;
        dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, foldOffset, r1Factor);
        uint64_t foldTailOffset = foldOffset + ubRDimLoopNum_ * nFactor_ * r0Dim_ * aDim_;
        if (r1TailTailFactor > 0) {
            PrepareInDy(dyLocal, halfXBufOffset_, foldTailOffset, r1TailTailFactor);
        }
        dyInQue_.EnQue(dyLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        CalcDbetaFold(dyLocal, dBetaFoldCacheLocal_, binaryAddTensor_, foldCacheIdx, r1Factor * r0Dim_,
            r1TailTailFactor * r0Dim_, tailBinaryAddParam_);

        xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        PrepareInX(xLocal, 0, foldOffset, r1Factor);
        if (r1TailTailFactor > 0) {
            PrepareInX(xLocal, halfXBufOffset_, foldTailOffset, r1TailTailFactor);
        }
        xInQue_.EnQue(xLocal);
        xLocal = xInQue_.template DeQue<DY_TYPE>();
        CalcDgammaFold(dyLocal, xLocal, meanLocal_, rstdLocal_, dGammaFoldCacheLocal_, binaryAddTensor_, foldCacheIdx,
            r1Factor * r0Dim_, r1TailTailFactor * r0Dim_, tailBinaryAddParam_);

        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessDx(uint64_t ni, uint64_t basicBlockIdx, uint64_t r1Factor)
    {
        // ProcessDx
        uint32_t bufferIdx = basicBlockIdx % BUFFER_NUM;
        uint64_t offset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_;
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, offset, r1Factor);
        dyInQue_.EnQue(dyLocal);
        if (basicBlockIdx >= BUFFER_NUM) {
            WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
        }
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        PrepareInX(xLocal, 0, offset, r1Factor);
        xInQue_.EnQue(xLocal);

        xLocal = xInQue_.template DeQue<DY_TYPE>();
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        CalcDx(xLocal, dyLocal, rstdLocal_, meanLocal_, gammaLocal_, dGammaLocal_, dBetaLocal_, r1Factor * r0Dim_,
            r1Dim_ * r0Dim_);

        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);
        CopyOutDx(xLocal, offset, r1Factor);
        if (ni < aDimLoopNum_ - 1 || basicBlockIdx < ubRDimLoopNum_ + ubRDimTailLoopNum_ - BUFFER_NUM) {
            SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
        }
        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= blockNum_) {
            return;
        }

        for (uint64_t ni = 0; ni < aDimLoopNum_; ni++) {
            PreProcess(ni);
            for (uint64_t basicBlockIdx = 0; basicBlockIdx < ubRDimTailLoopNum_; basicBlockIdx++) {
                if (basicBlockIdx < ubRDimTailLoopNum_ - 1) {
                    ProcessFoldBlock(ni, basicBlockIdx, nFactor_ / DIGIT_TWO, nFactor_ / DIGIT_TWO,
                        nFactor_ / DIGIT_TWO);
                } else {
                    uint64_t tailRFactor = nFactor_ / DIGIT_TWO;
                    uint64_t tailTailRFactor = tailNFactor_ - tailRFactor;
                    if (tailNFactor_ < tailRFactor) {
                        tailRFactor = tailNFactor_;
                        tailTailRFactor = 0;
                    }
                    ProcessFoldBlock(ni, basicBlockIdx, nFactor_ / DIGIT_TWO, tailRFactor, tailTailRFactor);
                }
                UpdateFoldCache(basicBlockIdx, ubRDimTailLoopNum_);
                UpdateCache(basicBlockIdx);
            }
            for (uint64_t basicBlockIdx = ubRDimTailLoopNum_; basicBlockIdx < ubRDimLoopNum_; basicBlockIdx++) {
                ProcessMainBlock(ni, basicBlockIdx, nFactor_);
                UpdateCache(basicBlockIdx);
            }
            ProcessSummation(ubRDimLoopNum_);
            for (uint64_t basicBlockIdx = 0; basicBlockIdx < ubRDimLoopNum_ + ubRDimTailLoopNum_ - 1; basicBlockIdx++) {
                ProcessDx(ni, basicBlockIdx, nFactor_);
            }
            ProcessDx(ni, ubRDimLoopNum_ + ubRDimTailLoopNum_ - 1, tailNFactor_);
            PostProcess(ni);
        }
    }

    __aicore__ inline void PrepareInDy(LocalTensor<DY_TYPE> &dy, uint64_t dstOffset, uint64_t srcOffset,
        uint64_t r1Factor)
    {
        CopyInRAR(dy, dyGm_, dstOffset, srcOffset, r1Factor);
    }

    __aicore__ inline void PrepareInX(LocalTensor<DY_TYPE> &x, uint64_t dstOffset, uint64_t srcOffset,
        uint64_t r1Factor)
    {
        CopyInRAR(x, xGm_, dstOffset, srcOffset, r1Factor);
    }

    __aicore__ inline void PrepareMean(LocalTensor<float> mean, uint64_t offset, uint64_t a)
    {
        CopyInA(mean, meanGm_, offset, a);
    }

    __aicore__ inline void PrepareRstd(LocalTensor<float> &rstd, uint64_t offset, uint64_t a)
    {
        CopyInA(rstd, rstdGm_, offset, a);
    }

    __aicore__ inline void PrepareInGamma(LocalTensor<WEIGHT_TYPE> &gamma, uint64_t offset, uint64_t a)
    {
        CopyInA(gamma, gammaGm_, offset, a);
    }

    __aicore__ inline void CalcDbetaVF(const __local_mem__ DY_TYPE *data, const __local_mem__ float *out,
        const __local_mem__ float *binaryAdd, uint32_t idx, uint32_t r, const BinaryAddParam &binaryAddParam)
    {
        uint32_t binaryAddQuotient = binaryAddParam.binaryAddQuotient;
        uint32_t binaryAddQuotientOffset = binaryAddQuotient;
        uint32_t binaryAddRemainder = r - binaryAddQuotient;
        uint16_t binaryAddRemainderLoop = CeilDiv(binaryAddRemainder, VL_FP32);
        uint16_t binaryAddQuotientLoop = CeilDiv(binaryAddQuotient, VL_FP32);

        uint16_t binaryAddKLoop = binaryAddParam.binaryAddk;
        uint16_t binaryAddLoop = ((binaryAddQuotient / VL_FP32) / VL_FP32);
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> x;
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> dbeta;

            MicroAPI::RegTensor<float> binaryAddQ;
            MicroAPI::RegTensor<float> binaryAddR;
            MicroAPI::RegTensor<float> vlSum;
            MicroAPI::RegTensor<float> tmp;

            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

            uint32_t sreg0 = binaryAddRemainder;
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(data, binaryAddQ, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(data, binaryAddR, pregLoop, i * VL_FP32 + binaryAddQuotientOffset);
                MicroAPI::Add(tmp, binaryAddQ, binaryAddR, pregLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ, tmp, pregLoop);
                MicroAPI::ReduceSum(vlSum, binaryAddQ, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAdd + i, vlSum, pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                LoadOneTensor<DY_TYPE>(data, x, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                MicroAPI::ReduceSum(vlSum, x, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAdd + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            uint16_t curBinaryAddLoop = binaryAddLoop;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddLoop = curBinaryAddLoop / DIGIT_TWO;
                for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                    MicroAPI::DataCopy(binaryAddQ, ((__local_mem__ float *)binaryAdd) + j * VL_FP32);
                    MicroAPI::DataCopy(binaryAddR,
                        ((__local_mem__ float *)binaryAdd) + (j + curBinaryAddLoop) * VL_FP32);
                    MicroAPI::Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                    MicroAPI::DataCopy((__local_mem__ float *)binaryAdd + j * VL_FP32, binaryAddQ, pregMain);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            }
            uint32_t sreg2 = binaryAddParam.binaryAddLastNum;
            MicroAPI::MaskReg pregLast = MicroAPI::UpdateMask<float>(sreg2);
            MicroAPI::DataCopy(sum, ((__local_mem__ float *)binaryAdd));
            MicroAPI::ReduceSum(dbeta, sum, pregLast);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)out + idxInLevel0Cache), dbeta, pregMerge);
        }
    }

    __aicore__ inline void CalcDbetaFoldVF(const __local_mem__ DY_TYPE *data, const __local_mem__ float *out,
        const __local_mem__ float *binaryAdd, uint32_t idx, uint32_t r, uint32_t tailR,
        const BinaryAddParam &binaryAddParam)
    {
        uint32_t binaryAddQuotient = binaryAddParam.binaryAddQuotient;
        uint32_t binaryAddQuotientOffset = binaryAddQuotient;
        uint32_t binaryAddRemainder = r - binaryAddQuotient;
        uint16_t binaryAddRemainderLoop = CeilDiv(binaryAddRemainder, VL_FP32);
        uint16_t binaryAddQuotientLoop = CeilDiv(binaryAddQuotient, VL_FP32);
        uint32_t tailMain = tailR > binaryAddQuotient ? binaryAddQuotient : tailR;
        uint32_t tailRemainder = tailR > binaryAddQuotient ? tailR - binaryAddQuotient : 0;
        uint16_t binaryAddKLoop = binaryAddParam.binaryAddk;
        uint16_t binaryAddLoop = ((binaryAddQuotient / VL_FP32) / VL_FP32);
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> x;
            MicroAPI::RegTensor<float> xTail;
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> dbeta;

            MicroAPI::RegTensor<float> binaryAddQ;
            MicroAPI::RegTensor<float> binaryAddQTail;
            MicroAPI::RegTensor<float> binaryAddR;
            MicroAPI::RegTensor<float> binaryAddRTail;
            MicroAPI::RegTensor<float> vlSum;
            MicroAPI::RegTensor<float> tmp1;
            MicroAPI::RegTensor<float> tmp2;

            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

            uint32_t sreg0 = binaryAddRemainder;
            uint32_t sreg1 = tailMain;
            uint32_t sreg2 = tailRemainder;
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                MicroAPI::MaskReg pregTailLoop = MicroAPI::UpdateMask<float>(sreg1);
                MicroAPI::MaskReg pregTailReminderLoop = MicroAPI::UpdateMask<float>(sreg2);
                LoadOneTensor<DY_TYPE>(data, binaryAddQ, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(data, binaryAddR, pregLoop, i * VL_FP32 + binaryAddQuotientOffset);
                LoadOneTensor<DY_TYPE>(data, binaryAddQTail, pregTailLoop, i * VL_FP32 + halfXBufOffset_);
                LoadOneTensor<DY_TYPE>(data, binaryAddRTail, pregTailReminderLoop,
                    i * VL_FP32 + binaryAddQuotientOffset + halfXBufOffset_);
                MicroAPI::Add(tmp1, binaryAddQ, binaryAddQTail, pregTailLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ, tmp1, pregTailLoop);
                MicroAPI::Add(tmp2, binaryAddR, binaryAddRTail, pregTailReminderLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddR, tmp2, pregTailReminderLoop);
                MicroAPI::Add(tmp1, binaryAddQ, binaryAddR, pregLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ, tmp1, pregLoop);
                MicroAPI::ReduceSum(vlSum, binaryAddQ, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAdd + i, vlSum, pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                MicroAPI::MaskReg pregTailLoop = MicroAPI::UpdateMask<float>(sreg1);
                LoadOneTensor<DY_TYPE>(data, x, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(data, xTail, pregTailLoop,
                    (i + binaryAddRemainderLoop) * VL_FP32 + halfXBufOffset_);
                MicroAPI::Add(tmp2, x, xTail, pregTailLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(x, tmp2, pregTailLoop);
                MicroAPI::ReduceSum(vlSum, x, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAdd + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            uint16_t curBinaryAddLoop = binaryAddLoop;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddLoop = curBinaryAddLoop / DIGIT_TWO;
                for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                    MicroAPI::DataCopy(binaryAddQ, ((__local_mem__ float *)binaryAdd) + j * VL_FP32);
                    MicroAPI::DataCopy(binaryAddR,
                        ((__local_mem__ float *)binaryAdd) + (j + curBinaryAddLoop) * VL_FP32);
                    MicroAPI::Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                    MicroAPI::DataCopy(((__local_mem__ float *)binaryAdd) + j * VL_FP32, binaryAddQ, pregMain);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            }
            uint32_t sreg3 = binaryAddParam.binaryAddLastNum;
            MicroAPI::MaskReg pregLast = MicroAPI::UpdateMask<float>(sreg3);
            MicroAPI::DataCopy(sum, ((__local_mem__ float *)binaryAdd));
            MicroAPI::ReduceSum(dbeta, sum, pregLast);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)out) + idxInLevel0Cache, dbeta, pregMerge);
        }
    }

    __aicore__ inline void CalcDbetaLessThanVL64VF(const __local_mem__ DY_TYPE *data, const __local_mem__ float *out,
        uint32_t idx, uint32_t r)
    {
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> x;
            MicroAPI::RegTensor<float> sum;
            uint32_t sreg0 = r;
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
            LoadOneTensor<DY_TYPE>(data, x, pregLoop, 0);
            MicroAPI::ReduceSum(sum, x, pregLoop);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)out) + idxInLevel0Cache, sum, pregMerge);
        }
    }

    __aicore__ inline void CalcDbetaLessThanVL64FoldVF(const __local_mem__ DY_TYPE *data,
        const __local_mem__ float *out, uint32_t idx, uint32_t r, uint32_t tailR)
    {
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> x;
            MicroAPI::RegTensor<float> xTail;
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> tmp;
            uint32_t sreg0 = r;
            uint32_t sreg1 = tailR;
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
            MicroAPI::MaskReg pregTailLoop = MicroAPI::UpdateMask<float>(sreg1);
            LoadOneTensor<DY_TYPE>(data, x, pregLoop, 0);
            LoadOneTensor<DY_TYPE>(data, xTail, pregTailLoop, halfXBufOffset_);
            MicroAPI::Add(tmp, x, xTail, pregTailLoop);
            MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(x, tmp, pregTailLoop);
            MicroAPI::ReduceSum(sum, x, pregLoop);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)out) + idxInLevel0Cache, sum, pregMerge);
        }
    }

    __aicore__ inline void CalcDbeta(LocalTensor<DY_TYPE> &dy, LocalTensor<float> &dbeta, LocalTensor<float> &binaryAdd,
        uint32_t idx, uint32_t r, const BinaryAddParam &binaryAddParam)
    {
        if (r <= VL_FP32) {
            CalcDbetaLessThanVL64VF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(), (__local_mem__ float *)dbeta.GetPhyAddr(),
                idx, r);
        } else {
            CalcDbetaVF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(), (__local_mem__ float *)dbeta.GetPhyAddr(),
                (__local_mem__ float *)binaryAdd.GetPhyAddr(), idx, r, binaryAddParam);
        }
    }

    __aicore__ inline void CalcDbetaFold(LocalTensor<DY_TYPE> &dy, LocalTensor<float> &dbeta,
        LocalTensor<float> &binaryAdd, uint32_t idx, uint32_t r, uint32_t tailR, const BinaryAddParam &binaryAddParam)
    {
        if (r <= VL_FP32) {
            CalcDbetaLessThanVL64FoldVF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(),
                (__local_mem__ float *)dbeta.GetPhyAddr(), idx, r, tailR);
        } else {
            CalcDbetaFoldVF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(), (__local_mem__ float *)dbeta.GetPhyAddr(),
                (__local_mem__ float *)binaryAdd.GetPhyAddr(), idx, r, tailR, binaryAddParam);
        }
    }

    __aicore__ inline void CalcDgammaVF(const __local_mem__ DY_TYPE *dyAddr, const __local_mem__ DY_TYPE *xAddr,
        const __local_mem__ float *meanAddr, const __local_mem__ float *rstdAddr, const __local_mem__ float *dgammaAddr,
        const __local_mem__ float *binaryAdd, uint32_t idx, uint32_t r, const BinaryAddParam &binaryAddParam)
    {
        uint32_t binaryAddQuotient = binaryAddParam.binaryAddQuotient;
        uint32_t binaryAddQuotientOffset = binaryAddQuotient;
        uint32_t binaryAddRemainder = r - binaryAddQuotient;
        uint16_t binaryAddRemainderLoop = CeilDiv(binaryAddRemainder, VL_FP32);
        uint16_t binaryAddQuotientLoop = CeilDiv(binaryAddQuotient, VL_FP32);

        uint16_t binaryAddKLoop = binaryAddParam.binaryAddk;
        uint16_t binaryAddLoop = ((binaryAddQuotient / VL_FP32) / VL_FP32);
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> dgamma;
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> binaryAddQ1;
            MicroAPI::RegTensor<float> binaryAddR1;
            MicroAPI::RegTensor<float> binaryAddQ2;
            MicroAPI::RegTensor<float> binaryAddR2;
            MicroAPI::RegTensor<float> vlSum;
            MicroAPI::RegTensor<float> tmp;

            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float *)(rstdAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));
            uint32_t sreg0 = binaryAddRemainder;
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR1, pregLoop, i * VL_FP32 + binaryAddQuotientOffset);

                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ2, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddR2, pregLoop, i * VL_FP32 + binaryAddQuotientOffset);

                MicroAPI::Sub(binaryAddQ2, binaryAddQ2, meanValue, pregMain);
                MicroAPI::Mul(binaryAddQ2, binaryAddQ2, binaryAddQ1, pregMain);

                MicroAPI::Sub(binaryAddR2, binaryAddR2, meanValue, pregLoop);
                MicroAPI::Mul(binaryAddR2, binaryAddR2, binaryAddR1, pregLoop);

                MicroAPI::Mul(binaryAddQ1, rstdValue, binaryAddQ2, pregMain);
                MicroAPI::Mul(binaryAddR1, rstdValue, binaryAddR2, pregLoop);
                MicroAPI::Add(tmp, binaryAddQ1, binaryAddR1, pregLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ1, tmp, pregLoop);
                MicroAPI::ReduceSum(vlSum, binaryAddQ1, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAdd + i, vlSum, pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ2, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                MicroAPI::Sub(binaryAddQ2, binaryAddQ2, meanValue, pregMain);
                MicroAPI::Mul(binaryAddQ2, binaryAddQ2, binaryAddQ1, pregMain);
                MicroAPI::Mul(binaryAddQ1, rstdValue, binaryAddQ2, pregMain);
                MicroAPI::ReduceSum(vlSum, binaryAddQ1, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAdd + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            uint16_t curBinaryAddLoop = binaryAddLoop;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddLoop = curBinaryAddLoop / DIGIT_TWO;
                for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                    MicroAPI::DataCopy(binaryAddQ1, (__local_mem__ float *)(binaryAdd) + j * VL_FP32);
                    MicroAPI::DataCopy(binaryAddR1,
                        (__local_mem__ float *)(binaryAdd) + (j + curBinaryAddLoop) * VL_FP32);
                    MicroAPI::Add(binaryAddQ1, binaryAddQ1, binaryAddR1, pregMain);
                    MicroAPI::DataCopy(((__local_mem__ float *)binaryAdd) + j * VL_FP32, binaryAddQ1, pregMain);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            }
            uint32_t sreg2 = binaryAddParam.binaryAddLastNum;
            MicroAPI::MaskReg pregLast = MicroAPI::UpdateMask<float>(sreg2);
            MicroAPI::DataCopy(sum, (__local_mem__ float *)(binaryAdd));
            MicroAPI::ReduceSum(dgamma, sum, pregLast);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)dgammaAddr) + idxInLevel0Cache, dgamma, pregMerge);
        }
    }

    __aicore__ inline void CalcDgammaFoldVF(const __local_mem__ DY_TYPE *dyAddr, const __local_mem__ DY_TYPE *xAddr,
        const __local_mem__ float *meanAddr, const __local_mem__ float *rstdAddr, const __local_mem__ float *dgammaAddr,
        const __local_mem__ float *binaryAdd, uint32_t idx, uint32_t r, uint32_t tailR,
        const BinaryAddParam &binaryAddParam)
    {
        uint32_t binaryAddQuotient = binaryAddParam.binaryAddQuotient;
        uint32_t binaryAddQuotientOffset = binaryAddQuotient;
        uint32_t binaryAddRemainder = r - binaryAddQuotient;
        uint16_t binaryAddRemainderLoop = CeilDiv(binaryAddRemainder, VL_FP32);
        uint16_t binaryAddQuotientLoop = CeilDiv(binaryAddQuotient, VL_FP32);
        uint32_t tailMain = tailR > binaryAddQuotient ? binaryAddQuotient : tailR;
        uint32_t tailRemainder = tailR > binaryAddQuotient ? tailR - binaryAddQuotient : 0;
        uint16_t binaryAddKLoop = binaryAddParam.binaryAddk;
        uint16_t binaryAddLoop = ((binaryAddQuotient / VL_FP32) / VL_FP32);
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> dgamma;
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> binaryAddQ1;
            MicroAPI::RegTensor<float> binaryAddR1;
            MicroAPI::RegTensor<float> binaryAddQ2;
            MicroAPI::RegTensor<float> binaryAddR2;
            MicroAPI::RegTensor<float> binaryAddQ1Tail;
            MicroAPI::RegTensor<float> binaryAddR1Tail;
            MicroAPI::RegTensor<float> binaryAddQ2Tail;
            MicroAPI::RegTensor<float> binaryAddR2Tail;
            MicroAPI::RegTensor<float> vlSum;
            MicroAPI::RegTensor<float> tmp1;
            MicroAPI::RegTensor<float> tmp2;

            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float *)(rstdAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));
            uint32_t sreg0 = binaryAddRemainder;
            uint32_t sreg1 = tailMain;
            uint32_t sreg2 = tailRemainder;
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                MicroAPI::MaskReg pregTailLoop = MicroAPI::UpdateMask<float>(sreg1);
                MicroAPI::MaskReg pregTailReminderLoop = MicroAPI::UpdateMask<float>(sreg2);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR1, pregLoop, i * VL_FP32 + binaryAddQuotientOffset);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1Tail, pregTailLoop, i * VL_FP32 + halfXBufOffset_);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR1Tail, pregTailReminderLoop,
                    i * VL_FP32 + binaryAddQuotientOffset + halfXBufOffset_);

                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ2, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddR2, pregLoop, i * VL_FP32 + binaryAddQuotientOffset);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ2Tail, pregTailLoop, i * VL_FP32 + halfXBufOffset_);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddR2Tail, pregTailReminderLoop,
                    i * VL_FP32 + binaryAddQuotientOffset + halfXBufOffset_);

                MicroAPI::Sub(binaryAddQ2, binaryAddQ2, meanValue, pregMain);
                MicroAPI::Mul(binaryAddQ2, binaryAddQ2, binaryAddQ1, pregMain);

                MicroAPI::Sub(binaryAddR2, binaryAddR2, meanValue, pregLoop);
                MicroAPI::Mul(binaryAddR2, binaryAddR2, binaryAddR1, pregLoop);

                MicroAPI::Mul(binaryAddQ1, rstdValue, binaryAddQ2, pregMain);
                MicroAPI::Mul(binaryAddR1, rstdValue, binaryAddR2, pregLoop);

                MicroAPI::Sub(binaryAddQ2Tail, binaryAddQ2Tail, meanValue, pregTailLoop);
                MicroAPI::Mul(binaryAddQ2Tail, binaryAddQ2Tail, binaryAddQ1Tail, pregTailLoop);

                MicroAPI::Sub(binaryAddR2Tail, binaryAddR2Tail, meanValue, pregTailReminderLoop);
                MicroAPI::Mul(binaryAddR2Tail, binaryAddR2Tail, binaryAddR1Tail, pregTailReminderLoop);

                MicroAPI::Mul(binaryAddQ1Tail, rstdValue, binaryAddQ2Tail, pregTailLoop);
                MicroAPI::Mul(binaryAddR1Tail, rstdValue, binaryAddR2Tail, pregTailReminderLoop);

                MicroAPI::Add(tmp1, binaryAddQ1, binaryAddQ1Tail, pregTailLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ1, tmp1, pregTailLoop);
                MicroAPI::Add(tmp2, binaryAddR1, binaryAddR1Tail, pregTailReminderLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddR1, tmp2, pregTailReminderLoop);
                MicroAPI::Add(tmp1, binaryAddQ1, binaryAddR1, pregLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ1, tmp1, pregLoop);
                MicroAPI::ReduceSum(vlSum, binaryAddQ1, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAdd + i, vlSum, pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                MicroAPI::MaskReg pregTailLoop = MicroAPI::UpdateMask<float>(sreg1);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1Tail, pregTailLoop,
                    (i + binaryAddRemainderLoop) * VL_FP32 + halfXBufOffset_);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ2, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ2Tail, pregTailLoop,
                    (i + binaryAddRemainderLoop) * VL_FP32 + halfXBufOffset_);
                MicroAPI::Sub(binaryAddQ2, binaryAddQ2, meanValue, pregMain);
                MicroAPI::Mul(binaryAddQ2, binaryAddQ2, binaryAddQ1, pregMain);
                MicroAPI::Mul(binaryAddQ1, rstdValue, binaryAddQ2, pregMain);

                MicroAPI::Sub(binaryAddQ2Tail, binaryAddQ2Tail, meanValue, pregTailLoop);
                MicroAPI::Mul(binaryAddQ2Tail, binaryAddQ2Tail, binaryAddQ1Tail, pregTailLoop);
                MicroAPI::Mul(binaryAddQ1Tail, rstdValue, binaryAddQ2Tail, pregTailLoop);
                MicroAPI::Add(tmp1, binaryAddQ1, binaryAddQ1Tail, pregTailLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ1, tmp1, pregTailLoop);

                MicroAPI::ReduceSum(vlSum, binaryAddQ1, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAdd + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            uint16_t curBinaryAddLoop = binaryAddLoop;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddLoop = curBinaryAddLoop / DIGIT_TWO;
                for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                    MicroAPI::DataCopy(binaryAddQ1, (__local_mem__ float *)(binaryAdd) + j * VL_FP32);
                    MicroAPI::DataCopy(binaryAddR1,
                        (__local_mem__ float *)(binaryAdd) + (j + curBinaryAddLoop) * VL_FP32);
                    MicroAPI::Add(binaryAddQ1, binaryAddQ1, binaryAddR1, pregMain);
                    MicroAPI::DataCopy(((__local_mem__ float *)binaryAdd) + j * VL_FP32, binaryAddQ1, pregMain);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            }
            uint32_t sreg3 = binaryAddParam.binaryAddLastNum;
            MicroAPI::MaskReg pregLast = MicroAPI::UpdateMask<float>(sreg3);
            MicroAPI::DataCopy(sum, (__local_mem__ float *)(binaryAdd));
            MicroAPI::ReduceSum(dgamma, sum, pregLast);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)dgammaAddr) + idxInLevel0Cache, dgamma, pregMerge);
        }
    }

    __aicore__ inline void CalcDgammaLessThanVLVF(const __local_mem__ DY_TYPE *dyAddr,
        const __local_mem__ DY_TYPE *xAddr, const __local_mem__ float *meanAddr, const __local_mem__ float *rstdAddr,
        const __local_mem__ float *dgammaAddr, uint32_t idx, uint32_t r)
    {
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> x;
            MicroAPI::RegTensor<float> y;
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> sum;
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0 = r;
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float *)(rstdAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));

            MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
            LoadOneTensor<DY_TYPE>(dyAddr, y, pregLoop, 0);
            LoadOneTensor<DY_TYPE>(xAddr, x, pregLoop, 0);
            MicroAPI::Sub(x, x, meanValue, pregLoop);
            MicroAPI::Mul(x, y, x, pregLoop);
            MicroAPI::Mul(x, x, rstdValue, pregLoop);
            MicroAPI::ReduceSum(sum, x, pregLoop);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)dgammaAddr) + idxInLevel0Cache, sum, pregMerge);
        }
    }

    __aicore__ inline void CalcDgammaLessThanVLFoldVF(const __local_mem__ DY_TYPE *dyAddr,
        const __local_mem__ DY_TYPE *xAddr, const __local_mem__ float *meanAddr, const __local_mem__ float *rstdAddr,
        const __local_mem__ float *dgammaAddr, uint32_t idx, uint32_t r, uint32_t tailR)
    {
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> x;
            MicroAPI::RegTensor<float> y;
            MicroAPI::RegTensor<float> xTail;
            MicroAPI::RegTensor<float> yTail;
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> tmp;
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0 = r;
            uint32_t sreg1 = tailR;
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float *)(rstdAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));

            MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
            MicroAPI::MaskReg pregTailLoop = MicroAPI::UpdateMask<float>(sreg1);
            LoadOneTensor<DY_TYPE>(dyAddr, y, pregLoop, 0);
            LoadOneTensor<DY_TYPE>(dyAddr, yTail, pregTailLoop, halfXBufOffset_);
            LoadOneTensor<DY_TYPE>(xAddr, x, pregLoop, 0);
            LoadOneTensor<DY_TYPE>(xAddr, xTail, pregTailLoop, halfXBufOffset_);

            MicroAPI::Sub(x, x, meanValue, pregLoop);
            MicroAPI::Mul(x, y, x, pregLoop);
            MicroAPI::Mul(x, x, rstdValue, pregLoop);

            MicroAPI::Sub(xTail, xTail, meanValue, pregTailLoop);
            MicroAPI::Mul(xTail, yTail, xTail, pregTailLoop);
            MicroAPI::Mul(xTail, xTail, rstdValue, pregTailLoop);
            MicroAPI::Add(tmp, x, xTail, pregTailLoop);
            MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(x, tmp, pregTailLoop);
            MicroAPI::ReduceSum(sum, x, pregLoop);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)dgammaAddr) + idxInLevel0Cache, sum, pregMerge);
        }
    }

    __aicore__ inline void CalcDgamma(LocalTensor<DY_TYPE> &dy, LocalTensor<DY_TYPE> &x, LocalTensor<float> &mean,
        LocalTensor<float> &rstd, LocalTensor<float> &dgamma, LocalTensor<float> &binaryAdd, uint32_t idx, uint32_t r,
        const BinaryAddParam &binaryAddParam)
    {
        if (r <= VL_FP32) {
            CalcDgammaLessThanVLVF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(), (__local_mem__ DY_TYPE *)x.GetPhyAddr(),
                (__local_mem__ float *)mean.GetPhyAddr(), (__local_mem__ float *)rstd.GetPhyAddr(),
                (__local_mem__ float *)dgamma.GetPhyAddr(), idx, r);
        } else {
            CalcDgammaVF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(), (__local_mem__ DY_TYPE *)x.GetPhyAddr(),
                (__local_mem__ float *)mean.GetPhyAddr(), (__local_mem__ float *)rstd.GetPhyAddr(),
                (__local_mem__ float *)dgamma.GetPhyAddr(), (__local_mem__ float *)binaryAdd.GetPhyAddr(), idx, r,
                binaryAddParam);
        }
    }

    __aicore__ inline void CalcDgammaFold(LocalTensor<DY_TYPE> &dy, LocalTensor<DY_TYPE> &x, LocalTensor<float> &mean,
        LocalTensor<float> &rstd, LocalTensor<float> &dgamma, LocalTensor<float> &binaryAdd, uint32_t idx, uint32_t r,
        uint32_t tailR, const BinaryAddParam &binaryAddParam)
    {
        if (r <= VL_FP32) {
            CalcDgammaLessThanVLFoldVF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(),
                (__local_mem__ DY_TYPE *)x.GetPhyAddr(), (__local_mem__ float *)mean.GetPhyAddr(),
                (__local_mem__ float *)rstd.GetPhyAddr(), (__local_mem__ float *)dgamma.GetPhyAddr(), idx, r, tailR);
        } else {
            CalcDgammaFoldVF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(), (__local_mem__ DY_TYPE *)x.GetPhyAddr(),
                (__local_mem__ float *)mean.GetPhyAddr(), (__local_mem__ float *)rstd.GetPhyAddr(),
                (__local_mem__ float *)dgamma.GetPhyAddr(), (__local_mem__ float *)binaryAdd.GetPhyAddr(), idx, r,
                tailR, binaryAddParam);
        }
    }

    __aicore__ inline void CalcDx(LocalTensor<DY_TYPE> &x, LocalTensor<DY_TYPE> &dy, LocalTensor<float> &rstd,
        LocalTensor<float> &mean, LocalTensor<WEIGHT_TYPE> &gamma, LocalTensor<float> &dgamma,
        LocalTensor<float> &dbeta, uint32_t xFactor, uint32_t r)
    {
        __local_mem__ DY_TYPE *xAddr = (__local_mem__ DY_TYPE *)x.GetPhyAddr();
        __local_mem__ DY_TYPE *dyAddr = (__local_mem__ DY_TYPE *)dy.GetPhyAddr();
        __local_mem__ float *rstdAddr = (__local_mem__ float *)rstd.GetPhyAddr();
        __local_mem__ float *meanAddr = (__local_mem__ float *)mean.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE *gammaAddr = (__local_mem__ WEIGHT_TYPE *)gamma.GetPhyAddr();
        __local_mem__ float *dgammaAddr = (__local_mem__ float *)dgamma.GetPhyAddr();
        __local_mem__ float *dbetaAddr = (__local_mem__ float *)dbeta.GetPhyAddr();

        uint16_t VL = VL_FP32;
        uint16_t loopTimes = (xFactor + VL - 1) / VL;
        float hRecipValue = 1.0f / (float)r;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> gammaValue;
            MicroAPI::RegTensor<float> dgammaValue;
            MicroAPI::RegTensor<float> dbetaValue;
            MicroAPI::RegTensor<float> rstdMulSubDy;
            MicroAPI::MaskReg pregR;
            MicroAPI::RegTensor<float> dy;
            MicroAPI::RegTensor<float> x;
            MicroAPI::RegTensor<float> mulDgamma;
            MicroAPI::RegTensor<float> addDbeta;
            MicroAPI::RegTensor<float> divH;
            MicroAPI::RegTensor<float> subDy;
            MicroAPI::RegTensor<float> dx;
            MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float *)(rstdAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                (__local_mem__ float *)(dgammaAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue, (__local_mem__ float *)(dbetaAddr));
            LoadOneElement<WEIGHT_TYPE>(gammaAddr, gammaValue, preg, 0);

            uint32_t sregMask = (uint32_t)xFactor;
            for (uint16_t j = 0; j < loopTimes; j++) {
                pregR = MicroAPI::UpdateMask<float>(sregMask);

                LoadOneTensor<DY_TYPE>(xAddr, x, pregR, VL * j);
                LoadOneTensor<DY_TYPE>(dyAddr, dy, pregR, VL * j);
                MicroAPI::Sub(x, x, meanValue, pregR);
                MicroAPI::Mul(x, x, rstdValue, pregR);

                MicroAPI::Mul(mulDgamma, x, dgammaValue, pregR);
                MicroAPI::Add(addDbeta, mulDgamma, dbetaValue, preg);
                MicroAPI::Muls(divH, addDbeta, hRecipValue, pregR);
                MicroAPI::Sub(subDy, dy, divH, pregR);
                MicroAPI::Mul(rstdMulSubDy, rstdValue, subDy, pregR);
                MicroAPI::Mul(dx, gammaValue, rstdMulSubDy, pregR);
                StoreOneTensor<DY_TYPE>(xAddr, dx, pregR, VL * j);
            }
        }
    }

    __aicore__ inline void ReSaveDGammaDBeta(LocalTensor<float> &dgamma, LocalTensor<float> &dbeta)
    {
        if constexpr (IsSameType<WEIGHT_TYPE, half>::value || IsSameType<WEIGHT_TYPE, bfloat16_t>::value) {
            __local_mem__ float *dgammaAddr = (__local_mem__ float *)dgamma.GetPhyAddr();
            __local_mem__ float *dbetaAddr = (__local_mem__ float *)dbeta.GetPhyAddr();

            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<float> dgammaValue;
                MicroAPI::RegTensor<float> dbetaValue;
                MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                    (__local_mem__ float *)(dgammaAddr));
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue,
                    (__local_mem__ float *)(dbetaAddr));
                StoreOneElement<WEIGHT_TYPE>(dbetaAddr, dbetaValue, pregMerge, 0);
                StoreOneElement<WEIGHT_TYPE>(dgammaAddr, dgammaValue, pregMerge, 0);
            }
        }
        return;
    }

    __aicore__ inline void CopyOutDx(LocalTensor<DY_TYPE> &dx, uint64_t offset, uint32_t r1Factor)
    {
        CopyOutRAR(dx, dxGm_, offset, r1Factor);
    }

    __aicore__ inline void CopyOutDgamma(LocalTensor<WEIGHT_TYPE> &dgamma, uint64_t meanOffset, uint32_t a)
    {
        CopyOutA(dgamma, dgammaGm_, meanOffset, a);
    }

    __aicore__ inline void CopyOutDbeta(LocalTensor<WEIGHT_TYPE> &dbeta, uint64_t meanOffset, uint32_t a)
    {
        CopyOutA(dbeta, dbetaGm_, meanOffset, a);
    }

    __aicore__ inline void CopyInRAR(LocalTensor<DY_TYPE> &localTensor, GlobalTensor<DY_TYPE> &globalTensor,
        uint64_t localOffset, uint64_t globalOffset, uint64_t r1Factor)
    {
        // RAR -> AR，R轴通过Compact Mode补pad到block对齐。
        DataCopyPadExtParams<DY_TYPE> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = r1Factor;
        copyInParams.blockLen = r0Dim_ * sizeof(DY_TYPE);
        copyInParams.srcStride = (aDim_ - 1) * r0Dim_ * sizeof(DY_TYPE);
        copyInParams.dstStride = 0;
        DataCopyPad<DY_TYPE, PaddingMode::Compact>(localTensor[localOffset], globalTensor[globalOffset], copyInParams,
            dataCopyPadExtParams);
    }

    template <typename U>
    __aicore__ inline void CopyInA(LocalTensor<U> &localTensor, GlobalTensor<U> &globalTensor, uint64_t offset,
        uint64_t a)
    {
        DataCopyPadExtParams<U> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = a * sizeof(U);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad<U, PaddingMode::Compact>(localTensor, globalTensor[offset], copyInParams, dataCopyPadExtParams);
    }

    __aicore__ inline void CopyOutRAR(LocalTensor<DY_TYPE> &localTensor, GlobalTensor<DY_TYPE> &globalTensor,
        uint64_t offset, uint64_t r1Factor)
    {
        // RR -> RAR
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = r1Factor;
        copyOutParams.blockLen = r0Dim_ * sizeof(DY_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = (aDim_ - 1) * r0Dim_ * sizeof(DY_TYPE);
        DataCopyPad<DY_TYPE, PaddingMode::Compact>(globalTensor[offset], localTensor, copyOutParams);
    }

    __aicore__ inline void CopyOutA(LocalTensor<WEIGHT_TYPE> &localTensor, GlobalTensor<WEIGHT_TYPE> &globalTensor,
        uint64_t offset, uint64_t a)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = a * sizeof(WEIGHT_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Compact>(globalTensor[offset], localTensor, copyOutParams);
    }

private:
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> binaryAddBuf_;
    TBuf<TPosition::VECCALC> cacheBuf_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;

    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, rstdGm_;

    event_t V_MTE3_EVENT;
    event_t MTE3_MTE2_EVENT;

    uint32_t r1Dim_{ 0 };
    uint32_t aDim_{ 0 };
    uint32_t r0Dim_{ 0 };
    uint32_t rAlign_{ 0 };
    uint32_t binaryAddQuotient_{ 0 };
    uint32_t binaryAddK_{ 0 };
    uint32_t binaryAddLastNum_{ 0 };
    uint32_t aTailCoreNum_{ 0 };
    uint32_t aDimTail_{ 0 };
    uint32_t gmOffset_{ 0 };
    uint32_t aDimLoopNum_{ 0 };
    uint32_t dyBufSize_{ 0 };
    uint32_t xBufSize_{ 0 };
    uint32_t halfXBufOffset_{ 0 };
    uint32_t xBufElemNum_{ 0 };
    uint32_t meanBufSize_{ 0 };
    uint32_t meanBufElemNum_{ 0 };
    uint32_t gammaBufSize_{ 0 };
    uint32_t gammaBufElemNum_{ 0 };
    uint32_t binaryAddBufSize_{ 0 };
    uint32_t binaryAddBufElemNum_{ 0 };
    uint32_t blockNum_{ 0 };
    uint64_t ubRDimLoopNum_{ 0 };
    uint64_t nFactor_{ 0 };
    uint64_t tailNFactor_{ 0 };
    uint64_t ubRDimTailFactor_{ 0 };
    uint64_t ubRDimTailTailFactor_{ 0 };

    uint32_t tailBinaryAddQuotient_{ 0 };
    uint32_t tailBinaryAddK_{ 0 };
    uint32_t tailBinaryAddLastNum_{ 0 };
    uint64_t ubRDimTailLoopNum_{ 0 };
    BinaryAddParam binaryAddParam_;
    BinaryAddParam tailBinaryAddParam_;
    LocalTensor<float> dBetaLocal_;
    LocalTensor<float> dGammaLocal_;
    LocalTensor<float> cacheLocal_;
    LocalTensor<float> dBetaCacheLocal_;
    LocalTensor<float> dGammaCacheLocal_;
    LocalTensor<float> dBetaFoldCacheLocal_;
    LocalTensor<float> dGammaFoldCacheLocal_;
    LocalTensor<float> binaryAddTensor_;
    LocalTensor<float> rstdLocal_;
    LocalTensor<float> meanLocal_;
    LocalTensor<WEIGHT_TYPE> gammaLocal_;
};
} // namespace BatchNormGrad
#endif // BATCH_NORM_GRAD_SPLIT_R1_REGBASE_H
