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

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1>
class BatchNormGradRARSplitR1 {
public:
    __aicore__ inline BatchNormGradRARSplitR1(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline uint32_t CalcBinaryAddLastNum(uint32_t binaryAddQuotient)
    {
        if (binaryAddQuotient <= VL_FP32 * VL_FP32) {
            return binaryAddQuotient / VL_FP32;
        }
        return VL_FP32;
    }

    __aicore__ inline void Init(__gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean, __gm__ uint8_t* rstd,
                                __gm__ uint8_t* gamma, __gm__ uint8_t* dx, __gm__ uint8_t* dgamma,
                                __gm__ uint8_t* dbeta, __gm__ uint8_t* workspace,
                                const BatchNormGradRARRecomputeTilingData* tilingData)
    {
        const BatchNormGradBaseTilingData& baseTilingData = tilingData->baseTilingData;
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

        aDimLoopNum_ = GetBlockIdx() < baseTilingData.tailBlockNum ? baseTilingData.tailBlockDim :
                                                                     baseTilingData.formerBlockDim;

        tailBinaryAddParam_.binaryAddQuotient = binaryAddParam_.binaryAddQuotient >> 1;
        tailBinaryAddParam_.binaryAddk = binaryAddParam_.binaryAddk > 1 ? binaryAddParam_.binaryAddk - 1 : 0;
        tailBinaryAddParam_.binaryAddLastNum = CalcBinaryAddLastNum(tailBinaryAddParam_.binaryAddQuotient);

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dy) + gmOffset_ * r0Dim_);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(x) + gmOffset_ * r0Dim_);
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean) + gmOffset_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(gamma) + gmOffset_);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dx) + gmOffset_ * r0Dim_);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dgamma) + gmOffset_);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dbeta) + gmOffset_);

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

    __aicore__ inline bool IsNeedUpdateLevel2Cache(uint64_t basiBlockIdx) { return ((basiBlockIdx + 1) & 0xffff) == 0; }

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
            uint64_t updateIdx = ((basiBlockIdx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1)) + 1) -
                                 FOLD_CACHE_CAPACITY;
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

    __aicore__ inline void CustomReduceSum(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                           uint32_t idx)
    {
        __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
        __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> x1;
            Reg::RegTensor<float> x2;
            Reg::RegTensor<float> x3;
            Reg::RegTensor<float> x4;
            Reg::RegTensor<float> sum1;
            Reg::RegTensor<float> sum2;
            Reg::RegTensor<float> sum12;
            Reg::RegTensor<float> vlSum;

            Reg::MaskReg pregAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::LoadAlign(x1, (__ubuf__ float*)(src));
            Reg::LoadAlign(x2, (__ubuf__ float*)(src) + 1 * VL_FP32);
            Reg::LoadAlign(x3, (__ubuf__ float*)(src) + DIGIT_TWO * VL_FP32);
            Reg::LoadAlign(x4, (__ubuf__ float*)(src) + DIGIT_THREE * VL_FP32);
            Reg::Add(sum1, x1, x3, pregAll);
            Reg::Add(sum2, x2, x4, pregAll);
            Reg::Add(sum12, sum1, sum2, pregAll);
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, sum12, pregAll);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dst + idx, vlSum, pregAll);
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
        SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
        WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
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

    __aicore__ inline void PrepareInDy(LocalTensor<DY_TYPE>& dy, uint64_t dstOffset, uint64_t srcOffset,
                                       uint64_t r1Factor)
    {
        CopyInRAR(dy, dyGm_, dstOffset, srcOffset, r1Factor);
    }

    __aicore__ inline void PrepareInX(LocalTensor<DY_TYPE>& x, uint64_t dstOffset, uint64_t srcOffset,
                                      uint64_t r1Factor)
    {
        CopyInRAR(x, xGm_, dstOffset, srcOffset, r1Factor);
    }

    __aicore__ inline void PrepareMean(LocalTensor<float> mean, uint64_t offset, uint64_t a)
    {
        CopyInA(mean, meanGm_, offset, a);
    }

    __aicore__ inline void PrepareRstd(LocalTensor<float>& rstd, uint64_t offset, uint64_t a)
    {
        CopyInA(rstd, rstdGm_, offset, a);
    }

    __aicore__ inline void PrepareInGamma(LocalTensor<WEIGHT_TYPE>& gamma, uint64_t offset, uint64_t a)
    {
        CopyInA(gamma, gammaGm_, offset, a);
    }

    __aicore__ inline void CalcDbetaVF(const __ubuf__ DY_TYPE* data, const __ubuf__ float* out,
                                       const __ubuf__ float* binaryAdd, uint32_t idx, uint32_t r,
                                       const BinaryAddParam& binaryAddParam)
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
            Reg::RegTensor<float> x;
            Reg::RegTensor<float> sum;
            Reg::RegTensor<float> dbeta;

            Reg::RegTensor<float> binaryAddQ;
            Reg::RegTensor<float> binaryAddR;
            Reg::RegTensor<float> vlSum;
            Reg::RegTensor<float> tmp;

            Reg::MaskReg pregMain = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();

            uint32_t sreg0 = binaryAddRemainder;
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(data, binaryAddQ, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(data, binaryAddR, pregLoop, i * VL_FP32 + binaryAddQuotientOffset);
                Reg::Add(tmp, binaryAddQ, binaryAddR, pregLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ, tmp, pregLoop);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)binaryAdd + i, vlSum,
                                                                               pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                LoadOneTensor<DY_TYPE>(data, x, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, x, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)binaryAdd + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            uint16_t curBinaryAddLoop = binaryAddLoop;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddLoop = curBinaryAddLoop / DIGIT_TWO;
                for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                    Reg::LoadAlign(binaryAddQ, ((__ubuf__ float*)binaryAdd) + j * VL_FP32);
                    Reg::LoadAlign(binaryAddR, ((__ubuf__ float*)binaryAdd) + (j + curBinaryAddLoop) * VL_FP32);
                    Reg::Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                    Reg::StoreAlign((__ubuf__ float*)binaryAdd + j * VL_FP32, binaryAddQ, pregMain);
                }
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            }
            uint32_t sreg2 = binaryAddParam.binaryAddLastNum;
            Reg::MaskReg pregLast = Reg::UpdateMask<float>(sreg2);
            Reg::LoadAlign(sum, ((__ubuf__ float*)binaryAdd));
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(dbeta, sum, pregLast);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)out + idxInLevel0Cache),
                                                                           dbeta, pregMerge);
        }
    }

    __aicore__ inline void CalcDbetaFoldVF(const __ubuf__ DY_TYPE* data, const __ubuf__ float* out,
                                           const __ubuf__ float* binaryAdd, uint32_t idx, uint32_t r, uint32_t tailR,
                                           const BinaryAddParam& binaryAddParam)
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
            Reg::RegTensor<float> x;
            Reg::RegTensor<float> xTail;
            Reg::RegTensor<float> sum;
            Reg::RegTensor<float> dbeta;

            Reg::RegTensor<float> binaryAddQ;
            Reg::RegTensor<float> binaryAddQTail;
            Reg::RegTensor<float> binaryAddR;
            Reg::RegTensor<float> binaryAddRTail;
            Reg::RegTensor<float> vlSum;
            Reg::RegTensor<float> tmp1;
            Reg::RegTensor<float> tmp2;

            Reg::MaskReg pregMain = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();

            uint32_t sreg0 = binaryAddRemainder;
            uint32_t sreg1 = tailMain;
            uint32_t sreg2 = tailRemainder;
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
                Reg::MaskReg pregTailLoop = Reg::UpdateMask<float>(sreg1);
                Reg::MaskReg pregTailReminderLoop = Reg::UpdateMask<float>(sreg2);
                LoadOneTensor<DY_TYPE>(data, binaryAddQ, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(data, binaryAddR, pregLoop, i * VL_FP32 + binaryAddQuotientOffset);
                LoadOneTensor<DY_TYPE>(data, binaryAddQTail, pregTailLoop, i * VL_FP32 + halfXBufOffset_);
                LoadOneTensor<DY_TYPE>(data, binaryAddRTail, pregTailReminderLoop,
                                       i * VL_FP32 + binaryAddQuotientOffset + halfXBufOffset_);
                Reg::Add(tmp1, binaryAddQ, binaryAddQTail, pregTailLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ, tmp1, pregTailLoop);
                Reg::Add(tmp2, binaryAddR, binaryAddRTail, pregTailReminderLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddR, tmp2, pregTailReminderLoop);
                Reg::Add(tmp1, binaryAddQ, binaryAddR, pregLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ, tmp1, pregLoop);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)binaryAdd + i, vlSum,
                                                                               pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                Reg::MaskReg pregTailLoop = Reg::UpdateMask<float>(sreg1);
                LoadOneTensor<DY_TYPE>(data, x, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(data, xTail, pregTailLoop,
                                       (i + binaryAddRemainderLoop) * VL_FP32 + halfXBufOffset_);
                Reg::Add(tmp2, x, xTail, pregTailLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(x, tmp2, pregTailLoop);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, x, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)binaryAdd + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            uint16_t curBinaryAddLoop = binaryAddLoop;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddLoop = curBinaryAddLoop / DIGIT_TWO;
                for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                    Reg::LoadAlign(binaryAddQ, ((__ubuf__ float*)binaryAdd) + j * VL_FP32);
                    Reg::LoadAlign(binaryAddR, ((__ubuf__ float*)binaryAdd) + (j + curBinaryAddLoop) * VL_FP32);
                    Reg::Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                    Reg::StoreAlign(((__ubuf__ float*)binaryAdd) + j * VL_FP32, binaryAddQ, pregMain);
                }
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            }
            uint32_t sreg3 = binaryAddParam.binaryAddLastNum;
            Reg::MaskReg pregLast = Reg::UpdateMask<float>(sreg3);
            Reg::LoadAlign(sum, ((__ubuf__ float*)binaryAdd));
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(dbeta, sum, pregLast);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)out) + idxInLevel0Cache,
                                                                           dbeta, pregMerge);
        }
    }

    __aicore__ inline void CalcDbetaLessThanVL64VF(const __ubuf__ DY_TYPE* data, const __ubuf__ float* out,
                                                   uint32_t idx, uint32_t r)
    {
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> x;
            Reg::RegTensor<float> sum;
            uint32_t sreg0 = r;
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
            LoadOneTensor<DY_TYPE>(data, x, pregLoop, 0);
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(sum, x, pregLoop);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)out) + idxInLevel0Cache,
                                                                           sum, pregMerge);
        }
    }

    __aicore__ inline void CalcDbetaLessThanVL64FoldVF(const __ubuf__ DY_TYPE* data, const __ubuf__ float* out,
                                                       uint32_t idx, uint32_t r, uint32_t tailR)
    {
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> x;
            Reg::RegTensor<float> xTail;
            Reg::RegTensor<float> sum;
            Reg::RegTensor<float> tmp;
            uint32_t sreg0 = r;
            uint32_t sreg1 = tailR;
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
            Reg::MaskReg pregTailLoop = Reg::UpdateMask<float>(sreg1);
            LoadOneTensor<DY_TYPE>(data, x, pregLoop, 0);
            LoadOneTensor<DY_TYPE>(data, xTail, pregTailLoop, halfXBufOffset_);
            Reg::Add(tmp, x, xTail, pregTailLoop);
            Reg::Move<float, Reg::MaskMergeMode::MERGING>(x, tmp, pregTailLoop);
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(sum, x, pregLoop);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)out) + idxInLevel0Cache,
                                                                           sum, pregMerge);
        }
    }

    __aicore__ inline void CalcDbeta(LocalTensor<DY_TYPE>& dy, LocalTensor<float>& dbeta, LocalTensor<float>& binaryAdd,
                                     uint32_t idx, uint32_t r, const BinaryAddParam& binaryAddParam)
    {
        if (r <= VL_FP32) {
            CalcDbetaLessThanVL64VF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ float*)dbeta.GetPhyAddr(), idx, r);
        } else {
            CalcDbetaVF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ float*)dbeta.GetPhyAddr(),
                        (__ubuf__ float*)binaryAdd.GetPhyAddr(), idx, r, binaryAddParam);
        }
    }

    __aicore__ inline void CalcDbetaFold(LocalTensor<DY_TYPE>& dy, LocalTensor<float>& dbeta,
                                         LocalTensor<float>& binaryAdd, uint32_t idx, uint32_t r, uint32_t tailR,
                                         const BinaryAddParam& binaryAddParam)
    {
        if (r <= VL_FP32) {
            CalcDbetaLessThanVL64FoldVF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ float*)dbeta.GetPhyAddr(), idx, r,
                                        tailR);
        } else {
            CalcDbetaFoldVF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ float*)dbeta.GetPhyAddr(),
                            (__ubuf__ float*)binaryAdd.GetPhyAddr(), idx, r, tailR, binaryAddParam);
        }
    }

    __aicore__ inline void CalcDgammaVF(const __ubuf__ DY_TYPE* dyAddr, const __ubuf__ DY_TYPE* xAddr,
                                        const __ubuf__ float* meanAddr, const __ubuf__ float* rstdAddr,
                                        const __ubuf__ float* dgammaAddr, const __ubuf__ float* binaryAdd, uint32_t idx,
                                        uint32_t r, const BinaryAddParam& binaryAddParam)
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
            Reg::RegTensor<float> sum;
            Reg::RegTensor<float> dgamma;
            Reg::RegTensor<float> meanValue;
            Reg::RegTensor<float> rstdValue;
            Reg::RegTensor<float> binaryAddQ1;
            Reg::RegTensor<float> binaryAddR1;
            Reg::RegTensor<float> binaryAddQ2;
            Reg::RegTensor<float> binaryAddR2;
            Reg::RegTensor<float> vlSum;
            Reg::RegTensor<float> tmp;

            Reg::MaskReg pregMain = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rstdValue, (__ubuf__ float*)(rstdAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(meanValue, (__ubuf__ float*)(meanAddr));
            uint32_t sreg0 = binaryAddRemainder;
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR1, pregLoop, i * VL_FP32 + binaryAddQuotientOffset);

                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ2, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddR2, pregLoop, i * VL_FP32 + binaryAddQuotientOffset);

                Reg::Sub(binaryAddQ2, binaryAddQ2, meanValue, pregMain);
                Reg::Mul(binaryAddQ2, binaryAddQ2, binaryAddQ1, pregMain);

                Reg::Sub(binaryAddR2, binaryAddR2, meanValue, pregLoop);
                Reg::Mul(binaryAddR2, binaryAddR2, binaryAddR1, pregLoop);

                Reg::Mul(binaryAddQ1, rstdValue, binaryAddQ2, pregMain);
                Reg::Mul(binaryAddR1, rstdValue, binaryAddR2, pregLoop);
                Reg::Add(tmp, binaryAddQ1, binaryAddR1, pregLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ1, tmp, pregLoop);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ1, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)binaryAdd + i, vlSum,
                                                                               pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ2, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                Reg::Sub(binaryAddQ2, binaryAddQ2, meanValue, pregMain);
                Reg::Mul(binaryAddQ2, binaryAddQ2, binaryAddQ1, pregMain);
                Reg::Mul(binaryAddQ1, rstdValue, binaryAddQ2, pregMain);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ1, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)binaryAdd + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            uint16_t curBinaryAddLoop = binaryAddLoop;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddLoop = curBinaryAddLoop / DIGIT_TWO;
                for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                    Reg::LoadAlign(binaryAddQ1, (__ubuf__ float*)(binaryAdd) + j * VL_FP32);
                    Reg::LoadAlign(binaryAddR1, (__ubuf__ float*)(binaryAdd) + (j + curBinaryAddLoop) * VL_FP32);
                    Reg::Add(binaryAddQ1, binaryAddQ1, binaryAddR1, pregMain);
                    Reg::StoreAlign(((__ubuf__ float*)binaryAdd) + j * VL_FP32, binaryAddQ1, pregMain);
                }
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            }
            uint32_t sreg2 = binaryAddParam.binaryAddLastNum;
            Reg::MaskReg pregLast = Reg::UpdateMask<float>(sreg2);
            Reg::LoadAlign(sum, (__ubuf__ float*)(binaryAdd));
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(dgamma, sum, pregLast);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__ubuf__ float*)dgammaAddr) + idxInLevel0Cache, dgamma, pregMerge);
        }
    }

    __aicore__ inline void CalcDgammaFoldVF(const __ubuf__ DY_TYPE* dyAddr, const __ubuf__ DY_TYPE* xAddr,
                                            const __ubuf__ float* meanAddr, const __ubuf__ float* rstdAddr,
                                            const __ubuf__ float* dgammaAddr, const __ubuf__ float* binaryAdd,
                                            uint32_t idx, uint32_t r, uint32_t tailR,
                                            const BinaryAddParam& binaryAddParam)
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
            Reg::RegTensor<float> sum;
            Reg::RegTensor<float> dgamma;
            Reg::RegTensor<float> meanValue;
            Reg::RegTensor<float> rstdValue;
            Reg::RegTensor<float> binaryAddQ1;
            Reg::RegTensor<float> binaryAddR1;
            Reg::RegTensor<float> binaryAddQ2;
            Reg::RegTensor<float> binaryAddR2;
            Reg::RegTensor<float> binaryAddQ1Tail;
            Reg::RegTensor<float> binaryAddR1Tail;
            Reg::RegTensor<float> binaryAddQ2Tail;
            Reg::RegTensor<float> binaryAddR2Tail;
            Reg::RegTensor<float> vlSum;
            Reg::RegTensor<float> tmp1;
            Reg::RegTensor<float> tmp2;

            Reg::MaskReg pregMain = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rstdValue, (__ubuf__ float*)(rstdAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(meanValue, (__ubuf__ float*)(meanAddr));
            uint32_t sreg0 = binaryAddRemainder;
            uint32_t sreg1 = tailMain;
            uint32_t sreg2 = tailRemainder;
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
                Reg::MaskReg pregTailLoop = Reg::UpdateMask<float>(sreg1);
                Reg::MaskReg pregTailReminderLoop = Reg::UpdateMask<float>(sreg2);
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

                Reg::Sub(binaryAddQ2, binaryAddQ2, meanValue, pregMain);
                Reg::Mul(binaryAddQ2, binaryAddQ2, binaryAddQ1, pregMain);

                Reg::Sub(binaryAddR2, binaryAddR2, meanValue, pregLoop);
                Reg::Mul(binaryAddR2, binaryAddR2, binaryAddR1, pregLoop);

                Reg::Mul(binaryAddQ1, rstdValue, binaryAddQ2, pregMain);
                Reg::Mul(binaryAddR1, rstdValue, binaryAddR2, pregLoop);

                Reg::Sub(binaryAddQ2Tail, binaryAddQ2Tail, meanValue, pregTailLoop);
                Reg::Mul(binaryAddQ2Tail, binaryAddQ2Tail, binaryAddQ1Tail, pregTailLoop);

                Reg::Sub(binaryAddR2Tail, binaryAddR2Tail, meanValue, pregTailReminderLoop);
                Reg::Mul(binaryAddR2Tail, binaryAddR2Tail, binaryAddR1Tail, pregTailReminderLoop);

                Reg::Mul(binaryAddQ1Tail, rstdValue, binaryAddQ2Tail, pregTailLoop);
                Reg::Mul(binaryAddR1Tail, rstdValue, binaryAddR2Tail, pregTailReminderLoop);

                Reg::Add(tmp1, binaryAddQ1, binaryAddQ1Tail, pregTailLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ1, tmp1, pregTailLoop);
                Reg::Add(tmp2, binaryAddR1, binaryAddR1Tail, pregTailReminderLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddR1, tmp2, pregTailReminderLoop);
                Reg::Add(tmp1, binaryAddQ1, binaryAddR1, pregLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ1, tmp1, pregLoop);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ1, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)binaryAdd + i, vlSum,
                                                                               pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                Reg::MaskReg pregTailLoop = Reg::UpdateMask<float>(sreg1);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1Tail, pregTailLoop,
                                       (i + binaryAddRemainderLoop) * VL_FP32 + halfXBufOffset_);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ2, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ2Tail, pregTailLoop,
                                       (i + binaryAddRemainderLoop) * VL_FP32 + halfXBufOffset_);
                Reg::Sub(binaryAddQ2, binaryAddQ2, meanValue, pregMain);
                Reg::Mul(binaryAddQ2, binaryAddQ2, binaryAddQ1, pregMain);
                Reg::Mul(binaryAddQ1, rstdValue, binaryAddQ2, pregMain);

                Reg::Sub(binaryAddQ2Tail, binaryAddQ2Tail, meanValue, pregTailLoop);
                Reg::Mul(binaryAddQ2Tail, binaryAddQ2Tail, binaryAddQ1Tail, pregTailLoop);
                Reg::Mul(binaryAddQ1Tail, rstdValue, binaryAddQ2Tail, pregTailLoop);
                Reg::Add(tmp1, binaryAddQ1, binaryAddQ1Tail, pregTailLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ1, tmp1, pregTailLoop);

                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ1, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)binaryAdd + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            uint16_t curBinaryAddLoop = binaryAddLoop;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddLoop = curBinaryAddLoop / DIGIT_TWO;
                for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                    Reg::LoadAlign(binaryAddQ1, (__ubuf__ float*)(binaryAdd) + j * VL_FP32);
                    Reg::LoadAlign(binaryAddR1, (__ubuf__ float*)(binaryAdd) + (j + curBinaryAddLoop) * VL_FP32);
                    Reg::Add(binaryAddQ1, binaryAddQ1, binaryAddR1, pregMain);
                    Reg::StoreAlign(((__ubuf__ float*)binaryAdd) + j * VL_FP32, binaryAddQ1, pregMain);
                }
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            }
            uint32_t sreg3 = binaryAddParam.binaryAddLastNum;
            Reg::MaskReg pregLast = Reg::UpdateMask<float>(sreg3);
            Reg::LoadAlign(sum, (__ubuf__ float*)(binaryAdd));
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(dgamma, sum, pregLast);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__ubuf__ float*)dgammaAddr) + idxInLevel0Cache, dgamma, pregMerge);
        }
    }

    __aicore__ inline void CalcDgammaLessThanVLVF(const __ubuf__ DY_TYPE* dyAddr, const __ubuf__ DY_TYPE* xAddr,
                                                  const __ubuf__ float* meanAddr, const __ubuf__ float* rstdAddr,
                                                  const __ubuf__ float* dgammaAddr, uint32_t idx, uint32_t r)
    {
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> x;
            Reg::RegTensor<float> y;
            Reg::RegTensor<float> meanValue;
            Reg::RegTensor<float> rstdValue;
            Reg::RegTensor<float> sum;
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            uint32_t sreg0 = r;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rstdValue, (__ubuf__ float*)(rstdAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(meanValue, (__ubuf__ float*)(meanAddr));

            Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
            LoadOneTensor<DY_TYPE>(dyAddr, y, pregLoop, 0);
            LoadOneTensor<DY_TYPE>(xAddr, x, pregLoop, 0);
            Reg::Sub(x, x, meanValue, pregLoop);
            Reg::Mul(x, y, x, pregLoop);
            Reg::Mul(x, x, rstdValue, pregLoop);
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(sum, x, pregLoop);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__ubuf__ float*)dgammaAddr) + idxInLevel0Cache, sum, pregMerge);
        }
    }

    __aicore__ inline void CalcDgammaLessThanVLFoldVF(const __ubuf__ DY_TYPE* dyAddr, const __ubuf__ DY_TYPE* xAddr,
                                                      const __ubuf__ float* meanAddr, const __ubuf__ float* rstdAddr,
                                                      const __ubuf__ float* dgammaAddr, uint32_t idx, uint32_t r,
                                                      uint32_t tailR)
    {
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> x;
            Reg::RegTensor<float> y;
            Reg::RegTensor<float> xTail;
            Reg::RegTensor<float> yTail;
            Reg::RegTensor<float> meanValue;
            Reg::RegTensor<float> rstdValue;
            Reg::RegTensor<float> sum;
            Reg::RegTensor<float> tmp;
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            uint32_t sreg0 = r;
            uint32_t sreg1 = tailR;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rstdValue, (__ubuf__ float*)(rstdAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(meanValue, (__ubuf__ float*)(meanAddr));

            Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
            Reg::MaskReg pregTailLoop = Reg::UpdateMask<float>(sreg1);
            LoadOneTensor<DY_TYPE>(dyAddr, y, pregLoop, 0);
            LoadOneTensor<DY_TYPE>(dyAddr, yTail, pregTailLoop, halfXBufOffset_);
            LoadOneTensor<DY_TYPE>(xAddr, x, pregLoop, 0);
            LoadOneTensor<DY_TYPE>(xAddr, xTail, pregTailLoop, halfXBufOffset_);

            Reg::Sub(x, x, meanValue, pregLoop);
            Reg::Mul(x, y, x, pregLoop);
            Reg::Mul(x, x, rstdValue, pregLoop);

            Reg::Sub(xTail, xTail, meanValue, pregTailLoop);
            Reg::Mul(xTail, yTail, xTail, pregTailLoop);
            Reg::Mul(xTail, xTail, rstdValue, pregTailLoop);
            Reg::Add(tmp, x, xTail, pregTailLoop);
            Reg::Move<float, Reg::MaskMergeMode::MERGING>(x, tmp, pregTailLoop);
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(sum, x, pregLoop);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__ubuf__ float*)dgammaAddr) + idxInLevel0Cache, sum, pregMerge);
        }
    }

    __aicore__ inline void CalcDgamma(LocalTensor<DY_TYPE>& dy, LocalTensor<DY_TYPE>& x, LocalTensor<float>& mean,
                                      LocalTensor<float>& rstd, LocalTensor<float>& dgamma,
                                      LocalTensor<float>& binaryAdd, uint32_t idx, uint32_t r,
                                      const BinaryAddParam& binaryAddParam)
    {
        if (r <= VL_FP32) {
            CalcDgammaLessThanVLVF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ DY_TYPE*)x.GetPhyAddr(),
                                   (__ubuf__ float*)mean.GetPhyAddr(), (__ubuf__ float*)rstd.GetPhyAddr(),
                                   (__ubuf__ float*)dgamma.GetPhyAddr(), idx, r);
        } else {
            CalcDgammaVF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ DY_TYPE*)x.GetPhyAddr(),
                         (__ubuf__ float*)mean.GetPhyAddr(), (__ubuf__ float*)rstd.GetPhyAddr(),
                         (__ubuf__ float*)dgamma.GetPhyAddr(), (__ubuf__ float*)binaryAdd.GetPhyAddr(), idx, r,
                         binaryAddParam);
        }
    }

    __aicore__ inline void CalcDgammaFold(LocalTensor<DY_TYPE>& dy, LocalTensor<DY_TYPE>& x, LocalTensor<float>& mean,
                                          LocalTensor<float>& rstd, LocalTensor<float>& dgamma,
                                          LocalTensor<float>& binaryAdd, uint32_t idx, uint32_t r, uint32_t tailR,
                                          const BinaryAddParam& binaryAddParam)
    {
        if (r <= VL_FP32) {
            CalcDgammaLessThanVLFoldVF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ DY_TYPE*)x.GetPhyAddr(),
                                       (__ubuf__ float*)mean.GetPhyAddr(), (__ubuf__ float*)rstd.GetPhyAddr(),
                                       (__ubuf__ float*)dgamma.GetPhyAddr(), idx, r, tailR);
        } else {
            CalcDgammaFoldVF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ DY_TYPE*)x.GetPhyAddr(),
                             (__ubuf__ float*)mean.GetPhyAddr(), (__ubuf__ float*)rstd.GetPhyAddr(),
                             (__ubuf__ float*)dgamma.GetPhyAddr(), (__ubuf__ float*)binaryAdd.GetPhyAddr(), idx, r,
                             tailR, binaryAddParam);
        }
    }

    __aicore__ inline void CalcDx(LocalTensor<DY_TYPE>& x, LocalTensor<DY_TYPE>& dy, LocalTensor<float>& rstd,
                                  LocalTensor<float>& mean, LocalTensor<WEIGHT_TYPE>& gamma, LocalTensor<float>& dgamma,
                                  LocalTensor<float>& dbeta, uint32_t xFactor, uint32_t r)
    {
        __ubuf__ DY_TYPE* xAddr = (__ubuf__ DY_TYPE*)x.GetPhyAddr();
        __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dy.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd.GetPhyAddr();
        __ubuf__ float* meanAddr = (__ubuf__ float*)mean.GetPhyAddr();
        __ubuf__ WEIGHT_TYPE* gammaAddr = (__ubuf__ WEIGHT_TYPE*)gamma.GetPhyAddr();
        __ubuf__ float* dgammaAddr = (__ubuf__ float*)dgamma.GetPhyAddr();
        __ubuf__ float* dbetaAddr = (__ubuf__ float*)dbeta.GetPhyAddr();

        uint16_t VL = VL_FP32;
        uint16_t loopTimes = (xFactor + VL - 1) / VL;
        float hRecipValue = 1.0f / (float)r;
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> rstdValue;
            Reg::RegTensor<float> meanValue;
            Reg::RegTensor<float> gammaValue;
            Reg::RegTensor<float> dgammaValue;
            Reg::RegTensor<float> dbetaValue;
            Reg::RegTensor<float> rstdMulSubDy;
            Reg::MaskReg pregR;
            Reg::RegTensor<float> dy;
            Reg::RegTensor<float> x;
            Reg::RegTensor<float> mulDgamma;
            Reg::RegTensor<float> addDbeta;
            Reg::RegTensor<float> divH;
            Reg::RegTensor<float> subDy;
            Reg::RegTensor<float> dx;
            Reg::MaskReg preg = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rstdValue, (__ubuf__ float*)(rstdAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(meanValue, (__ubuf__ float*)(meanAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(dgammaValue, (__ubuf__ float*)(dgammaAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(dbetaValue, (__ubuf__ float*)(dbetaAddr));
            LoadOneElement<WEIGHT_TYPE>(gammaAddr, gammaValue, preg, 0);

            uint32_t sregMask = (uint32_t)xFactor;
            for (uint16_t j = 0; j < loopTimes; j++) {
                pregR = Reg::UpdateMask<float>(sregMask);

                LoadOneTensor<DY_TYPE>(xAddr, x, pregR, VL * j);
                LoadOneTensor<DY_TYPE>(dyAddr, dy, pregR, VL * j);
                Reg::Sub(x, x, meanValue, pregR);
                Reg::Mul(x, x, rstdValue, pregR);

                Reg::Mul(mulDgamma, x, dgammaValue, pregR);
                Reg::Add(addDbeta, mulDgamma, dbetaValue, preg);
                Reg::Muls(divH, addDbeta, hRecipValue, pregR);
                Reg::Sub(subDy, dy, divH, pregR);
                Reg::Mul(rstdMulSubDy, rstdValue, subDy, pregR);
                Reg::Mul(dx, gammaValue, rstdMulSubDy, pregR);
                StoreOneTensor<DY_TYPE>(xAddr, dx, pregR, VL * j);
            }
        }
    }

    __aicore__ inline void ReSaveDGammaDBeta(LocalTensor<float>& dgamma, LocalTensor<float>& dbeta)
    {
        if constexpr (IsSameType<WEIGHT_TYPE, half>::value || IsSameType<WEIGHT_TYPE, bfloat16_t>::value) {
            __ubuf__ float* dgammaAddr = (__ubuf__ float*)dgamma.GetPhyAddr();
            __ubuf__ float* dbetaAddr = (__ubuf__ float*)dbeta.GetPhyAddr();

            __VEC_SCOPE__
            {
                Reg::RegTensor<float> dgammaValue;
                Reg::RegTensor<float> dbetaValue;
                Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(dgammaValue, (__ubuf__ float*)(dgammaAddr));
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(dbetaValue, (__ubuf__ float*)(dbetaAddr));
                StoreOneElement<WEIGHT_TYPE>(dbetaAddr, dbetaValue, pregMerge, 0);
                StoreOneElement<WEIGHT_TYPE>(dgammaAddr, dgammaValue, pregMerge, 0);
            }
        }
        return;
    }

    __aicore__ inline void CopyOutDx(LocalTensor<DY_TYPE>& dx, uint64_t offset, uint32_t r1Factor)
    {
        CopyOutRAR(dx, dxGm_, offset, r1Factor);
    }

    __aicore__ inline void CopyOutDgamma(LocalTensor<WEIGHT_TYPE>& dgamma, uint64_t meanOffset, uint32_t a)
    {
        CopyOutA(dgamma, dgammaGm_, meanOffset, a);
    }

    __aicore__ inline void CopyOutDbeta(LocalTensor<WEIGHT_TYPE>& dbeta, uint64_t meanOffset, uint32_t a)
    {
        CopyOutA(dbeta, dbetaGm_, meanOffset, a);
    }

    __aicore__ inline void CopyInRAR(LocalTensor<DY_TYPE>& localTensor, GlobalTensor<DY_TYPE>& globalTensor,
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
    __aicore__ inline void CopyInA(LocalTensor<U>& localTensor, GlobalTensor<U>& globalTensor, uint64_t offset,
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

    __aicore__ inline void CopyOutRAR(LocalTensor<DY_TYPE>& localTensor, GlobalTensor<DY_TYPE>& globalTensor,
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

    __aicore__ inline void CopyOutA(LocalTensor<WEIGHT_TYPE>& localTensor, GlobalTensor<WEIGHT_TYPE>& globalTensor,
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
    TPipe* pipe_ = nullptr;
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

    uint32_t r1Dim_{0};
    uint32_t aDim_{0};
    uint32_t r0Dim_{0};
    uint32_t rAlign_{0};
    uint32_t binaryAddQuotient_{0};
    uint32_t binaryAddK_{0};
    uint32_t binaryAddLastNum_{0};
    uint32_t aTailCoreNum_{0};
    uint32_t aDimTail_{0};
    uint32_t gmOffset_{0};
    uint32_t aDimLoopNum_{0};
    uint32_t dyBufSize_{0};
    uint32_t xBufSize_{0};
    uint32_t halfXBufOffset_{0};
    uint32_t xBufElemNum_{0};
    uint32_t meanBufSize_{0};
    uint32_t meanBufElemNum_{0};
    uint32_t gammaBufSize_{0};
    uint32_t gammaBufElemNum_{0};
    uint32_t binaryAddBufSize_{0};
    uint32_t binaryAddBufElemNum_{0};
    uint32_t blockNum_{0};
    uint64_t ubRDimLoopNum_{0};
    uint64_t nFactor_{0};
    uint64_t tailNFactor_{0};
    uint64_t ubRDimTailFactor_{0};
    uint64_t ubRDimTailTailFactor_{0};

    uint32_t tailBinaryAddQuotient_{0};
    uint32_t tailBinaryAddK_{0};
    uint32_t tailBinaryAddLastNum_{0};
    uint64_t ubRDimTailLoopNum_{0};
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
