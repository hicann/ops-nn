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
 * \file batch_norm_grad_v3_recompute_split_r0_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_V3_RECOMPUTE_SPLIT_R0_REGBASE_H_
#define __BATCH_NORM_GRAD_V3_RECOMPUTE_SPLIT_R0_REGBASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_v3_common.h"

namespace BNGV3RARRecomputeSplitR0 {
using namespace AscendC;
using namespace BatchNormGradV3;
static constexpr uint32_t UB_ADD_LEN = 4 * VL_FP32;                   // ub内二分累加每级存放个数
static constexpr int64_t UB_ADD_BUF = 3 * UB_ADD_LEN * sizeof(float); // ub间二分累加固定buf

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1>
class BatchNormGradV3RARRecomputeSplitR0 {
public:
    __aicore__ inline BatchNormGradV3RARRecomputeSplitR0(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace,
                                const BatchNormGradV3RARRecomputeTilingData* tilingData)
    {
        const BatchNormGradV3BaseTilingData baseTilingData = tilingData->baseTilingData;
        r1Dim_ = baseTilingData.r1Dim;
        aDim_ = baseTilingData.aDim;
        r0Dim_ = baseTilingData.r0Dim;
        blockNum_ = baseTilingData.blockNum;
        /* ub内二分累加相关参数 */
        halfBinaryAddQuotient_ = tilingData->tailBinAddTilingData.binaryAddQuotient;
        halfBinaryAddK_ = tilingData->tailBinAddTilingData.binaryAddk;
        halfBinaryAddLastNum_ = tilingData->tailBinAddTilingData.binaryAddLastNum;
        binaryAddQuotient_ = tilingData->generalBinAddTilingData.binaryAddQuotient;
        binaryAddK_ = tilingData->generalBinAddTilingData.binaryAddk;
        binaryAddLastNum_ = tilingData->generalBinAddTilingData.binaryAddLastNum;
        /* A轴开多核的相关参数 */
        aDimPerCore_ = GetBlockIdx() < baseTilingData.tailBlockNum ? baseTilingData.tailBlockDim :
                                                                     baseTilingData.formerBlockDim;
        gmOffset_ = GetBlockIdx() < baseTilingData.tailBlockNum ?
                        (GetBlockIdx() * baseTilingData.tailBlockDim) :
                        (baseTilingData.tailBlockNum * baseTilingData.tailBlockDim +
                         (GetBlockIdx() - baseTilingData.tailBlockNum) * baseTilingData.formerBlockDim);
        /* ub切r0相关参数，涉及ub间二分累加 */
        ubRDimFactor_ = tilingData->ubRDimFactor;
        ubRDimLoopNum_ = tilingData->ubRDimLoopNum;
        ubRDimTailFactor_ = tilingData->ubRDimTailFactor;
        ubRDimTailFactorAlign_ = tilingData->ubRDimTailFactorAlign;
        ubRDimTailLoopNum_ = tilingData->ubRDimTailLoopNum;
        ubRDimTailTail_ = tilingData->ubRDimTailTail;
        ubRDimTailTailLoopNum_ = tilingData->ubRDimTailTailLoopNum;
        isTailLoop2_ = ubRDimTailTail_ > 0 && ubRDimTailTailLoopNum_ == BatchNormGradV3::TWO;
        tailLoop_ = ubRDimTailTailLoopNum_ == BatchNormGradV3::TWO ? ubRDimTailLoopNum_ / BatchNormGradV3::TWO :
                                                                     ubRDimTailLoopNum_;
        tailLoop_ = ubRDimTailTail_ > 0 ? (tailLoop_ + 1) : tailLoop_;
        sumLoop_ = ubRDimLoopNum_ + tailLoop_;
        tailFactorAlign_ = RoundUpOneBlock(ubRDimTailFactor_ * sizeof(DY_TYPE)) / sizeof(DY_TYPE);

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dy) + gmOffset_ * r0Dim_);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(x) + gmOffset_ * r0Dim_);
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean) + gmOffset_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(gamma) + gmOffset_);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dx) + gmOffset_ * r0Dim_);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dgamma) + gmOffset_);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dbeta) + gmOffset_);

        int64_t dyBufSize = tailFactorAlign_ * BatchNormGradV3::TWO * sizeof(DY_TYPE);
        int64_t xBufSize = ubRDimTailFactorAlign_ * BatchNormGradV3::TWO * sizeof(float);
        xBufElemNum_ = xBufSize / sizeof(DY_TYPE);
        int64_t binaryAddBufSize = RoundUpOneBlock(binaryAddQuotient_ / VL_FP32 * sizeof(float));

        pipe_.InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize);
        pipe_.InitBuffer(xInQue_, BUFFER_NUM, xBufSize);
        pipe_.InitBuffer(meanInQue_, BUFFER_NUM, BatchNormGradV3::ONE_BLK_SIZE);
        pipe_.InitBuffer(rstdInQue_, BUFFER_NUM, BatchNormGradV3::ONE_BLK_SIZE);
        pipe_.InitBuffer(gammaInQue_, BUFFER_NUM, BatchNormGradV3::ONE_BLK_SIZE);
        pipe_.InitBuffer(dbetaOutQue_, 1, UB_ADD_BUF);
        pipe_.InitBuffer(dgammaOutQue_, 1, UB_ADD_BUF);
        pipe_.InitBuffer(binaryAddBuf_, binaryAddBufSize);

        V_MTE3_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        MTE3_MTE2_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= blockNum_) {
            return;
        }
        for (int64_t i = 0; i < aDimPerCore_; i++) {
            int64_t meanOffset = i;
            ProcessPre(meanOffset);
            CalcDbetaAndDgammaPerA(meanOffset);
            CalDxPerA(meanOffset);
            ProcessPost(meanOffset);
        }
    }

    __aicore__ inline void ProcessPre(int64_t meanOffset)
    {
        CopyInMeanAndRstd(meanOffset);
        CopyInGamma(meanOffset);
        mean_ = meanInQue_.template DeQue<float>();
        rstd_ = rstdInQue_.template DeQue<float>();
        gamma_ = gammaInQue_.template DeQue<WEIGHT_TYPE>();
        dgamma_ = dgammaOutQue_.template AllocTensor<float>();
        dbeta_ = dbetaOutQue_.template AllocTensor<float>();
    }

    __aicore__ inline void ProcessPost(int64_t meanOffset)
    {
        meanInQue_.FreeTensor(mean_);
        rstdInQue_.FreeTensor(rstd_);
        gammaInQue_.FreeTensor(gamma_);
        CopyOutDbetaAndDgamma(meanOffset);
    }

    __aicore__ inline void ProcessTail(int64_t offset1, int64_t offset2, uint32_t level1Idx, uint32_t tailLen)
    {
        CopyInDyTwo(offset1, offset2, tailLen);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        CalcDbetaTailPre(dyLocal, level1Idx, tailLen);
        CopyInXTwo(offset1, offset2, tailLen);
        LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
        CalcDgammaTailPre(xLocal, dyLocal, level1Idx, tailLen);
        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessHalfMain(int64_t offset1, uint32_t level1Idx)
    {
        CopyInDyOne(offset1, ubRDimTailFactor_);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        CalcDbetaPre(dyLocal, level1Idx, true);
        CopyInXOne(offset1, ubRDimTailFactor_);
        LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
        CalcDgammaPre(xLocal, dyLocal, level1Idx, true);
        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessMain(int64_t offset1, uint32_t level1Idx)
    {
        CopyInDyOne(offset1, ubRDimFactor_);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        CalcDbetaPre(dyLocal, level1Idx, false);
        CopyInXOne(offset1, ubRDimFactor_);
        LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
        CalcDgammaPre(xLocal, dyLocal, level1Idx, false);
        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessTailTail(int64_t offset1, int64_t offset2, uint32_t& totalLoop, uint32_t& level2Idx,
                                           uint32_t& level3Idx)
    {
        uint32_t level1Idx = 0;
        if (ubRDimTailTail_ > 0 && ubRDimTailTailLoopNum_ == 1) {
            level1Idx = totalLoop % UB_ADD_LEN;
            ProcessTail(offset1, offset2, level1Idx, ubRDimTailTail_);
            totalLoop += 1;
            ReduceToNextBuf(level1Idx, level2Idx, level3Idx);
        }
        if (isTailLoop2_ && (ubRDimTailTail_ <= ubRDimTailFactor_)) {
            level1Idx = totalLoop % UB_ADD_LEN;
            ProcessTail(offset1, offset2, level1Idx, ubRDimTailTail_);
            totalLoop += 1;
            ReduceToNextBuf(level1Idx, level2Idx, level3Idx);

            offset1 += ubRDimTailFactor_;
            level1Idx = totalLoop % UB_ADD_LEN;
            ProcessHalfMain(offset1, level1Idx);
            totalLoop += 1;
            ReduceToNextBuf(level1Idx, level2Idx, level3Idx);
        }
        if (isTailLoop2_ && (ubRDimTailTail_ > ubRDimTailFactor_)) {
            level1Idx = totalLoop % UB_ADD_LEN;
            ProcessTail(offset1, offset2, level1Idx, ubRDimTailFactor_);
            totalLoop += 1;
            ReduceToNextBuf(level1Idx, level2Idx, level3Idx);

            offset1 += ubRDimTailFactor_;
            offset2 += ubRDimTailFactor_;
            level1Idx = totalLoop % UB_ADD_LEN;
            uint32_t tailLen = ubRDimTailTail_ - ubRDimTailFactor_;
            ProcessTail(offset1, offset2, level1Idx, tailLen);
            totalLoop += 1;
            ReduceToNextBuf(level1Idx, level2Idx, level3Idx);
        }
    }

    __aicore__ inline void CalcDbetaAndDgammaPerA(int64_t meanOffset)
    {
        int64_t offset = meanOffset * r0Dim_;
        int64_t tailOffset = ubRDimLoopNum_ * ubRDimFactor_;
        int64_t r1Offset = 0;
        int64_t offset1 = 0;
        int64_t offset2 = 0;
        uint32_t totalLoop = 0;
        uint32_t level1Idx = 0;
        uint32_t level2Idx = 0;
        uint32_t level3Idx = 0;
        for (int64_t k = 0; k < r1Dim_; k++) {
            r1Offset = k * r0Dim_ * aDim_;
            for (int64_t j = 0; j < ubRDimTailLoopNum_; j++) {
                offset1 = offset + r1Offset + ubRDimTailFactor_ * j;
                offset2 = offset1 + tailOffset;
                level1Idx = totalLoop % UB_ADD_LEN;
                ProcessTail(offset1, offset2, level1Idx, ubRDimTailFactor_);
                totalLoop += 1;
                ReduceToNextBuf(level1Idx, level2Idx, level3Idx);
            }

            int64_t tailtailOffset = tailOffset + ubRDimTailLoopNum_ * ubRDimTailFactor_;
            offset1 = offset + r1Offset + ubRDimTailFactor_ * ubRDimTailLoopNum_;
            offset2 = offset + r1Offset + tailtailOffset;
            ProcessTailTail(offset1, offset2, totalLoop, level2Idx, level3Idx);

            for (int64_t j = tailLoop_; j < ubRDimLoopNum_; j++) {
                offset1 = offset + r1Offset + ubRDimFactor_ * j;
                level1Idx = totalLoop % UB_ADD_LEN;
                ProcessMain(offset1, level1Idx);
                totalLoop += 1;
                ReduceToNextBuf(level1Idx, level2Idx, level3Idx);
            }
        }
        level1Idx = totalLoop % UB_ADD_LEN;
        ReduceDbetaAndDgamma(level1Idx, level2Idx, level3Idx);
    }

    __aicore__ inline void CalDxPerA(int64_t meanOffset)
    {
        for (int64_t k = 0; k < r1Dim_; k++) {
            for (int64_t j = 0; j < sumLoop_; j++) {
                int64_t bufferIdx = j % BUFFER_NUM;
                int64_t offset1 = meanOffset * r0Dim_ + k * r0Dim_ * aDim_ + ubRDimFactor_ * j;
                uint32_t processR = ubRDimFactor_;
                bool isLast = j == (sumLoop_ - 1);
                if (unlikely(isLast)) {
                    processR = ubRDimTailTail_ == 0 ? ubRDimFactor_ : ubRDimTailTail_;
                }
                CopyInDyOne(offset1, processR);
                if (k > 0 || j >= BUFFER_NUM) {
                    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
                }
                CopyInXOne(offset1, processR);
                LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
                LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
                CalcDx(dyLocal, xLocal, processR);
                SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);
                WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);
                CopyOutDx(xLocal, processR, offset1);
                if (k < r1Dim_ - 1 || j < sumLoop_ - BUFFER_NUM) {
                    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
                }
                dyInQue_.FreeTensor(dyLocal);
                xInQue_.FreeTensor(xLocal);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInMeanAndRstd(int64_t offset)
    {
        mean_ = meanInQue_.template AllocTensor<float>();
        rstd_ = rstdInQue_.template AllocTensor<float>();
        DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(mean_, meanGm_[offset], copyInParams, dataCopyPadExtParams);
        DataCopyPad(rstd_, rstdGm_[offset], copyInParams, dataCopyPadExtParams);
        meanInQue_.EnQue(mean_);
        rstdInQue_.EnQue(rstd_);
    }

    __aicore__ inline void CopyInGamma(int64_t offset)
    {
        LocalTensor<WEIGHT_TYPE> gammaLocal = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
        CopyIn(gammaLocal, gammaGm_, 0, offset, 1);
        gammaInQue_.EnQue(gammaLocal);
    }

    template <typename U>
    __aicore__ inline void CopyIn(LocalTensor<U>& localTensor, GlobalTensor<U>& globalTensor, uint32_t ubOffset,
                                  int64_t gmOffset, uint32_t copyLen)
    {
        DataCopyPadExtParams<U> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = copyLen * sizeof(U);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad<U, PaddingMode::Compact>(localTensor[ubOffset], globalTensor[gmOffset], copyInParams,
                                             dataCopyPadExtParams);
    }

    __aicore__ inline void CopyInDyTwo(int64_t offset1, int64_t offset2, uint32_t tailLen)
    {
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(dyLocal, dyGm_, 0, offset1, ubRDimTailFactor_);
        CopyIn(dyLocal, dyGm_, tailFactorAlign_, offset2, tailLen);
        dyInQue_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInDyOne(int64_t offset1, uint32_t copyLen)
    {
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(dyLocal, dyGm_, 0, offset1, copyLen);
        dyInQue_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInXTwo(int64_t offset1, int64_t offset2, uint32_t tailLen)
    {
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        if constexpr (IsSameType<DY_TYPE, float>::value) {
            CopyIn(xLocal, xGm_, 0, offset1, ubRDimTailFactor_);
            CopyIn(xLocal, xGm_, tailFactorAlign_, offset2, tailLen);
        } else {
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNormGradV3::TWO, offset1, ubRDimTailFactor_);
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNormGradV3::TWO + tailFactorAlign_, offset2, tailLen);
        }
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInXOne(int64_t offset1, uint32_t copyLen)
    {
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        if constexpr (IsSameType<DY_TYPE, float>::value) {
            CopyIn(xLocal, xGm_, 0, offset1, copyLen);
        } else {
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNormGradV3::TWO, offset1, copyLen);
        }
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CastDbetaAndDgamma()
    {
        __ubuf__ float* dgammaAddr = (__ubuf__ float*)dgamma_.GetPhyAddr();
        __ubuf__ float* dbetaAddr = (__ubuf__ float*)dbeta_.GetPhyAddr();

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

    __aicore__ inline void CopyOutDbetaAndDgamma(int64_t offset)
    {
        if constexpr (IsSameType<WEIGHT_TYPE, half>::value || IsSameType<WEIGHT_TYPE, bfloat16_t>::value) {
            CastDbetaAndDgamma();
        }
        dbetaOutQue_.EnQue(dbeta_);
        dgammaOutQue_.EnQue(dgamma_);
        LocalTensor<WEIGHT_TYPE> dbetaOut = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
        LocalTensor<WEIGHT_TYPE> dgammaOut = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = sizeof(WEIGHT_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Compact>(dbetaGm_[offset], dbetaOut, copyOutParams);
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Compact>(dgammaGm_[offset], dgammaOut, copyOutParams);
        dbetaOutQue_.FreeTensor(dbetaOut);
        dgammaOutQue_.FreeTensor(dgammaOut);
    }

    __aicore__ inline void CopyOutDx(LocalTensor<DY_TYPE>& dx, uint32_t copyLen, int64_t offset)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = copyLen * sizeof(DY_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad<DY_TYPE, PaddingMode::Compact>(dxGm_[offset], dx, copyOutParams);
    }

    __aicore__ inline void BinaryAdd(Reg::RegTensor<float>& rst, __ubuf__ float* binaryAddAddr, uint16_t binaryAddLoop,
                                     uint16_t binaryAddKLoop, uint32_t binaryAddLastNum)
    {
        Reg::RegTensor<float> binaryAddQ1;
        Reg::RegTensor<float> binaryAddR1;
        Reg::RegTensor<float> sum;
        Reg::MaskReg pregMain = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        uint16_t curBinaryAddLoop = binaryAddLoop;
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddLoop = curBinaryAddLoop / BatchNormGradV3::TWO;
            for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                Reg::LoadAlign(binaryAddQ1, ((__ubuf__ float*)binaryAddAddr + j * VL_FP32));
                Reg::LoadAlign(binaryAddR1, ((__ubuf__ float*)binaryAddAddr + (j + curBinaryAddLoop) * VL_FP32));
                Reg::Add(binaryAddQ1, binaryAddQ1, binaryAddR1, pregMain);
                Reg::StoreAlign(((__ubuf__ float*)binaryAddAddr) + j * VL_FP32, binaryAddQ1, pregMain);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
        uint32_t sreg2 = binaryAddLastNum;
        Reg::MaskReg pregLast = Reg::UpdateMask<float>(sreg2);
        Reg::LoadAlign(sum, ((__ubuf__ float*)binaryAddAddr));
        Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rst, sum, pregLast);
    }

    __aicore__ inline void CalcDbetaTailPre(LocalTensor<DY_TYPE>& dy, uint32_t idx, uint32_t tailLen)
    {
        LocalTensor<float> binaryAdd = binaryAddBuf_.Get<float>();
        const __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dy.GetPhyAddr();
        const __ubuf__ float* binaryAddAddr = (__ubuf__ float*)binaryAdd.GetPhyAddr();
        const __ubuf__ float* outAddr = (__ubuf__ float*)dbeta_.GetPhyAddr();
        int64_t binaryAddRemainder = ubRDimTailFactor_ - halfBinaryAddQuotient_;
        uint16_t binaryAddRemainderLoop = CeilDiv(binaryAddRemainder, VL_FP32);
        uint16_t binaryAddQuotientLoop = CeilDiv(halfBinaryAddQuotient_, VL_FP32);

        uint32_t tailQ = tailLen <= halfBinaryAddQuotient_ ? tailLen : halfBinaryAddQuotient_;
        uint32_t tailR = tailLen <= halfBinaryAddQuotient_ ? 0 : (tailLen - halfBinaryAddQuotient_);
        uint16_t binaryAddKLoop = halfBinaryAddK_;
        uint16_t binaryAddLoop = ((halfBinaryAddQuotient_ / VL_FP32) / VL_FP32);

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> rst;
            Reg::RegTensor<float> binaryAddQ1;
            Reg::RegTensor<float> binaryAddQ2;
            Reg::RegTensor<float> binaryAddR1;
            Reg::RegTensor<float> binaryAddR2;
            Reg::RegTensor<float> vlSum;
            Reg::RegTensor<float> tmp;

            Reg::MaskReg pregMain = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            uint32_t sreg0 = binaryAddRemainder;
            uint32_t sreg1 = tailQ;
            uint32_t sreg2 = tailR;

            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
                Reg::MaskReg pregQ = Reg::UpdateMask<float>(sreg1);
                Reg::MaskReg pregR = Reg::UpdateMask<float>(sreg2);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr + tailFactorAlign_, binaryAddQ2, pregQ, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR1, pregLoop, i * VL_FP32 + halfBinaryAddQuotient_);
                LoadOneTensor<DY_TYPE>(dyAddr + tailFactorAlign_, binaryAddR2, pregR,
                                       i * VL_FP32 + halfBinaryAddQuotient_);
                Reg::Add(tmp, binaryAddQ1, binaryAddQ2, pregQ);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ1, tmp, pregQ);
                Reg::Add(tmp, binaryAddR1, binaryAddR2, pregR);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddR1, tmp, pregR);
                Reg::Add(tmp, binaryAddQ1, binaryAddR1, pregLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ1, tmp, pregLoop);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ1, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)binaryAddAddr + i,
                                                                               vlSum, pregMerge);
            }
            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                Reg::MaskReg pregQ = Reg::UpdateMask<float>(sreg1);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr + tailFactorAlign_, binaryAddQ2, pregQ,
                                       (i + binaryAddRemainderLoop) * VL_FP32);
                Reg::Add(tmp, binaryAddQ1, binaryAddQ2, pregQ);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ1, tmp, pregQ);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ1, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            BinaryAdd(rst, (__ubuf__ float*)binaryAddAddr, binaryAddLoop, binaryAddKLoop, halfBinaryAddLastNum_);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)outAddr + idx), rst,
                                                                           pregMerge);
        }
    }

    __aicore__ inline void CalcDbetaPre(LocalTensor<DY_TYPE>& dy, uint32_t idx, bool isHalf)
    {
        LocalTensor<float> binaryAdd = binaryAddBuf_.Get<float>();
        const __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dy.GetPhyAddr();
        const __ubuf__ float* binaryAddAddr = (__ubuf__ float*)binaryAdd.GetPhyAddr();
        const __ubuf__ float* outAddr = (__ubuf__ float*)dbeta_.GetPhyAddr();
        int64_t processR = ubRDimFactor_;
        uint32_t binaryAddQuotient = binaryAddQuotient_;
        uint32_t binaryAddK = binaryAddK_;
        uint32_t binaryAddLastNum = binaryAddLastNum_;
        if (isHalf) {
            processR = ubRDimTailFactor_;
            binaryAddQuotient = halfBinaryAddQuotient_;
            binaryAddK = halfBinaryAddK_;
            binaryAddLastNum = halfBinaryAddLastNum_;
        }
        int64_t binaryAddRemainder = processR - binaryAddQuotient;
        uint16_t binaryAddRemainderLoop = CeilDiv(binaryAddRemainder, VL_FP32);
        uint16_t binaryAddQuotientLoop = CeilDiv(binaryAddQuotient, VL_FP32);

        uint16_t binaryAddKLoop = binaryAddK;
        uint16_t binaryAddLoop = ((binaryAddQuotient / VL_FP32) / VL_FP32);

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> rst;
            Reg::RegTensor<float> binaryAddQ;
            Reg::RegTensor<float> binaryAddR;
            Reg::RegTensor<float> vlSum;
            Reg::RegTensor<float> tmp;

            Reg::MaskReg pregMain = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            uint32_t sreg0 = binaryAddRemainder;
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR, pregLoop, i * VL_FP32 + binaryAddQuotient);
                Reg::Add(tmp, binaryAddQ, binaryAddR, pregLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ, tmp, pregLoop);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)binaryAddAddr + i,
                                                                               vlSum, pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            BinaryAdd(rst, (__ubuf__ float*)binaryAddAddr, binaryAddLoop, binaryAddKLoop, binaryAddLastNum);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)outAddr + idx), rst,
                                                                           pregMerge);
        }
    }

    __aicore__ inline void CalcDgammaTailPre(LocalTensor<DY_TYPE>& x, LocalTensor<DY_TYPE>& dy, uint32_t idx,
                                             uint32_t tailLen)
    {
        LocalTensor<float> binaryAdd = binaryAddBuf_.Get<float>();
        const __ubuf__ DY_TYPE* xAddr = (__ubuf__ DY_TYPE*)x.GetPhyAddr();
        const __ubuf__ float* meanAddr = (__ubuf__ float*)mean_.GetPhyAddr();
        const __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd_.GetPhyAddr();
        const __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dy.GetPhyAddr();
        const __ubuf__ float* outAddr = (__ubuf__ float*)dgamma_.GetPhyAddr();
        const __ubuf__ float* binaryAddAddr = (__ubuf__ float*)binaryAdd.GetPhyAddr();
        if constexpr (IsSameType<DY_TYPE, half>::value || IsSameType<DY_TYPE, bfloat16_t>::value) {
            xAddr += xBufElemNum_ / BatchNormGradV3::TWO;
        }
        int64_t binaryAddRemainder = ubRDimTailFactor_ - halfBinaryAddQuotient_;
        uint16_t binaryAddRemainderLoop = CeilDiv(binaryAddRemainder, VL_FP32);
        uint16_t binaryAddQuotientLoop = CeilDiv(halfBinaryAddQuotient_, VL_FP32);

        uint32_t tailQ = tailLen <= halfBinaryAddQuotient_ ? tailLen : halfBinaryAddQuotient_;
        uint32_t tailR = tailLen <= halfBinaryAddQuotient_ ? 0 : (tailLen - halfBinaryAddQuotient_);
        uint16_t binaryAddKLoop = halfBinaryAddK_;
        uint16_t binaryAddLoop = ((halfBinaryAddQuotient_ / VL_FP32) / VL_FP32);

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> rst;
            Reg::RegTensor<float> meanValue;
            Reg::RegTensor<float> rstdValue;
            Reg::RegTensor<float> binaryAddXQ1;
            Reg::RegTensor<float> binaryAddXR1;
            Reg::RegTensor<float> binaryAddXQ2;
            Reg::RegTensor<float> binaryAddXR2;
            Reg::RegTensor<float> binaryAddDyQ1;
            Reg::RegTensor<float> binaryAddDyR1;
            Reg::RegTensor<float> binaryAddDyQ2;
            Reg::RegTensor<float> binaryAddDyR2;
            Reg::RegTensor<float> vlSum;
            Reg::RegTensor<float> tmp;

            Reg::MaskReg pregMain = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(meanValue, (__ubuf__ float*)(meanAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rstdValue, (__ubuf__ float*)(rstdAddr));
            uint32_t sreg0 = binaryAddRemainder;
            uint32_t sreg1 = tailQ;
            uint32_t sreg2 = tailR;

            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
                Reg::MaskReg pregQ = Reg::UpdateMask<float>(sreg1);
                Reg::MaskReg pregR = Reg::UpdateMask<float>(sreg2);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddXQ1, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr + tailFactorAlign_, binaryAddXQ2, pregQ, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddXR1, pregLoop, i * VL_FP32 + halfBinaryAddQuotient_);
                LoadOneTensor<DY_TYPE>(xAddr + tailFactorAlign_, binaryAddXR2, pregR,
                                       i * VL_FP32 + halfBinaryAddQuotient_);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddDyQ1, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr + tailFactorAlign_, binaryAddDyQ2, pregQ, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddDyR1, pregLoop, i * VL_FP32 + halfBinaryAddQuotient_);
                LoadOneTensor<DY_TYPE>(dyAddr + tailFactorAlign_, binaryAddDyR2, pregR,
                                       i * VL_FP32 + halfBinaryAddQuotient_);

                Reg::Sub(binaryAddXQ1, binaryAddXQ1, meanValue, pregMain);
                Reg::Sub(binaryAddXR1, binaryAddXR1, meanValue, pregLoop);
                Reg::Sub(binaryAddXQ2, binaryAddXQ2, meanValue, pregQ);
                Reg::Sub(binaryAddXR2, binaryAddXR2, meanValue, pregR);

                Reg::Mul(binaryAddXQ1, binaryAddXQ1, binaryAddDyQ1, pregMain);
                Reg::Mul(binaryAddXR1, binaryAddXR1, binaryAddDyR1, pregLoop);
                Reg::Mul(binaryAddXQ2, binaryAddXQ2, binaryAddDyQ2, pregQ);
                Reg::Mul(binaryAddXR2, binaryAddXR2, binaryAddDyR2, pregR);

                Reg::Mul(binaryAddDyQ1, rstdValue, binaryAddXQ1, pregMain);
                Reg::Mul(binaryAddDyR1, rstdValue, binaryAddXR1, pregLoop);
                Reg::Mul(binaryAddDyQ2, rstdValue, binaryAddXQ2, pregQ);
                Reg::Mul(binaryAddDyR2, rstdValue, binaryAddXR2, pregR);

                Reg::Add(tmp, binaryAddDyQ1, binaryAddDyQ2, pregQ);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddDyQ1, tmp, pregQ);
                Reg::Add(tmp, binaryAddDyR1, binaryAddDyR2, pregR);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddDyR1, tmp, pregR);
                Reg::Add(tmp, binaryAddDyQ1, binaryAddDyR1, pregLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddDyQ1, tmp, pregLoop);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddDyQ1, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)binaryAddAddr + i,
                                                                               vlSum, pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                Reg::MaskReg pregQ = Reg::UpdateMask<float>(sreg1);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddXQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr + tailFactorAlign_, binaryAddXQ2, pregQ,
                                       (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddDyQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr + tailFactorAlign_, binaryAddDyQ2, pregQ,
                                       (i + binaryAddRemainderLoop) * VL_FP32);

                Reg::Sub(binaryAddXQ1, binaryAddXQ1, meanValue, pregMain);
                Reg::Sub(binaryAddXQ2, binaryAddXQ2, meanValue, pregQ);

                Reg::Mul(binaryAddXQ1, binaryAddXQ1, binaryAddDyQ1, pregMain);
                Reg::Mul(binaryAddXQ2, binaryAddXQ2, binaryAddDyQ2, pregQ);

                Reg::Mul(binaryAddDyQ1, rstdValue, binaryAddXQ1, pregMain);
                Reg::Mul(binaryAddDyQ2, rstdValue, binaryAddXQ2, pregQ);

                Reg::Add(tmp, binaryAddDyQ1, binaryAddDyQ2, pregQ);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddDyQ1, tmp, pregQ);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddDyQ1, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            BinaryAdd(rst, (__ubuf__ float*)binaryAddAddr, binaryAddLoop, binaryAddKLoop, halfBinaryAddLastNum_);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)outAddr + idx), rst,
                                                                           pregMerge);
        }
    }

    __aicore__ inline void CalcDgammaPre(LocalTensor<DY_TYPE>& x, LocalTensor<DY_TYPE>& dy, uint32_t idx, bool isHalf)
    {
        LocalTensor<float> binaryAdd = binaryAddBuf_.Get<float>();
        const __ubuf__ DY_TYPE* xAddr = (__ubuf__ DY_TYPE*)x.GetPhyAddr();
        const __ubuf__ float* meanAddr = (__ubuf__ float*)mean_.GetPhyAddr();
        const __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd_.GetPhyAddr();
        const __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dy.GetPhyAddr();
        const __ubuf__ float* outAddr = (__ubuf__ float*)dgamma_.GetPhyAddr();
        const __ubuf__ float* binaryAddAddr = (__ubuf__ float*)binaryAdd.GetPhyAddr();
        if constexpr (IsSameType<DY_TYPE, half>::value || IsSameType<DY_TYPE, bfloat16_t>::value) {
            xAddr += xBufElemNum_ / BatchNormGradV3::TWO;
        }
        int64_t processR = ubRDimFactor_;
        uint32_t binaryAddQuotient = binaryAddQuotient_;
        uint32_t binaryAddLastNum = binaryAddLastNum_;
        uint32_t binaryAddK = binaryAddK_;
        if (isHalf) {
            processR = ubRDimTailFactor_;
            binaryAddQuotient = halfBinaryAddQuotient_;
            binaryAddLastNum = halfBinaryAddLastNum_;
            binaryAddK = halfBinaryAddK_;
        }
        int64_t binaryAddRemainder = processR - binaryAddQuotient;
        uint16_t binaryAddRemainderLoop = CeilDiv(binaryAddRemainder, VL_FP32);
        uint16_t binaryAddQuotientLoop = CeilDiv(binaryAddQuotient, VL_FP32);

        uint16_t binaryAddKLoop = binaryAddK;
        uint16_t binaryAddLoop = ((binaryAddQuotient / VL_FP32) / VL_FP32);

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> rst;
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
            uint32_t sreg0 = binaryAddRemainder;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(meanValue, (__ubuf__ float*)(meanAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rstdValue, (__ubuf__ float*)(rstdAddr));
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ1, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddR1, pregLoop, i * VL_FP32 + binaryAddQuotient);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ2, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR2, pregLoop, i * VL_FP32 + binaryAddQuotient);
                Reg::Sub(binaryAddQ1, binaryAddQ1, meanValue, pregMain);
                Reg::Sub(binaryAddR1, binaryAddR1, meanValue, pregLoop);
                Reg::Mul(binaryAddQ1, binaryAddQ1, binaryAddQ2, pregMain);
                Reg::Mul(binaryAddR1, binaryAddR1, binaryAddR2, pregLoop);
                Reg::Mul(binaryAddQ2, rstdValue, binaryAddQ1, pregMain);
                Reg::Mul(binaryAddR2, rstdValue, binaryAddR1, pregLoop);
                Reg::Add(tmp, binaryAddQ2, binaryAddR2, pregLoop);
                Reg::Move<float, Reg::MaskMergeMode::MERGING>(binaryAddQ2, tmp, pregLoop);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ2, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)binaryAddAddr + i,
                                                                               vlSum, pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ2, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                Reg::Sub(binaryAddQ1, binaryAddQ1, meanValue, pregMain);
                Reg::Mul(binaryAddQ1, binaryAddQ1, binaryAddQ2, pregMain);
                Reg::Mul(binaryAddQ2, rstdValue, binaryAddQ1, pregMain);
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ2, pregMain);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            BinaryAdd(rst, (__ubuf__ float*)binaryAddAddr, binaryAddLoop, binaryAddKLoop, binaryAddLastNum);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)outAddr + idx), rst,
                                                                           pregMerge);
        }
    }

    __aicore__ inline void ReduceToLevel1(LocalTensor<float>& out, uint32_t num, uint32_t srcOffset, uint32_t dstIdx)
    {
        if (num <= VL_FP32) {
            ReduceOutLessVF(out, num, srcOffset, dstIdx);
        } else {
            ReduceOut(out, num, srcOffset, dstIdx);
        }
    }

    __aicore__ inline void ReduceAllLevel(LocalTensor<float>& out, uint32_t level1Idx, uint32_t level2Idx,
                                          uint32_t level3Idx)
    {
        uint32_t reduce = 0;
        if (level1Idx > 0) {
            ReduceToLevel1(out, level1Idx, 0, reduce);
            reduce += 1;
        }
        if (level2Idx > 0) {
            ReduceToLevel1(out, level2Idx, UB_ADD_LEN, reduce);
            reduce += 1;
        }
        if (level3Idx > 0) {
            ReduceToLevel1(out, level3Idx, UB_ADD_LEN * BatchNormGradV3::TWO, reduce);
            reduce += 1;
        }
        ReduceOutLessVF(out, reduce, 0, 0);
    }

    __aicore__ inline void ReduceDbetaAndDgamma(uint32_t level1Idx, uint32_t level2Idx, uint32_t level3Idx)
    {
        ReduceAllLevel(dbeta_, level1Idx, level2Idx, level3Idx);
        ReduceAllLevel(dgamma_, level1Idx, level2Idx, level3Idx);
    }

    __aicore__ inline void ReduceToNextBuf(uint32_t& level1Idx, uint32_t& level2Idx, uint32_t& level3Idx)
    {
        if (level2Idx >= UB_ADD_LEN) {
            ReduceOut(dbeta_, UB_ADD_LEN, UB_ADD_LEN, UB_ADD_LEN * BatchNormGradV3::TWO + level3Idx);
            ReduceOut(dgamma_, UB_ADD_LEN, UB_ADD_LEN, UB_ADD_LEN * BatchNormGradV3::TWO + level3Idx);
            level3Idx += 1;
            level2Idx = 0;
        }
        if (level1Idx >= UB_ADD_LEN - 1) {
            ReduceOut(dbeta_, UB_ADD_LEN, 0, UB_ADD_LEN + level2Idx);
            ReduceOut(dgamma_, UB_ADD_LEN, 0, UB_ADD_LEN + level2Idx);
            level2Idx += 1;
        }
    }

    __aicore__ inline void ReduceOutLessVF(LocalTensor<float>& out, uint32_t reduceLen, uint32_t srcOffset,
                                           uint32_t dstIdx)
    {
        const __ubuf__ float* outAddr = (__ubuf__ float*)out.GetPhyAddr();
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> sum;
            Reg::RegTensor<float> rst;

            uint32_t sreg0 = reduceLen;
            Reg::MaskReg pregLast = Reg::UpdateMask<float>(sreg0);
            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            Reg::LoadAlign(sum, ((__ubuf__ float*)outAddr + srcOffset));
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rst, sum, pregLast);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)outAddr + dstIdx), rst,
                                                                           pregMerge);
        }
    }

    __aicore__ inline void ReduceOut(LocalTensor<float>& out, uint32_t reduceLen, uint32_t srcOffset, uint32_t dstIdx)
    {
        const __ubuf__ float* outAddr = (__ubuf__ float*)out.GetPhyAddr();
        LocalTensor<float> binaryAdd = binaryAddBuf_.Get<float>();
        const __ubuf__ float* binaryAddAddr = (__ubuf__ float*)binaryAdd.GetPhyAddr();
        uint16_t binaryAddLoop = CeilDiv(reduceLen, VL_FP32);
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> sum;
            Reg::RegTensor<float> rst;
            Reg::RegTensor<float> binaryAddQ;

            Reg::MaskReg pregMerge = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            uint32_t sreg0 = reduceLen;
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            for (uint16_t j = 0; j < binaryAddLoop; j++) {
                Reg::MaskReg pregLoop = Reg::UpdateMask<float>(sreg0);
                Reg::LoadAlign(binaryAddQ, ((__ubuf__ float*)outAddr + srcOffset + j * VL_FP32));
                Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rst, binaryAddQ, pregLoop);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + j),
                                                                               rst, pregMerge);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            uint32_t sreg1 = binaryAddLoop;
            Reg::MaskReg pregLast = Reg::UpdateMask<float>(sreg1);
            Reg::LoadAlign(sum, ((__ubuf__ float*)binaryAddAddr));
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rst, sum, pregLast);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)outAddr + dstIdx), rst,
                                                                           pregMerge);
        }
    }

    __aicore__ inline void CalcDx(LocalTensor<DY_TYPE>& dyLocal, LocalTensor<DY_TYPE>& xLocal, uint32_t r)
    {
        __ubuf__ DY_TYPE* xSrcAddr = (__ubuf__ DY_TYPE*)xLocal.GetPhyAddr();
        __ubuf__ DY_TYPE* xDstAddr = (__ubuf__ DY_TYPE*)xLocal.GetPhyAddr();
        __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dyLocal.GetPhyAddr();
        __ubuf__ float* meanAddr = (__ubuf__ float*)mean_.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd_.GetPhyAddr();
        __ubuf__ WEIGHT_TYPE* gammaAddr = (__ubuf__ WEIGHT_TYPE*)gamma_.GetPhyAddr();
        __ubuf__ float* dgammaAddr = (__ubuf__ float*)dgamma_.GetPhyAddr();
        __ubuf__ float* dbetaAddr = (__ubuf__ float*)dbeta_.GetPhyAddr();

        if constexpr (IsSameType<DY_TYPE, half>::value || IsSameType<DY_TYPE, bfloat16_t>::value) {
            xSrcAddr += xBufElemNum_ / BatchNormGradV3::TWO;
        }

        uint16_t loopTimes = CeilDiv(r, VL_FP32);
        float hRecipValue = 1.0f / (float)(r1Dim_ * r0Dim_);
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> meanValue;
            Reg::RegTensor<float> rstdValue;
            Reg::RegTensor<float> xValue;
            Reg::RegTensor<float> xSubMean;
            Reg::RegTensor<float> xHat;
            Reg::RegTensor<float> gammaValue;
            Reg::RegTensor<float> dgammaValue;
            Reg::RegTensor<float> dbetaValue;
            Reg::RegTensor<float> gammaMulSubDy;
            Reg::MaskReg pregR;
            Reg::RegTensor<float> dy;
            Reg::RegTensor<float> mulDgamma;
            Reg::RegTensor<float> addDbeta;
            Reg::RegTensor<float> divH;
            Reg::RegTensor<float> subDy;
            Reg::RegTensor<float> dx;
            Reg::MaskReg preg = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(meanValue, (__ubuf__ float*)(meanAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rstdValue, (__ubuf__ float*)(rstdAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(dgammaValue, (__ubuf__ float*)(dgammaAddr));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(dbetaValue, (__ubuf__ float*)(dbetaAddr));
            LoadOneElement<WEIGHT_TYPE>(gammaAddr, gammaValue, preg, 0);
            uint32_t sregMask = r;
            for (uint16_t j = 0; j < loopTimes; j++) {
                pregR = Reg::UpdateMask<float>(sregMask);
                LoadOneTensor<DY_TYPE>(xSrcAddr, xValue, pregR, VL_FP32 * j);
                LoadOneTensor<DY_TYPE>(dyAddr, dy, pregR, VL_FP32 * j);
                Reg::Sub(xSubMean, xValue, meanValue, pregR);
                Reg::Mul(xHat, xSubMean, rstdValue, pregR);
                Reg::Mul(mulDgamma, xHat, dgammaValue, pregR);
                Reg::Add(addDbeta, mulDgamma, dbetaValue, preg);
                Reg::Muls(divH, addDbeta, hRecipValue, pregR);
                Reg::Sub(subDy, dy, divH, pregR);
                Reg::Mul(gammaMulSubDy, gammaValue, subDy, pregR);
                Reg::Mul(dx, rstdValue, gammaMulSubDy, pregR);
                StoreOneTensor<DY_TYPE>(xDstAddr, dx, pregR, VL_FP32 * j);
            }
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;
    TBuf<TPosition::VECCALC> binaryAddBuf_;

    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, rstdGm_;

    LocalTensor<float> mean_;
    LocalTensor<float> rstd_;
    LocalTensor<WEIGHT_TYPE> gamma_;
    LocalTensor<float> dgamma_;
    LocalTensor<float> dbeta_;

    event_t V_MTE3_EVENT;
    event_t MTE3_MTE2_EVENT;

    int64_t r1Dim_{0};
    int64_t aDim_{0};
    int64_t r0Dim_{0};
    int64_t blockNum_{0};
    int64_t aDimPerCore_{0};
    int64_t gmOffset_{0};
    int64_t ubRDimLoopNum_{0};
    int64_t ubRDimTailTailLoopNum_{0};
    int64_t ubRDimTailLoopNum_{0};
    int64_t tailLoop_{0};
    int64_t sumLoop_{0};
    bool isTailLoop2_{true};

    uint32_t binaryAddQuotient_{0};
    uint32_t binaryAddK_{0};
    uint32_t binaryAddLastNum_{0};
    uint32_t halfBinaryAddQuotient_{0};
    uint32_t halfBinaryAddK_{0};
    uint32_t halfBinaryAddLastNum_{0};
    uint32_t ubRDimFactor_{0};
    uint32_t ubRDimTailFactor_{0};
    uint32_t ubRDimTailFactorAlign_{0};
    uint32_t ubRDimTailTail_{0};
    uint32_t xBufElemNum_{0};
    uint32_t tailFactorAlign_{0};
};
} // namespace BNGV3RARRecomputeSplitR0
#endif // __BATCH_NORM_GRAD_V3_RECOMPUTE_SPLIT_RO_REGBASE_H_
