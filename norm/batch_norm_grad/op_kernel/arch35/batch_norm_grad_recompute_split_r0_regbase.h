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
 * \file batch_norm_grad_recompute_split_r0_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_RECOMPUTE_SPLIT_R0_REGBASE_H_
#define __BATCH_NORM_GRAD_RECOMPUTE_SPLIT_R0_REGBASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_common.h"

namespace BNGRARRecomputeSplitR0 {
using namespace AscendC;
using namespace BatchNormGrad;
static constexpr int64_t UB_ADD_BUF = 3 * 1024;              // ub间二分累加固定buf
static constexpr uint32_t UB_ADD_LEN = 1024 / sizeof(float); // ub内二分累加每级存放个数

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1> class BatchNormGradRARRecomputeSplitR0 {
public:
    __aicore__ inline BatchNormGradRARRecomputeSplitR0(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx,
        GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace, const BatchNormGradRARRecomputeTilingData *tilingData)
    {
        const BatchNormGradBaseTilingData baseTilingData = tilingData->baseTilingData;
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
        aDimPerCore_ =
            GetBlockIdx() < baseTilingData.tailBlockNum ? baseTilingData.tailBlockDim : baseTilingData.formerBlockDim;
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
        isTailLoop2_ = ubRDimTailTail_ > 0 && ubRDimTailTailLoopNum_ == BatchNormGrad::TWO;
        tailLoop_ = ubRDimTailTailLoopNum_ == BatchNormGrad::TWO ? ubRDimTailLoopNum_ / BatchNormGrad::TWO :
                                                                     ubRDimTailLoopNum_;
        tailLoop_ = ubRDimTailTail_ > 0 ? (tailLoop_ + 1) : tailLoop_;
        sumLoop_ = ubRDimLoopNum_ + tailLoop_;
        tailFactorAlign_ = RoundUpOneBlock(ubRDimTailFactor_ * sizeof(DY_TYPE)) / sizeof(DY_TYPE);

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE *)(dy) + gmOffset_ * r0Dim_);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE *)(x) + gmOffset_ * r0Dim_);
        meanGm_.SetGlobalBuffer((__gm__ float *)(mean) + gmOffset_);
        rstdGm_.SetGlobalBuffer((__gm__ float *)(rstd) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE *)(gamma) + gmOffset_);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE *)(dx) + gmOffset_ * r0Dim_);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE *)(dgamma) + gmOffset_);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE *)(dbeta) + gmOffset_);

        int64_t dyBufSize = tailFactorAlign_ * BatchNormGrad::TWO * sizeof(DY_TYPE);
        int64_t xBufSize = ubRDimTailFactorAlign_ * BatchNormGrad::TWO * sizeof(float);
        xBufElemNum_ = xBufSize / sizeof(DY_TYPE);
        int64_t binaryAddBufSize = RoundUpOneBlock(binaryAddQuotient_ / VL_FP32 * sizeof(float));

        pipe_.InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize);
        pipe_.InitBuffer(xInQue_, BUFFER_NUM, xBufSize);
        pipe_.InitBuffer(meanInQue_, BUFFER_NUM, BatchNormGrad::ONE_BLK_SIZE);
        pipe_.InitBuffer(rstdInQue_, BUFFER_NUM, BatchNormGrad::ONE_BLK_SIZE);
        pipe_.InitBuffer(gammaInQue_, BUFFER_NUM, BatchNormGrad::ONE_BLK_SIZE);
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

    __aicore__ inline void ProcessTailTail(int64_t offset1, int64_t offset2, uint32_t &totalLoop, uint32_t &level2Idx,
        uint32_t &level3Idx)
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
        pipe_barrier(PIPE_ALL);
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
    __aicore__ inline void CopyIn(LocalTensor<U> &localTensor, GlobalTensor<U> &globalTensor, uint32_t ubOffset,
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
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNormGrad::TWO, offset1, ubRDimTailFactor_);
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNormGrad::TWO + tailFactorAlign_, offset2, tailLen);
        }
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInXOne(int64_t offset1, uint32_t copyLen)
    {
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        if constexpr (IsSameType<DY_TYPE, float>::value) {
            CopyIn(xLocal, xGm_, 0, offset1, copyLen);
        } else {
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNormGrad::TWO, offset1, copyLen);
        }
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CastDbetaAndDgamma()
    {
        __local_mem__ float *dgammaAddr = (__local_mem__ float *)dgamma_.GetPhyAddr();
        __local_mem__ float *dbetaAddr = (__local_mem__ float *)dbeta_.GetPhyAddr();

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> dgammaValue;
            MicroAPI::RegTensor<float> dbetaValue;
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                (__local_mem__ float *)(dgammaAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue, (__local_mem__ float *)(dbetaAddr));
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

    __aicore__ inline void CopyOutDx(LocalTensor<DY_TYPE> &dx, uint32_t copyLen, int64_t offset)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = copyLen * sizeof(DY_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad<DY_TYPE, PaddingMode::Compact>(dxGm_[offset], dx, copyOutParams);
    }

    __aicore__ inline void BinaryAdd(MicroAPI::RegTensor<float> &rst, __local_mem__ float *binaryAddAddr,
        uint16_t binaryAddLoop, uint16_t binaryAddKLoop, uint32_t binaryAddLastNum)
    {
        MicroAPI::RegTensor<float> binaryAddQ1;
        MicroAPI::RegTensor<float> binaryAddR1;
        MicroAPI::RegTensor<float> sum;
        MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        uint16_t curBinaryAddLoop = binaryAddLoop;
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddLoop = curBinaryAddLoop / BatchNormGrad::TWO;
            for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                MicroAPI::DataCopy(binaryAddQ1, ((__local_mem__ float *)binaryAddAddr + j * VL_FP32));
                MicroAPI::DataCopy(binaryAddR1,
                    ((__local_mem__ float *)binaryAddAddr + (j + curBinaryAddLoop) * VL_FP32));
                MicroAPI::Add(binaryAddQ1, binaryAddQ1, binaryAddR1, pregMain);
                MicroAPI::DataCopy(((__local_mem__ float *)binaryAddAddr) + j * VL_FP32, binaryAddQ1, pregMain);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }
        uint32_t sreg2 = binaryAddLastNum;
        MicroAPI::MaskReg pregLast = MicroAPI::UpdateMask<float>(sreg2);
        MicroAPI::DataCopy(sum, ((__local_mem__ float *)binaryAddAddr));
        MicroAPI::ReduceSum(rst, sum, pregLast);
    }

    __aicore__ inline void CalcDbetaTailPre(LocalTensor<DY_TYPE> &dy, uint32_t idx, uint32_t tailLen)
    {
        LocalTensor<float> binaryAdd = binaryAddBuf_.Get<float>();
        const __local_mem__ DY_TYPE *dyAddr = (__local_mem__ DY_TYPE *)dy.GetPhyAddr();
        const __local_mem__ float *binaryAddAddr = (__local_mem__ float *)binaryAdd.GetPhyAddr();
        const __local_mem__ float *outAddr = (__local_mem__ float *)dbeta_.GetPhyAddr();
        int64_t binaryAddRemainder = ubRDimTailFactor_ - halfBinaryAddQuotient_;
        uint16_t binaryAddRemainderLoop = CeilDiv(binaryAddRemainder, VL_FP32);
        uint16_t binaryAddQuotientLoop = CeilDiv(halfBinaryAddQuotient_, VL_FP32);

        uint32_t tailQ = tailLen <= halfBinaryAddQuotient_ ? tailLen : halfBinaryAddQuotient_;
        uint32_t tailR = tailLen <= halfBinaryAddQuotient_ ? 0 : (tailLen - halfBinaryAddQuotient_);
        uint16_t binaryAddKLoop = halfBinaryAddK_;
        uint16_t binaryAddLoop = ((halfBinaryAddQuotient_ / VL_FP32) / VL_FP32);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> rst;
            MicroAPI::RegTensor<float> binaryAddQ1;
            MicroAPI::RegTensor<float> binaryAddQ2;
            MicroAPI::RegTensor<float> binaryAddR1;
            MicroAPI::RegTensor<float> binaryAddR2;
            MicroAPI::RegTensor<float> vlSum;
            MicroAPI::RegTensor<float> tmp;

            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0 = binaryAddRemainder;
            uint32_t sreg1 = tailQ;
            uint32_t sreg2 = tailR;

            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                MicroAPI::MaskReg pregQ = MicroAPI::UpdateMask<float>(sreg1);
                MicroAPI::MaskReg pregR = MicroAPI::UpdateMask<float>(sreg2);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr + tailFactorAlign_, binaryAddQ2, pregQ, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR1, pregLoop, i * VL_FP32 + halfBinaryAddQuotient_);
                LoadOneTensor<DY_TYPE>(dyAddr + tailFactorAlign_, binaryAddR2, pregR,
                    i * VL_FP32 + halfBinaryAddQuotient_);
                MicroAPI::Add(tmp, binaryAddQ1, binaryAddQ2, pregQ);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ1, tmp, pregQ);
                MicroAPI::Add(tmp, binaryAddR1, binaryAddR2, pregR);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddR1, tmp, pregR);
                MicroAPI::Add(tmp, binaryAddQ1, binaryAddR1, pregLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ1, tmp, pregLoop);
                MicroAPI::ReduceSum(vlSum, binaryAddQ1, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAddAddr + i, vlSum, pregMerge);
            }
            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                MicroAPI::MaskReg pregQ = MicroAPI::UpdateMask<float>(sreg1);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr + tailFactorAlign_, binaryAddQ2, pregQ,
                    (i + binaryAddRemainderLoop) * VL_FP32);
                MicroAPI::Add(tmp, binaryAddQ1, binaryAddQ2, pregQ);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ1, tmp, pregQ);
                MicroAPI::ReduceSum(vlSum, binaryAddQ1, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAddAddr + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            BinaryAdd(rst, (__local_mem__ float *)binaryAddAddr, binaryAddLoop, binaryAddKLoop, halfBinaryAddLastNum_);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)outAddr + idx), rst, pregMerge);
        }
    }

    __aicore__ inline void CalcDbetaPre(LocalTensor<DY_TYPE> &dy, uint32_t idx, bool isHalf)
    {
        LocalTensor<float> binaryAdd = binaryAddBuf_.Get<float>();
        const __local_mem__ DY_TYPE *dyAddr = (__local_mem__ DY_TYPE *)dy.GetPhyAddr();
        const __local_mem__ float *binaryAddAddr = (__local_mem__ float *)binaryAdd.GetPhyAddr();
        const __local_mem__ float *outAddr = (__local_mem__ float *)dbeta_.GetPhyAddr();
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
            MicroAPI::RegTensor<float> rst;
            MicroAPI::RegTensor<float> binaryAddQ;
            MicroAPI::RegTensor<float> binaryAddR;
            MicroAPI::RegTensor<float> vlSum;
            MicroAPI::RegTensor<float> tmp;

            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0 = binaryAddRemainder;
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR, pregLoop, i * VL_FP32 + binaryAddQuotient);
                MicroAPI::Add(tmp, binaryAddQ, binaryAddR, pregLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ, tmp, pregLoop);
                MicroAPI::ReduceSum(vlSum, binaryAddQ, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAddAddr + i, vlSum, pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                MicroAPI::ReduceSum(vlSum, binaryAddQ, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAddAddr + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            BinaryAdd(rst, (__local_mem__ float *)binaryAddAddr, binaryAddLoop, binaryAddKLoop, binaryAddLastNum);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)outAddr + idx), rst, pregMerge);
        }
    }

    __aicore__ inline void CalcDgammaTailPre(LocalTensor<DY_TYPE> &x, LocalTensor<DY_TYPE> &dy, uint32_t idx,
        uint32_t tailLen)
    {
        LocalTensor<float> binaryAdd = binaryAddBuf_.Get<float>();
        const __local_mem__ DY_TYPE *xAddr = (__local_mem__ DY_TYPE *)x.GetPhyAddr();
        const __local_mem__ float *meanAddr = (__local_mem__ float *)mean_.GetPhyAddr();
        const __local_mem__ float *rstdAddr = (__local_mem__ float *)rstd_.GetPhyAddr();
        const __local_mem__ DY_TYPE *dyAddr = (__local_mem__ DY_TYPE *)dy.GetPhyAddr();
        const __local_mem__ float *outAddr = (__local_mem__ float *)dgamma_.GetPhyAddr();
        const __local_mem__ float *binaryAddAddr = (__local_mem__ float *)binaryAdd.GetPhyAddr();
        if constexpr (IsSameType<DY_TYPE, half>::value || IsSameType<DY_TYPE, bfloat16_t>::value) {
            xAddr += xBufElemNum_ / BatchNormGrad::TWO;
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
            MicroAPI::RegTensor<float> rst;
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> binaryAddXQ1;
            MicroAPI::RegTensor<float> binaryAddXR1;
            MicroAPI::RegTensor<float> binaryAddXQ2;
            MicroAPI::RegTensor<float> binaryAddXR2;
            MicroAPI::RegTensor<float> binaryAddDyQ1;
            MicroAPI::RegTensor<float> binaryAddDyR1;
            MicroAPI::RegTensor<float> binaryAddDyQ2;
            MicroAPI::RegTensor<float> binaryAddDyR2;
            MicroAPI::RegTensor<float> vlSum;
            MicroAPI::RegTensor<float> tmp;

            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float *)(rstdAddr));
            uint32_t sreg0 = binaryAddRemainder;
            uint32_t sreg1 = tailQ;
            uint32_t sreg2 = tailR;

            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                MicroAPI::MaskReg pregQ = MicroAPI::UpdateMask<float>(sreg1);
                MicroAPI::MaskReg pregR = MicroAPI::UpdateMask<float>(sreg2);
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

                MicroAPI::Sub(binaryAddXQ1, binaryAddXQ1, meanValue, pregMain);
                MicroAPI::Sub(binaryAddXR1, binaryAddXR1, meanValue, pregLoop);
                MicroAPI::Sub(binaryAddXQ2, binaryAddXQ2, meanValue, pregQ);
                MicroAPI::Sub(binaryAddXR2, binaryAddXR2, meanValue, pregR);

                MicroAPI::Mul(binaryAddXQ1, binaryAddXQ1, binaryAddDyQ1, pregMain);
                MicroAPI::Mul(binaryAddXR1, binaryAddXR1, binaryAddDyR1, pregLoop);
                MicroAPI::Mul(binaryAddXQ2, binaryAddXQ2, binaryAddDyQ2, pregQ);
                MicroAPI::Mul(binaryAddXR2, binaryAddXR2, binaryAddDyR2, pregR);

                MicroAPI::Mul(binaryAddDyQ1, rstdValue, binaryAddXQ1, pregMain);
                MicroAPI::Mul(binaryAddDyR1, rstdValue, binaryAddXR1, pregLoop);
                MicroAPI::Mul(binaryAddDyQ2, rstdValue, binaryAddXQ2, pregQ);
                MicroAPI::Mul(binaryAddDyR2, rstdValue, binaryAddXR2, pregR);

                MicroAPI::Add(tmp, binaryAddDyQ1, binaryAddDyQ2, pregQ);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddDyQ1, tmp, pregQ);
                MicroAPI::Add(tmp, binaryAddDyR1, binaryAddDyR2, pregR);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddDyR1, tmp, pregR);
                MicroAPI::Add(tmp, binaryAddDyQ1, binaryAddDyR1, pregLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddDyQ1, tmp, pregLoop);
                MicroAPI::ReduceSum(vlSum, binaryAddDyQ1, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAddAddr + i, vlSum, pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                MicroAPI::MaskReg pregQ = MicroAPI::UpdateMask<float>(sreg1);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddXQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr + tailFactorAlign_, binaryAddXQ2, pregQ,
                    (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddDyQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr + tailFactorAlign_, binaryAddDyQ2, pregQ,
                    (i + binaryAddRemainderLoop) * VL_FP32);

                MicroAPI::Sub(binaryAddXQ1, binaryAddXQ1, meanValue, pregMain);
                MicroAPI::Sub(binaryAddXQ2, binaryAddXQ2, meanValue, pregQ);

                MicroAPI::Mul(binaryAddXQ1, binaryAddXQ1, binaryAddDyQ1, pregMain);
                MicroAPI::Mul(binaryAddXQ2, binaryAddXQ2, binaryAddDyQ2, pregQ);

                MicroAPI::Mul(binaryAddDyQ1, rstdValue, binaryAddXQ1, pregMain);
                MicroAPI::Mul(binaryAddDyQ2, rstdValue, binaryAddXQ2, pregQ);

                MicroAPI::Add(tmp, binaryAddDyQ1, binaryAddDyQ2, pregQ);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddDyQ1, tmp, pregQ);
                MicroAPI::ReduceSum(vlSum, binaryAddDyQ1, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAddAddr + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            BinaryAdd(rst, (__local_mem__ float *)binaryAddAddr, binaryAddLoop, binaryAddKLoop, halfBinaryAddLastNum_);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)outAddr + idx), rst, pregMerge);
        }
    }

    __aicore__ inline void CalcDgammaPre(LocalTensor<DY_TYPE> &x, LocalTensor<DY_TYPE> &dy, uint32_t idx, bool isHalf)
    {
        LocalTensor<float> binaryAdd = binaryAddBuf_.Get<float>();
        const __local_mem__ DY_TYPE *xAddr = (__local_mem__ DY_TYPE *)x.GetPhyAddr();
        const __local_mem__ float *meanAddr = (__local_mem__ float *)mean_.GetPhyAddr();
        const __local_mem__ float *rstdAddr = (__local_mem__ float *)rstd_.GetPhyAddr();
        const __local_mem__ DY_TYPE *dyAddr = (__local_mem__ DY_TYPE *)dy.GetPhyAddr();
        const __local_mem__ float *outAddr = (__local_mem__ float *)dgamma_.GetPhyAddr();
        const __local_mem__ float *binaryAddAddr = (__local_mem__ float *)binaryAdd.GetPhyAddr();
        if constexpr (IsSameType<DY_TYPE, half>::value || IsSameType<DY_TYPE, bfloat16_t>::value) {
            xAddr += xBufElemNum_ / BatchNormGrad::TWO;
        }
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
            MicroAPI::RegTensor<float> rst;
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
            uint32_t sreg0 = binaryAddRemainder;
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float *)(rstdAddr));
            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ1, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddR1, pregLoop, i * VL_FP32 + binaryAddQuotient);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ2, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR2, pregLoop, i * VL_FP32 + binaryAddQuotient);
                MicroAPI::Sub(binaryAddQ1, binaryAddQ1, meanValue, pregMain);
                MicroAPI::Sub(binaryAddR1, binaryAddR1, meanValue, pregLoop);
                MicroAPI::Mul(binaryAddQ1, binaryAddQ1, binaryAddQ2, pregMain);
                MicroAPI::Mul(binaryAddR1, binaryAddR1, binaryAddR2, pregLoop);
                MicroAPI::Mul(binaryAddQ2, rstdValue, binaryAddQ1, pregMain);
                MicroAPI::Mul(binaryAddR2, rstdValue, binaryAddR1, pregLoop);
                MicroAPI::Add(tmp, binaryAddQ2, binaryAddR2, pregLoop);
                MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(binaryAddQ2, tmp, pregLoop);
                MicroAPI::ReduceSum(vlSum, binaryAddQ2, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAddAddr + i, vlSum, pregMerge);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ1, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ2, pregMain, (i + binaryAddRemainderLoop) * VL_FP32);
                MicroAPI::Sub(binaryAddQ1, binaryAddQ1, meanValue, pregMain);
                MicroAPI::Mul(binaryAddQ1, binaryAddQ1, binaryAddQ2, pregMain);
                MicroAPI::Mul(binaryAddQ2, rstdValue, binaryAddQ1, pregMain);
                MicroAPI::ReduceSum(vlSum, binaryAddQ2, pregMain);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float *)binaryAddAddr + binaryAddRemainderLoop + i, vlSum, pregMerge);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            BinaryAdd(rst, (__local_mem__ float *)binaryAddAddr, binaryAddLoop, binaryAddKLoop, binaryAddLastNum);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)outAddr + idx), rst, pregMerge);
        }
    }

    __aicore__ inline void ReduceToLevel1(LocalTensor<float> &out, uint32_t num, uint32_t srcOffset, uint32_t dstIdx)
    {
        if (num <= VL_FP32) {
            ReduceOutLessVF(out, num, srcOffset, dstIdx);
        } else {
            ReduceOut(out, num, srcOffset, dstIdx);
        }
    }

    __aicore__ inline void ReduceAllLevel(LocalTensor<float> &out, uint32_t level1Idx, uint32_t level2Idx,
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
            ReduceToLevel1(out, level3Idx, UB_ADD_LEN * BatchNormGrad::TWO, reduce);
            reduce += 1;
        }
        ReduceOutLessVF(out, reduce, 0, 0);
    }

    __aicore__ inline void ReduceDbetaAndDgamma(uint32_t level1Idx, uint32_t level2Idx, uint32_t level3Idx)
    {
        ReduceAllLevel(dbeta_, level1Idx, level2Idx, level3Idx);
        ReduceAllLevel(dgamma_, level1Idx, level2Idx, level3Idx);
    }

    __aicore__ inline void ReduceToNextBuf(uint32_t &level1Idx, uint32_t &level2Idx, uint32_t &level3Idx)
    {
        if (level2Idx >= UB_ADD_LEN) {
            ReduceOut(dbeta_, UB_ADD_LEN, UB_ADD_LEN, UB_ADD_LEN * BatchNormGrad::TWO + level3Idx);
            ReduceOut(dgamma_, UB_ADD_LEN, UB_ADD_LEN, UB_ADD_LEN * BatchNormGrad::TWO + level3Idx);
            level3Idx += 1;
            level2Idx = 0;
        }
        if (level1Idx >= UB_ADD_LEN - 1) {
            ReduceOut(dbeta_, UB_ADD_LEN, 0, UB_ADD_LEN + level2Idx);
            ReduceOut(dgamma_, UB_ADD_LEN, 0, UB_ADD_LEN + level2Idx);
            level2Idx += 1;
        }
    }

    __aicore__ inline void ReduceOutLessVF(LocalTensor<float> &out, uint32_t reduceLen, uint32_t srcOffset,
        uint32_t dstIdx)
    {
        const __local_mem__ float *outAddr = (__local_mem__ float *)out.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> rst;

            uint32_t sreg0 = reduceLen;
            MicroAPI::MaskReg pregLast = MicroAPI::UpdateMask<float>(sreg0);
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::DataCopy(sum, ((__local_mem__ float *)outAddr + srcOffset));
            MicroAPI::ReduceSum(rst, sum, pregLast);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)outAddr + dstIdx), rst, pregMerge);
        }
    }

    __aicore__ inline void ReduceOut(LocalTensor<float> &out, uint32_t reduceLen, uint32_t srcOffset, uint32_t dstIdx)
    {
        const __local_mem__ float *outAddr = (__local_mem__ float *)out.GetPhyAddr();
        LocalTensor<float> binaryAdd = binaryAddBuf_.Get<float>();
        const __local_mem__ float *binaryAddAddr = (__local_mem__ float *)binaryAdd.GetPhyAddr();
        uint16_t binaryAddLoop = CeilDiv(reduceLen, VL_FP32);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> rst;
            MicroAPI::RegTensor<float> binaryAddQ;

            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0 = reduceLen;
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t j = 0; j < binaryAddLoop; j++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                MicroAPI::DataCopy(binaryAddQ, ((__local_mem__ float *)outAddr + srcOffset + j * VL_FP32));
                MicroAPI::ReduceSum(rst, binaryAddQ, pregLoop);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float *)binaryAddAddr + j), rst, pregMerge);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            uint32_t sreg1 = binaryAddLoop;
            MicroAPI::MaskReg pregLast = MicroAPI::UpdateMask<float>(sreg1);
            MicroAPI::DataCopy(sum, ((__local_mem__ float *)binaryAddAddr));
            MicroAPI::ReduceSum(rst, sum, pregLast);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ((__local_mem__ float *)outAddr + dstIdx), rst, pregMerge);
        }
    }

    __aicore__ inline void CalcDx(LocalTensor<DY_TYPE> &dyLocal, LocalTensor<DY_TYPE> &xLocal, uint32_t r)
    {
        __local_mem__ DY_TYPE *xSrcAddr = (__local_mem__ DY_TYPE *)xLocal.GetPhyAddr();
        __local_mem__ DY_TYPE *xDstAddr = (__local_mem__ DY_TYPE *)xLocal.GetPhyAddr();
        __local_mem__ DY_TYPE *dyAddr = (__local_mem__ DY_TYPE *)dyLocal.GetPhyAddr();
        __local_mem__ float *meanAddr = (__local_mem__ float *)mean_.GetPhyAddr();
        __local_mem__ float *rstdAddr = (__local_mem__ float *)rstd_.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE *gammaAddr = (__local_mem__ WEIGHT_TYPE *)gamma_.GetPhyAddr();
        __local_mem__ float *dgammaAddr = (__local_mem__ float *)dgamma_.GetPhyAddr();
        __local_mem__ float *dbetaAddr = (__local_mem__ float *)dbeta_.GetPhyAddr();

        if constexpr (IsSameType<DY_TYPE, half>::value || IsSameType<DY_TYPE, bfloat16_t>::value) {
            xSrcAddr += xBufElemNum_ / BatchNormGrad::TWO;
        }

        uint16_t loopTimes = CeilDiv(r, VL_FP32);
        float hRecipValue = 1.0f / (float)(r1Dim_ * r0Dim_);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> xValue;
            MicroAPI::RegTensor<float> xSubMean;
            MicroAPI::RegTensor<float> xHat;
            MicroAPI::RegTensor<float> gammaValue;
            MicroAPI::RegTensor<float> dgammaValue;
            MicroAPI::RegTensor<float> dbetaValue;
            MicroAPI::RegTensor<float> rstdMulSubDy;
            MicroAPI::MaskReg pregR;
            MicroAPI::RegTensor<float> dy;
            MicroAPI::RegTensor<float> mulDgamma;
            MicroAPI::RegTensor<float> addDbeta;
            MicroAPI::RegTensor<float> divH;
            MicroAPI::RegTensor<float> subDy;
            MicroAPI::RegTensor<float> dx;
            MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float *)(rstdAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                (__local_mem__ float *)(dgammaAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue, (__local_mem__ float *)(dbetaAddr));
            LoadOneElement<WEIGHT_TYPE>(gammaAddr, gammaValue, preg, 0);
            uint32_t sregMask = r;
            for (uint16_t j = 0; j < loopTimes; j++) {
                pregR = MicroAPI::UpdateMask<float>(sregMask);
                LoadOneTensor<DY_TYPE>(xSrcAddr, xValue, pregR, VL_FP32 * j);
                LoadOneTensor<DY_TYPE>(dyAddr, dy, pregR, VL_FP32 * j);
                MicroAPI::Sub(xSubMean, xValue, meanValue, pregR);
                MicroAPI::Mul(xHat, xSubMean, rstdValue, pregR);
                MicroAPI::Mul(mulDgamma, xHat, dgammaValue, pregR);
                MicroAPI::Add(addDbeta, mulDgamma, dbetaValue, preg);
                MicroAPI::Muls(divH, addDbeta, hRecipValue, pregR);
                MicroAPI::Sub(subDy, dy, divH, pregR);
                MicroAPI::Mul(rstdMulSubDy, rstdValue, subDy, pregR);
                MicroAPI::Mul(dx, gammaValue, rstdMulSubDy, pregR);
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

    int64_t r1Dim_{ 0 };
    int64_t aDim_{ 0 };
    int64_t r0Dim_{ 0 };
    int64_t blockNum_{ 0 };
    int64_t aDimPerCore_{ 0 };
    int64_t gmOffset_{ 0 };
    int64_t ubRDimLoopNum_{ 0 };
    int64_t ubRDimTailTailLoopNum_{ 0 };
    int64_t ubRDimTailLoopNum_{ 0 };
    int64_t tailLoop_{ 0 };
    int64_t sumLoop_{ 0 };
    bool isTailLoop2_{ true };

    uint32_t binaryAddQuotient_{ 0 };
    uint32_t binaryAddK_{ 0 };
    uint32_t binaryAddLastNum_{ 0 };
    uint32_t halfBinaryAddQuotient_{ 0 };
    uint32_t halfBinaryAddK_{ 0 };
    uint32_t halfBinaryAddLastNum_{ 0 };
    uint32_t ubRDimFactor_{ 0 };
    uint32_t ubRDimTailFactor_{ 0 };
    uint32_t ubRDimTailFactorAlign_{ 0 };
    uint32_t ubRDimTailTail_{ 0 };
    uint32_t xBufElemNum_{ 0 };
    uint32_t tailFactorAlign_{ 0 };
};
} // namespace BNGRARRecomputeSplitR0
#endif // __BATCH_NORM_GRAD_RECOMPUTE_SPLIT_RO_REGBASE_H_
