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
 * \file batch_norm_grad_infer.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_INFER_H
#define BATCH_NORM_GRAD_INFER_H

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "batch_norm_grad_common.h"

namespace BatchNormGrad
{
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

static constexpr int64_t UB_ADD_BUF = 3 * 1024;              // ub间二分累加固定buf
static constexpr uint32_t UB_ADD_LEN = 1024 / sizeof(float); // ub内二分累加每级存放个数

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1, bool EN_DGAMMA = true, bool EN_DBETA = true>
class BatchNormGradInferDgammaDbetaSplitR0 {
public:
    __aicore__ inline BatchNormGradInferDgammaDbetaSplitR0(TPipe *pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR var, GM_ADDR gamma, GM_ADDR dx,
        GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace, const BatchNormGradInferTilingData *tilingData)
    {
        epsilon = tilingData->baseTilingData.epsilon;
        r1Dim_ = tilingData->baseTilingData.r1Dim;
        aDim_ = tilingData->baseTilingData.aDim;
        r0Dim_ = tilingData->baseTilingData.r0Dim;
        blockNum_ = tilingData->blockNum;
        /* ub内二分累加相关参数 */
        halfBinaryAddQuotient_ = tilingData->tailBinAddTilingData.binaryAddQuotient;
        halfBinaryAddK_ = tilingData->tailBinAddTilingData.binaryAddk;
        halfBinaryAddLastNum_ = tilingData->tailBinAddTilingData.binaryAddLastNum;
        binaryAddQuotient_ = tilingData->generalBinAddTilingData.binaryAddQuotient;
        binaryAddK_ = tilingData->generalBinAddTilingData.binaryAddk;
        binaryAddLastNum_ = tilingData->generalBinAddTilingData.binaryAddLastNum;
        /* A轴开多核的相关参数 */
        aDimPerCore_ =
            GetBlockIdx() < tilingData->tailBlockNum ? tilingData->tailBlockDim : tilingData->formerBlockDim;
        gmOffset_ = GetBlockIdx() < tilingData->tailBlockNum ?
            (GetBlockIdx() * tilingData->tailBlockDim) :
            (tilingData->tailBlockNum * tilingData->tailBlockDim +
            (GetBlockIdx() - tilingData->tailBlockNum) * tilingData->formerBlockDim);
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
        varGm_.SetGlobalBuffer((__gm__ float *)(var) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE *)(gamma) + gmOffset_);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE *)(dx) + gmOffset_ * r0Dim_);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE *)(dgamma) + gmOffset_);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE *)(dbeta) + gmOffset_);

        int64_t dyBufSize = tailFactorAlign_ * BatchNormGrad::TWO * sizeof(DY_TYPE);
        int64_t xBufSize = ubRDimTailFactorAlign_ * BatchNormGrad::TWO * sizeof(float);
        xBufElemNum_ = xBufSize / sizeof(DY_TYPE);
        int64_t binaryAddBufSize = RoundUpOneBlock(binaryAddQuotient_ / VL_FP32 * sizeof(float));

        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, xBufSize);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, BatchNormGrad::ONE_BLK_SIZE);
        pipe_->InitBuffer(varInQue_, BUFFER_NUM, BatchNormGrad::ONE_BLK_SIZE);
        pipe_->InitBuffer(dbetaOutQue_, 1, UB_ADD_BUF);
        pipe_->InitBuffer(dgammaOutQue_, 1, UB_ADD_BUF);
        pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize);
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
            CopyOutDbetaAndDgamma(meanOffset);
        }
    }

    __aicore__ inline void ProcessPre(int64_t meanOffset)
    {
        if constexpr (EN_DBETA) {
            dbeta_ = dbetaOutQue_.template AllocTensor<float>();
        }
        if constexpr (EN_DGAMMA) {
            CopyInMeanAndRstd(meanOffset);
            dgamma_ = dgammaOutQue_.template AllocTensor<float>();
            var_ = varInQue_.template DeQue<float>();
            mean_ = meanInQue_.template DeQue<float>();
        }
    }

    __aicore__ inline void ProcessTail(int64_t offset1, int64_t offset2, uint32_t level1Idx, uint32_t tailLen)
    {
        CopyInDyTwo(offset1, offset2, tailLen);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        if constexpr (EN_DBETA) {
            CalcDbetaTailPre(dyLocal, level1Idx, tailLen);
        }
        if constexpr (EN_DGAMMA) {
            CopyInXTwo(offset1, offset2, tailLen);
            LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
            CalcDgammaTailPre(xLocal, dyLocal, level1Idx, tailLen);
            xInQue_.FreeTensor(xLocal);
        }

        dyInQue_.FreeTensor(dyLocal);
    }

    __aicore__ inline void ProcessHalfMain(int64_t offset1, uint32_t level1Idx)
    {
        CopyInDyOne(offset1, ubRDimTailFactor_);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        if constexpr (EN_DBETA) {
            CalcDbetaPre(dyLocal, level1Idx, true);
        }
        if constexpr (EN_DGAMMA) {
            CopyInXOne(offset1, ubRDimTailFactor_);
            LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
            CalcDgammaPre(xLocal, dyLocal, level1Idx, true);
            xInQue_.FreeTensor(xLocal);
        }
        dyInQue_.FreeTensor(dyLocal);
    }

    __aicore__ inline void ProcessMain(int64_t offset1, uint32_t level1Idx)
    {
        CopyInDyOne(offset1, ubRDimFactor_);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        if constexpr (EN_DBETA) {
            CalcDbetaPre(dyLocal, level1Idx, false);
        }
        if constexpr (EN_DGAMMA) {
            CopyInXOne(offset1, ubRDimFactor_);
            LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
            CalcDgammaPre(xLocal, dyLocal, level1Idx, false);
            xInQue_.FreeTensor(xLocal);
        }
        dyInQue_.FreeTensor(dyLocal);
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
        int64_t offset = meanOffset * r0Dim_;   // A R0 里面的偏移
        int64_t tailOffset = ubRDimLoopNum_ * ubRDimFactor_;
        int64_t r1Offset = 0;
        int64_t offset1 = 0;
        int64_t offset2 = 0;
        uint32_t totalLoop = 0;
        uint32_t level1Idx = 0;
        uint32_t level2Idx = 0;
        uint32_t level3Idx = 0;
        for (int64_t k = 0; k < r1Dim_; k++) {      // 遍历 r1 
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

    __aicore__ inline void CopyInMeanAndRstd(int64_t offset)
    {
        mean_ = meanInQue_.template AllocTensor<float>();
        var_ = varInQue_.template AllocTensor<float>();
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
        DataCopyPad(var_, varGm_[offset], copyInParams, dataCopyPadExtParams);
        meanInQue_.EnQue(mean_);
        varInQue_.EnQue(var_);
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
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            if constexpr (EN_DGAMMA) {
                MicroAPI::RegTensor<float> dgammaValue;
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                    (__local_mem__ float *)(dgammaAddr));
                StoreOneElement<WEIGHT_TYPE>(dgammaAddr, dgammaValue, pregMerge, 0);
            }
            if constexpr (EN_DBETA) {
                MicroAPI::RegTensor<float> dbetaValue;
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue, (__local_mem__ float *)(dbetaAddr));                
                StoreOneElement<WEIGHT_TYPE>(dbetaAddr, dbetaValue, pregMerge, 0);
            }
        }
    }

    __aicore__ inline void CopyOutDbetaAndDgamma(int64_t offset)
    {
        if constexpr (EN_DGAMMA) {
            meanInQue_.FreeTensor(mean_);
            varInQue_.FreeTensor(var_);
        }

        if constexpr (IsSameType<WEIGHT_TYPE, half>::value || IsSameType<WEIGHT_TYPE, bfloat16_t>::value) {
            CastDbetaAndDgamma();
        }

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = sizeof(WEIGHT_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        if constexpr (EN_DBETA) {
            dbetaOutQue_.EnQue(dbeta_);
            LocalTensor<WEIGHT_TYPE> dbetaOut = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
            DataCopyPad<WEIGHT_TYPE, PaddingMode::Compact>(dbetaGm_[offset], dbetaOut, copyOutParams);
            dbetaOutQue_.FreeTensor(dbetaOut);
        }
        if constexpr (EN_DGAMMA) {
            dgammaOutQue_.EnQue(dgamma_);
            LocalTensor<WEIGHT_TYPE> dgammaOut = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
            DataCopyPad<WEIGHT_TYPE, PaddingMode::Compact>(dgammaGm_[offset], dgammaOut, copyOutParams);
            dgammaOutQue_.FreeTensor(dgammaOut);
        }
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
        const __local_mem__ float *varAddr = (__local_mem__ float *)var_.GetPhyAddr();
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
            MicroAPI::RegTensor<float> runningVarValue;
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
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(runningVarValue, (__local_mem__ float *)(varAddr));

            CalRstdByHighPrecision(runningVarValue, rstdValue, epsilon);

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

                MicroAPI::Mul(binaryAddXQ1, binaryAddXQ1, rstdValue, pregMain);
                MicroAPI::Mul(binaryAddXR1, binaryAddXR1, rstdValue, pregLoop);
                MicroAPI::Mul(binaryAddXQ2, binaryAddXQ2, rstdValue, pregQ);
                MicroAPI::Mul(binaryAddXR2, binaryAddXR2, rstdValue, pregR);

                MicroAPI::Mul(binaryAddDyQ1, binaryAddDyQ1, binaryAddXQ1, pregMain);
                MicroAPI::Mul(binaryAddDyR1, binaryAddDyR1, binaryAddXR1, pregLoop);
                MicroAPI::Mul(binaryAddDyQ2, binaryAddDyQ2, binaryAddXQ2, pregQ);
                MicroAPI::Mul(binaryAddDyR2, binaryAddDyR2, binaryAddXR2, pregR);

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

                MicroAPI::Mul(binaryAddXQ1, binaryAddXQ1, rstdValue, pregMain);
                MicroAPI::Mul(binaryAddXQ2, binaryAddXQ2, rstdValue, pregQ);

                MicroAPI::Mul(binaryAddDyQ1, binaryAddDyQ1, binaryAddXQ1, pregMain);
                MicroAPI::Mul(binaryAddDyQ2, binaryAddDyQ2, binaryAddXQ2, pregQ);

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
        const __local_mem__ float *varAddr = (__local_mem__ float *)var_.GetPhyAddr();
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
            MicroAPI::RegTensor<float> runningVarValue;
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
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(runningVarValue, (__local_mem__ float *)(varAddr));

            CalRstdByHighPrecision(runningVarValue, rstdValue, epsilon);

            for (uint16_t i = 0; i < binaryAddRemainderLoop; i++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddQ1, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(xAddr, binaryAddR1, pregLoop, i * VL_FP32 + binaryAddQuotient);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ2, pregMain, i * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR2, pregLoop, i * VL_FP32 + binaryAddQuotient);
                MicroAPI::Sub(binaryAddQ1, binaryAddQ1, meanValue, pregMain);
                MicroAPI::Sub(binaryAddR1, binaryAddR1, meanValue, pregLoop);
                MicroAPI::Mul(binaryAddQ1, binaryAddQ1, rstdValue, pregMain);
                MicroAPI::Mul(binaryAddR1, binaryAddR1, rstdValue, pregLoop);
                MicroAPI::Mul(binaryAddQ2, binaryAddQ2, binaryAddQ1, pregMain);
                MicroAPI::Mul(binaryAddR2, binaryAddR2, binaryAddR1, pregLoop);
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
                MicroAPI::Mul(binaryAddQ1, binaryAddQ1, rstdValue, pregMain);
                MicroAPI::Mul(binaryAddQ2, binaryAddQ2, binaryAddQ1, pregMain);
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
        if constexpr (EN_DBETA) {
            ReduceAllLevel(dbeta_, level1Idx, level2Idx, level3Idx);
        }
        if constexpr (EN_DGAMMA) {
            ReduceAllLevel(dgamma_, level1Idx, level2Idx, level3Idx);
        }
    }

    __aicore__ inline void ReduceToNextBuf(uint32_t &level1Idx, uint32_t &level2Idx, uint32_t &level3Idx)
    {
        if (level2Idx >= UB_ADD_LEN) {
            if constexpr (EN_DBETA) {
                ReduceOut(dbeta_, UB_ADD_LEN, UB_ADD_LEN, UB_ADD_LEN * BatchNormGrad::TWO + level3Idx);
            }
            if constexpr (EN_DGAMMA) {
                ReduceOut(dgamma_, UB_ADD_LEN, UB_ADD_LEN, UB_ADD_LEN * BatchNormGrad::TWO + level3Idx);
            }
            level3Idx += 1;
            level2Idx = 0;
        }
        if (level1Idx >= UB_ADD_LEN - 1) {
            if constexpr (EN_DBETA) {
                ReduceOut(dbeta_, UB_ADD_LEN, 0, UB_ADD_LEN + level2Idx);
            }
            if constexpr (EN_DGAMMA) {
                ReduceOut(dgamma_, UB_ADD_LEN, 0, UB_ADD_LEN + level2Idx);
            }
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

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> varInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;
    TBuf<TPosition::VECCALC> binaryAddBuf_;

    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, varGm_;

    LocalTensor<float> mean_;
    LocalTensor<float> var_;
    LocalTensor<float> dgamma_;
    LocalTensor<float> dbeta_;

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
    float epsilon;

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

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1, bool EN_DGAMMA = true, bool EN_DBETA = true>
class BatchNormGradInferDgammaDbetaSplitR1 {
public:
    __aicore__ inline BatchNormGradInferDgammaDbetaSplitR1(TPipe *pipe)
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

    __aicore__ inline void Init(__gm__ uint8_t *dy, __gm__ uint8_t *x, __gm__ uint8_t *mean, __gm__ uint8_t *var,
        __gm__ uint8_t *gamma, __gm__ uint8_t *dx, __gm__ uint8_t *dgamma, __gm__ uint8_t *dbeta,
        __gm__ uint8_t *workspace, const BatchNormGradInferTilingData *tilingData)
    {
        r1Dim_ = tilingData->baseTilingData.r1Dim;
        aDim_ = tilingData->baseTilingData.aDim;
        r0Dim_ = tilingData->baseTilingData.r0Dim;
        epsilon = tilingData->baseTilingData.epsilon;
        blockNum_ = tilingData->blockNum;
        rAlign_ = tilingData->ubRDimFactor;

        binaryAddParam_.binaryAddQuotient = tilingData->generalBinAddTilingData.binaryAddQuotient;
        binaryAddParam_.binaryAddk = tilingData->generalBinAddTilingData.binaryAddk;
        binaryAddParam_.binaryAddLastNum = tilingData->generalBinAddTilingData.binaryAddLastNum;
        aTailCoreNum_ = tilingData->tailBlockNum;
        ubRDimLoopNum_ = tilingData->ubRDimLoopNum;

        nFactor_ = tilingData->ubRDimFactor / r0Dim_;
        tailNFactor_ = r1Dim_ % nFactor_ == 0 ? nFactor_ : r1Dim_ % nFactor_;
        ubRDimTailLoopNum_ = CeilDiv(r1Dim_, nFactor_) - ubRDimLoopNum_;

        gmOffset_ = GetBlockIdx() < tilingData->tailBlockNum ?
            GetBlockIdx() * tilingData->tailBlockDim :
            tilingData->tailBlockNum * tilingData->tailBlockDim +
            (GetBlockIdx() - tilingData->tailBlockNum) * tilingData->formerBlockDim;

        aDimLoopNum_ =
            GetBlockIdx() < tilingData->tailBlockNum ? tilingData->tailBlockDim : tilingData->formerBlockDim;

        tailBinaryAddParam_.binaryAddQuotient = binaryAddParam_.binaryAddQuotient >> 1;
        tailBinaryAddParam_.binaryAddk = binaryAddParam_.binaryAddk > 1 ? binaryAddParam_.binaryAddk - 1 : 0;
        tailBinaryAddParam_.binaryAddLastNum = CalcBinaryAddLastNum(tailBinaryAddParam_.binaryAddQuotient);

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE *)(dy) + gmOffset_ * r0Dim_);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE *)(x) + gmOffset_ * r0Dim_);
        meanGm_.SetGlobalBuffer((__gm__ float *)(mean) + gmOffset_);
        varGm_.SetGlobalBuffer((__gm__ float *)(var) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE *)(gamma) + gmOffset_);
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
        pipe_->InitBuffer(varInQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize_ * BUFFER_NUM);
        pipe_->InitBuffer(cacheBuf_, CACHE_BUFF_SIZE);

        binaryAddTensor_ = binaryAddBuf_.Get<float>();
        cacheLocal_ = cacheBuf_.Get<float>();
        dBetaCacheLocal_ = cacheLocal_[0];
        dGammaCacheLocal_ = cacheLocal_[DGAMA_CACHE_INDEX];
        dBetaFoldCacheLocal_ = cacheLocal_[DBETA_FOLD_CACHE_INDEX];
        dGammaFoldCacheLocal_ = cacheLocal_[DGAMA_FOLD_CACHE_INDEX];
    }

    __aicore__ inline void PreProcess(uint64_t ni)
    {
        if constexpr (EN_DGAMMA) {
            dGammaLocal_ = dgammaOutQue_.template AllocTensor<float>();
        }
        if constexpr (EN_DBETA) {
            dBetaLocal_ = dbetaOutQue_.template AllocTensor<float>();
        }
        Duplicate(cacheLocal_, 0.0f, CACHE_BUFF_SIZE / sizeof(float));
        if constexpr (EN_DGAMMA) {
            meanLocal_ = meanInQue_.template AllocTensor<float>();
            PrepareMean(meanLocal_, ni, 1);
            meanInQue_.EnQue(meanLocal_);

            varLocal_ = varInQue_.template AllocTensor<float>();
            PrepareRstd(varLocal_, ni, 1);
            varInQue_.EnQue(varLocal_);

            meanLocal_ = meanInQue_.template DeQue<float>();
            varLocal_ = varInQue_.template DeQue<float>();
        }
    }

    __aicore__ inline void PostProcess(uint64_t ni)
    {
        // PostProcess
        ReSaveDGammaDBeta(dGammaLocal_, dBetaLocal_);

        if constexpr (EN_DBETA) {
            dbetaOutQue_.EnQue(dBetaLocal_);
            LocalTensor<WEIGHT_TYPE> dBetaLocal = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
            CopyOutDbeta(dBetaLocal, ni, 1);
            dbetaOutQue_.FreeTensor(dBetaLocal);
        }

        if constexpr (EN_DGAMMA) {
            dgammaOutQue_.EnQue(dGammaLocal_);
            LocalTensor<WEIGHT_TYPE> dGammaLocal = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
            CopyOutDgamma(dGammaLocal, ni, 1);
            dgammaOutQue_.FreeTensor(dGammaLocal);
            
            meanInQue_.FreeTensor(meanLocal_);
            varInQue_.FreeTensor(varLocal_);
        }
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
            if constexpr (EN_DBETA) {
                CustomReduceSum(dBetaCacheLocal_[CACHE_LEVEL1_INDEX], dBetaCacheLocal_[CACHE_LEVEL0_INDEX], updateIdx);
            }
            if constexpr (EN_DGAMMA) {
                CustomReduceSum(dGammaCacheLocal_[CACHE_LEVEL1_INDEX], dGammaCacheLocal_[CACHE_LEVEL0_INDEX], updateIdx);
            }
        }
        if (IsNeedUpdateLevel2Cache(basiBlockIdx)) {
            uint64_t updateIdx = (basiBlockIdx >> 16);
            if constexpr (EN_DBETA) {
                CustomReduceSum(dBetaCacheLocal_[CACHE_LEVEL2_INDEX], dBetaCacheLocal_[CACHE_LEVEL1_INDEX], updateIdx);
            }
            if constexpr (EN_DGAMMA) {
                CustomReduceSum(dGammaCacheLocal_[CACHE_LEVEL2_INDEX], dGammaCacheLocal_[CACHE_LEVEL1_INDEX], updateIdx);
            }
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
            if constexpr (EN_DBETA) {
                Add(dBetaCacheLocal_[updateIdx], dBetaCacheLocal_[updateIdx], dBetaFoldCacheLocal_, FOLD_CACHE_CAPACITY);
            }
            if constexpr (EN_DGAMMA) {
                Add(dGammaCacheLocal_[updateIdx], dGammaCacheLocal_[updateIdx], dGammaFoldCacheLocal_, FOLD_CACHE_CAPACITY);
            }
        } else if (basiBlockIdx + 1 == FoldLoopNum) {
            uint32_t updateNum = static_cast<uint32_t>((basiBlockIdx + 1) & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1));
            uint32_t updateIdx = CeilDiv(updateNum, FOLD_CACHE_CAPACITY) * FOLD_CACHE_CAPACITY - FOLD_CACHE_CAPACITY;
                updateNum = updateNum & (FOLD_CACHE_CAPACITY - 1);
            if constexpr (EN_DBETA) {
                Add(dBetaCacheLocal_[updateIdx], dBetaCacheLocal_[updateIdx], dBetaFoldCacheLocal_, updateNum);
            }
            if constexpr (EN_DGAMMA) {
                Add(dGammaCacheLocal_[updateIdx], dGammaCacheLocal_[updateIdx], dGammaFoldCacheLocal_, updateNum);
            }
        }
    }

    __aicore__ inline void ProcessSummation(uint64_t basicBlockNum)
    {
        if (basicBlockNum < ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY) {
            if constexpr (EN_DBETA) {
                CustomReduceSum(dBetaLocal_, dBetaCacheLocal_[CACHE_LEVEL0_INDEX], 0);
            }
            if constexpr (EN_DGAMMA) {
                CustomReduceSum(dGammaLocal_, dGammaCacheLocal_[CACHE_LEVEL0_INDEX], 0);
            }
        } else if (basicBlockNum < ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY * ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY) {
            if constexpr (EN_DBETA) {
                CustomReduceSum(dBetaLocal_, dBetaCacheLocal_[CACHE_LEVEL1_INDEX], 0);
            }
            if constexpr (EN_DGAMMA) {
                CustomReduceSum(dGammaLocal_, dGammaCacheLocal_[CACHE_LEVEL1_INDEX], 0);
            }
        } else {
            if constexpr (EN_DBETA) {
                CustomReduceSum(dBetaLocal_, dBetaCacheLocal_[CACHE_LEVEL2_INDEX], 0);
            }
            if constexpr (EN_DGAMMA) {
                CustomReduceSum(dGammaLocal_, dGammaCacheLocal_[CACHE_LEVEL2_INDEX], 0);
            }
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
        if constexpr (EN_DBETA) {
            CalcDbeta(dyLocal, dBetaCacheLocal_, binaryAddTensor_, basicBlockIdx, r1Factor * r0Dim_, binaryAddParam_);
        }
        if constexpr (EN_DGAMMA) {
            LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
            PrepareInX(xLocal, 0, offset, r1Factor);
            xInQue_.EnQue(xLocal);
            xLocal = xInQue_.template DeQue<DY_TYPE>();
            CalcDgamma(dyLocal, xLocal, meanLocal_, varLocal_, dGammaCacheLocal_, binaryAddTensor_, basicBlockIdx,
                r1Factor * r0Dim_, binaryAddParam_);
            xInQue_.FreeTensor(xLocal);
        }
        dyInQue_.FreeTensor(dyLocal);
    }

    __aicore__ inline void ProcessFoldBlock(uint64_t ni, uint64_t basicBlockIdx, uint64_t r1Factor,
        uint64_t r1TailFactor, uint64_t r1TailTailFactor)
    {
        uint64_t offset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_;
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, offset, r1Factor);
        uint64_t tailOffset = offset + ubRDimLoopNum_ * nFactor_ * r0Dim_ * aDim_;
        PrepareInDy(dyLocal, halfXBufOffset_, tailOffset, r1TailFactor);
        dyInQue_.EnQue(dyLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        
        if constexpr (EN_DBETA) {
            CalcDbetaFold(dyLocal, dBetaCacheLocal_, binaryAddTensor_, basicBlockIdx, r1Factor * r0Dim_,
                r1TailFactor * r0Dim_, tailBinaryAddParam_);
        }
        if constexpr (EN_DGAMMA) {
            LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
            PrepareInX(xLocal, 0, offset, r1Factor);
            PrepareInX(xLocal, halfXBufOffset_, tailOffset, r1TailFactor);
            xInQue_.EnQue(xLocal);
            xLocal = xInQue_.template DeQue<DY_TYPE>();
            CalcDgammaFold(dyLocal, xLocal, meanLocal_, varLocal_, dGammaCacheLocal_, binaryAddTensor_, basicBlockIdx,
                r1Factor * r0Dim_, r1TailFactor * r0Dim_, tailBinaryAddParam_);

            xInQue_.FreeTensor(xLocal);
        }
        dyInQue_.FreeTensor(dyLocal);

        uint64_t foldOffset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_ + r1Factor * r0Dim_ * aDim_;
        uint32_t foldCacheIdx = static_cast<uint32_t>(basicBlockIdx & (FOLD_CACHE_CAPACITY - 1));
        dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, foldOffset, r1Factor);
        uint64_t foldTailOffset = foldOffset + ubRDimLoopNum_ * nFactor_ * r0Dim_ * aDim_;
        if (r1TailTailFactor > 0) {
            PrepareInDy(dyLocal, halfXBufOffset_, foldTailOffset, r1TailTailFactor);
        }
        dyInQue_.EnQue(dyLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        if constexpr (EN_DBETA) {
            CalcDbetaFold(dyLocal, dBetaFoldCacheLocal_, binaryAddTensor_, foldCacheIdx, r1Factor * r0Dim_,
                r1TailTailFactor * r0Dim_, tailBinaryAddParam_);
        }
        if constexpr (EN_DGAMMA) {
            LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
            PrepareInX(xLocal, 0, foldOffset, r1Factor);
            if (r1TailTailFactor > 0) {
                PrepareInX(xLocal, halfXBufOffset_, foldTailOffset, r1TailTailFactor);
            }
            xInQue_.EnQue(xLocal);
            xLocal = xInQue_.template DeQue<DY_TYPE>();
            CalcDgammaFold(dyLocal, xLocal, meanLocal_, varLocal_, dGammaFoldCacheLocal_, binaryAddTensor_, foldCacheIdx,
                r1Factor * r0Dim_, r1TailTailFactor * r0Dim_, tailBinaryAddParam_);
            
            xInQue_.FreeTensor(xLocal);
        }
        dyInQue_.FreeTensor(dyLocal);
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
        CopyInA(rstd, varGm_, offset, a);
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
        const __local_mem__ float *meanAddr, const __local_mem__ float *varAddr, const __local_mem__ float *dgammaAddr,
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
            MicroAPI::RegTensor<float> runningVarValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> binaryAddQ1;
            MicroAPI::RegTensor<float> binaryAddR1;
            MicroAPI::RegTensor<float> binaryAddQ2;
            MicroAPI::RegTensor<float> binaryAddR2;
            MicroAPI::RegTensor<float> vlSum;
            MicroAPI::RegTensor<float> tmp;

            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(runningVarValue, (__local_mem__ float *)(varAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));

            CalRstdByHighPrecision(runningVarValue, rstdValue, epsilon);

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
        const __local_mem__ float *meanAddr, const __local_mem__ float *varAddr, const __local_mem__ float *dgammaAddr,
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
            MicroAPI::RegTensor<float> runningVarValue;
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
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(runningVarValue, (__local_mem__ float *)(varAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));

            CalRstdByHighPrecision(runningVarValue, rstdValue, epsilon);

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
        const __local_mem__ DY_TYPE *xAddr, const __local_mem__ float *meanAddr, const __local_mem__ float *varAddr,
        const __local_mem__ float *dgammaAddr, uint32_t idx, uint32_t r)
    {
        uint32_t idxInLevel0Cache = idx & (ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY - 1);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> x;
            MicroAPI::RegTensor<float> y;
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> runningVarValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> sum;
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0 = r;
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(runningVarValue, (__local_mem__ float *)(varAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));

            CalRstdByHighPrecision(runningVarValue, rstdValue, epsilon);

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
        const __local_mem__ DY_TYPE *xAddr, const __local_mem__ float *meanAddr, const __local_mem__ float *varAddr,
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
            MicroAPI::RegTensor<float> runingVarValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> tmp;
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0 = r;
            uint32_t sreg1 = tailR;

            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(runingVarValue, (__local_mem__ float *)(varAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float *)(meanAddr));

            CalRstdByHighPrecision(runingVarValue, rstdValue, epsilon);

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
        LocalTensor<float> &var, LocalTensor<float> &dgamma, LocalTensor<float> &binaryAdd, uint32_t idx, uint32_t r,
        const BinaryAddParam &binaryAddParam)
    {
        if (r <= VL_FP32) {
            CalcDgammaLessThanVLVF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(), (__local_mem__ DY_TYPE *)x.GetPhyAddr(),
                (__local_mem__ float *)mean.GetPhyAddr(), (__local_mem__ float *)var.GetPhyAddr(),
                (__local_mem__ float *)dgamma.GetPhyAddr(), idx, r);
        } else {
            CalcDgammaVF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(), (__local_mem__ DY_TYPE *)x.GetPhyAddr(),
                (__local_mem__ float *)mean.GetPhyAddr(), (__local_mem__ float *)var.GetPhyAddr(),
                (__local_mem__ float *)dgamma.GetPhyAddr(), (__local_mem__ float *)binaryAdd.GetPhyAddr(), idx, r,
                binaryAddParam);
        }
    }

    __aicore__ inline void CalcDgammaFold(LocalTensor<DY_TYPE> &dy, LocalTensor<DY_TYPE> &x, LocalTensor<float> &mean,
        LocalTensor<float> &var, LocalTensor<float> &dgamma, LocalTensor<float> &binaryAdd, uint32_t idx, uint32_t r,
        uint32_t tailR, const BinaryAddParam &binaryAddParam)
    {
        if (r <= VL_FP32) {
            CalcDgammaLessThanVLFoldVF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(),
                (__local_mem__ DY_TYPE *)x.GetPhyAddr(), (__local_mem__ float *)mean.GetPhyAddr(),
                (__local_mem__ float *)var.GetPhyAddr(), (__local_mem__ float *)dgamma.GetPhyAddr(), idx, r, tailR);
        } else {
            CalcDgammaFoldVF((__local_mem__ DY_TYPE *)dy.GetPhyAddr(), (__local_mem__ DY_TYPE *)x.GetPhyAddr(),
                (__local_mem__ float *)mean.GetPhyAddr(), (__local_mem__ float *)var.GetPhyAddr(),
                (__local_mem__ float *)dgamma.GetPhyAddr(), (__local_mem__ float *)binaryAdd.GetPhyAddr(), idx, r,
                tailR, binaryAddParam);
        }
    }

    __aicore__ inline void ReSaveDGammaDBeta(LocalTensor<float> &dgamma, LocalTensor<float> &dbeta)
    {
        if constexpr (IsSameType<WEIGHT_TYPE, half>::value || IsSameType<WEIGHT_TYPE, bfloat16_t>::value) {
            __local_mem__ float *dgammaAddr = (__local_mem__ float *)dgamma.GetPhyAddr();
            __local_mem__ float *dbetaAddr = (__local_mem__ float *)dbeta.GetPhyAddr();

            __VEC_SCOPE__
            {
                MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
                if constexpr (EN_DGAMMA) {
                    MicroAPI::RegTensor<float> dgammaValue;
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                        (__local_mem__ float *)(dgammaAddr));
                    StoreOneElement<WEIGHT_TYPE>(dgammaAddr, dgammaValue, pregMerge, 0);
                }
                if constexpr (EN_DBETA) {
                    MicroAPI::RegTensor<float> dbetaValue;                    
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue,
                        (__local_mem__ float *)(dbetaAddr));
                    StoreOneElement<WEIGHT_TYPE>(dbetaAddr, dbetaValue, pregMerge, 0);
                }
            }
        }
        return;
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
    TQue<QuePosition::VECIN, BUFFER_NUM> varInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;

    GlobalTensor<DY_TYPE> dyGm_, xGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, varGm_;

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

    float epsilon;
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
    LocalTensor<float> varLocal_;
    LocalTensor<float> meanLocal_;
    LocalTensor<WEIGHT_TYPE> gammaLocal_;
};


template <typename T1, typename T2>
class BatchNormGradInferDx
{
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradInferDx(TPipe* pipeIn, const BatchNormGradInferDxTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
        pipe_ = pipeIn;
        epsilon = tilingData_->epsilon;
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR gamma, GM_ADDR runningVar, GM_ADDR dx)
    {
        dyGm_.SetGlobalBuffer((__gm__ T1*)dy);
        gammaGm_.SetGlobalBuffer((__gm__ T2*)gamma);
        runningVarGm_.SetGlobalBuffer((__gm__ T2*)runningVar);
        dxGm_.SetGlobalBuffer((__gm__ T1*)dx);

        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T2));
        pipe_->InitBuffer(runningVarQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T2));

        int64_t xShapeLen = tilingData_->tileBlockR0Len * tilingData_->tileBlockALen * tilingData_->tileBlockR1Len;
        pipe_->InitBuffer(dxQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
        pipe_->InitBuffer(dyQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= tilingData_->usedCoreNums) {
            return;
        }

        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        int64_t paddingANumSizeT1 = tilingData_->tileBlockAPaddingNum * sizeof(T1) / BLOCK_SIZE;

        // pattern is [B0, A, B1]
        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curB1Idx = curIdx % tilingData_->r0Outer;
            int64_t curAIdx = (curIdx / tilingData_->r0Outer) / tilingData_->r1Outer;
            int64_t curB0Idx = (curIdx / tilingData_->r0Outer) % tilingData_->r1Outer;

            // ping、pang搬运首次或者tile块沿B轴换列时需要拷贝gamma、invstd
            // curIdx 整数倍(tilingData_->r1Outer * tilingData_->r0Outer) 表示一轮循环, 出现换行
            bool needCopy = (curIdx <= beginIdx + 1) || (curIdx % (tilingData_->r1Outer * tilingData_->r0Outer) <= 1);

            // Tile整尾块
            int64_t curTileB0Len =
                curB0Idx == (tilingData_->r1Outer - 1) ? tilingData_->tileBlockR1Tail : tilingData_->tileBlockR1Len;
            int64_t curTileALen =
                curAIdx == (tilingData_->aOuter - 1) ? tilingData_->tileBlockATail : tilingData_->tileBlockALen;
            int64_t curTileB1Len =
                curB1Idx == (tilingData_->r0Outer - 1) ? tilingData_->tileBlockR0Tail : tilingData_->tileBlockR0Len;

            int64_t ubStrideT1 = curAIdx == (tilingData_->aOuter - 1) ? paddingANumSizeT1 : 0;

            // x、y偏移一致，gamma、runningVar偏移一致
            int64_t weightOffset = curAIdx * tilingData_->tileBlockALen;
            int64_t dyOffset =
                // b0 offset
                curB0Idx * tilingData_->tileBlockR1Len * tilingData_->aDim * tilingData_->r0Dim +
                // a offset
                curAIdx * tilingData_->tileBlockALen * tilingData_->r0Dim +
                // b1 offset
                curB1Idx * tilingData_->tileBlockR0Len;

            CopyInDy(dyOffset, curTileB0Len, curTileALen, curTileB1Len, ubStrideT1);
            CopyInGammaVar(needCopy, weightOffset, curTileALen);
            Compute(curTileB0Len, curTileALen, curTileB1Len);
            CopyOutDx(dyOffset, curTileB0Len, curTileALen, curTileB1Len, ubStrideT1);
        }
    }

private:
    __aicore__ inline void CopyInDy(int64_t dyGmOffset, int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len,
                                    int64_t ubStrideT1)
    {
        LocalTensor<T1> dyLocal = dyQueue_.AllocTensor<T1>();
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curTileB0Len;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->aDim * tilingData_->r0Dim * sizeof(T1);
        loopParams.loop1DstStride = tilingData_->tileBlockR0Len * tilingData_->tileBlockALen * sizeof(T1);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPadExtParams<T1> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileALen;
        copyInParams.blockLen = curTileB1Len * sizeof(T1);
        copyInParams.srcStride = (tilingData_->r0Dim - curTileB1Len) * sizeof(T1);
        copyInParams.dstStride = (tilingData_->tileBlockR0Len - curTileB1Len) * sizeof(T1) / BLOCK_SIZE;
        DataCopyPad<T1, PaddingMode::Normal>(dyLocal, dyGm_[dyGmOffset], copyInParams, dataCopyPadExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        dyQueue_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInGammaVar(bool needCopy, int64_t offset, int64_t curTileALen)
    {
        LocalTensor<T2> gammaLocal = gammaQueue_.AllocTensor<T2>();
        LocalTensor<T2> varLocal = runningVarQueue_.AllocTensor<T2>();

        if (needCopy) {
            DataCopyExtParams extParam;
            extParam.blockCount = 1;

            // gamma
            extParam.blockLen = curTileALen * sizeof(T2);
            DataCopyPadExtParams<T2> padExtParam;
            padExtParam.isPad = false;
            DataCopyPad(gammaLocal, gammaGm_[offset], extParam, padExtParam);

            // runningVar
            extParam.blockLen = curTileALen * sizeof(T2);
            DataCopyPadExtParams<T2> padExtParams1;
            padExtParams1.isPad = false;
            DataCopyPad(varLocal, runningVarGm_[offset], extParam, padExtParams1);
        }

        gammaQueue_.EnQue(gammaLocal);
        runningVarQueue_.EnQue(varLocal);
    }

    __aicore__ inline void Compute(int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len)
    {
        LocalTensor<T1> dy = dyQueue_.DeQue<T1>();
        LocalTensor<T2> gamma = gammaQueue_.DeQue<T2>();
        LocalTensor<T2> runningVar = runningVarQueue_.DeQue<T2>();
        LocalTensor<T1> dx = dxQueue_.AllocTensor<T1>();

        __local_mem__ T1* dyLocal = (__local_mem__ T1*)dy.GetPhyAddr();
        __local_mem__ T2* gammaLocal = (__local_mem__ T2*)gamma.GetPhyAddr();
        __local_mem__ T2* varLocal = (__local_mem__ T2*)runningVar.GetPhyAddr();
        __local_mem__ T1* dxLocal = (__local_mem__ T1*)dx.GetPhyAddr();

        VFNormalize(dyLocal, gammaLocal, varLocal, dxLocal, curTileB0Len, curTileALen, curTileB1Len);

        dxQueue_.EnQue(dx);

        dyQueue_.FreeTensor<T1>(dy);
        gammaQueue_.FreeTensor<T2>(gamma);
        runningVarQueue_.FreeTensor<T2>(runningVar);
    }

    __aicore__ inline void VFNormalize(__local_mem__ T1* dyLocal, __local_mem__ T2* gammaLocal,
                                       __local_mem__ T2* varLocal, __local_mem__ T1* dxLocal, uint16_t curTileB0Len,
                                       uint16_t curTileALen, uint16_t curTileB1Len)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> dy;
            RegTensor<float> gamma;
            RegTensor<float> runningVar;
            RegTensor<float> dx;
            RegTensor<float> invstd;

            MaskReg pregMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

            uint16_t loopNum = ops::CeilDiv(curTileB1Len, VL_FP32);
            uint32_t tileBlockALenTmp = static_cast<uint32_t>(tilingData_->tileBlockALen);
            uint32_t tileBlockR0LenTmp = static_cast<uint32_t>(tilingData_->tileBlockR0Len);
            for (uint16_t i = 0; i < curTileALen; i++) {
                // loads runningVar  1->64
                LoadsTensorForDtypeT<T2>(varLocal, runningVar, pregMask, i);
                CalRstdByHighPrecision(runningVar, invstd, epsilon);

                // load gamma 1->64
                LoadsTensorForDtypeT<T2>(gammaLocal, gamma, pregMask, i);
                for (uint16_t j = 0; j < curTileB0Len; j++) {
                    for (uint16_t k = 0; k < loopNum; k++) {
                        uint32_t dyOffset = (j * tileBlockALenTmp + i) * tileBlockR0LenTmp + k * VL_FP32;

                        // load dy
                        LoadOneTensor<T1>(dyLocal, dy, pregMask, dyOffset);

                        // compute
                        Mul(dx, dy, invstd, pregMask);
                        Mul(dx, dx, gamma, pregMask);

                        // copy out
                        StoreOneTensor<T1>(dxLocal, dx, pregMask, dyOffset);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutDx(int64_t dxGmOffset, int64_t curTileB0Len, int64_t curTileALen,
                                     int64_t curTileB1Len, int64_t ubStrideT1)
    {
        LocalTensor<T1> dx = dxQueue_.DeQue<T1>();
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curTileB0Len;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->tileBlockR0Len * tilingData_->tileBlockALen * sizeof(T1);
        loopParams.loop1DstStride = tilingData_->aDim * tilingData_->r0Dim * sizeof(T1);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileALen;
        copyInParams.blockLen = curTileB1Len * sizeof(T1);
        copyInParams.srcStride = (tilingData_->tileBlockR0Len - curTileB1Len) * sizeof(T1) / BLOCK_SIZE;
        copyInParams.dstStride = (tilingData_->r0Dim - curTileB1Len) * sizeof(T1);
        DataCopyPad<T1, PaddingMode::Normal>(dxGm_[dxGmOffset], dx, copyInParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        dxQueue_.FreeTensor(dx);
    }

private:
    const BatchNormGradInferDxTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> gammaQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> runningVarQueue_;

    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dxQueue_;

    GlobalTensor<T1> dyGm_;
    GlobalTensor<T2> gammaGm_;
    GlobalTensor<T2> runningVarGm_;
    GlobalTensor<T1> dxGm_;

    float epsilon;
};

template <typename T1, typename T2, bool SPLIT_R0>
class BatchNormGradInfer
{
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradInfer(){};

    __aicore__ inline BatchNormGradInfer(TPipe* pipeIn, const BatchNormGradInferTilingData* tilingDataIn)
    {
        pipe_ = pipeIn;
        tilingData_ = tilingDataIn;
        enableDx = tilingData_->enableDx;
        enableDgamma = tilingData_->enableDgamma;
        enableDbeta = tilingData_->enableDbeta;
    }

    __aicore__ inline void Process(GM_ADDR dy, GM_ADDR x, GM_ADDR gamma, GM_ADDR save_mean, GM_ADDR runningVar, 
            GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR usrWorkspace) {
        if (enableDx) {
            BatchNormGradInferDx<T1, T2> opDx(pipe_, &tilingData_->baseTilingData);
            opDx.Init(dy, gamma, runningVar, dx);
            opDx.Process();
            PipeBarrier<PIPE_ALL>();
            pipe_->Reset();
        }
        if constexpr (SPLIT_R0) { // split R0
            if (enableDgamma && enableDbeta) {
                BatchNormGradInferDgammaDbetaSplitR0<T1, T2, BUFFER_NUM, true, true> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            } else if (enableDgamma && !enableDbeta) {
                BatchNormGradInferDgammaDbetaSplitR0<T1, T2, BUFFER_NUM, true, false> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            } else if (!enableDgamma && enableDbeta) {
                BatchNormGradInferDgammaDbetaSplitR0<T1, T2, BUFFER_NUM, false, true> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            }
        } else {
            if (enableDgamma && enableDbeta) {
                BatchNormGradInferDgammaDbetaSplitR1<T1, T2, BUFFER_NUM, true, true> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            } else if (enableDgamma && !enableDbeta) {
                BatchNormGradInferDgammaDbetaSplitR1<T1, T2, BUFFER_NUM, true, false> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            } else if (!enableDgamma && enableDbeta) {
                BatchNormGradInferDgammaDbetaSplitR1<T1, T2, BUFFER_NUM, false, true> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            }
        }
    }
private:
    const BatchNormGradInferTilingData* tilingData_;
    TPipe* pipe_;

    bool enableDx;
    bool enableDgamma;
    bool enableDbeta;
};
}  // namespace BatchNormGrad

#endif
