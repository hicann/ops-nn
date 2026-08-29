/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_mx_quant_split_r.h
 * \brief Unified SPLIT_R kernel for both FP8 and FP4 output types.
 *        IS_FP4 is derived at compile-time from T_Y via IsFP4Type trait.
 */
#ifndef ADD_RMS_NORM_DYNAMIC_MX_QUANT_SPLIT_R_H
#define ADD_RMS_NORM_DYNAMIC_MX_QUANT_SPLIT_R_H

#include "add_rms_norm_dynamic_mx_quant_common.h"

namespace AddRmsNormDynamicMxQuant {

template <typename T_X, typename T_GAMMA, typename T_Y, bool HAS_X3>
class AddRmsNormDynamicMxQuantSplitR {
public:
    __aicore__ inline AddRmsNormDynamicMxQuantSplitR(TPipe* pipe) { pPipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR x3, GM_ADDR y, GM_ADDR x,
                                GM_ADDR mxscale, GM_ADDR workspace, GM_ADDR rstd,
                                const AddRmsNormDynamicMxQuantSplitRTilingData* tiling)
    {
#if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");

        numCol_ = tiling->numCol;
        numColAlign_ = tiling->numColAlign;
        blockFactor_ = tiling->blockFactor;
        mLastCore_ = tiling->mLastCore;
        baseN_ = tiling->baseN;
        baseNBlockSize_ = tiling->baseNBlockSize;
        baseM_ = tiling->baseM;
        nUbLoops_ = tiling->nUbLoops;
        binAddQuotient_ = tiling->binAddQuotient;
        powerSplit_ = tiling->powerSplit;
        mainFoldCount_ = tiling->mainFoldCount;
        foldTail_ = tiling->foldTail;
        epsilon_ = tiling->epsilon;
        avgFactor_ = tiling->avgFactor;
        roundMode_ = tiling->roundMode;
        mxBlockSize_ = tiling->mxBlockSize;
        scaleAlg_ = tiling->scaleAlg;
        mxScaleSize_ = tiling->mxScaleSize;
        betaFlag_ = tiling->betaFlag;
        rstdFlag_ = tiling->rstdFlag;

        resultCacheID_ = GetCacheId(powerSplit_ - 1);
        mCurCore_ = (GetBlockIdx() == GetBlockNum() - 1) ? mLastCore_ : blockFactor_;

        uint64_t blockOffset = GetBlockIdx() * blockFactor_ * numCol_;
        x1Gm_.SetGlobalBuffer((__gm__ T_X*)x1 + blockOffset, mCurCore_ * numCol_);
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + blockOffset, mCurCore_ * numCol_);
        if constexpr (HAS_X3) {
            x3Gm_.SetGlobalBuffer((__gm__ T_X*)x3 + blockOffset, mCurCore_ * numCol_);
        }
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, numCol_);
        if (betaFlag_ != 0) {
            betaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)beta, numCol_);
        }
        xOutGm_.SetGlobalBuffer((__gm__ T_X*)x + blockOffset, mCurCore_ * numCol_);
        if (rstdFlag_ != 0) {
            rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor_, blockFactor_);
        }

        if constexpr (IsFP4Type<T_Y>::value) {
            yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + blockOffset / DIGIT_TWO, mCurCore_ * numCol_ / DIGIT_TWO);
        } else {
            yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + blockOffset, mCurCore_ * numCol_);
        }
        mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxscale + GetBlockIdx() * blockFactor_ * mxScaleSize_,
                                   mCurCore_ * mxScaleSize_);

        uint64_t xBufSize = CeilAlign(baseN_ * sizeof(T_X), UB_BLOCK_SIZE);
        uint64_t xFp32BufSize = CeilAlign(baseN_ * sizeof(float), UB_BLOCK_SIZE);
        uint64_t yTmpBufSize = CeilAlign(baseN_ * sizeof(T_X), UB_BLOCK_SIZE);
        uint64_t rstdBufSize = CeilAlign(baseM_ * sizeof(float), UB_BLOCK_SIZE);
        uint64_t cacheBufSize = CeilAlign(
            static_cast<uint64_t>((resultCacheID_ + 1) * sizeof(float)) * AR_RECOMPUTE_SUM_LEN, UB_BLOCK_SIZE);
        uint64_t binaryAddBufSize = CeilAlign(VL_F32 * DIGIT_TWO * sizeof(float), UB_BLOCK_SIZE);

        uint64_t quantYBufSize;
        if constexpr (IsFP4Type<T_Y>::value) {
            quantYBufSize = CeilAlign(baseN_ / DIGIT_TWO, UB_BLOCK_SIZE);
        } else {
            quantYBufSize = CeilAlign(baseN_ * sizeof(T_Y), UB_BLOCK_SIZE);
        }
        // MxQuantComputeScaleOCP loads VL_F32 uint16_t elements (128 bytes) via LoadAlign per iteration.
        // Ensure buffer can accommodate at least one such load even when baseNBlockSize is small.
        constexpr uint64_t minScaleBufSize = static_cast<uint64_t>(VL_F32) * sizeof(uint16_t);
        uint64_t maxExpBufSize = CeilAlign(baseNBlockSize_ * sizeof(uint16_t), UB_BLOCK_SIZE);
        if (maxExpBufSize < minScaleBufSize) {
            maxExpBufSize = minScaleBufSize;
        }
        uint64_t halfScaleBufSize = maxExpBufSize;
        uint64_t scaleBufSize = CeilAlign(baseNBlockSize_ * sizeof(uint8_t), UB_BLOCK_SIZE);

        pPipe_->InitBuffer(inQueueX1_, DOUBLE_BUFFER_NUM, xBufSize);
        pPipe_->InitBuffer(inQueueX2_, DOUBLE_BUFFER_NUM, xBufSize);
        if constexpr (HAS_X3) {
            pPipe_->InitBuffer(inQueueX3_, DOUBLE_BUFFER_NUM, xBufSize);
        }
        if (betaFlag_ != 0) {
            pPipe_->InitBuffer(inQueueGammabeta_, DOUBLE_BUFFER_NUM,
                               DIGIT_TWO * CeilAlign(baseN_ * sizeof(T_GAMMA), UB_BLOCK_SIZE));
        } else {
            pPipe_->InitBuffer(inQueueGammabeta_, DOUBLE_BUFFER_NUM,
                               CeilAlign(baseN_ * sizeof(T_GAMMA), UB_BLOCK_SIZE));
        }

        pPipe_->InitBuffer(outQueueX_, DOUBLE_BUFFER_NUM, xBufSize);
        pPipe_->InitBuffer(outQueueRstd_, DOUBLE_BUFFER_NUM, rstdBufSize);
        pPipe_->InitBuffer(outQueueQuantY_, DOUBLE_BUFFER_NUM, quantYBufSize);
        pPipe_->InitBuffer(mxScaleQueue_, DOUBLE_BUFFER_NUM, scaleBufSize);

        pPipe_->InitBuffer(xFp32Buf_, xFp32BufSize);
        pPipe_->InitBuffer(yTmpBuf_, yTmpBufSize);
        pPipe_->InitBuffer(cacheBuf_, cacheBufSize);
        pPipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize);
        pPipe_->InitBuffer(maxExpBuff_, maxExpBufSize);
        pPipe_->InitBuffer(halfScaleBuff_, halfScaleBufSize);

        eventMTE3MTE2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        uint32_t mCnt = CeilDiv(mCurCore_, baseM_);
        for (uint64_t i = 0; i < mCnt; ++i) {
            uint32_t curM = (i == mCnt - 1) ? static_cast<uint32_t>(mCurCore_ - (mCnt - 1) * baseM_) :
                                              static_cast<uint32_t>(baseM_);

            LocalTensor<float> rstdLocal = outQueueRstd_.AllocTensor<float>();
            for (uint32_t j = 0; j < curM; ++j) {
                int64_t gmRowOffset = (i * baseM_ + j) * numCol_;
                ComputeOneLineXSquareSum(rstdLocal, gmRowOffset, j);
            }
            NormCommon::ComputeRstdNewtonRaphson<true, true>(rstdLocal, rstdLocal, curM, epsilon_, avgFactor_, VL_F32);
            outQueueRstd_.EnQue<float>(rstdLocal);
            rstdLocal = outQueueRstd_.DeQue<float>();

            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
            for (uint64_t j = 0; j < nUbLoops_; ++j) {
                uint32_t curN = (j == nUbLoops_ - 1) ? static_cast<uint32_t>(numCol_ - (nUbLoops_ - 1) * baseN_) :
                                                       static_cast<uint32_t>(baseN_);

                LocalTensor<T_GAMMA> gammabetaLocal = inQueueGammabeta_.AllocTensor<T_GAMMA>();
                CopyInGammabeta(gammabetaLocal, j * baseN_, curN);
                inQueueGammabeta_.EnQue(gammabetaLocal);
                inQueueGammabeta_.DeQue<T_GAMMA>();

                for (uint32_t k = 0; k < curM; ++k) {
                    int64_t gmOffset = (i * baseM_ + k) * numCol_ + j * baseN_;

                    LocalTensor<float> xFp32Local = xFp32Buf_.Get<float>();

                    LocalTensor<T_X> x1Local = inQueueX1_.AllocTensor<T_X>();
                    DataCopyPadExtParams<T_X> padParams{false, 0, 0, 0};
                    DataCopyExtParams extParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curN * sizeof(T_X)), 0,
                                                0, 0};
                    DataCopyPad(x1Local, x1Gm_[gmOffset], extParams, padParams);
                    inQueueX1_.EnQue<T_X>(x1Local);
                    x1Local = inQueueX1_.DeQue<T_X>();

                    LocalTensor<T_X> x2Local = inQueueX2_.AllocTensor<T_X>();
                    DataCopyPad(x2Local, x2Gm_[gmOffset], extParams, padParams);
                    inQueueX2_.EnQue<T_X>(x2Local);
                    x2Local = inQueueX2_.DeQue<T_X>();

                    LocalTensor<T_X> x3Local;
                    if constexpr (HAS_X3) {
                        x3Local = inQueueX3_.AllocTensor<T_X>();
                        DataCopyPad(x3Local, x3Gm_[gmOffset], extParams, padParams);
                        inQueueX3_.EnQue<T_X>(x3Local);
                        x3Local = inQueueX3_.DeQue<T_X>();
                    }

                    LocalTensor<T_X> yLocal = yTmpBuf_.Get<T_X>();
                    CalculateXAdd<T_X, HAS_X3, false>(x1Local, x2Local, yLocal, xFp32Local, curN,
                                                      HAS_X3 ? &x3Local : nullptr);

                    inQueueX1_.FreeTensor(x1Local);
                    inQueueX2_.FreeTensor(x2Local);
                    if constexpr (HAS_X3) {
                        inQueueX3_.FreeTensor(x3Local);
                    }
                    if ((j == nUbLoops_ - 1) && (numCol_ != numColAlign_)) {
                        Duplicate<T_X>(yLocal, static_cast<T_X>(0), baseN_);
                        PipeBarrier<PIPE_V>();
                    }
                    if (betaFlag_ != 0) {
                        CalculateY<true>(xFp32Local, yLocal, rstdLocal, curN, k);
                    } else {
                        CalculateY<false>(xFp32Local, yLocal, rstdLocal, curN, k);
                    }
                    DispatchMxQuant(yLocal, j);
                    CopyOutQuantY(gmOffset, curN, j);
                    CopyOutMxScale(i * baseM_ + k, j);
                }

                inQueueGammabeta_.FreeTensor(gammabetaLocal);
            }
            if (rstdFlag_ != 0) {
                DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curM * sizeof(float)),
                                             static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
                DataCopyPad(rstdGm_[i * baseM_], rstdLocal, copyParams);
            }
            outQueueRstd_.FreeTensor(rstdLocal);
        }
    }

private:
    __aicore__ inline void ComputeOneLineXSquareSum(LocalTensor<float>& rstdLocal, int64_t gmRowOffset,
                                                    uint32_t rowIndex)
    {
        DataCopyPadExtParams<T_X> padParams{false, 0, 0, 0};
        DataCopyExtParams xDataCopyExtParams;
        xDataCopyExtParams.blockCount = 1;
        xDataCopyExtParams.srcStride = 0;
        xDataCopyExtParams.dstStride = 0;
        DataCopyExtParams xFoldDataCopyExtParams;
        xFoldDataCopyExtParams.blockCount = 1;
        xFoldDataCopyExtParams.srcStride = 0;
        xFoldDataCopyExtParams.dstStride = 0;

        LocalTensor<float> cacheLocal = cacheBuf_.Get<float>();
        LocalTensor<float> xFp32Tmp = xFp32Buf_.Get<float>();

        for (int64_t r = 0; r < powerSplit_; ++r) {
            int64_t xGmOffset1 = gmRowOffset + baseN_ * r;
            int64_t xGmOffset2 = gmRowOffset + baseN_ * (r + powerSplit_);

            xDataCopyExtParams.blockLen = baseN_ * sizeof(T_X);
            LocalTensor<T_X> x1Local = inQueueX1_.AllocTensor<T_X>();
            DataCopyPad(x1Local, x1Gm_[xGmOffset1], xDataCopyExtParams, padParams);
            inQueueX1_.EnQue<T_X>(x1Local);
            x1Local = inQueueX1_.DeQue<T_X>();

            LocalTensor<T_X> x2Local = inQueueX2_.AllocTensor<T_X>();
            DataCopyPad(x2Local, x2Gm_[xGmOffset1], xDataCopyExtParams, padParams);
            inQueueX2_.EnQue<T_X>(x2Local);
            x2Local = inQueueX2_.DeQue<T_X>();

            LocalTensor<T_X> x3Local;
            if constexpr (HAS_X3) {
                x3Local = inQueueX3_.AllocTensor<T_X>();
                DataCopyPad(x3Local, x3Gm_[xGmOffset1], xDataCopyExtParams, padParams);
                inQueueX3_.EnQue<T_X>(x3Local);
                x3Local = inQueueX3_.DeQue<T_X>();
            }

            LocalTensor<T_X> xOutMainLocal = outQueueX_.AllocTensor<T_X>();
            MainBlockSquareAndWriteXOutVF<T_X, HAS_X3>(x1Local, x2Local, xFp32Tmp, xOutMainLocal, baseN_,
                                                       HAS_X3 ? &x3Local : nullptr);
            outQueueX_.EnQue<T_X>(xOutMainLocal);
            LocalTensor<T_X> xOutMainReady = outQueueX_.DeQue<T_X>();
            DataCopyExtParams mainCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(baseN_ * sizeof(T_X)), 0,
                                             0, 0};
            DataCopyPad(xOutGm_[xGmOffset1], xOutMainReady, mainCopyParams);
            outQueueX_.FreeTensor(xOutMainReady);

            inQueueX1_.FreeTensor(x1Local);
            inQueueX2_.FreeTensor(x2Local);
            if constexpr (HAS_X3) {
                inQueueX3_.FreeTensor(x3Local);
            }

            if (r < mainFoldCount_) {
                xFoldDataCopyExtParams.blockLen = baseN_ * sizeof(T_X);
                LocalTensor<T_X> x1FoldLocal = inQueueX1_.AllocTensor<T_X>();
                DataCopyPad(x1FoldLocal, x1Gm_[xGmOffset2], xFoldDataCopyExtParams, padParams);
                inQueueX1_.EnQue<T_X>(x1FoldLocal);
                x1FoldLocal = inQueueX1_.DeQue<T_X>();

                LocalTensor<T_X> x2FoldLocal = inQueueX2_.AllocTensor<T_X>();
                DataCopyPad(x2FoldLocal, x2Gm_[xGmOffset2], xFoldDataCopyExtParams, padParams);
                inQueueX2_.EnQue<T_X>(x2FoldLocal);
                x2FoldLocal = inQueueX2_.DeQue<T_X>();

                LocalTensor<T_X> x3FoldLocal;
                if constexpr (HAS_X3) {
                    x3FoldLocal = inQueueX3_.AllocTensor<T_X>();
                    DataCopyPad(x3FoldLocal, x3Gm_[xGmOffset2], xFoldDataCopyExtParams, padParams);
                    inQueueX3_.EnQue<T_X>(x3FoldLocal);
                    x3FoldLocal = inQueueX3_.DeQue<T_X>();
                }

                LocalTensor<T_X> xOutFoldLocal = outQueueX_.AllocTensor<T_X>();
                FoldBlockSquareAddAndWriteXOutVF<T_X, HAS_X3>(x1FoldLocal, x2FoldLocal, xFp32Tmp, xOutFoldLocal, baseN_,
                                                              HAS_X3 ? &x3FoldLocal : nullptr);
                outQueueX_.EnQue<T_X>(xOutFoldLocal);
                LocalTensor<T_X> xOutFoldReady = outQueueX_.DeQue<T_X>();
                DataCopyExtParams foldCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(baseN_ * sizeof(T_X)),
                                                 0, 0, 0};
                DataCopyPad(xOutGm_[xGmOffset2], xOutFoldReady, foldCopyParams);
                outQueueX_.FreeTensor(xOutFoldReady);

                inQueueX1_.FreeTensor(x1FoldLocal);
                inQueueX2_.FreeTensor(x2FoldLocal);
                if constexpr (HAS_X3) {
                    inQueueX3_.FreeTensor(x3FoldLocal);
                }
            } else if (r == mainFoldCount_ && foldTail_ > 0) {
                xFoldDataCopyExtParams.blockLen = foldTail_ * sizeof(T_X);
                LocalTensor<T_X> x1FoldLocal = inQueueX1_.AllocTensor<T_X>();
                DataCopyPad(x1FoldLocal, x1Gm_[xGmOffset2], xFoldDataCopyExtParams, padParams);
                inQueueX1_.EnQue<T_X>(x1FoldLocal);
                x1FoldLocal = inQueueX1_.DeQue<T_X>();

                LocalTensor<T_X> x2FoldLocal = inQueueX2_.AllocTensor<T_X>();
                DataCopyPad(x2FoldLocal, x2Gm_[xGmOffset2], xFoldDataCopyExtParams, padParams);
                inQueueX2_.EnQue<T_X>(x2FoldLocal);
                x2FoldLocal = inQueueX2_.DeQue<T_X>();

                LocalTensor<T_X> x3FoldLocal;
                if constexpr (HAS_X3) {
                    x3FoldLocal = inQueueX3_.AllocTensor<T_X>();
                    DataCopyPad(x3FoldLocal, x3Gm_[xGmOffset2], xFoldDataCopyExtParams, padParams);
                    inQueueX3_.EnQue<T_X>(x3FoldLocal);
                    x3FoldLocal = inQueueX3_.DeQue<T_X>();
                }

                LocalTensor<T_X> xOutTailLocal = outQueueX_.AllocTensor<T_X>();
                FoldBlockSquareAddAndWriteXOutVF<T_X, HAS_X3>(x1FoldLocal, x2FoldLocal, xFp32Tmp, xOutTailLocal,
                                                              foldTail_, HAS_X3 ? &x3FoldLocal : nullptr);
                outQueueX_.EnQue<T_X>(xOutTailLocal);
                LocalTensor<T_X> xOutTailReady = outQueueX_.DeQue<T_X>();
                DataCopyExtParams tailCopyParams{static_cast<uint16_t>(1),
                                                 static_cast<uint32_t>(foldTail_ * sizeof(T_X)), 0, 0, 0};
                DataCopyPad(xOutGm_[xGmOffset2], xOutTailReady, tailCopyParams);
                outQueueX_.FreeTensor(xOutTailReady);

                inQueueX1_.FreeTensor(x1FoldLocal);
                inQueueX2_.FreeTensor(x2FoldLocal);
                if constexpr (HAS_X3) {
                    inQueueX3_.FreeTensor(x3FoldLocal);
                }
            }
            NormCommon::NormCommonRegbase::CalculateReduceSum(xFp32Tmp, xFp32Tmp, binaryAddBuf_, baseN_,
                                                              static_cast<uint32_t>(binAddQuotient_));
            int64_t cacheId = GetCacheId(r);
            UpdateCache(cacheLocal, xFp32Tmp, cacheId, AR_RECOMPUTE_SUM_LEN);
        }

        __ubuf__ float* dstPtr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __ubuf__ float* cachePtr = (__ubuf__ float*)cacheLocal.GetPhyAddr() + resultCacheID_ * AR_RECOMPUTE_SUM_LEN;
        __VEC_SCOPE__
        {
            RegTensor<float> a;
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            LoadAlign<float, LoadDist::DIST_NORM>(a, cachePtr);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + rowIndex, a, pregOne);
        }
    }

    __aicore__ inline void DispatchMxQuant(LocalTensor<T_X>& yLocal, uint64_t ubLoopIdx)
    {
        if constexpr (IsFP4Type<T_Y>::value) {
            if (roundMode_ == MODE_RINT) {
                MxQuantPhaseFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(yLocal, ubLoopIdx);
            } else if (roundMode_ == MODE_ROUND) {
                MxQuantPhaseFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(yLocal, ubLoopIdx);
            } else if (roundMode_ == MODE_FLOOR) {
                MxQuantPhaseFP4<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(yLocal, ubLoopIdx);
            }
        } else {
            MxQuantPhaseFP8<RoundMode::CAST_RINT>(yLocal, ubLoopIdx);
        }
    }

    template <bool hasBeta>
    __aicore__ inline void CalculateY(LocalTensor<float>& xFp32Local, LocalTensor<T_X>& yLocal,
                                      LocalTensor<float>& rstdLocal, uint32_t curN, uint32_t rowIdx)
    {
        __ubuf__ float* xFp32Tmp = (__ubuf__ float*)xFp32Local.GetPhyAddr();
        __ubuf__ T_GAMMA* gammaInUb = (__ubuf__ T_GAMMA*)gammaLocal_.GetPhyAddr();
        __ubuf__ T_X* yInUb = (__ubuf__ T_X*)yLocal.GetPhyAddr();
        __ubuf__ float* rstdInUb = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __ubuf__ T_GAMMA* betaInUb;
        if constexpr (hasBeta) {
            betaInUb = (__ubuf__ T_GAMMA*)betaLocal_.GetPhyAddr();
        }

        uint16_t loopCols = static_cast<uint16_t>((curN + VL_F32 - 1) / VL_F32);

        __VEC_SCOPE__
        {
            RegTensor<float> xRegFp32, gammaRegFp32, rstdReg, betaRegFp32;
            MaskReg maskReg;

            AscendC::MicroAPI::LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdInUb + rowIdx);
            uint32_t sregCount = curN;
            for (uint16_t r = 0; r < loopCols; ++r) {
                uint32_t offset = r * VL_F32;
                maskReg = UpdateMask<float>(sregCount);
                LoadTensorForDtypeTIn<float>(xFp32Tmp, xRegFp32, maskReg, offset);
                LoadTensorForDtypeTIn<T_GAMMA>(gammaInUb, gammaRegFp32, maskReg, offset);
                AscendC::MicroAPI::Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
                AscendC::MicroAPI::Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
                if constexpr (hasBeta) {
                    LoadTensorForDtypeTIn<T_GAMMA>(betaInUb, betaRegFp32, maskReg, offset);
                    AscendC::MicroAPI::Add(xRegFp32, xRegFp32, betaRegFp32, maskReg);
                }
                StoreTensorForDtypeTOut<T_X>(yInUb, xRegFp32, maskReg, offset);
            }
        }
    }

    template <AscendC::RoundMode roundMode>
    __aicore__ inline void MxQuantPhaseFP8(LocalTensor<T_X>& yLocal, uint64_t ubLoopIdx)
    {
        uint32_t curBlockNumInColAxis;
        uint32_t curN;
        if (ubLoopIdx == nUbLoops_ - 1) {
            curN = static_cast<uint32_t>(numColAlign_ - (nUbLoops_ - 1) * baseN_);
            curBlockNumInColAxis = CeilDiv(static_cast<uint64_t>(curN), static_cast<uint64_t>(mxBlockSize_));
        } else {
            curN = static_cast<uint32_t>(baseN_);
            curBlockNumInColAxis = CeilDiv(baseN_, mxBlockSize_);
        }

        uint32_t totalScaleInUB = curBlockNumInColAxis;
        uint32_t totalCountInUB = curBlockNumInColAxis * mxBlockSize_;

        uint16_t loopNum = (totalCountInUB + VL_B16 * DIGIT_TWO - 1) / (VL_B16 * DIGIT_TWO);
        uint16_t loopNumScale = (totalScaleInUB + VL_B16 - 1) / VL_B16;
        uint16_t loopNumScale4NV = (totalScaleInUB + VL_F32 - 1) / VL_F32;

        LocalTensor<uint16_t> maxExpLocal = maxExpBuff_.Get<uint16_t>();
        auto srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());

        LocalTensor<uint16_t> mxScaleLocal = mxScaleQueue_.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxScaleLocal.GetPhyAddr());

        LocalTensor<uint16_t> halfScaleLocal = halfScaleBuff_.Get<uint16_t>();
        auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        LocalTensor<int8_t> outLocal = outQueueQuantY_.AllocTensor<int8_t>();
        auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outLocal.GetPhyAddr());

        if (scaleAlg_ == 0) {
            MxQuantComputeMaxExpOCP<T_X>(srcAddr, maxExpAddr, loopNum);
            MxQuantComputeScaleOCP<T_Y>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);
        } else {
            MxQuantComputeMaxExpcuBLAS<T_X>(srcAddr, maxExpAddr, loopNum);
            MxQuantComputeScalecuBLAS<T_X, T_Y>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB,
                                                loopNumScale4NV);
        }

        srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        MxQuantComputeData<roundMode, T_X, T_Y>(srcAddr, halfScaleLocalAddr, outLocalAddr, loopNum);

        outQueueQuantY_.EnQue(outLocal);
        mxScaleQueue_.EnQue(mxScaleLocal);
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void MxQuantPhaseFP4(LocalTensor<T_X>& yLocal, uint64_t ubLoopIdx)
    {
        uint32_t curBlockNumInColAxis;
        uint32_t curN;
        if (ubLoopIdx == nUbLoops_ - 1) {
            curN = static_cast<uint32_t>(numColAlign_ - (nUbLoops_ - 1) * baseN_);
            curBlockNumInColAxis = CeilDiv(static_cast<uint64_t>(curN), static_cast<uint64_t>(mxBlockSize_));
        } else {
            curN = static_cast<uint32_t>(baseN_);
            curBlockNumInColAxis = CeilDiv(baseN_, mxBlockSize_);
        }

        uint32_t totalScaleInUB = curBlockNumInColAxis;
        uint32_t totalCountInUB = curBlockNumInColAxis * mxBlockSize_;

        uint16_t loopNum = (totalCountInUB + VL_B16 * DIGIT_TWO - 1) / (VL_B16 * DIGIT_TWO);
        uint16_t loopNumScale = (totalScaleInUB + VL_B16 - 1) / VL_B16;

        LocalTensor<uint16_t> maxExpLocal = maxExpBuff_.Get<uint16_t>();
        auto srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());

        LocalTensor<uint16_t> mxScaleLocal = mxScaleQueue_.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxScaleLocal.GetPhyAddr());

        LocalTensor<uint16_t> halfScaleLocal = halfScaleBuff_.Get<uint16_t>();
        auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        LocalTensor<int8_t> outLocal = outQueueQuantY_.AllocTensor<int8_t>();
        auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outLocal.GetPhyAddr());

        MxQuantComputeMaxExpOCP<T_X>(srcAddr, maxExpAddr, loopNum);
        MxQuantComputeScaleOCP<T_Y>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);

        srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        MxQuantComputeDataFP4<toBf16RoundMode, roundMode, T_X, T_Y>(srcAddr, halfScaleLocalAddr, outLocalAddr,
                                                                    totalCountInUB, loopNum);

        outQueueQuantY_.EnQue(outLocal);
        mxScaleQueue_.EnQue(mxScaleLocal);
    }

    __aicore__ inline void CopyInGammabeta(LocalTensor<T_GAMMA>& gammabetaLocal, int64_t offset, uint32_t len)
    {
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(T_GAMMA)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPadExtParams<T_GAMMA> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                static_cast<T_GAMMA>(0.0)};
        gammaLocal_ = gammabetaLocal;
        DataCopyPad<T_GAMMA>(gammaLocal_, gammaGm_[offset], copyParams, padParams);
        if (betaFlag_ != 0) {
            betaLocal_ = gammabetaLocal[CeilAlign(baseN_ * sizeof(T_GAMMA), UB_BLOCK_SIZE) / sizeof(T_GAMMA)];
            DataCopyPad<T_GAMMA>(betaLocal_, betaGm_[offset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOutQuantY(int64_t gmOffset, uint32_t curN, uint64_t ubLoopIdx)
    {
        LocalTensor<uint8_t> quantYLocal = outQueueQuantY_.DeQue<uint8_t>();
        if constexpr (IsFP4Type<T_Y>::value) {
            uint32_t fp4ByteLen = curN / DIGIT_TWO;
            uint32_t srcStride = 0;
            if ((ubLoopIdx == nUbLoops_ - 1) && (numCol_ != numColAlign_)) {
                srcStride = (numColAlign_ - numCol_) / DIGIT_TWO / UB_BLOCK_SIZE;
            }
            DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(fp4ByteLen),
                                         static_cast<uint32_t>(srcStride), static_cast<uint32_t>(0), 0};
            DataCopyPad<uint8_t>(yGm_[gmOffset / DIGIT_TWO], quantYLocal, copyParams);
        } else {
            uint32_t srcStride = 0;
            if ((ubLoopIdx == nUbLoops_ - 1) && (numCol_ != numColAlign_)) {
                srcStride = (numColAlign_ - numCol_) * sizeof(uint8_t) / UB_BLOCK_SIZE;
            }
            DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curN),
                                         static_cast<uint32_t>(srcStride), static_cast<uint32_t>(0), 0};
            DataCopyPad<uint8_t>(yGm_[gmOffset], quantYLocal, copyParams);
        }
        outQueueQuantY_.FreeTensor(quantYLocal);
    }

    __aicore__ inline void CopyOutMxScale(uint64_t rowIdx, uint64_t tileIdx)
    {
        LocalTensor<uint8_t> mxScaleLocal = mxScaleQueue_.DeQue<uint8_t>();
        uint32_t curScaleSize;
        if (tileIdx == nUbLoops_ - 1) {
            uint32_t curN = static_cast<uint32_t>(numColAlign_ - (nUbLoops_ - 1) * baseN_);
            curScaleSize = CeilDiv(static_cast<uint64_t>(curN), mxBlockSize_);
        } else {
            curScaleSize = CeilDiv(baseN_, mxBlockSize_);
        }
        uint64_t scaleGmOffset = rowIdx * mxScaleSize_ + tileIdx * CeilDiv(baseN_, mxBlockSize_);
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curScaleSize),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPad<uint8_t, PaddingMode::Compact>(mxScaleGm_[scaleGmOffset], mxScaleLocal, copyParams);
        mxScaleQueue_.FreeTensor(mxScaleLocal);
    }

private:
    TPipe* pPipe_ = nullptr;

    TQue<QuePosition::VECIN, 1> inQueueX1_;
    TQue<QuePosition::VECIN, 1> inQueueX2_;
    TQue<QuePosition::VECIN, 1> inQueueX3_;
    TQue<QuePosition::VECIN, 1> inQueueGammabeta_;

    LocalTensor<T_GAMMA> gammaLocal_;
    LocalTensor<T_GAMMA> betaLocal_;

    TQue<QuePosition::VECOUT, 1> outQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueRstd_;
    TQue<QuePosition::VECOUT, 1> outQueueQuantY_;
    TQue<QuePosition::VECOUT, 1> mxScaleQueue_;

    TBuf<TPosition::VECCALC> xFp32Buf_;
    TBuf<TPosition::VECCALC> cacheBuf_;
    TBuf<TPosition::VECCALC> binaryAddBuf_;
    TBuf<TPosition::VECCALC> yTmpBuf_;
    TBuf<TPosition::VECCALC> maxExpBuff_;
    TBuf<TPosition::VECCALC> halfScaleBuff_;

    GlobalTensor<T_X> x1Gm_, x2Gm_, x3Gm_, xOutGm_;
    GlobalTensor<T_GAMMA> gammaGm_, betaGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<uint8_t> yGm_, mxScaleGm_;

    uint64_t numCol_{0}, numColAlign_{0};
    uint64_t blockFactor_{0}, mLastCore_{0}, mCurCore_{0};
    uint64_t baseN_{0}, baseM_{0}, baseNBlockSize_{0};
    uint64_t nUbLoops_{0};
    uint64_t binAddQuotient_{0}, powerSplit_{0};
    uint64_t mainFoldCount_{0}, foldTail_{0};
    int64_t resultCacheID_{0};
    float epsilon_{1e-6}, avgFactor_{0.0f};
    uint64_t roundMode_{4};
    uint64_t mxBlockSize_{32};
    int64_t scaleAlg_{0};
    uint64_t mxScaleSize_{0};
    uint32_t betaFlag_{0}, rstdFlag_{0};

    int32_t eventMTE3MTE2_ = 0;
};

} // namespace AddRmsNormDynamicMxQuant
#endif // ADD_RMS_NORM_DYNAMIC_MX_QUANT_SPLIT_R_H
