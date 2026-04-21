/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_mx_quant_fp4_split_r.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_MX_QUANT_FP4_SPLIT_R_H
#define ADD_RMS_NORM_DYNAMIC_MX_QUANT_FP4_SPLIT_R_H

#include "add_rms_norm_dynamic_mx_quant_common.h"

namespace AddRmsNormDynamicMxQuant {

template <typename T_X, typename T_GAMMA, typename T_Y>
class AddRmsNormDynamicMxQuantFP4SplitR {
public:
    __aicore__ inline AddRmsNormDynamicMxQuantFP4SplitR(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
        GM_ADDR x, GM_ADDR mxscale, GM_ADDR workspace, GM_ADDR rstd,
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

        // === Setup GM tensors ===
        uint64_t blockOffset = GetBlockIdx() * blockFactor_ * numCol_;
        x1Gm.SetGlobalBuffer((__gm__ T_X*)x1 + blockOffset, mCurCore_ * numCol_);
        x2Gm.SetGlobalBuffer((__gm__ T_X*)x2 + blockOffset, mCurCore_ * numCol_);
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, numCol_);
        if (betaFlag_ != 0) {
            betaGm.SetGlobalBuffer((__gm__ T_GAMMA*)beta, numCol_);
        }
        xOutGm.SetGlobalBuffer((__gm__ T_X*)x + blockOffset, mCurCore_ * numCol_);
        if (rstdFlag_ != 0) {
            rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor_, blockFactor_);
        }
        yFp4Gm.SetGlobalBuffer((__gm__ uint8_t*)y + blockOffset / DIGIT_TWO, mCurCore_ * numCol_ / DIGIT_TWO);
        mxScaleGm.SetGlobalBuffer(
            (__gm__ uint8_t*)mxscale + GetBlockIdx() * blockFactor_ * mxScaleSize_, mCurCore_ * mxScaleSize_);

        // === Compute buffer sizes ===
        uint64_t xBufSize = CeilAlign(baseN_ * sizeof(T_X), UB_BLOCK_SIZE);
        uint64_t xFp32BufSize = CeilAlign(baseN_ * sizeof(float), UB_BLOCK_SIZE);
        uint64_t yTmpBufSize = CeilAlign(baseN_ * sizeof(T_X), UB_BLOCK_SIZE);
        uint64_t rstdBufSize = CeilAlign(baseM_ * sizeof(float), UB_BLOCK_SIZE);
        uint64_t cacheBufSize = CeilAlign(
            static_cast<uint64_t>((resultCacheID_ + 1) * sizeof(float)) * AR_RECOMPUTE_SUM_LEN, UB_BLOCK_SIZE);
        uint64_t binaryAddBufSize = CeilAlign(VL_F32 * DIGIT_TWO * sizeof(float), UB_BLOCK_SIZE);
        // FP4: baseN_ elements -> baseN_/2 bytes (fp4x2 packing)
        uint64_t quantYBufSize = CeilAlign(baseN_ / DIGIT_TWO, UB_BLOCK_SIZE);
        uint64_t maxExpBufSize = CeilAlign(baseNBlockSize_ * sizeof(uint16_t), UB_BLOCK_SIZE);
        uint64_t halfScaleBufSize = maxExpBufSize;
        uint64_t scaleBufSize = CeilAlign(baseNBlockSize_ * sizeof(uint8_t), UB_BLOCK_SIZE);

        // === Init buffers ===
        pPipe->InitBuffer(inQueueX1, DOUBLE_BUFFER_NUM, xBufSize);
        pPipe->InitBuffer(inQueueX2, DOUBLE_BUFFER_NUM, xBufSize);
        if (betaFlag_ != 0) {
            pPipe->InitBuffer(
                inQueueGammabeta, DOUBLE_BUFFER_NUM, DIGIT_TWO * CeilAlign(baseN_ * sizeof(T_GAMMA), UB_BLOCK_SIZE));
        } else {
            pPipe->InitBuffer(inQueueGammabeta, DOUBLE_BUFFER_NUM, CeilAlign(baseN_ * sizeof(T_GAMMA), UB_BLOCK_SIZE));
        }

        pPipe->InitBuffer(outQueueX, DOUBLE_BUFFER_NUM, xBufSize);
        pPipe->InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rstdBufSize);
        pPipe->InitBuffer(outQueueQuantY, DOUBLE_BUFFER_NUM, quantYBufSize);
        pPipe->InitBuffer(mxScaleQueue, DOUBLE_BUFFER_NUM, scaleBufSize);

        pPipe->InitBuffer(xFp32Buf, xFp32BufSize);
        pPipe->InitBuffer(yTmpBuf, yTmpBufSize);
        pPipe->InitBuffer(cacheBuf, cacheBufSize);
        pPipe->InitBuffer(binaryAddBuf, binaryAddBufSize);
        pPipe->InitBuffer(maxExpBuff, maxExpBufSize);
        pPipe->InitBuffer(halfScaleBuff, halfScaleBufSize);
    }

    __aicore__ inline void Process()
    {
        uint32_t mCnt = CeilDiv(mCurCore_, baseM_);
        for (uint64_t i = 0; i < mCnt; ++i) {
            uint32_t curM = (i == mCnt - 1) ? static_cast<uint32_t>(mCurCore_ - (mCnt - 1) * baseM_)
                                             : static_cast<uint32_t>(baseM_);

            // Phase1: compute rstd 
            LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
            for (uint32_t j = 0; j < curM; ++j) {
                int64_t gmRowOffset = (i * baseM_ + j) * numCol_;
                ComputeOneLineXSquareSum(rstdLocal, gmRowOffset, j);
            }
            CalculateRstd(rstdLocal, rstdLocal, curM, avgFactor_, epsilon_);
            outQueueRstd.EnQue<float>(rstdLocal);
            rstdLocal = outQueueRstd.DeQue<float>();

            //  Phase2: compute y、MxQuant、x_out 
            for (uint64_t j = 0; j < nUbLoops_; ++j) {
                uint32_t curN = (j == nUbLoops_ - 1)
                    ? static_cast<uint32_t>(numCol_ - (nUbLoops_ - 1) * baseN_)
                    : static_cast<uint32_t>(baseN_);

                // Load gamma (and beta) per-tile
                LocalTensor<T_GAMMA> gammabetaLocal = inQueueGammabeta.AllocTensor<T_GAMMA>();
                CopyInGammabeta(gammabetaLocal, j * baseN_, curN);
                inQueueGammabeta.EnQue(gammabetaLocal);
                inQueueGammabeta.DeQue<T_GAMMA>();

                for (uint32_t k = 0; k < curM; ++k) {
                    int64_t gmOffset = (i * baseM_ + k) * numCol_ + j * baseN_;

                    // Re-load x1, x2
                    CopyInX(gmOffset, curN, j);
                    LocalTensor<T_X> xLocal1 = inQueueX1.DeQue<T_X>();
                    LocalTensor<T_X> xLocal2 = inQueueX2.DeQue<T_X>();
                    LocalTensor<T_X> xOutLocal = outQueueX.AllocTensor<T_X>();
                    LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
                    CalculateXAdd(xLocal1, xLocal2, xOutLocal, xFp32Local, curN);
                    inQueueX1.FreeTensor(xLocal1);
                    inQueueX2.FreeTensor(xLocal2);
                    outQueueX.EnQue<T_X>(xOutLocal);

                    // CopyOut x_out
                    CopyOutX(gmOffset, curN, j);

                    // Compute y_local
                    LocalTensor<T_X> yLocal = yTmpBuf.Get<T_X>();
                    if ((j == nUbLoops_ - 1) && (numCol_ != numColAlign_)) {
                        Duplicate<T_X>(yLocal, static_cast<T_X>(0), baseN_);
                        PipeBarrier<PIPE_V>();
                    }
                    if (betaFlag_ != 0) {
                        CalculateY<true>(xFp32Local, yLocal, rstdLocal, curN, k);
                    } else {
                        CalculateY<false>(xFp32Local, yLocal, rstdLocal, curN, k);
                    }

                    // MxQuant FP4 with roundMode dispatch
                    if (roundMode_ == MODE_RINT) {
                        DynamicMxQuantPhaseFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(yLocal, j);
                    } else if (roundMode_ == MODE_ROUND) {
                        DynamicMxQuantPhaseFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(yLocal, j);
                    } else if (roundMode_ == MODE_FLOOR) {
                        DynamicMxQuantPhaseFP4<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(yLocal, j);
                    }

                    // CopyOut y_quant and mxscale
                    CopyOutQuantYFP4(gmOffset, curN, j);
                    CopyOutMxScale(i * baseM_ + k, j);
                }

                inQueueGammabeta.FreeTensor(gammabetaLocal);
            }
            // CopyOut rstd
            if (rstdFlag_ != 0) {
                DataCopyExtParams copyParams{
                    static_cast<uint16_t>(1), static_cast<uint32_t>(curM * sizeof(float)), static_cast<uint32_t>(0),
                    static_cast<uint32_t>(0), 0};
                DataCopyPad(rstdGm[i * baseM_], rstdLocal, copyParams);
            }
            outQueueRstd.FreeTensor(rstdLocal);
        }
    }

private:
    __aicore__ inline void ComputeOneLineXSquareSum(
        LocalTensor<float>& rstdLocal, int64_t gmRowOffset, uint32_t rowIndex)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams xDataCopyParams;
        xDataCopyParams.blockCount = 1;
        xDataCopyParams.srcStride = 0;
        xDataCopyParams.dstStride = 0;
        DataCopyParams xFoldDataCopyParams;
        xFoldDataCopyParams.blockCount = 1;
        xFoldDataCopyParams.srcStride = 0;
        xFoldDataCopyParams.dstStride = 0;

        LocalTensor<float> cacheLocal = cacheBuf.Get<float>();
        LocalTensor<float> xFp32Tmp = xFp32Buf.Get<float>();

        for (int64_t r = 0; r < powerSplit_; ++r) {
            int64_t xGmOffset1 = gmRowOffset + baseN_ * r;
            int64_t xGmOffset2 = gmRowOffset + baseN_ * (r + powerSplit_);

            // Step1: Load main tile, compute (x1+x2)² → xFp32Tmp, then Free
            xDataCopyParams.blockLen = baseN_ * sizeof(T_X);
            LocalTensor<T_X> x1Local = inQueueX1.AllocTensor<T_X>();
            DataCopyPad(x1Local, x1Gm[xGmOffset1], xDataCopyParams, padParams);
            inQueueX1.EnQue<T_X>(x1Local);
            x1Local = inQueueX1.DeQue<T_X>();

            LocalTensor<T_X> x2Local = inQueueX2.AllocTensor<T_X>();
            DataCopyPad(x2Local, x2Gm[xGmOffset1], xDataCopyParams, padParams);
            inQueueX2.EnQue<T_X>(x2Local);
            x2Local = inQueueX2.DeQue<T_X>();

            MainBlockSquareVF(x1Local, x2Local, xFp32Tmp, baseN_);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);

            // Step2: Load fold tile, compute (x1Fold+x2Fold)² and accumulate to xFp32Tmp, then Free
            if (r < mainFoldCount_) {
                xFoldDataCopyParams.blockLen = baseN_ * sizeof(T_X);
                LocalTensor<T_X> x1FoldLocal = inQueueX1.AllocTensor<T_X>();
                DataCopyPad(x1FoldLocal, x1Gm[xGmOffset2], xFoldDataCopyParams, padParams);
                inQueueX1.EnQue<T_X>(x1FoldLocal);
                x1FoldLocal = inQueueX1.DeQue<T_X>();

                LocalTensor<T_X> x2FoldLocal = inQueueX2.AllocTensor<T_X>();
                DataCopyPad(x2FoldLocal, x2Gm[xGmOffset2], xFoldDataCopyParams, padParams);
                inQueueX2.EnQue<T_X>(x2FoldLocal);
                x2FoldLocal = inQueueX2.DeQue<T_X>();

                FoldBlockSquareAddVF(x1FoldLocal, x2FoldLocal, xFp32Tmp, baseN_);
                inQueueX1.FreeTensor(x1FoldLocal);
                inQueueX2.FreeTensor(x2FoldLocal);
            } else if (r == mainFoldCount_ && foldTail_ > 0) {
                xFoldDataCopyParams.blockLen = foldTail_ * sizeof(T_X);
                LocalTensor<T_X> x1FoldLocal = inQueueX1.AllocTensor<T_X>();
                DataCopyPad(x1FoldLocal, x1Gm[xGmOffset2], xFoldDataCopyParams, padParams);
                inQueueX1.EnQue<T_X>(x1FoldLocal);
                x1FoldLocal = inQueueX1.DeQue<T_X>();

                LocalTensor<T_X> x2FoldLocal = inQueueX2.AllocTensor<T_X>();
                DataCopyPad(x2FoldLocal, x2Gm[xGmOffset2], xFoldDataCopyParams, padParams);
                inQueueX2.EnQue<T_X>(x2FoldLocal);
                x2FoldLocal = inQueueX2.DeQue<T_X>();

                FoldBlockSquareAddVF(x1FoldLocal, x2FoldLocal, xFp32Tmp, foldTail_);
                inQueueX1.FreeTensor(x1FoldLocal);
                inQueueX2.FreeTensor(x2FoldLocal);
            }
            // reduce sum
            CalculateSquareReduceSum(xFp32Tmp, xFp32Tmp, baseN_);
            int64_t cacheId = GetCacheId(r);
            UpdateCache(cacheLocal, xFp32Tmp, cacheId, AR_RECOMPUTE_SUM_LEN);
        }

        // final accumulated result to rstdLocal
        __local_mem__ float* dstPtr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* cachePtr =
            (__local_mem__ float*)cacheLocal.GetPhyAddr() + resultCacheID_ * AR_RECOMPUTE_SUM_LEN;
        __VEC_SCOPE__
        {
            RegTensor<float> a;
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            DataCopy<float, LoadDist::DIST_NORM>(a, cachePtr);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + rowIndex, a, pregOne);
        }
    }

    __aicore__ inline void CalculateSquareReduceSum(
        LocalTensor<float>& xFp32Local, LocalTensor<float>& xReduceLocal, uint32_t reduceNum)
    {
        LocalTensor<float> binaryAddBuffTmp = binaryAddBuf.Get<float>();
        __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        __local_mem__ float* tmpUb = (__local_mem__ float*)binaryAddBuffTmp.GetPhyAddr();
        __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();

        if (reduceNum <= VL_F32) {
            CalculateSquareReduceSumLessThanVL(xFp32Tmp, xReduceUb, reduceNum);
        } else if (reduceNum <= VL_F32 + VL_F32) {
            CalculateSquareReduceSumLessThanTwoVL(xFp32Tmp, xReduceUb, reduceNum);
        } else if (reduceNum <= VL_F32 * VL_F32 * DIGIT_TWO) {
            CalculateSquareReduceSumCommon<DIGIT_ONE>(xFp32Tmp, xReduceUb, tmpUb, reduceNum);
        } else {
            CalculateSquareReduceSumCommon<DIGIT_TWO>(xFp32Tmp, xReduceUb, tmpUb, reduceNum);
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanVL(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, uint32_t reduceNum)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x, vMean, onesReg;
            uint32_t sreg0 = reduceNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            AscendC::MicroAPI::Duplicate(onesReg, float(1.0), pregOne);

            LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregLoop, 0);
            AscendC::MicroAPI::ReduceSum(vMean, x, pregLoop);
            AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, uint32_t reduceNum)
    {
        uint32_t tailLen = reduceNum - VL_F32;
        __VEC_SCOPE__
        {
            RegTensor<float> x, xFold, sumReg, vMean, onesReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregTail = UpdateMask<float>(tailLen);
            AscendC::MicroAPI::Duplicate(onesReg, float(1.0), pregOne);

            LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregFull, 0);
            LoadTensorForDtypeTIn<float>(xFp32Tmp + VL_F32, xFold, pregTail, 0);
            AscendC::MicroAPI::ShiftLefts(
                (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregTail);
            AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
            AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
            AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateSquareReduceSumCommon(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, __local_mem__ float* tmpUb,
        uint32_t reduceNum)
    {
        uint32_t binaryAddQuotient = binAddQuotient_;
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + VL_F32 - 1) / VL_F32;
        uint32_t lastBinaryAddNum = binaryAddQuotient / VL_F32;

        uint32_t binaryAddRemainder = reduceNum - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_F32 - 1) / VL_F32;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_F32;
        __VEC_SCOPE__
        {
            RegTensor<float> x, xFold, sumReg, vMean, onesReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;
            AscendC::MicroAPI::Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; ++r) {
                uint32_t offset = r * VL_F32;
                LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregFull, offset);
                LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddQuotient, xFold, pregFull, offset);
                AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
                AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
                AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    tmpUb + static_cast<uint32_t>(r), vMean, pregOne);
            }
            uint32_t sregRemainder = binaryAddRemainder - binaryAddRemainderFloorLoop * VL_F32;
            for (uint16_t r = 0;
                 r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); ++r) {
                pregLoop = UpdateMask<float>(sregRemainder);
                LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderFloorLoop * VL_F32, x, pregFull, 0);
                LoadTensorForDtypeTIn<float>(
                    xFp32Tmp + binaryAddRemainderFloorLoop * VL_F32 + binaryAddQuotient, xFold, pregLoop, 0);
                AscendC::MicroAPI::ShiftLefts(
                    (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLoop);
                AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
                AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
                AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    tmpUb + static_cast<uint32_t>(binaryAddRemainderFloorLoop), vMean, pregOne);
            }
            for (uint16_t r = 0;
                 r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop); ++r) {
                LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderCeilLoop * VL_F32, x, pregFull, 0);
                AscendC::MicroAPI::ReduceSum(vMean, x, pregFull);
                AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    tmpUb + static_cast<uint32_t>(binaryAddRemainderCeilLoop + r), vMean, pregOne);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                DataCopy(x, tmpUb);
                AscendC::MicroAPI::ReduceSum(vMean, x, pregLast);
                AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                lastBinaryAddNum -= VL_F32;
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                DataCopy(x, tmpUb);
                DataCopy(xFold, tmpUb + VL_F32);
                AscendC::MicroAPI::ShiftLefts(
                    (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLast);
                AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
                AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
                AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
            }
        }
    }


    template <bool hasBeta>
    __aicore__ inline void CalculateY(
        LocalTensor<float>& xFp32Local, LocalTensor<T_X>& yLocal, LocalTensor<float>& rstdLocal,
        uint32_t curN, uint32_t rowIdx)
    {
        __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaInUb = (__local_mem__ T_GAMMA*)gammaLocal_.GetPhyAddr();
        __local_mem__ T_X* yInUb = (__local_mem__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_GAMMA* betaInUb;
        if constexpr (hasBeta) {
            betaInUb = (__local_mem__ T_GAMMA*)betaLocal_.GetPhyAddr();
        }

        uint16_t loopCols = static_cast<uint16_t>((curN + VL_F32 - 1) / VL_F32);

        __VEC_SCOPE__
        {
            RegTensor<float> xRegFp32, gammaRegFp32, rstdReg, betaRegFp32;
            MaskReg maskReg;

            AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdInUb + rowIdx);
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

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void DynamicMxQuantPhaseFP4(LocalTensor<T_X>& yLocal, uint64_t ubLoopIdx)
    {
        // each call processes a single row's single tile
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

        LocalTensor<uint16_t> maxExpLocal = maxExpBuff.Get<uint16_t>();
        auto srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());

        LocalTensor<uint16_t> mxScaleLocal = mxScaleQueue.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxScaleLocal.GetPhyAddr());

        LocalTensor<uint16_t> halfScaleLocal = halfScaleBuff.Get<uint16_t>();
        auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        LocalTensor<int8_t> outLocal = outQueueQuantY.AllocTensor<int8_t>();
        auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outLocal.GetPhyAddr());

        MxQuantComputeMaxExpOCP<T_X>(srcAddr, maxExpAddr, loopNum);
        MxQuantComputeScaleOCP<T_Y>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);

        srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        MxQuantComputeDataFP4<toBf16RoundMode, roundMode, T_X, T_Y>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, totalCountInUB, loopNum);

        outQueueQuantY.EnQue(outLocal);
        mxScaleQueue.EnQue(mxScaleLocal);
    }

    __aicore__ inline void CopyInGammabeta(LocalTensor<T_GAMMA>& gammabetaLocal, int64_t offset, uint32_t len)
    {
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(len * sizeof(T_GAMMA)),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            0
        };
        DataCopyPadExtParams<T_GAMMA> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T_GAMMA>(0.0)};
        gammaLocal_ = gammabetaLocal;
        DataCopyPad<T_GAMMA>(gammaLocal_, gammaGm[offset], copyParams, padParams);
        if (betaFlag_ != 0) {
            betaLocal_ = gammabetaLocal[CeilAlign(baseN_ * sizeof(T_GAMMA), UB_BLOCK_SIZE) / sizeof(T_GAMMA)];
            DataCopyPad<T_GAMMA>(betaLocal_, betaGm[offset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyInX(int64_t gmOffset, uint32_t curN, uint64_t ubLoopIdx)
    {
        LocalTensor<T_X> xLocal1 = inQueueX1.AllocTensor<T_X>();
        LocalTensor<T_X> xLocal2 = inQueueX2.AllocTensor<T_X>();

        DataCopyExtParams extParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(curN * sizeof(T_X)),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            0
        };
        DataCopyPadExtParams<T_X> padParams{
            false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T_X>(0.0)};

        DataCopyPad(xLocal1, x1Gm[gmOffset], extParams, padParams);
        DataCopyPad(xLocal2, x2Gm[gmOffset], extParams, padParams);
        inQueueX1.EnQue(xLocal1);
        inQueueX2.EnQue(xLocal2);
    }

    __aicore__ inline void CopyOutX(int64_t gmOffset, uint32_t curN, uint64_t ubLoopIdx)
    {
        LocalTensor<T_X> xLocal = outQueueX.DeQue<T_X>();

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(curN * sizeof(T_X)),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            0
        };
        DataCopyPad(xOutGm[gmOffset], xLocal, copyParams);
        outQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutQuantYFP4(int64_t gmOffset, uint32_t curN, uint64_t ubLoopIdx)
    {
        LocalTensor<uint8_t> quantYLocal = outQueueQuantY.DeQue<uint8_t>();
        uint32_t fp4ByteLen = curN / DIGIT_TWO;
        uint32_t srcStride = 0;
        if ((ubLoopIdx == nUbLoops_ - 1) && (numCol_ != numColAlign_)) {
            srcStride = (numColAlign_ - numCol_) / DIGIT_TWO / UB_BLOCK_SIZE;
        }
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(fp4ByteLen),
            static_cast<uint32_t>(srcStride),
            static_cast<uint32_t>(0),
            0
        };
        DataCopyPad<uint8_t>(yFp4Gm[gmOffset / DIGIT_TWO], quantYLocal, copyParams);
        outQueueQuantY.FreeTensor(quantYLocal);
    }

    __aicore__ inline void CopyOutMxScale(uint64_t rowIdx, uint64_t tileIdx)
    {
        LocalTensor<uint8_t> mxScaleLocal = mxScaleQueue.DeQue<uint8_t>();
        uint32_t curScaleSize;
        if (tileIdx == nUbLoops_ - 1) {
            uint32_t curN = static_cast<uint32_t>(numColAlign_ - (nUbLoops_ - 1) * baseN_);
            curScaleSize = CeilDiv(static_cast<uint64_t>(curN), mxBlockSize_);
        } else {
            curScaleSize = CeilDiv(baseN_, mxBlockSize_);
        }
        uint64_t scaleGmOffset = rowIdx * mxScaleSize_ + tileIdx * CeilDiv(baseN_, mxBlockSize_);
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(curScaleSize),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            0
        };
        DataCopyPad<uint8_t, PaddingMode::Compact>(mxScaleGm[scaleGmOffset], mxScaleLocal, copyParams);
        mxScaleQueue.FreeTensor(mxScaleLocal);
    }

private:
    TPipe* pPipe = nullptr;

    // Input Queues
    TQue<QuePosition::VECIN, 1> inQueueX1;
    TQue<QuePosition::VECIN, 1> inQueueX2;
    TQue<QuePosition::VECIN, 1> inQueueGammabeta;

    LocalTensor<T_GAMMA> gammaLocal_;
    LocalTensor<T_GAMMA> betaLocal_;

    // Output Queues
    TQue<QuePosition::VECOUT, 1> outQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;
    TQue<QuePosition::VECOUT, 1> outQueueQuantY;
    TQue<QuePosition::VECOUT, 1> mxScaleQueue;

    // TBuf
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> cacheBuf;
    TBuf<TPosition::VECCALC> binaryAddBuf;
    TBuf<TPosition::VECCALC> yTmpBuf;
    TBuf<TPosition::VECCALC> maxExpBuff;
    TBuf<TPosition::VECCALC> halfScaleBuff;

    // GM Tensors
    GlobalTensor<T_X> x1Gm, x2Gm, xOutGm;
    GlobalTensor<T_GAMMA> gammaGm, betaGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<uint8_t> yFp4Gm, mxScaleGm;

    // Tiling Parameters
    uint64_t numCol_, numColAlign_;
    uint64_t blockFactor_, mLastCore_, mCurCore_;
    uint64_t baseN_, baseM_, baseNBlockSize_;
    uint64_t nUbLoops_;
    uint64_t binAddQuotient_, powerSplit_;
    uint64_t mainFoldCount_, foldTail_;
    int64_t resultCacheID_;
    float epsilon_, avgFactor_;
    uint64_t roundMode_, mxBlockSize_;
    int64_t scaleAlg_;
    uint64_t mxScaleSize_;
    uint32_t betaFlag_, rstdFlag_;
};

} // namespace AddRmsNormDynamicMxQuant
#endif // ADD_RMS_NORM_DYNAMIC_MX_QUANT_FP4_SPLIT_R_H
