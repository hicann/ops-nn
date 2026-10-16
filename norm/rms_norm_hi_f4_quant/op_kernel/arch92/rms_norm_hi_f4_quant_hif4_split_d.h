/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_HI_F4_QUANT_HIF4_SPLIT_D_H
#define RMS_NORM_HI_F4_QUANT_HIF4_SPLIT_D_H

#include "rms_norm_hi_f4_quant_common.h"

namespace RmsNormHiF4Quant {

template <typename T_X, typename T_GAMMA>
class RmsNormHiF4QuantHif4SplitD {
public:
    __aicore__ inline RmsNormHiF4QuantHif4SplitD(TPipe* pipe) { pPipe = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR scale, GM_ADDR rstd,
                                GM_ADDR workspace, const RmsNormHiF4QuantSplitDTilingData* tiling)
    {
#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __CCE_AICORE__ == 920)
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
        mxBlockSize_ = tiling->mxBlockSize;
        mxScaleSize_ = tiling->mxScaleSize;
        betaFlag_ = tiling->betaFlag;
        rstdFlag_ = tiling->rstdFlag;
        scaleAlg_ = tiling->scaleAlg;
        innerLoopEle_ = tiling->innerLoopEle;
        max8LoopEle_ = tiling->max8LoopEle;
        scaleLoopEle_ = tiling->scaleLoopEle;
        outLoopEle_ = tiling->outLoopEle;
        alignGroupNum_ = tiling->alignGroupNum;
        innerLoopTail_ = tiling->innerLoopTail;
        max8LoopTail_ = tiling->max8LoopTail;
        scaleLoopTail_ = tiling->scaleLoopTail;
        outLoopTail_ = tiling->outLoopTail;

        resultCacheID_ = GetCacheId(powerSplit_ - 1);
        mCurCore_ = (GetBlockIdx() == GetBlockNum() - 1) ? mLastCore_ : blockFactor_;

        uint64_t blockOffset = GetBlockIdx() * blockFactor_ * numCol_;
        xGm.SetGlobalBuffer((__gm__ T_X*)x + blockOffset, mCurCore_ * numCol_);
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, numCol_);
        if (betaFlag_ != 0) {
            betaGm.SetGlobalBuffer((__gm__ T_GAMMA*)beta, numCol_);
        }
        if (rstdFlag_ != 0) {
            rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor_, blockFactor_);
        }
        yFp4Gm.SetGlobalBuffer((__gm__ uint8_t*)y + blockOffset / DIGIT_TWO, mCurCore_ * numCol_ / DIGIT_TWO);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale + GetBlockIdx() * blockFactor_ * mxScaleSize_,
                                mCurCore_ * mxScaleSize_);

        uint64_t xBufSize = CeilAlign(baseN_ * sizeof(T_X), UB_BLOCK_SIZE);
        uint64_t xFp32BufSize = CeilAlign(baseN_ * sizeof(float), UB_BLOCK_SIZE);
        uint64_t rstdBufSize = CeilAlign(baseM_ * sizeof(float), UB_BLOCK_SIZE);
        uint64_t cacheBufSize = CeilAlign(
            static_cast<uint64_t>((resultCacheID_ + 1) * sizeof(float)) * AR_RECOMPUTE_SUM_LEN, UB_BLOCK_SIZE);
        uint64_t binaryAddBufSize = CeilAlign(VL_F32 * DIGIT_TWO * sizeof(float), UB_BLOCK_SIZE);
        uint64_t quantYBufSize = CeilAlign(baseN_ / DIGIT_TWO, UB_BLOCK_SIZE);

        pPipe->InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, xBufSize);
        if (betaFlag_ != 0) {
            pPipe->InitBuffer(inQueueGammabeta, DOUBLE_BUFFER_NUM,
                              DIGIT_TWO * CeilAlign(baseN_ * sizeof(T_GAMMA), UB_BLOCK_SIZE));
        } else {
            pPipe->InitBuffer(inQueueGammabeta, DOUBLE_BUFFER_NUM, CeilAlign(baseN_ * sizeof(T_GAMMA), UB_BLOCK_SIZE));
        }
        pPipe->InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rstdBufSize);
        pPipe->InitBuffer(outQueueQuantY, DOUBLE_BUFFER_NUM, quantYBufSize);

        int64_t totalGroups = static_cast<int64_t>(1) * static_cast<int64_t>(baseNBlockSize_);
        int64_t alignTotalGroups = CeilAlign(totalGroups, HIF4_BLOCK_SIZE);
        int64_t alignTo256UbFormer = CeilAlign(alignTotalGroups, DQ_HIFP4_SCALE_ALIGN);
        pPipe->InitBuffer(outQueueScale, DOUBLE_BUFFER_NUM, alignTo256UbFormer * sizeof(float));

        pPipe->InitBuffer(xFp32Buf, xFp32BufSize);
        pPipe->InitBuffer(cacheBuf, cacheBufSize);
        pPipe->InitBuffer(binaryAddBuf, binaryAddBufSize);
        pPipe->InitBuffer(yTmpBuf, xBufSize);

        bufE8Buffer_ = CeilAlign(totalGroups, HIF4_BLOCK_SIZE);
        buf1E8Buffer_ = CeilAlign(totalGroups, HIF4_BLOCK_SIZE);
        buf1E16Buffer_ = CeilAlign(totalGroups, HIF4_BLOCK_SIZE);
        maxPer4Buffer_ = CeilAlign(totalGroups, DQ_HIFP4_MAXPER4_ALIGN);
        maxPer8Buffer_ = CeilAlign(totalGroups, DQ_HIFP4_MAXPER8_ALIGN);
        tmpScaleBuffer_ = CeilAlign(totalGroups, DQ_HIFP4_SCALETMP_ALIGN);

        pPipe->InitBuffer(bufE8_, bufE8Buffer_ * sizeof(int16_t));
        pPipe->InitBuffer(buf1E8_, buf1E8Buffer_ * sizeof(int32_t));
        pPipe->InitBuffer(buf1E16_, buf1E16Buffer_ * sizeof(int16_t));
        pPipe->InitBuffer(bufMaxPer4_, maxPer4Buffer_ * 16 * sizeof(int16_t));
        pPipe->InitBuffer(bufMaxPer8_, maxPer8Buffer_ * 8 * sizeof(int16_t));
        pPipe->InitBuffer(bufTmpScale_, tmpScaleBuffer_ * 32 * sizeof(int16_t));

        sysRegLen_ = 256;
        gmCopyInRegStep_ = sysRegLen_ / sizeof(float) * 2;
        gmCopyOutRegStep_ = sysRegLen_ / sizeof(int32_t) / DQ_HIFP4_CAST_COPYOUT_PACK4_NUM;
        gmCopyOutScaleRegStep_ = sysRegLen_ / sizeof(float);
        gmCopyInRegStep2_ = sysRegLen_ / sizeof(T_X);
        gmCopyTmpE1_ = sysRegLen_ * DIGIT_FOUR / HIF4_BLOCK_SIZE / DIGIT_TWO;
        gmCopyTmpE8_ = sysRegLen_ * DIGIT_FOUR / DIGIT_EIGHT;
        gmCopyTmpE16_ = sysRegLen_ * DIGIT_FOUR / DIGIT_FOUR;
        gmCopyTmpBufE8Step_ = sysRegLen_ * DIGIT_TWO;
        gmCopyTmpBufOtherStep_ = sysRegLen_;
    }

    __aicore__ inline void Process()
    {
        uint32_t mCnt = CeilDiv(mCurCore_, baseM_);
        for (uint64_t i = 0; i < mCnt; ++i) {
            uint32_t curM = (i == mCnt - 1) ? static_cast<uint32_t>(mCurCore_ - (mCnt - 1) * baseM_) :
                                              static_cast<uint32_t>(baseM_);

            LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
            for (uint32_t j = 0; j < curM; ++j) {
                int64_t gmRowOffset = (i * baseM_ + j) * numCol_;
                ComputeOneLineXSquareSum(rstdLocal, gmRowOffset, j);
            }
            CalculateRstd(rstdLocal, rstdLocal, curM, avgFactor_, epsilon_);
            outQueueRstd.EnQue<float>(rstdLocal);
            rstdLocal = outQueueRstd.DeQue<float>();

            for (uint64_t j = 0; j < nUbLoops_; ++j) {
                uint32_t curN = (j == nUbLoops_ - 1) ? static_cast<uint32_t>(numCol_ - (nUbLoops_ - 1) * baseN_) :
                                                       static_cast<uint32_t>(baseN_);

                LocalTensor<T_GAMMA> gammabetaLocal = inQueueGammabeta.AllocTensor<T_GAMMA>();
                CopyInGammabeta(gammabetaLocal, j * baseN_, curN);
                inQueueGammabeta.EnQue(gammabetaLocal);
                inQueueGammabeta.DeQue<T_GAMMA>();

                for (uint32_t k = 0; k < curM; ++k) {
                    int64_t gmOffset = (i * baseM_ + k) * numCol_ + j * baseN_;

                    CopyInX(gmOffset, curN);
                    LocalTensor<T_X> xLocal = inQueueX.DeQue<T_X>();
                    LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
                    CalculateCopyInToFP32(xLocal, xFp32Local, curN);
                    inQueueX.FreeTensor(xLocal);

                    LocalTensor<T_X> yLocal = yTmpBuf.Get<T_X>();
                    if (j == nUbLoops_ - 1 && numCol_ != numColAlign_) {
                        Duplicate<T_X>(yLocal, static_cast<T_X>(0), baseN_);
                        PipeBarrier<PIPE_V>();
                    }
                    if (betaFlag_ != 0) {
                        CalculateY<true>(xFp32Local, yLocal, rstdLocal, curN, k);
                    } else {
                        CalculateY<false>(xFp32Local, yLocal, rstdLocal, curN, k);
                    }

                    int64_t curGroups = CeilDiv(static_cast<uint64_t>(curN), mxBlockSize_);
                    int64_t hif4Inner = (curGroups + HIF4_GROUP_PER_INNER_LOOP - 1) / HIF4_GROUP_PER_INNER_LOOP;
                    int64_t hif4Max8 = (curGroups + HIF4_GROUP_PER_MAX8_LOOP - 1) / HIF4_GROUP_PER_MAX8_LOOP;
                    int64_t hif4Scale = (curGroups + HIF4_GROUP_PER_SCALE_LOOP - 1) / HIF4_GROUP_PER_SCALE_LOOP;
                    int64_t hif4Out = (curGroups + HIF4_GROUP_PER_OUT_LOOP - 1) / HIF4_GROUP_PER_OUT_LOOP;

                    HiF4Quant(yLocal, hif4Inner, hif4Max8, hif4Scale, hif4Out);

                    CopyOutQuantYFP4(gmOffset, curN);
                    CopyOutScale(i * baseM_ + k, j);
                }

                inQueueGammabeta.FreeTensor(gammabetaLocal);
            }

            if (rstdFlag_ != 0) {
                DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curM * sizeof(float)),
                                             static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
                DataCopyPad(rstdGm[i * baseM_], rstdLocal, copyParams);
            }
            outQueueRstd.FreeTensor(rstdLocal);
        }
    }

private:
    __aicore__ inline void ComputeOneLineXSquareSum(LocalTensor<float>& rstdLocal, int64_t gmRowOffset,
                                                    uint32_t rowIndex)
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
            int64_t xGmOffset = gmRowOffset + baseN_ * r;
            int64_t xGmOffsetFold = gmRowOffset + baseN_ * (r + powerSplit_);

            xDataCopyParams.blockLen = baseN_ * sizeof(T_X);
            LocalTensor<T_X> xLocal = inQueueX.AllocTensor<T_X>();
            DataCopyPad(xLocal, xGm[xGmOffset], xDataCopyParams, padParams);
            inQueueX.EnQue<T_X>(xLocal);
            xLocal = inQueueX.DeQue<T_X>();

            CalculateCopyInToFP32(xLocal, xFp32Tmp, baseN_);
            inQueueX.FreeTensor(xLocal);

            if (r < mainFoldCount_) {
                xFoldDataCopyParams.blockLen = baseN_ * sizeof(T_X);
                LocalTensor<T_X> xFoldLocal = inQueueX.AllocTensor<T_X>();
                DataCopyPad(xFoldLocal, xGm[xGmOffsetFold], xFoldDataCopyParams, padParams);
                inQueueX.EnQue<T_X>(xFoldLocal);
                xFoldLocal = inQueueX.DeQue<T_X>();

                FoldBlockSquareAdd(xFoldLocal, xFp32Tmp, baseN_);
                inQueueX.FreeTensor(xFoldLocal);
            } else if (r == mainFoldCount_ && foldTail_ > 0) {
                xFoldDataCopyParams.blockLen = foldTail_ * sizeof(T_X);
                LocalTensor<T_X> xFoldLocal = inQueueX.AllocTensor<T_X>();
                DataCopyPad(xFoldLocal, xGm[xGmOffsetFold], xFoldDataCopyParams, padParams);
                inQueueX.EnQue<T_X>(xFoldLocal);
                xFoldLocal = inQueueX.DeQue<T_X>();

                FoldBlockSquareAdd(xFoldLocal, xFp32Tmp, foldTail_);
                inQueueX.FreeTensor(xFoldLocal);
            }

            LocalTensor<float> xReduceDummy = outQueueRstd.AllocTensor<float>();
            LocalTensor<float> binaryAddLocal = binaryAddBuf.Get<float>();
            CalculateSquareReduceSum(xFp32Tmp, xReduceDummy, binaryAddLocal, static_cast<uint32_t>(baseN_),
                                     static_cast<uint32_t>(binAddQuotient_));
            outQueueRstd.FreeTensor(xReduceDummy);

            int64_t cacheId = GetCacheId(r);
            UpdateCache(cacheLocal, xReduceDummy, cacheId, AR_RECOMPUTE_SUM_LEN);
        }

        __local_mem__ float* dstPtr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* cachePtr = (__local_mem__ float*)cacheLocal.GetPhyAddr() +
                                        resultCacheID_ * AR_RECOMPUTE_SUM_LEN;
        __VEC_SCOPE__
        {
            RegTensor<float> a;
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            DataCopy<float, LoadDist::DIST_NORM>(a, cachePtr);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + rowIndex, a, pregOne);
        }
    }

    __aicore__ inline void FoldBlockSquareAdd(LocalTensor<T_X>& xFoldLocal, LocalTensor<float>& xFp32Tmp,
                                              uint32_t tailCount)
    {
        __local_mem__ T_X* xFoldInUb = (__local_mem__ T_X*)xFoldLocal.GetPhyAddr();
        __local_mem__ float* xFp32TmpBuf = (__local_mem__ float*)xFp32Tmp.GetPhyAddr();
        uint16_t tailLoops = (tailCount + VL_F32 - 1) / VL_F32;
        uint32_t sregTail = tailCount;
        __VEC_SCOPE__
        {
            RegTensor<float> xFoldReg, foldSquare, mainReg, sum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < tailLoops; ++i) {
                pregLoop = UpdateMask<float>(sregTail);
                uint32_t offset = i * VL_F32;
                LoadTensorForDtypeTIn<T_X>(xFoldInUb, xFoldReg, pregLoop, offset);
                AscendC::MicroAPI::Mul(foldSquare, xFoldReg, xFoldReg, pregLoop);
                AscendC::MicroAPI::DataCopy(mainReg, xFp32TmpBuf + offset);
                AscendC::MicroAPI::Add(sum, mainReg, foldSquare, pregLoop);
                AscendC::MicroAPI::Select(sum, sum, mainReg, pregLoop);
                AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_NORM_B32>(xFp32TmpBuf + offset, sum, pregLoop);
            }
        }
    }

    template <bool hasBeta>
    __aicore__ inline void CalculateY(LocalTensor<float>& xFp32Local, LocalTensor<T_X>& yLocal,
                                      LocalTensor<float>& rstdLocal, uint32_t curN, uint32_t rowIdx)
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

    __aicore__ inline void CopyInX(int64_t gmOffset, uint32_t curN)
    {
        LocalTensor<T_X> xLocal = inQueueX.AllocTensor<T_X>();

        DataCopyExtParams extParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curN * sizeof(T_X)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPadExtParams<T_X> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                            static_cast<T_X>(0.0)};

        DataCopyPad(xLocal, xGm[gmOffset], extParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGammabeta(LocalTensor<T_GAMMA>& gammabetaLocal, int64_t offset, uint32_t len)
    {
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(T_GAMMA)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPadExtParams<T_GAMMA> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                static_cast<T_GAMMA>(0.0)};
        gammaLocal_ = gammabetaLocal;
        DataCopyPad<T_GAMMA>(gammaLocal_, gammaGm[offset], copyParams, padParams);
        if (betaFlag_ != 0) {
            betaLocal_ = gammabetaLocal[CeilAlign(baseN_ * sizeof(T_GAMMA), UB_BLOCK_SIZE) / sizeof(T_GAMMA)];
            DataCopyPad<T_GAMMA>(betaLocal_, betaGm[offset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOutQuantYFP4(int64_t gmOffset, uint32_t curN)
    {
        LocalTensor<uint8_t> quantYLocal = outQueueQuantY.DeQue<uint8_t>();
        uint32_t fp4ByteLen = curN / DIGIT_TWO;
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(fp4ByteLen),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPad<uint8_t>(yFp4Gm[gmOffset / DIGIT_TWO], quantYLocal, copyParams);
        outQueueQuantY.FreeTensor(quantYLocal);
    }

    __aicore__ inline void CopyOutScale(uint64_t rowIdx, uint64_t tileIdx)
    {
        LocalTensor<float> scaleLocal = outQueueScale.DeQue<float>();
        uint32_t curScaleSize;
        if (tileIdx == nUbLoops_ - 1) {
            uint32_t curN = static_cast<uint32_t>(numCol_ - (nUbLoops_ - 1) * baseN_);
            curScaleSize = CeilDiv(static_cast<uint64_t>(curN), mxBlockSize_);
        } else {
            curScaleSize = CeilDiv(baseN_, mxBlockSize_);
        }
        uint64_t scaleGmOffset = rowIdx * mxScaleSize_ + tileIdx * CeilDiv(baseN_, mxBlockSize_);
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curScaleSize * sizeof(float)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPad<float, PaddingMode::Compact>(scaleGm[scaleGmOffset], scaleLocal, copyParams);
        outQueueScale.FreeTensor(scaleLocal);
    }

    __aicore__ inline void HiF4Quant(LocalTensor<T_X>& yLocal, int64_t innerLoop, int64_t max8Loop, int64_t scaleLoop,
                                     int64_t outLoop)
    {
        LocalTensor<uint16_t> e8buf = bufE8_.Get<uint16_t>();
        LocalTensor<uint8_t> e1p8buf = buf1E8_.Get<uint8_t>();
        LocalTensor<uint8_t> e1p16buf = buf1E16_.Get<uint8_t>();
        LocalTensor<uint16_t> tmpscale = bufTmpScale_.Get<uint16_t>();
        LocalTensor<uint16_t> maxper4 = bufMaxPer4_.Get<uint16_t>();
        LocalTensor<uint16_t> maxper8 = bufMaxPer8_.Get<uint16_t>();

        auto xAddr1 = (__ubuf__ float*)yLocal.GetPhyAddr();
        auto xAddr2 = (__ubuf__ T_X*)yLocal.GetPhyAddr();

        LocalTensor<uint8_t> quantYLocal = outQueueQuantY.AllocTensor<uint8_t>();
        auto yAddr = (__ubuf__ int32_t*)quantYLocal.GetPhyAddr();

        LocalTensor<float> scaleLocal = outQueueScale.AllocTensor<float>();
        auto scaleAddr = (__ubuf__ uint16_t*)scaleLocal.GetPhyAddr();

        auto e8BufAddr = (__ubuf__ uint16_t*)e8buf.GetPhyAddr();
        auto e8BufAddr2 = (__ubuf__ uint8_t*)e8buf.GetPhyAddr();
        auto e1p8BufAddr = (__ubuf__ uint32_t*)e1p8buf.GetPhyAddr();
        auto e1p16BufAddr = (__ubuf__ uint8_t*)e1p16buf.GetPhyAddr();
        auto tmpScaleAddr = (__ubuf__ uint16_t*)tmpscale.GetPhyAddr();
        auto maxPer4BufAddr = (__ubuf__ uint16_t*)maxper4.GetPhyAddr();
        auto maxPer8BufAddr = (__ubuf__ uint16_t*)maxper8.GetPhyAddr();

        auto e8BufRead1Addr = e8BufAddr;
        auto e8BufRead2Addr = e8BufAddr2;
        auto e1p8BufReadAddr = (__ubuf__ uint8_t*)e1p8buf.GetPhyAddr();
        auto e1p16BufReadAddr = e1p16BufAddr;
        auto tmpScaleReadAddr = tmpScaleAddr;
        auto maxPer4BufReadAddr = maxPer4BufAddr;
        auto maxPer4BufRead2Addr = maxPer4BufAddr;
        auto maxPer8BufReadAddr = maxPer8BufAddr;

        ComputeMaxPer4(static_cast<uint16_t>(innerLoop), xAddr1, maxPer4BufAddr);
        ComputeMaxPer8(static_cast<uint16_t>(max8Loop), maxPer4BufReadAddr, e8BufAddr, maxPer8BufAddr);
        ComputeTmpScale(static_cast<uint16_t>(innerLoop), maxPer4BufRead2Addr, maxPer8BufReadAddr, e8BufRead1Addr,
                        e1p8BufAddr, e1p16BufAddr, tmpScaleAddr);
        ComputeScale(static_cast<uint16_t>(scaleLoop), e8BufRead2Addr, e1p8BufReadAddr, e1p16BufReadAddr, scaleAddr);
        ComputeOut(static_cast<uint16_t>(outLoop), xAddr2, tmpScaleReadAddr, yAddr);

        outQueueQuantY.EnQue(quantYLocal);
        outQueueScale.EnQue(scaleLocal);
    }

    __aicore__ inline void ComputeMaxPer4(uint16_t regLoop, __ubuf__ float* xAddr1, __ubuf__ uint16_t* maxPer4BufAddr)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T_X> vregIn1;
            MicroAPI::RegTensor<T_X> vregIn2;
            MicroAPI::RegTensor<T_X> vregIn3;
            MicroAPI::RegTensor<T_X> vregIn4;
            MicroAPI::RegTensor<T_X> vregMax1;
            MicroAPI::RegTensor<T_X> vregMax2;
            MicroAPI::RegTensor<T_X> vregOut;
            MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
            AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
            AscendC::MicroAPI::Duplicate(expMaskBF16, 0x7fff);
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
            static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
                MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    (MicroAPI::RegTensor<float>&)vregIn1, (MicroAPI::RegTensor<float>&)vregIn2, xAddr1,
                    gmCopyInRegStep_);
                AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1,
                                       (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1, expMaskBF16, maskAll);
                AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2,
                                       (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2, expMaskBF16, maskAll);
                MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    (MicroAPI::RegTensor<float>&)vregIn3, (MicroAPI::RegTensor<float>&)vregIn4, xAddr1,
                    gmCopyInRegStep_);
                AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3,
                                       (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3, expMaskBF16, maskAll);
                AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4,
                                       (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4, expMaskBF16, maskAll);
                MicroAPI::Max(vregMax1, vregIn1, vregIn2, maskAll);
                MicroAPI::Max(vregMax2, vregIn3, vregIn4, maskAll);
                MicroAPI::DeInterleave(vregMax1, vregMax2, vregMax1, vregMax2);
                MicroAPI::Max(vregOut, vregMax1, vregMax2, maskAll);
                if constexpr (IsSame<T_X, half>::value) {
                    AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vregOutbf16, vregOut, maskAll);
                    AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        maxPer4BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregOutbf16, gmCopyInRegStep2_, maskAll);
                } else {
                    AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        maxPer4BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregOut, gmCopyInRegStep2_, maskAll);
                }
            }
        }
    }

    __aicore__ inline void ComputeMaxPer8(uint16_t maxPer8RegLoop, __ubuf__ uint16_t* maxPer4BufReadAddr,
                                          __ubuf__ uint16_t* e8BufAddr, __ubuf__ uint16_t* maxPer8BufAddr)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper4;
            MicroAPI::RegTensor<bfloat16_t> vreg1Maxper4;
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper8;
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper64;
            MicroAPI::RegTensor<bfloat16_t> vreg1Maxper64;
            MicroAPI::RegTensor<uint16_t> vreg0E6m2;
            MicroAPI::RegTensor<uint16_t> vreg1E6m2;
            MicroAPI::RegTensor<uint16_t> vregNumscale;
            MicroAPI::Duplicate(vregNumscale, 0x3e12);
            MicroAPI::UnalignReg ureg1;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
            static constexpr AscendC::MicroAPI::CastTrait castTraitE6M2 = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
            for (uint16_t loopIdx = 0; loopIdx < maxPer8RegLoop; loopIdx++) {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                   MicroAPI::LoadDist::DIST_DINTLV_B16>(
                    (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, (MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4,
                    maxPer4BufReadAddr, gmCopyInRegStep2_ * DIGIT_TWO);
                MicroAPI::Max(vreg0Maxper8, vreg0Maxper4, vreg1Maxper4, maskAll);
                MicroAPI::Interleave(vreg0Maxper4, vreg1Maxper4, vreg0Maxper4, vreg1Maxper4);

                MicroAPI::ReduceMaxWithDataBlock<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64,
                                                           (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, maskAll);
                MicroAPI::ReduceMaxWithDataBlock<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg1Maxper64,
                                                           (MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4, maskAll);

                MicroAPI::Mul(vreg0Maxper64, vreg0Maxper64, (MicroAPI::RegTensor<bfloat16_t>&)vregNumscale, maskAll);
                MicroAPI::Mul(vreg1Maxper64, vreg1Maxper64, (MicroAPI::RegTensor<bfloat16_t>&)vregNumscale, maskAll);

                MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0E6m2,
                                                                      vreg0Maxper64, maskAll);
                MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg1E6m2,
                                                                      vreg1Maxper64, maskAll);

                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxPer8BufAddr, (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper8, gmCopyInRegStep2_, maskAll);
                AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    e8BufAddr, vreg0E6m2, ureg1, 8);
                AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    e8BufAddr, vreg1E6m2, ureg1, 8);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(e8BufAddr, ureg1, 0);
        }
    }

    __aicore__ inline void ComputeTmpScale(uint16_t regLoop, __ubuf__ uint16_t* maxPer4BufRead2Addr,
                                           __ubuf__ uint16_t* maxPer8BufReadAddr, __ubuf__ uint16_t* e8BufRead1Addr,
                                           __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
                                           __ubuf__ uint16_t* tmpScaleAddr)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper4;
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper8;
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper64;
            MicroAPI::RegTensor<bfloat16_t> vb16_Ea_rec_bf16;
            MicroAPI::RegTensor<bfloat16_t> vreg0E6m2;
            MicroAPI::RegTensor<bfloat16_t> vregTmpScale;

            float number1 = 1.0;
            MicroAPI::RegTensor<bfloat16_t> vregNum1;
            MicroAPI::Duplicate(vregNum1, number1);
            float numberhalf = 0.5;
            MicroAPI::RegTensor<bfloat16_t> vregNumhalf;
            MicroAPI::Duplicate(vregNumhalf, static_cast<bfloat16_t>(numberhalf));

            // Eb/Ec 门限：scale_alg==1(unbiased) 用 3.75/1.875，否则(mse_first)用 4/2。
            float thdEb = (scaleAlg_ == 1) ? 3.75f : 4.0f;
            float thdEc = (scaleAlg_ == 1) ? 1.875f : 2.0f;

            MicroAPI::MaskReg vb1E8Comp;
            MicroAPI::MaskReg vb1E8CompTmp;
            MicroAPI::MaskReg vb1E16Comp;
            MicroAPI::MaskReg vb1E16CompTmp;
            MicroAPI::UnalignReg ureg1;

            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
            for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                    (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64, e8BufRead1Addr, 8);
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
                    (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper8, maxPer8BufReadAddr, 64);
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, maxPer4BufRead2Addr, 128);

                ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitFp82B16>(
                    (MicroAPI::RegTensor<bfloat16_t>&)vb16_Ea_rec_bf16, (MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0Maxper64,
                    maskAll);
                MicroAPI::Mul(vreg0Maxper8, vreg0Maxper8, vb16_Ea_rec_bf16, maskAll);
                MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vreg0Maxper8, thdEb, maskAll);
                MicroAPI::Select(vreg0Maxper8, vregNumhalf, vregNum1, vb1E8Comp);
                MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);
                MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
                MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vreg0Maxper4, thdEc, maskAll);
                MicroAPI::Select(vreg0Maxper4, vregNumhalf, vregNum1, vb1E16Comp);
                MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
                MicroAPI::Mul(vregTmpScale, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);

                MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                    tmpScaleAddr + loopIdx * gmCopyTmpBufOtherStep_, (MicroAPI::RegTensor<uint16_t>&)vregTmpScale,
                    (MicroAPI::RegTensor<uint16_t>&)vregTmpScale, maskAll);

                MicroAPI::DataCopyUnAlign<uint32_t>(e1p8BufAddr, vb1E8Comp, ureg1);

                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_PACK>(
                    e1p16BufAddr, vb1E16Comp, 16);
            }
            MicroAPI::DataCopyUnAlignPost<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(e1p8BufAddr, ureg1, 0);
        }
    }

    __aicore__ inline void ComputeScale(uint16_t scaleRegLoop, __ubuf__ uint8_t* e8BufRead2Addr,
                                        __ubuf__ uint8_t* e1p8BufReadAddr, __ubuf__ uint8_t* e1p16BufReadAddr,
                                        __ubuf__ uint16_t* scaleAddr)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<uint8_t> vregE8;
            MicroAPI::RegTensor<uint8_t> vreg1E8;
            MicroAPI::RegTensor<uint8_t> vreg1E16First;
            MicroAPI::RegTensor<uint8_t> vreg1E16Second;

            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
            for (uint16_t loopIdx = 0; loopIdx < scaleRegLoop; loopIdx++) {
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DS_B8>(
                    vregE8, e8BufRead2Addr, gmCopyTmpBufE8Step_);
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    vreg1E8, e1p8BufReadAddr, gmCopyTmpBufOtherStep_);
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    vreg1E16First, e1p16BufReadAddr, gmCopyTmpBufOtherStep_);
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    vreg1E16Second, e1p16BufReadAddr, gmCopyTmpBufOtherStep_);

                MicroAPI::Interleave<uint8_t>(vregE8, vreg1E8, vregE8, vreg1E8);
                MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                    scaleAddr + loopIdx * gmCopyTmpBufE8Step_, (MicroAPI::RegTensor<uint16_t>&)vregE8,
                    (MicroAPI::RegTensor<uint16_t>&)vreg1E16First, maskAll);
                MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                    scaleAddr + loopIdx * gmCopyTmpBufE8Step_ + gmCopyTmpBufOtherStep_,
                    (MicroAPI::RegTensor<uint16_t>&)vreg1E8, (MicroAPI::RegTensor<uint16_t>&)vreg1E16Second, maskAll);
            }
        }
    }

    __aicore__ inline void ComputeOut(uint16_t outRegLoop, __ubuf__ T_X* xAddr2, __ubuf__ uint16_t* tmpScaleReadAddr,
                                      __ubuf__ int32_t* yAddr)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T_X> vregOut;
            MicroAPI::RegTensor<uint16_t> vregScaleTmp;
            MicroAPI::RegTensor<fp4x2_e1m2_t> vbOut;
            MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
            static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            for (uint16_t loopIdx = 0; loopIdx < outRegLoop; loopIdx++) {
                if constexpr (IsSame<T_X, half>::value) {
                    MicroAPI::DataCopy<T_X, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                        vregOut, xAddr2, gmCopyInRegStep2_);
                    AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(
                        (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16, vregOut, maskAll);
                } else {
                    MicroAPI::DataCopy<T_X, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                        vregOutbf16, xAddr2, gmCopyInRegStep2_);
                }
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
                    vregScaleTmp, tmpScaleReadAddr, gmCopyOutScaleRegStep_);
                MicroAPI::Mul((MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16,
                              (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16,
                              (MicroAPI::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
                MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut, vregOutbf16, maskAll);
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                   MicroAPI::StoreDist::DIST_PACK4_B32>(yAddr, (MicroAPI::RegTensor<int32_t>&)vbOut,
                                                                        gmCopyOutRegStep_, maskAll);
            }
        }
    }

private:
    TPipe* pPipe = nullptr;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueGammabeta;
    LocalTensor<T_GAMMA> gammaLocal_;
    LocalTensor<T_GAMMA> betaLocal_;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueRstd;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueQuantY;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueScale;

    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> cacheBuf;
    TBuf<TPosition::VECCALC> binaryAddBuf;
    TBuf<TPosition::VECCALC> yTmpBuf;
    TBuf<TPosition::VECCALC> bufE8_;
    TBuf<TPosition::VECCALC> buf1E8_;
    TBuf<TPosition::VECCALC> buf1E16_;
    TBuf<TPosition::VECCALC> bufTmpScale_;
    TBuf<TPosition::VECCALC> bufMaxPer4_;
    TBuf<TPosition::VECCALC> bufMaxPer8_;

    GlobalTensor<T_X> xGm;
    GlobalTensor<T_GAMMA> gammaGm, betaGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<uint8_t> yFp4Gm;
    GlobalTensor<float> scaleGm;

    uint64_t numCol_, numColAlign_;
    uint64_t blockFactor_, mLastCore_, mCurCore_;
    uint64_t baseN_, baseM_, baseNBlockSize_;
    uint64_t nUbLoops_;
    uint64_t binAddQuotient_, powerSplit_;
    uint64_t mainFoldCount_, foldTail_;
    int64_t resultCacheID_;
    float epsilon_, avgFactor_;
    uint64_t mxBlockSize_, mxScaleSize_;
    uint32_t betaFlag_, rstdFlag_;
    int64_t scaleAlg_{0};

    int64_t innerLoopEle_, max8LoopEle_, scaleLoopEle_, outLoopEle_, alignGroupNum_;
    int64_t innerLoopTail_, max8LoopTail_, scaleLoopTail_, outLoopTail_;

    int64_t sysRegLen_ = 256;
    int64_t gmCopyInRegStep_ = 0;
    int64_t gmCopyOutRegStep_ = 0;
    int64_t gmCopyOutScaleRegStep_ = 0;
    int64_t gmCopyInRegStep2_ = 0;
    int64_t gmCopyTmpE1_ = 0;
    int64_t gmCopyTmpE8_ = 0;
    int64_t gmCopyTmpE16_ = 0;
    int64_t gmCopyTmpBufE8Step_ = 0;
    int64_t gmCopyTmpBufOtherStep_ = 0;

    int64_t bufE8Buffer_ = 0;
    int64_t buf1E8Buffer_ = 0;
    int64_t buf1E16Buffer_ = 0;
    int64_t maxPer4Buffer_ = 0;
    int64_t maxPer8Buffer_ = 0;
    int64_t tmpScaleBuffer_ = 0;

    constexpr static MicroAPI::CastTrait castTrait_ = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
};
} // namespace RmsNormHiF4Quant
#endif
