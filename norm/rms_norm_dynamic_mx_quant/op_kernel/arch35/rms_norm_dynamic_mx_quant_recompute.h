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
 * \file rms_norm_dynamic_mx_quant_recompute.h
 * \brief RmsNormDynamicMxQuant recompute template (TilingKey=2000/20000)
 */

#ifndef RMS_NORM_DYNAMIC_MX_QUANT_RECOMPUTE_H
#define RMS_NORM_DYNAMIC_MX_QUANT_RECOMPUTE_H

#include "rms_norm_dynamic_mx_quant_common.h"

namespace RmsNormDynamicMxQuantNs {

using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename T_X, typename T_GAMMA, typename T_Y, bool isOptimizeMode = false>
class RmsNormDynamicMxQuantRecompute {
public:
    __aicore__ inline RmsNormDynamicMxQuantRecompute(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxScale, GM_ADDR rstd,
        const RmsNormDynamicMxQuantRecomputeTilingData* tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();

#if (__NPU_ARCH__ == 3510)
        oriOverflowMode_ = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

        int64_t usedTailCores = blockIdx_ < tilingData_->mTailCores ? blockIdx_ : tilingData_->mTailCores;
        int64_t rowOffset = (blockIdx_ * tilingData_->mPerCore + usedTailCores);
        int64_t xOffset = rowOffset * tilingData_->numN;
        int64_t scaleOffset = rowOffset * tilingData_->nMxblockNumAlignedTwo;

        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)beta);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + xOffset);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + rowOffset);
        mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale + scaleOffset);
        if constexpr (IsSameType<T_Y, fp4x2_e2m1_t>::value || IsSameType<T_Y, fp4x2_e1m2_t>::value) {
            yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xOffset / NUM_TWO);
        } else {
            yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xOffset);
        }

        int64_t xBufSize = tilingData_->baseN * sizeof(T_X);
        pipe_->InitBuffer(xInQueue_, DOUBLE_BUFFER, xBufSize);
        pipe_->InitBuffer(xFoldInQueue_, DOUBLE_BUFFER, xBufSize);

        pipe_->InitBuffer(xFp32TmpBuf_, tilingData_->baseN * sizeof(float));
        pipe_->InitBuffer(binaryAddBuf_, VL_FP32 * NUM_TWO * sizeof(float));
        pipe_->InitBuffer(cacheBuf_, (tilingData_->resultCacheId + 1) * UB_BLOCK_SIZE);

        pipe_->InitBuffer(rstdOutQueue_, DOUBLE_BUFFER, tilingData_->baseM * sizeof(float));

        int64_t gammaBufSize = tilingData_->baseN * sizeof(T_GAMMA);
        pipe_->InitBuffer(gammaInQueue_, DOUBLE_BUFFER, gammaBufSize);
        if (tilingData_->hasInputBeta) {
            pipe_->InitBuffer(betaInQueue_, DOUBLE_BUFFER, gammaBufSize);
        }

        pipe_->InitBuffer(normOutBuffer_, xBufSize);

        int64_t scaleBufSize =
            ops::CeilAlign(static_cast<uint32_t>(tilingData_->baseN / tilingData_->mxBlockSize), VL_B16) * sizeof(T_X);
        pipe_->InitBuffer(scaleOutQueue_, DOUBLE_BUFFER, scaleBufSize);
        pipe_->InitBuffer(maxExpBuffer_, scaleBufSize);
        pipe_->InitBuffer(maxhalfScaleBuffer_, scaleBufSize);

        int64_t yOutBufSize;
        if constexpr (IsSameType<T_Y, fp4x2_e2m1_t>::value || IsSameType<T_Y, fp4x2_e1m2_t>::value) {
            yOutBufSize = ops::CeilAlign(
                static_cast<int64_t>(tilingData_->baseN / NUM_TWO * sizeof(uint8_t)), static_cast<int64_t>(VL_B16));
        } else {
            yOutBufSize = ops::CeilAlign(static_cast<int64_t>(4 * tilingData_->baseN * sizeof(uint8_t)), UB_BLOCK_SIZE);
        }
        pipe_->InitBuffer(yOutQueue_, DOUBLE_BUFFER, yOutBufSize);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tilingData_->usedCoreNum) {
            return;
        }

        int64_t mCurCore = blockIdx_ < tilingData_->mTailCores ? (tilingData_->mPerCore + 1) : tilingData_->mPerCore;
        int64_t mBatchCnt = ops::CeilDiv(mCurCore, tilingData_->baseM);
        int64_t mBatchLast = mCurCore - (mBatchCnt - 1) * tilingData_->baseM;

        for (int64_t batchIdx = 0; batchIdx < mBatchCnt; ++batchIdx) {
            int64_t curM = (batchIdx == mBatchCnt - 1) ? mBatchLast : tilingData_->baseM;
            int64_t batchOffset = batchIdx * tilingData_->baseM * tilingData_->numN;

            LocalTensor<float> rstdLocal = rstdOutQueue_.AllocTensor<float>();
            ComputeBatchRstd(rstdLocal, batchOffset, curM);
            CopyOutRstd(batchIdx, curM, rstdLocal);

            int64_t batchScaleOffset = batchIdx * tilingData_->baseM * tilingData_->nMxblockNumAlignedTwo;
            ProcessNSplitLoop(batchOffset, batchScaleOffset, curM, rstdLocal);

            rstdOutQueue_.FreeTensor(rstdLocal);
        }
    }

private:
    __aicore__ inline void ComputeBatchRstd(LocalTensor<float>& rstdLocal, int64_t batchOffset, int64_t curM)
    {
        for (int64_t rowIdx = 0; rowIdx < curM; ++rowIdx) {
            int64_t rowGmOffset = batchOffset + rowIdx * tilingData_->numN;
            ComputeOneLineXSquareSum(rstdLocal, rowGmOffset, rowIdx);
        }
        ComputeRstd(rstdLocal, rstdLocal, static_cast<uint64_t>(curM), tilingData_->epsilon, tilingData_->avgFactor);
        rstdOutQueue_.EnQue(rstdLocal);
        rstdLocal = rstdOutQueue_.DeQue<float>();
    }

    __aicore__ inline void CopyOutRstd(int64_t batchIdx, int64_t curM, LocalTensor<float>& rstdLocal)
    {
        if (tilingData_->hasOutputRstd) {
            int64_t rstdGmOffset = batchIdx * tilingData_->baseM;
            DataCopyExtParams copyRstdOut;
            copyRstdOut.blockCount = 1;
            copyRstdOut.blockLen = static_cast<uint32_t>(curM * sizeof(float));
            copyRstdOut.srcStride = 0;
            copyRstdOut.dstStride = 0;
            DataCopyPad(rstdGm_[rstdGmOffset], rstdLocal, copyRstdOut);
        }
    }

    __aicore__ inline void ProcessNSplitLoop(
        int64_t batchOffset, int64_t batchScaleOffset, int64_t curM, LocalTensor<float>& rstdLocal)
    {
        for (int64_t nIdx = 0; nIdx < tilingData_->nUbLoops; ++nIdx) {
            bool isLastN = nIdx == (tilingData_->nUbLoops - 1) ? true : false;
            int64_t curN = isLastN ? tilingData_->baseNTail : tilingData_->baseN;
            int64_t nOffset = nIdx * tilingData_->baseN;

            CopyInGammaBeta(nOffset, curN);
            LocalTensor<T_GAMMA> gammaLocal = gammaInQueue_.DeQue<T_GAMMA>();
            LocalTensor<T_GAMMA> betaLocal;
            if (tilingData_->hasInputBeta) {
                betaLocal = betaInQueue_.DeQue<T_GAMMA>();
            }

            for (int64_t rowIdx = 0; rowIdx < curM; ++rowIdx) {
                int64_t yOutOffset = batchOffset + rowIdx * tilingData_->numN + nOffset;
                int64_t scaleOutOffset =
                    batchScaleOffset + rowIdx * tilingData_->nMxblockNumAlignedTwo + nOffset / tilingData_->mxBlockSize;
                ComputeNormAndQuantOneRow(
                    rstdLocal, gammaLocal, betaLocal, rowIdx, isLastN, curN, yOutOffset, scaleOutOffset);
            }

            gammaInQueue_.FreeTensor(gammaLocal);
            if (tilingData_->hasInputBeta) {
                betaInQueue_.FreeTensor(betaLocal);
            }
        }
    }

    // ============================================================
    // 辅助：获取 ub 间二分合并的 cache slot id
    // ============================================================
    __aicore__ inline int64_t GetCacheId(const int64_t idx)
    {
        return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1;
    }

    // ============================================================
    // 搬入 gamma / beta（分块拷贝）
    // ============================================================
    __aicore__ inline void CopyInGammaBeta(int64_t offset, int64_t len)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(len * sizeof(T_GAMMA));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T_GAMMA> padParams{false, 0, 0, static_cast<T_GAMMA>(0.0)};

        LocalTensor<T_GAMMA> gammaLocal = gammaInQueue_.AllocTensor<T_GAMMA>();
        DataCopyPad(gammaLocal, gammaGm_[offset], copyParams, padParams);
        gammaInQueue_.EnQue(gammaLocal);

        if (tilingData_->hasInputBeta) {
            LocalTensor<T_GAMMA> betaLocal = betaInQueue_.AllocTensor<T_GAMMA>();
            DataCopyPad(betaLocal, betaGm_[offset], copyParams, padParams);
            betaInQueue_.EnQue(betaLocal);
        }
    }

    __aicore__ inline void ComputeOneLineXSquareSum(
        LocalTensor<float>& rstdLocal, int64_t rowGmOffset, int64_t rowIndex)
    {
        LocalTensor<float> cacheLocal = cacheBuf_.Get<float>();
        LocalTensor<float> xFp32Tmp = xFp32TmpBuf_.Get<float>();

        DataCopyPadParams padParams{false, 0, 0, 0};

        DataCopyParams xCopyParams;
        xCopyParams.blockCount = 1;
        xCopyParams.srcStride = 0;
        xCopyParams.dstStride = 0;
        DataCopyParams xFoldCopyParams;
        xFoldCopyParams.blockCount = 1;
        xFoldCopyParams.srcStride = 0;
        xFoldCopyParams.dstStride = 0;

        for (int64_t r = 0; r < tilingData_->powerSplit; ++r) {
            int64_t gmOff1 = rowGmOffset + tilingData_->baseN * r;
            int64_t gmOff2 = rowGmOffset + tilingData_->baseN * (r + tilingData_->powerSplit);

            // 搬入 x[r*tilingData_->baseN]
            xCopyParams.blockLen = static_cast<uint32_t>(tilingData_->baseN * sizeof(T_X));
            LocalTensor<T_X> xLocal = xInQueue_.AllocTensor<T_X>();
            DataCopyPad(xLocal, xGm_[gmOff1], xCopyParams, padParams);
            xInQueue_.EnQue<T_X>(xLocal);
            xLocal = xInQueue_.DeQue<T_X>();

            if (r < tilingData_->mainFoldCount) {
                // 完整尾折块
                xFoldCopyParams.blockLen = static_cast<uint32_t>(tilingData_->baseN * sizeof(T_X));
                LocalTensor<T_X> xFoldLocal = xFoldInQueue_.AllocTensor<T_X>();
                DataCopyPad(xFoldLocal, xGm_[gmOff2], xFoldCopyParams, padParams);
                xFoldInQueue_.EnQue<T_X>(xFoldLocal);
                xFoldLocal = xFoldInQueue_.DeQue<T_X>();
                FoldBlockVF(
                    xLocal, xFoldLocal, xFp32Tmp, static_cast<uint32_t>(tilingData_->baseN),
                    static_cast<uint32_t>(tilingData_->baseN));
                xFoldInQueue_.FreeTensor(xFoldLocal);
            } else if (r == tilingData_->mainFoldCount && tilingData_->foldTail > 0) {
                // 尾部非整块
                xFoldCopyParams.blockLen = static_cast<uint32_t>(tilingData_->foldTail * sizeof(T_X));
                LocalTensor<T_X> xFoldLocal = xFoldInQueue_.AllocTensor<T_X>();
                DataCopyPad(xFoldLocal, xGm_[gmOff2], xFoldCopyParams, padParams);
                xFoldInQueue_.EnQue<T_X>(xFoldLocal);
                xFoldLocal = xFoldInQueue_.DeQue<T_X>();
                FoldBlockVF(
                    xLocal, xFoldLocal, xFp32Tmp, static_cast<uint32_t>(tilingData_->foldTail),
                    static_cast<uint32_t>(tilingData_->baseN));
                xFoldInQueue_.FreeTensor(xFoldLocal);
            } else {
                // 无尾折，只对本块平方
                CastAndSquare(xLocal, xFp32Tmp, static_cast<uint32_t>(tilingData_->baseN));
            }

            // 块内二分归约
            CalculateSquareReduceSum(xFp32Tmp, xFp32Tmp, static_cast<uint32_t>(tilingData_->baseN));

            // 树形 cache 合并
            int64_t cacheId = GetCacheId(r);
            UpdateCache(cacheLocal, xFp32Tmp, cacheId, UB_BLOCK_SIZE_FP32);

            xInQueue_.FreeTensor(xLocal);
        }

        // 从 cache 取出整行结果写入 rstdLocal[rowIndex]
        __local_mem__ float* dstPtr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* cachePtr =
            (__local_mem__ float*)cacheLocal.GetPhyAddr() + tilingData_->resultCacheId * UB_BLOCK_SIZE_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> a;
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            DataCopy<float, LoadDist::DIST_NORM>(a, cachePtr);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + rowIndex, a, pregOne);
        }
    }

    __aicore__ inline void CastAndSquare(LocalTensor<T_X>& xLocal, LocalTensor<float>& xFp32Tmp, uint32_t count)
    {
        __local_mem__ T_X* xAddr = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* dstAddr = (__local_mem__ float*)xFp32Tmp.GetPhyAddr();
        uint16_t loops = static_cast<uint16_t>(ops::CeilDiv(count, VL_FP32));

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = count;
            RegTensor<float> xReg;
            for (uint16_t i = 0; i < loops; ++i) {
                uint32_t offset = i * VL_FP32;
                pregMask = UpdateMask<float>(sreg);
                LoadTensorForDtypeT(xAddr, xReg, pregMask, i * VL_FP32);
                Mul(xReg, xReg, xReg, pregMask);
                DataCopy<float, StoreDist::DIST_NORM_B32>(dstAddr + i * VL_FP32, xReg, pregMask);
            }
        }
    }

    __aicore__ inline void FoldBlockVF(
        LocalTensor<T_X>& xLocal, LocalTensor<T_X>& xFoldLocal, LocalTensor<float>& xFp32Tmp, uint32_t tailCount,
        uint32_t count)
    {
        __local_mem__ T_X* xInUb = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_X* xFoldInUb = (__local_mem__ T_X*)xFoldLocal.GetPhyAddr();
        __local_mem__ float* dstBuf = (__local_mem__ float*)xFp32Tmp.GetPhyAddr();

        uint16_t loops = static_cast<uint16_t>(ops::CeilDiv(count, VL_FP32));
        uint16_t tailLoops = static_cast<uint16_t>(ops::CeilDiv(tailCount, VL_FP32));
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, xFoldReg, sumReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;
            uint32_t sregTail = tailCount;

            // 有尾块的部分：x^2 + xFold^2
            for (uint16_t i = 0; i < tailLoops; ++i) {
                pregLoop = UpdateMask<float>(sregTail);
                uint32_t offset = i * VL_FP32;
                LoadTensorForDtypeT(xInUb, xReg, pregFull, offset);
                LoadTensorForDtypeT(xFoldInUb, xFoldReg, pregLoop, offset);
                Mul(xReg, xReg, xReg, pregFull);
                Mul(xFoldReg, xFoldReg, xFoldReg, pregLoop);
                Add(sumReg, xReg, xFoldReg, pregLoop);
                Select(sumReg, sumReg, xReg, pregLoop); // 超出尾块范围用 xReg^2
                DataCopy<float, StoreDist::DIST_NORM_B32>(dstBuf + offset, sumReg, pregFull);
            }
            // 无尾块的部分：只有 x^2
            for (uint16_t i = tailLoops; i < loops; ++i) {
                uint32_t offset = i * VL_FP32;
                LoadTensorForDtypeT(xInUb, xReg, pregFull, offset);
                Mul(xReg, xReg, xReg, pregFull);
                DataCopy<float, StoreDist::DIST_NORM_B32>(dstBuf + offset, xReg, pregFull);
            }
        }
    }

    // ============================================================
    // UpdateCache：树形合并，将 xFp32Tmp 中的局部和累加进 cache[cacheId]
    // ============================================================
    __aicore__ inline void UpdateCache(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const int64_t cacheId,
        const int64_t stride)
    {
        uint16_t innerLoopTimes = cacheId;
        uint32_t innerLoopStride = stride;
        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* cache = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheId * stride;
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            RegTensor<float> aReg, bReg;
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            DataCopy(aReg, (__local_mem__ float*)src);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                DataCopy(bReg, dst + j * innerLoopStride);
                Add(aReg, aReg, bReg, pregOne);
            }
            DataCopy((__local_mem__ float*)cache, aReg, pregOne);
        }
    }

    // ============================================================
    // CalculateSquareReduceSum：块内二分归约，将 xFp32Local 归约为一个标量
    // ============================================================
    __aicore__ inline void CalculateSquareReduceSum(
        LocalTensor<float>& xFp32Local, LocalTensor<float>& xReduceLocal, uint32_t reduceNum)
    {
        LocalTensor<float> binaryAddTmp = binaryAddBuf_.Get<float>();
        __local_mem__ float* xFp32Ptr = (__local_mem__ float*)xFp32Local.GetPhyAddr();
        __local_mem__ float* xReducePtr = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        __local_mem__ float* tmpPtr = (__local_mem__ float*)binaryAddTmp.GetPhyAddr();

        if (reduceNum <= VL_FP32) {
            CalculateReduceSumLessThanVL(xFp32Ptr, xReducePtr, reduceNum);
        } else if (reduceNum <= VL_FP32 + VL_FP32) {
            CalculateReduceSumLessThanTwoVL(xFp32Ptr, xReducePtr, reduceNum);
        } else if (reduceNum <= VL_FP32 * VL_FP32 * NUM_TWO) {
            CalculateReduceSumCommon<1>(xFp32Ptr, xReducePtr, tmpPtr, reduceNum);
        } else {
            CalculateReduceSumCommon<2>(xFp32Ptr, xReducePtr, tmpPtr, reduceNum);
        }
    }

    __aicore__ inline void CalculateReduceSumLessThanVL(
        __local_mem__ float* xPtr, __local_mem__ float* dstPtr, uint32_t num)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x, sum;
            uint32_t sreg0 = num;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            LoadTensorForDtypeT(xPtr, x, pregLoop, 0u);
            ReduceSum(sum, x, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, sum, pregOne);
        }
    }

    __aicore__ inline void CalculateReduceSumLessThanTwoVL(
        __local_mem__ float* xPtr, __local_mem__ float* dstPtr, uint32_t num)
    {
        uint32_t tail = num - VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, xFoldReg, addReg, sumReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregTail = UpdateMask<float>(tail);
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            LoadTensorForDtypeT(xPtr, xReg, pregFull, 0u);
            LoadTensorForDtypeT(xPtr + VL_FP32, xFoldReg, pregTail, 0u);
            ShiftLefts(
                (RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0), pregTail);
            Add(addReg, xReg, xFoldReg, pregFull);
            ReduceSum(sumReg, addReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, sumReg, pregOne);
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateReduceSumCommon(
        __local_mem__ float* xPtr, __local_mem__ float* dstPtr, __local_mem__ float* tmpPtr, uint32_t reduceNum)
    {
        uint32_t baq = static_cast<uint32_t>(tilingData_->binAddQuotient);
        uint16_t baqLoop = static_cast<uint16_t>((baq + VL_FP32 - 1) / VL_FP32);
        uint32_t lastNum = baq / VL_FP32;
        uint32_t remainder = reduceNum - baq;
        uint16_t remCeil = static_cast<uint16_t>((remainder + VL_FP32 - 1) / VL_FP32);
        uint16_t remFloor = static_cast<uint16_t>(remainder / VL_FP32);

        __VEC_SCOPE__
        {
            RegTensor<float> x, xFold, sumReg, vMean;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;
            uint32_t sregRem = remainder - remFloor * VL_FP32;

            for (uint16_t r = 0; r < remFloor; ++r) {
                uint32_t off = r * VL_FP32;
                LoadTensorForDtypeT(xPtr, x, pregFull, off);
                LoadTensorForDtypeT(xPtr + baq, xFold, pregFull, off);
                Add(sumReg, x, xFold, pregFull);
                ReduceSum(vMean, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + static_cast<uint32_t>(r), vMean, pregOne);
            }
            // 余数部分
            for (uint16_t r = 0; r < static_cast<uint16_t>(remCeil - remFloor); ++r) {
                pregLoop = UpdateMask<float>(sregRem);
                LoadTensorForDtypeT(xPtr + remFloor * VL_FP32, x, pregFull, 0u);
                LoadTensorForDtypeT(xPtr + remFloor * VL_FP32 + baq, xFold, pregLoop, 0u);
                ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLoop);
                Add(sumReg, x, xFold, pregFull);
                ReduceSum(vMean, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    tmpPtr + static_cast<uint32_t>(remFloor), vMean, pregOne);
            }
            // baq 前半部分单独归约
            for (uint16_t r = 0; r < static_cast<uint16_t>(baqLoop - remCeil); ++r) {
                LoadTensorForDtypeT(xPtr + remCeil * VL_FP32, x, pregFull, 0u);
                ReduceSum(vMean, x, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    tmpPtr + static_cast<uint32_t>(remCeil + r), vMean, pregOne);
            }

            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastNum);
                DataCopy<float, LoadDist::DIST_NORM>(x, tmpPtr);
                ReduceSum(vMean, x, pregLast);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, vMean, pregOne);
            } else {
                lastNum -= VL_FP32;
                MaskReg pregLast = UpdateMask<float>(lastNum);
                DataCopy<float, LoadDist::DIST_NORM>(x, tmpPtr);
                DataCopy<float, LoadDist::DIST_NORM>(xFold, tmpPtr + VL_FP32);
                ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLast);
                Add(sumReg, x, xFold, pregFull);
                ReduceSum(vMean, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, vMean, pregOne);
            }
        }
    }

    __aicore__ inline void ComputeNormAndQuantOneRow(
        LocalTensor<float>& rstdLocal, LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<T_GAMMA>& betaLocal,
        int64_t rowIdx, bool isLastN, int64_t curN, int64_t xOffset, int64_t scaleOutOffset)
    {
        // 搬入 x（当前N分块，curN个元素）
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams xCopy;
        xCopy.blockCount = 1;
        xCopy.blockLen = static_cast<uint32_t>(curN * sizeof(T_X));
        xCopy.srcStride = 0;
        xCopy.dstStride = 0;

        LocalTensor<T_X> xLocal = xInQueue_.AllocTensor<T_X>();
        DataCopyPad(xLocal, xGm_[xOffset], xCopy, padParams);
        xInQueue_.EnQue<T_X>(xLocal);
        xLocal = xInQueue_.DeQue<T_X>();

        // 计算 norm：y = x * rstd * gamma [+ beta]，结果放入 normOutBuffer_
        // ComputeYOneRow 内部使用 ZEROING + full mask store，最后分块可自动补0到对齐长度
        LocalTensor<T_X> normOutLocal = normOutBuffer_.Get<T_X>();
        if (tilingData_->hasInputBeta) {
            ComputeYOneRow<true>(
                xLocal, gammaLocal, betaLocal, rstdLocal, normOutLocal, static_cast<uint32_t>(rowIdx),
                static_cast<uint32_t>(curN));
        } else {
            ComputeYOneRow<false>(
                xLocal, gammaLocal, betaLocal, rstdLocal, normOutLocal, static_cast<uint32_t>(rowIdx),
                static_cast<uint32_t>(curN));
        }

        xInQueue_.FreeTensor(xLocal);

#if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
        if (tilingData_->roundMode == MODE_RINT) {
            ComputeMxQuantAndOutputOneRow<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
                normOutLocal, xOffset, scaleOutOffset, static_cast<int64_t>(curN), isLastN);
        } else if (tilingData_->roundMode == MODE_ROUND) {
            ComputeMxQuantAndOutputOneRow<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(
                normOutLocal, xOffset, scaleOutOffset, static_cast<int64_t>(curN), isLastN);
        } else if (tilingData_->roundMode == MODE_FLOOR) {
            ComputeMxQuantAndOutputOneRow<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(
                normOutLocal, xOffset, scaleOutOffset, static_cast<int64_t>(curN), isLastN);
        }
#if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode_);
#endif
    }

    template <bool hasInputBeta = false>
    __aicore__ inline void ComputeYOneRow(
        LocalTensor<T_X>& xLocal, LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<T_GAMMA>& betaLocal,
        LocalTensor<float>& rstdLocal, LocalTensor<T_X>& yLocal, uint32_t rstdOffset, uint32_t curN)
    {
        __local_mem__ T_X* xAddr = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaAddr = (__local_mem__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_GAMMA* betaAddr;
        if constexpr (hasInputBeta) {
            betaAddr = (__local_mem__ T_GAMMA*)betaLocal.GetPhyAddr();
        }
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_X* yAddr = (__local_mem__ T_X*)yLocal.GetPhyAddr();

        uint16_t nloops =
            static_cast<uint16_t>(ops::CeilDiv(static_cast<uint64_t>(curN), static_cast<uint64_t>(VL_FP32)));

        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> RstdReg;
            RegTensor<float> gammaReg;
            RegTensor<float> betaReg;
            RegTensor<float> yReg;
            MaskReg pregMask;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();

            uint32_t sreg = curN;
            DataCopy<float, LoadDist::DIST_BRC_B32>(RstdReg, rstdAddr + rstdOffset);
            for (uint16_t j = 0; j < nloops; ++j) {
                pregMask = UpdateMask<float>(sreg);
                uint32_t off = j * VL_FP32;
                LoadTensorForDtypeT(xAddr, xReg, pregMask, off);
                Mul(yReg, xReg, RstdReg, pregMask);
                LoadTensorForDtypeT(gammaAddr, gammaReg, pregMask, off);
                Mul<float, MaskMergeMode::ZEROING>(yReg, yReg, gammaReg, pregMask);
                if constexpr (hasInputBeta) {
                    LoadTensorForDtypeT(betaAddr, betaReg, pregMask, off);
                    Add<float, MaskMergeMode::ZEROING>(yReg, yReg, betaReg, pregMask);
                }
                StoreTensorForDtypeT(yAddr, yReg, pregFull, off);
            }
        }
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeMxQuantAndOutputOneRow(
        LocalTensor<T_X>& yLocal, int64_t xOffset, int64_t scaleOutOffset, int64_t curN, bool isLastN)
    {
        int64_t curNAligned = isLastN ? tilingData_->baseNTailAligned : curN;
        int64_t curNMxblockNum = ops::CeilDiv(curNAligned, static_cast<int64_t>(tilingData_->mxBlockSize));
        uint32_t totalScaleInUB = static_cast<uint32_t>(curNMxblockNum);
        uint32_t totalCountInUB = static_cast<uint32_t>(curNMxblockNum * tilingData_->mxBlockSize);
        uint16_t loopNum = static_cast<uint16_t>((totalCountInUB + VL_B16 * NUM_TWO - 1) / (VL_B16 * NUM_TWO));
        uint16_t loopNumScale = static_cast<uint16_t>((totalScaleInUB + VL_B16 - 1) / VL_B16);

        LocalTensor<uint16_t> scaleOutLocal = scaleOutQueue_.AllocTensor<uint16_t>();
        LocalTensor<int8_t> yOutLocal = yOutQueue_.AllocTensor<int8_t>();
        LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
        LocalTensor<uint16_t> halfScaleLocal = maxhalfScaleBuffer_.Get<uint16_t>();

        auto srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
        auto halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());
        auto scaleOutAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleOutLocal.GetPhyAddr());
        auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(yOutLocal.GetPhyAddr());

        if constexpr (IsSameType<T_Y, fp4x2_e2m1_t>::value || IsSameType<T_Y, fp4x2_e1m2_t>::value) {
            ComputeMaxExpOCP<T_X>(srcAddr, maxExpAddr, totalCountInUB, loopNum);
            ComputeScaleOCP<T_X, T_Y>(maxExpAddr, scaleOutAddr, halfScaleAddr, totalScaleInUB, loopNumScale);
            if constexpr (isOptimizeMode) {
                ComputeDataMxfp4Optimize<T_X, T_Y, toBf16RoundMode, roundMode>(
                    srcAddr, halfScaleAddr, outLocalAddr, totalCountInUB, loopNum);
            } else {
                ComputeDataMxfp4General<T_X, T_Y, toBf16RoundMode, roundMode>(
                    srcAddr, halfScaleAddr, outLocalAddr, totalCountInUB, loopNum);
            }
        } else {
            if (tilingData_->scaleAlg == 0) {
                ComputeMaxExpOCP<T_X>(srcAddr, maxExpAddr, totalCountInUB, loopNum);
                ComputeScaleOCP<T_X, T_Y>(maxExpAddr, scaleOutAddr, halfScaleAddr, totalScaleInUB, loopNumScale);
            } else {
                uint16_t loopNumScale4NV = static_cast<uint16_t>((totalScaleInUB + VL_FP32 - 1) / VL_FP32);
                ComputeMaxExpcuBLAS<T_X>(srcAddr, maxExpAddr, totalCountInUB, loopNum);
                ComputeScalecuBLAS<T_X, T_Y>(maxExpAddr, scaleOutAddr, halfScaleAddr, totalScaleInUB, loopNumScale4NV);
            }
            ComputeData<toBf16RoundMode, roundMode, T_X, T_Y>(
                srcAddr, halfScaleAddr, outLocalAddr, totalCountInUB, loopNum);
        }

        yOutQueue_.EnQue(yOutLocal);
        scaleOutQueue_.EnQue(scaleOutLocal);

        CopyOutScaleAndYOneRow(xOffset, scaleOutOffset, curN, isLastN);
    }

    __aicore__ inline void CopyOutScaleAndYOneRow(int64_t xOffset, int64_t scaleOutOffset, int64_t curN, bool isLastN)
    {
        int64_t yOffset = xOffset;
        int64_t ySrcStride = isLastN ? (tilingData_->baseNTailAligned - curN) : 0;
        int64_t yDstStride = tilingData_->numN - curN;

        DataCopyExtParams copyOutYParam;
        copyOutYParam.blockCount = 1;
        copyOutYParam.srcStride = 0;
        copyOutYParam.dstStride = 0;
        if constexpr (IsSameType<T_Y, fp4x2_e2m1_t>::value || IsSameType<T_Y, fp4x2_e1m2_t>::value) {
            yOffset = yOffset / NUM_TWO;
            copyOutYParam.blockLen = static_cast<uint32_t>(curN / NUM_TWO);
        } else {
            copyOutYParam.blockLen = static_cast<uint32_t>(curN);
        }

        LocalTensor<uint8_t> outLocal = yOutQueue_.DeQue<uint8_t>();
        DataCopyPad<uint8_t, PaddingMode::Compact>(yGm_[yOffset], outLocal, copyOutYParam);
        yOutQueue_.FreeTensor(outLocal);

        int64_t curNAligned = isLastN ? tilingData_->baseNTailAligned : curN;
        int64_t curNMxblockNum = curNAligned / tilingData_->mxBlockSize;

        DataCopyExtParams copyOutScale;
        copyOutScale.blockCount = 1;
        copyOutScale.blockLen = curNMxblockNum;
        copyOutScale.srcStride = 0;
        copyOutScale.dstStride = 0;

        LocalTensor<uint8_t> scaleLocal = scaleOutQueue_.DeQue<uint8_t>();
        DataCopyPad<uint8_t, PaddingMode::Compact>(mxScaleGm_[scaleOutOffset], scaleLocal, copyOutScale);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }

    // ============================================================
    // member variables
    // ============================================================
    TPipe* pipe_{nullptr};
    const RmsNormDynamicMxQuantRecomputeTilingData* tilingData_{nullptr};

    // GM
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<T_GAMMA> betaGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> mxScaleGm_;

    // UB queues & buffers
    TQue<QuePosition::VECIN, 1> gammaInQueue_;
    TQue<QuePosition::VECIN, 1> betaInQueue_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> xFoldInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue_;
    TQue<QuePosition::VECOUT, 1> rstdOutQueue_;

    TBuf<TPosition::VECCALC> xFp32TmpBuf_;
    TBuf<TPosition::VECCALC> binaryAddBuf_;
    TBuf<TPosition::VECCALC> cacheBuf_;

    TBuf<TPosition::VECCALC> normOutBuffer_;
    TBuf<TPosition::VECCALC> maxExpBuffer_;
    TBuf<TPosition::VECCALC> maxhalfScaleBuffer_;

    int64_t blockIdx_{0};
#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode_{0};
#endif
};

} // namespace RmsNormDynamicMxQuantNs

#endif // RMS_NORM_DYNAMIC_MX_QUANT_RECOMPUTE_H
