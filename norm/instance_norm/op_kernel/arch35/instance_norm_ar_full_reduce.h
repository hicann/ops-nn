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
 * \file instance_norm_ar_full_reduce.h
 * \brief
 */
#ifndef INSTANCE_NORM_AR_FULL_REDUCE_H_
#define INSTANCE_NORM_AR_FULL_REDUCE_H_

#include "instance_norm_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace InstanceNormOps {
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

constexpr uint64_t ALIGN_32_FACTOR = 32;
constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;

template <typename T_X, typename T_BETA, typename T_MEAN>
class InstanceNormARFullReduce {
public:
    __aicore__ inline InstanceNormARFullReduce(const InstanceNormARFullReduceTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();

        cInner_ = tilingData->cInner;
        cOuter_ = tilingData->cOuter;
        cTail_ = tilingData->cTail;
        numN_ = tilingData->numN;
        numC_ = tilingData->numC;
        numR_ = tilingData->numR;
        rAlign_ = tilingData->rAlign;
        binaryAddQuotient_ = tilingData->binaryAddQuotient;
        perCoreCnt_ = tilingData->perCoreCnt;
        epsilon_ = tilingData->epsilon;
        avgFactor_ = tilingData->avgFactor;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance)
    {
        uint64_t gmLen = numN_ * numC_ * numR_;
        uint16_t meanLen = numN_ * numC_;
        xGm_.SetGlobalBuffer((__gm__ T_X*)x, gmLen);
        gammaGm_.SetGlobalBuffer((__gm__ T_BETA*)gamma, numC_);
        betaGm_.SetGlobalBuffer((__gm__ T_BETA*)beta, numC_);
        yGm_.SetGlobalBuffer((__gm__ T_X*)y, gmLen);
        meanGm_.SetGlobalBuffer((__gm__ T_MEAN*)mean, meanLen);
        varianceGm_.SetGlobalBuffer((__gm__ T_MEAN*)variance, meanLen);

        uint64_t rstdUbSizeAlignSize =
            ops::CeilAlign(static_cast<uint64_t>(cInner_), static_cast<uint64_t>(VL_FP32)) * sizeof(float);
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient_ + VL_FP32 - 1) / VL_FP32;
        uint32_t binaryAddBufLen = (binaryAddQuotientLoop + BLK_B32 - 1) / BLK_B32 * BLK_B32 * sizeof(float) * cInner_;

        pipe_.InitBuffer(
            inQueueX_, DOUBLE_BUFFER_NUM,
            ops::CeilAlign(rAlign_ * cInner_ * sizeof(T_X), static_cast<uint64_t>(BLOCK_SIZE)));
        pipe_.InitBuffer(
            inQueueGamma_, DOUBLE_BUFFER_NUM,
            ops::CeilAlign(cInner_ * sizeof(T_BETA), static_cast<uint64_t>(BLOCK_SIZE)));
        pipe_.InitBuffer(
            inQueueBeta_, DOUBLE_BUFFER_NUM,
            ops::CeilAlign(cInner_ * sizeof(T_BETA), static_cast<uint64_t>(BLOCK_SIZE)));
        pipe_.InitBuffer(
            outQueueY_, DOUBLE_BUFFER_NUM,
            ops::CeilAlign(rAlign_ * cInner_ * sizeof(T_X), static_cast<uint64_t>(BLOCK_SIZE)));
        pipe_.InitBuffer(
            outQueueMean_, DOUBLE_BUFFER_NUM,
            ops::CeilAlign(cInner_ * sizeof(T_MEAN), static_cast<uint64_t>(BLOCK_SIZE)));
        pipe_.InitBuffer(
            outQueueVariance_, DOUBLE_BUFFER_NUM,
            ops::CeilAlign(cInner_ * sizeof(T_MEAN), static_cast<uint64_t>(BLOCK_SIZE)));
        pipe_.InitBuffer(rstdBuf_, rstdUbSizeAlignSize);
        pipe_.InitBuffer(meanFp32Buff_, ops::CeilAlign(cInner_ * sizeof(float), static_cast<uint64_t>(BLOCK_SIZE)));
        pipe_.InitBuffer(binaryAddBuf_, binaryAddBufLen);
    }

    __aicore__ inline void Process()
    {
        int64_t totalCnt = numN_ * cOuter_;
        int64_t startIndex = blockIdx_ * perCoreCnt_;
        int64_t endIndex = ((blockIdx_ + 1) * perCoreCnt_ > totalCnt) ? totalCnt : (blockIdx_ + 1) * perCoreCnt_;
        uint64_t cIdxPre = -1;
        LocalTensor<T_BETA> gammaLocal;
        LocalTensor<T_BETA> betaLocal;
        for (uint64_t i = startIndex; i < endIndex; ++i) {
            uint64_t nIdx = i % numN_;
            uint64_t cIdx = i / numN_;
            uint64_t cOffset = cIdx * cInner_;
            uint32_t curCLen = (cIdx == cOuter_ - 1) ? cTail_ : cInner_;
            if (cIdxPre != cIdx) {
                CopyInGammaBeta(cOffset, curCLen); // 拷贝输入gamma、beta到UB
                gammaLocal = inQueueGamma_.DeQue<T_BETA>();
                betaLocal = inQueueBeta_.DeQue<T_BETA>();
            }
            uint64_t offset = numC_ * nIdx + cOffset;
            CopyInX(offset * numR_, curCLen, numR_, rAlign_); // 拷贝输入x到UB

            LocalTensor<T_X> xLocal = inQueueX_.DeQue<T_X>();
            LocalTensor<T_MEAN> meanLocal = outQueueMean_.AllocTensor<T_MEAN>();
            LocalTensor<T_MEAN> varLocal = outQueueVariance_.AllocTensor<T_MEAN>();
            LocalTensor<float> meanFp32Local = meanFp32Buff_.Get<float>();
            LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
            CalculateSquareReduceSum(
                xLocal, meanLocal, varLocal, meanFp32Local, rstdLocal, curCLen, rAlign_, numR_, avgFactor_);
            outQueueMean_.EnQue<T_MEAN>(meanLocal);
            outQueueVariance_.EnQue<T_MEAN>(varLocal);
            CopyOutMeanVar(offset, curCLen);
            CalculateRstd(rstdLocal, curCLen, epsilon_, rstdLocal);
            // 计算 y
            LocalTensor<T_X> yLocal = outQueueY_.AllocTensor<T_X>();
            CalculateY(xLocal, gammaLocal, betaLocal, yLocal, meanFp32Local, rstdLocal, curCLen, rAlign_, numR_);
            inQueueX_.FreeTensor(xLocal);
            outQueueY_.EnQue<T_X>(yLocal);
            CopyOutY(offset * numR_, curCLen, numR_, rAlign_);

            if (cIdxPre != cIdx) {
                inQueueGamma_.FreeTensor(gammaLocal);
                inQueueBeta_.FreeTensor(betaLocal);
                cIdxPre = cIdx;
            }
        }
    }

private:
    __aicore__ inline void CopyInX(uint64_t offset, uint32_t cnt, uint32_t r, uint32_t rAlign)
    {
        LocalTensor<T_X> xLocal = inQueueX_.AllocTensor<T_X>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(cnt),                                          // blockCount
            static_cast<uint32_t>(r * sizeof(T_X)),                              // blockLen
            static_cast<uint32_t>(0),                                            // srcStride
            static_cast<uint32_t>((rAlign - r) * sizeof(T_X) / ALIGN_32_FACTOR), // dstStride
            0                                                                    // rsv
        };
        DataCopyPadExtParams<T_X> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T_X>(0.0)    // paddingValue
        };
        DataCopyPad(xLocal, xGm_[offset], extParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGammaBeta(uint64_t cOffset, uint32_t tileLen)
    {
        LocalTensor<T_BETA> gammaLocal = inQueueGamma_.AllocTensor<T_BETA>();
        LocalTensor<T_BETA> betaLocal = inQueueBeta_.AllocTensor<T_BETA>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),                        // blockCount
            static_cast<uint32_t>(tileLen * sizeof(T_BETA)), // blockLen
            static_cast<uint32_t>(0),                        // srcStride
            static_cast<uint32_t>(0),                        // dstStride
            0                                                // rsv
        };
        DataCopyPadExtParams<T_BETA> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T_BETA>(0)   // paddingValue
        };
        DataCopyPad(gammaLocal, gammaGm_[cOffset], copyParams, padParams);
        DataCopyPad(betaLocal, betaGm_[cOffset], copyParams, padParams);
        inQueueGamma_.EnQue(gammaLocal);
        inQueueBeta_.EnQue(betaLocal);
    }

    __aicore__ inline void CopyOutMeanVar(uint64_t offset, uint32_t cnt)
    {
        LocalTensor<T_MEAN> meanLocal = outQueueMean_.DeQue<T_MEAN>();
        LocalTensor<T_MEAN> varLocal = outQueueVariance_.DeQue<T_MEAN>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),                    // blockCount
            static_cast<uint32_t>(cnt * sizeof(T_MEAN)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPad(meanGm_[offset], meanLocal, copyParams);
        DataCopyPad(varianceGm_[offset], varLocal, copyParams);
        outQueueMean_.FreeTensor(meanLocal);
        outQueueVariance_.FreeTensor(varLocal);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint32_t cnt, uint32_t r, uint32_t rAlign)
    {
        LocalTensor<T_X> yLocal = outQueueY_.DeQue<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(cnt),                                          // blockCount
            static_cast<uint32_t>(r * sizeof(T_X)),                              // blockLen
            static_cast<uint32_t>((rAlign - r) * sizeof(T_X) / ALIGN_32_FACTOR), // srcStride
            static_cast<uint32_t>(0),                                            // dstStride
            0                                                                    // rsv
        };
        DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CalculateY(
        LocalTensor<T_X>& xLocal, LocalTensor<T_BETA>& gammaLocal, LocalTensor<T_BETA>& betaLocal,
        LocalTensor<T_X>& yLocal, LocalTensor<float>& meanFp32Local, LocalTensor<float>& rstdLocal, uint32_t curRows,
        uint32_t numColAlign, uint32_t reduceNum)
    {
        __local_mem__ T_X* xInUb = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_X* yInUb = (__local_mem__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ float* meanFp32Ub = (__local_mem__ float*)meanFp32Local.GetPhyAddr();
        __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_BETA* gammaInUb = (__local_mem__ T_BETA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_BETA* betaInUb = (__local_mem__ T_BETA*)betaLocal.GetPhyAddr();

        uint16_t loopRows = static_cast<uint16_t>(curRows);
        uint16_t loopCols = static_cast<uint16_t>((reduceNum + VL_FP32 - 1) / VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> rstdReg;
            RegTensor<float> meanReg;
            RegTensor<float> gammaReg;
            RegTensor<float> betaReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            for (uint16_t i = 0; i < loopRows; ++i) {
                uint32_t sregCount = reduceNum;
                LoadScalarForDtypeTIn<float>(rstdInUb, rstdReg, pregFull, i);
                LoadScalarForDtypeTIn<float>(meanFp32Ub, meanReg, pregFull, i);
                LoadScalarForDtypeTIn<T_BETA>(gammaInUb, gammaReg, pregFull, i);
                LoadScalarForDtypeTIn<T_BETA>(betaInUb, betaReg, pregFull, i);
                for (uint16_t r = 0; r < loopCols; ++r) {
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    uint32_t offset = i * numColAlign + r * VL_FP32;
                    LoadTensorForDtypeTIn<T_X>(xInUb, xReg, regCurLoop, offset);
                    Sub(xReg, xReg, meanReg, regCurLoop);
                    Mul(xReg, xReg, rstdReg, regCurLoop);
                    Mul(xReg, xReg, gammaReg, regCurLoop);
                    Add(xReg, xReg, betaReg, regCurLoop);
                    StoreTensorForDtypeTOut<T_X>(yInUb, xReg, regCurLoop, offset);
                }
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSum(
        LocalTensor<T_X>& xLocal, LocalTensor<T_MEAN>& meanLocal, LocalTensor<T_MEAN>& varLocal,
        LocalTensor<float>& meanFp32Local, LocalTensor<float>& varFp32Local, uint32_t curRows, uint32_t numColAlign,
        uint32_t reduceNum, float avgFactor)
    {
        LocalTensor<float> binaryAddBuffTmp = binaryAddBuf_.Get<float>();
        __local_mem__ T_X* xInUb = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_MEAN* meanUb = (__local_mem__ T_MEAN*)meanLocal.GetPhyAddr();
        __local_mem__ T_MEAN* varUb = (__local_mem__ T_MEAN*)varLocal.GetPhyAddr();

        __local_mem__ float* meanFp32Ub = (__local_mem__ float*)meanFp32Local.GetPhyAddr();
        __local_mem__ float* varFp32Ub = (__local_mem__ float*)varFp32Local.GetPhyAddr();
        __local_mem__ float* tmpUb = (__local_mem__ float*)binaryAddBuffTmp.GetPhyAddr(); // 二分累加 tmp buffer

        if (reduceNum <= VL_FP32) {
            CalculateMeanLessThanVL(xInUb, meanUb, meanFp32Ub, curRows, numColAlign, reduceNum, avgFactor);
            CalculateSquareReduceSumLessThanVL(
                xInUb, meanFp32Ub, varUb, varFp32Ub, curRows, numColAlign, reduceNum, avgFactor);
        } else if (reduceNum <= VL_FP32 + VL_FP32) {
            CalculateMeanLessThanTwoVL(xInUb, meanUb, meanFp32Ub, curRows, numColAlign, reduceNum, avgFactor);
            CalculateSquareReduceSumLessThanTwoVL(
                xInUb, meanFp32Ub, varUb, varFp32Ub, curRows, numColAlign, reduceNum, avgFactor);
        } else if (reduceNum <= VL_FP32 * VL_FP32 * NUM_TWO) {
            CalculateMeanSumCommon<NUM_ONE>(
                xInUb, meanUb, meanFp32Ub, tmpUb, curRows, numColAlign, reduceNum, avgFactor);
            CalculateSquareReduceSumCommon<NUM_ONE>(
                xInUb, meanFp32Ub, varUb, tmpUb, varFp32Ub, curRows, numColAlign, reduceNum, avgFactor);
        } else {
            CalculateMeanSumCommon<NUM_TWO>(
                xInUb, meanUb, meanFp32Ub, tmpUb, curRows, numColAlign, reduceNum, avgFactor);
            CalculateSquareReduceSumCommon<NUM_TWO>(
                xInUb, meanFp32Ub, varUb, tmpUb, varFp32Ub, curRows, numColAlign, reduceNum, avgFactor);
        }
    }

    // LessThanVL
    __aicore__ inline void CalculateMeanLessThanVL(
        __local_mem__ T_X* xInUb, __local_mem__ T_MEAN* meanUb, __local_mem__ float* meanFp32Ub, uint16_t curRows,
        uint32_t numColAlign, uint32_t reduceNum, float avgFactor)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> mean;

            uint32_t sreg0 = reduceNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            for (uint16_t i = 0; i < curRows; i++) {
                LoadTensorForDtypeTIn<T_X>(xInUb, x, pregLoop, i * numColAlign);
                ReduceSum(mean, x, pregLoop);
                Muls(mean, mean, avgFactor, pregOne);
                StoreOneElementForDtypeTOut<T_MEAN>(meanUb, mean, pregOne, i);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanFp32Ub + i, mean, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanVL(
        __local_mem__ T_X* xInUb, __local_mem__ float* meanFp32Ub, __local_mem__ T_MEAN* varUb,
        __local_mem__ float* varFp32Ub, uint16_t curRows, uint32_t numColAlign, uint32_t reduceNum, float avgFactor)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> mean;
            RegTensor<float> vMean;

            uint32_t sreg0 = reduceNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            for (uint16_t i = 0; i < curRows; i++) {
                LoadTensorForDtypeTIn<T_X>(xInUb, x, pregLoop, i * numColAlign);
                LoadScalarForDtypeTIn<float>(meanFp32Ub, mean, pregLoop, i);
                Sub(x, x, mean, pregLoop);
                Mul(x, x, x, pregLoop);
                ReduceSum(vMean, x, pregLoop);
                Muls(vMean, vMean, avgFactor, pregOne);
                StoreOneElementForDtypeTOut<T_MEAN>(varUb, vMean, pregOne, i);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varFp32Ub + i, vMean, pregOne);
            }
        }
    }

    // LessThanTwoVL
    __aicore__ inline void CalculateMeanLessThanTwoVL(
        __local_mem__ T_X* xInUb, __local_mem__ T_MEAN* meanUb, __local_mem__ float* meanFp32Ub, uint16_t curRows,
        uint32_t numColAlign, uint32_t reduceNum, float avgFactor)
    {
        uint32_t tailLen = reduceNum - VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> mean;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregTail = UpdateMask<float>(tailLen);
            for (uint16_t i = 0; i < curRows; ++i) {
                LoadTensorForDtypeTIn<T_X>(xInUb, x, pregFull, i * numColAlign);
                LoadTensorForDtypeTIn<T_X>(xInUb + VL_FP32, xFold, pregTail, i * numColAlign);
                ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregTail);
                Add(x, x, xFold, pregFull);
                ReduceSum(mean, x, pregFull);
                Muls(mean, mean, avgFactor, pregOne);
                StoreOneElementForDtypeTOut<T_MEAN>(meanUb, mean, pregOne, i);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanFp32Ub + i, mean, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(
        __local_mem__ T_X* xInUb, __local_mem__ float* meanFp32Ub, __local_mem__ T_MEAN* varUb,
        __local_mem__ float* varFp32Ub, uint16_t curRows, uint32_t numColAlign, uint32_t reduceNum, float avgFactor)
    {
        uint32_t tailLen = reduceNum - VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> mean;
            RegTensor<float> vMean;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregTail = UpdateMask<float>(tailLen);
            for (uint16_t i = 0; i < curRows; ++i) {
                LoadTensorForDtypeTIn<T_X>(xInUb, x, pregFull, i * numColAlign);
                LoadTensorForDtypeTIn<T_X>(xInUb + VL_FP32, xFold, pregTail, i * numColAlign);
                LoadScalarForDtypeTIn<float>(meanFp32Ub, mean, pregFull, i);
                ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregTail);
                Sub(x, x, mean, pregFull);
                Sub(xFold, xFold, mean, pregTail);
                Mul(x, x, x, pregFull);
                Mul(xFold, xFold, xFold, pregTail);
                Add(x, x, xFold, pregFull);
                ReduceSum(vMean, x, pregFull);
                Muls(vMean, vMean, avgFactor, pregOne);
                StoreOneElementForDtypeTOut<T_MEAN>(varUb, vMean, pregOne, i);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varFp32Ub + i, vMean, pregOne);
            }
        }
    }

    //
    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateMeanSumCommon(
        __local_mem__ T_X* xInUb, __local_mem__ T_MEAN* meanUb, __local_mem__ float* meanFp32Ub,
        __local_mem__ float* tmpUb, uint16_t curRows, uint32_t numColAlign, uint32_t reduceNum, float avgFactor)
    {
        uint32_t binaryAddQuotient = binaryAddQuotient_; // TODO:
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + VL_FP32 - 1) / VL_FP32;

        uint32_t lastBinaryAddNum = binaryAddQuotient / VL_FP32;
        uint32_t lastBinaryAddNumAlign = (binaryAddQuotientLoop + BLK_B32 - 1) / BLK_B32 * BLK_B32;

        uint32_t binaryAddRemainder = reduceNum - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_FP32 - 1) / VL_FP32;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> mean;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;
            for (uint16_t i = 0; i < curRows; ++i) {
                uint32_t baseOffset = i * numColAlign;
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; ++r) {
                    uint32_t offset = r * VL_FP32 + baseOffset;
                    LoadTensorForDtypeTIn<T_X>(xInUb, x, pregFull, offset);
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddQuotient, xFold, pregFull, offset);
                    Add(x, x, xFold, pregFull);
                    ReduceSum(mean, x, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + r), mean, pregOne);
                }
                uint32_t sregRemainder = binaryAddRemainder - binaryAddRemainderFloorLoop * VL_FP32;
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    uint32_t offset = baseOffset;
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddRemainderFloorLoop * VL_FP32, x, pregFull, offset);
                    LoadTensorForDtypeTIn<T_X>(
                        xInUb + binaryAddRemainderFloorLoop * VL_FP32 + binaryAddQuotient, xFold, pregLoop, offset);
                    ShiftLefts(
                        (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLoop);
                    Add(x, x, xFold, pregFull);
                    ReduceSum(mean, x, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), mean,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    uint32_t offset = r * VL_FP32 + baseOffset;
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddRemainderCeilLoop * VL_FP32, x, pregFull, offset);
                    ReduceSum(mean, x, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r), mean,
                        pregOne);
                }
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < curRows; ++i) {
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    ReduceSum(vMean, x, pregLast);
                    Muls(vMean, vMean, avgFactor, pregOne);
                    StoreOneElementForDtypeTOut<T_MEAN>(meanUb, vMean, pregOne, i);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanFp32Ub + i, vMean, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                lastBinaryAddNum -= VL_FP32;
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < curRows; ++i) {
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    DataCopy(xFold, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + VL_FP32));
                    ShiftLefts(
                        (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLast);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    Muls(vMean, vMean, avgFactor, pregOne);
                    StoreOneElementForDtypeTOut<T_MEAN>(meanUb, vMean, pregOne, i);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanFp32Ub + i, vMean, pregOne);
                }
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateSquareReduceSumCommon(
        __local_mem__ T_X* xInUb, __local_mem__ float* meanFp32Ub, __local_mem__ T_MEAN* varUb,
        __local_mem__ float* tmpUb, __local_mem__ float* varFp32Ub, uint16_t curRows, uint32_t numColAlign,
        uint32_t reduceNum, float avgFactor)
    {
        uint32_t binaryAddQuotient = binaryAddQuotient_;
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + VL_FP32 - 1) / VL_FP32;

        uint32_t lastBinaryAddNum = binaryAddQuotient / VL_FP32;
        uint32_t lastBinaryAddNumAlign = (binaryAddQuotientLoop + BLK_B32 - 1) / BLK_B32 * BLK_B32;

        uint32_t binaryAddRemainder = reduceNum - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_FP32 - 1) / VL_FP32;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> mean;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t i = 0; i < curRows; ++i) {
                uint32_t baseOffset = i * numColAlign;
                LoadScalarForDtypeTIn<float>(meanFp32Ub, mean, pregFull, i);
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; ++r) {
                    uint32_t offset = r * VL_FP32 + baseOffset;
                    LoadTensorForDtypeTIn<T_X>(xInUb, x, pregFull, offset);
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddQuotient, xFold, pregFull, offset);
                    Sub(x, x, mean, pregFull);
                    Sub(xFold, xFold, mean, pregFull);
                    Mul(x, x, x, pregFull);
                    Mul(xFold, xFold, xFold, pregFull);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + r), vMean, pregOne);
                }
                uint32_t sregRemainder = binaryAddRemainder - binaryAddRemainderFloorLoop * VL_FP32;
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    uint32_t offset = baseOffset;
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddRemainderFloorLoop * VL_FP32, x, pregFull, offset);
                    LoadTensorForDtypeTIn<T_X>(
                        xInUb + binaryAddRemainderFloorLoop * VL_FP32 + binaryAddQuotient, xFold, pregLoop, offset);
                    Sub(x, x, mean, pregFull);
                    Sub(xFold, xFold, mean, pregLoop);
                    Mul(x, x, x, pregFull);
                    Mul(xFold, xFold, xFold, pregLoop);
                    ShiftLefts(
                        (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLoop);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), vMean,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    uint32_t offset = r * VL_FP32 + baseOffset;
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddRemainderCeilLoop * VL_FP32, x, pregFull, offset);
                    Sub(x, x, mean, pregFull);
                    Mul(x, x, x, pregFull);
                    ReduceSum(vMean, x, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r),
                        vMean, pregOne);
                }
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < curRows; ++i) {
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    ReduceSum(vMean, x, pregLast);
                    Muls(vMean, vMean, avgFactor, pregOne);
                    StoreOneElementForDtypeTOut<T_MEAN>(varUb, vMean, pregOne, i);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varFp32Ub + i, vMean, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                lastBinaryAddNum -= VL_FP32;
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < curRows; ++i) {
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    DataCopy(xFold, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + VL_FP32));
                    ShiftLefts(
                        (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLast);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    Muls(vMean, vMean, avgFactor, pregOne);
                    StoreOneElementForDtypeTOut<T_MEAN>(varUb, vMean, pregOne, i);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varFp32Ub + i, vMean, pregOne);
                }
            }
        }
    }

    __aicore__ inline void CalculateRstd(
        LocalTensor<float>& varLocal, uint32_t curRows, float epsilon, LocalTensor<float>& rstdLocal)
    {
        static constexpr float POS_INF = 3.40282366920938E+38;
        static constexpr float SCALAR1 = -0.5;
        static constexpr float SCALAR2 = 1.5;
        static constexpr float SCALAR3 = 0.5;
        static constexpr float SCALAR0 = -99.99;

        __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* varUb = (__local_mem__ float*)varLocal.GetPhyAddr();
        uint16_t loopRows = static_cast<uint16_t>((curRows + VL_FP32 - 1) / VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> var;
            RegTensor<float> rstd;
            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> one;
            RegTensor<float> scalar1;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;

            uint32_t sreg = static_cast<uint32_t>(curRows);
            for (uint16_t i = 0; i < loopRows; ++i) {
                pregLoop = UpdateMask<float>(sreg);
                Duplicate(scalarInf, POS_INF, pregLoop);
                Duplicate(scalarZero, float(0.0), pregLoop);
                Duplicate(one, float(1.0), pregLoop);
                Duplicate(scalar1, SCALAR3, pregLoop);
                Duplicate(t1, SCALAR2, pregLoop);
                Duplicate(s, float(1.0), pregLoop);
                // rstd
                DataCopy(var, varUb + i * VL_FP32);
                Adds(var, var, epsilon, pregLoop);
                Maxs(var, var, SCALAR0, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, SCALAR1, pregLoop);
                Mul(t, t, y, pregLoop);                // -0.5 * x * y
                Mula(t1, t, y, pregLoop);              // 1.5 + (-0.5 * x * y) * y
                Mul(rstd, y, t1, pregLoop);            // y = y * (1.5 - 0.5 * x * y)
                Muls(t3, var, float(-1.0), pregLoop);  // -1 * x
                Mula(s, t3, r, pregLoop);              // 1 + (-1) * x * r
                Muls(t4, rstd, float(-1.0), pregLoop); // (-1) * y
                Mula(r, t4, rstd, pregLoop);           // r + (-1) * y * y
                Mula(s, var, r, pregLoop);             // s + x * t
                Mul(s, s, rstd, pregLoop);             // e * y
                Mula(rstd, s, scalar1, pregLoop);      // y + y * e * 0.5
                CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
                Select(rstd, scalarZero, rstd, cmpRegZero);
                CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
                Select(rstd, scalarInf, rstd, cmpRegInf);
                DataCopy(rstdInUb + i * VL_FP32, rstd, pregLoop);
            }
        }
    }

private:
    /*
     * inQueueX_: cInner_ * numR_
     *
     */
    TPipe pipe_;
    GlobalTensor<T_X> xGm_, yGm_;
    GlobalTensor<T_BETA> gammaGm_, betaGm_;
    GlobalTensor<T_MEAN> meanGm_, varianceGm_;
    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX_, inQueueGamma_, inQueueBeta_;
    TQue<QuePosition::VECOUT, 1> outQueueY_, outQueueMean_, outQueueVariance_;
    // Buffer
    TBuf<TPosition::VECCALC> rstdBuf_;
    TBuf<TPosition::VECCALC> meanFp32Buff_;
    TBuf<TPosition::VECCALC> binaryAddBuf_;

    // Platform
    int64_t blockIdx_{0};
    uint64_t blockNum_{0};
    // Tiling data
    int64_t cInner_;
    int64_t cOuter_;
    int64_t cTail_;
    int64_t numN_;
    int64_t numC_;
    int64_t numR_;
    uint64_t rAlign_;            // r 对齐
    uint32_t binaryAddQuotient_; // 二分累加折叠点

    uint32_t perCoreCnt_;
    float epsilon_{0};
    float avgFactor_{0};
};
} // namespace InstanceNormOps
#endif // INSTANCE_NORM_AR_FULL_REDUCE_H_
