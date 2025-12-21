/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_layer_norm_regbase_full_load.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_REGBASE_FULL_LOAD_H
#define ADD_LAYER_NORM_REGBASE_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "add_layer_norm_regbase_common.h"

namespace AddLayerNorm {
template <
    typename X1_TYPE, typename X2_TYPE, typename GAMMA_TYPE, typename BETA_TYPE, typename BIAS_TYPE, int TILING_KEY,
    int BUFFER_NUM = 1>
class RegbaseFullLoad {
public:
    static constexpr bool isMix =
        !(IsSameType<X1_TYPE, X2_TYPE>::value && IsSameType<X1_TYPE, GAMMA_TYPE>::value &&
          IsSameType<X1_TYPE, BETA_TYPE>::value && IsSameType<X1_TYPE, BIAS_TYPE>::value);

    __aicore__ inline RegbaseFullLoad(const AddLayerNormRegbaseTilingData* tilingData) : tiling_(tilingData)
    {}

    __aicore__ inline void Init(
        __gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* gamma, __gm__ uint8_t* beta, __gm__ uint8_t* bias,
        __gm__ uint8_t* y, __gm__ uint8_t* mean, __gm__ uint8_t* rstd, __gm__ uint8_t* x)
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= tiling_->usedCoreNum) {
            return;
        }

        blockSize_ = tiling_->blockSize;
        vlFp32_ = tiling_->vlFp32;
        tailCoreStartIndex_ = tiling_->tailCoreStartIndex;
        colsPerLoop_ = tiling_->colsPerLoop;
        eps_ = tiling_->eps;
        binaryAddNum_ = tiling_->binaryAddNum;
        binaryAddK_ = tiling_->binaryAddK;
        binaryAddLastNum_ = tiling_->binaryAddLastNum;

        powerOfTwo_ = 1;
        while (powerOfTwo_ < colsPerLoop_) {
            powerOfTwo_ *= TWO;
        }

        uint64_t gmOffset;
        uint64_t meanOffset;
        if (coreIdx < tailCoreStartIndex_) {
            // non-tail cores
            rowsPerCore_ = tiling_->rowsPerCore;
            rowsPerLoop_ = tiling_->rowsPerLoop;
            gmOffset = (tiling_->rowsPerCore * colsPerLoop_) * coreIdx;
            meanOffset = gmOffset / colsPerLoop_;
        } else {
            // tail cores
            rowsPerCore_ = tiling_->rowsPerTailCore;
            rowsPerLoop_ = tiling_->rowsPerLoop;
            gmOffset = tailCoreStartIndex_ * tiling_->rowsPerCore * colsPerLoop_ +
                       (coreIdx - tailCoreStartIndex_) * tiling_->rowsPerTailCore * colsPerLoop_;
            meanOffset = gmOffset / colsPerLoop_;
        }
        rowsTail_ = (rowsPerCore_ % rowsPerLoop_ == 0) ? rowsPerLoop_ : (rowsPerCore_ % rowsPerLoop_);
        rowsLoopCount_ = CEIL_DIV(rowsPerCore_, rowsPerLoop_);

        x1Gm_.SetGlobalBuffer((__gm__ X1_TYPE*)(x1) + gmOffset);
        x2Gm_.SetGlobalBuffer((__gm__ X2_TYPE*)(x2) + gmOffset);
        if constexpr (IS_BIAS_ELEWISE) {
            biasGm_.SetGlobalBuffer((__gm__ BIAS_TYPE*)(bias) + gmOffset);
        } else if constexpr (IS_BIAS_BROADCAST) {
            biasGm_.SetGlobalBuffer((__gm__ BIAS_TYPE*)bias);
        }
        gammaGm_.SetGlobalBuffer((__gm__ GAMMA_TYPE*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ BETA_TYPE*)beta);
        yGm_.SetGlobalBuffer((__gm__ BIAS_TYPE*)(y) + gmOffset);
        xGm_.SetGlobalBuffer((__gm__ BIAS_TYPE*)(x) + gmOffset);
        // mean/rstd always output fp32
        meanGm_.SetGlobalBuffer((__gm__ float*)mean + meanOffset);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + meanOffset);

        colsPerLoopAlign_ = BLOCK_ALIGN(colsPerLoop_ * sizeof(X1_TYPE), blockSize_) / sizeof(X1_TYPE);

        if constexpr (isMix) {
            colsPerLoopAlignB16_ = BLOCK_ALIGN(colsPerLoop_ * sizeof(half), blockSize_) / sizeof(half);
            colsPerLoopAlignBias_ = BLOCK_ALIGN(colsPerLoop_ * sizeof(BIAS_TYPE), blockSize_) / sizeof(BIAS_TYPE);
            colsPerLoopAlignB32_ = BLOCK_ALIGN(colsPerLoop_ * sizeof(float), blockSize_) / sizeof(float);
            pipe_.InitBuffer(x1Queue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlignB16_ * sizeof(float)));
            pipe_.InitBuffer(x2Queue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlignB16_ * sizeof(float)));
            if constexpr (IS_BIAS_ELEWISE) {
                pipe_.InitBuffer(biasQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlignB16_ * sizeof(float)));
            } else if constexpr (IS_BIAS_BROADCAST) {
                pipe_.InitBuffer(biasQueue_, 1, (colsPerLoopAlignB16_ * sizeof(float)));
            }
            pipe_.InitBuffer(xQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlignB16_ * sizeof(float)));
            pipe_.InitBuffer(yQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlignB16_ * sizeof(float)));
            pipe_.InitBuffer(x32Queue_, (rowsPerLoop_ * colsPerLoopAlignB16_ * sizeof(float)));
            pipe_.InitBuffer(betaQueue_, 1, (colsPerLoopAlignB16_ * sizeof(float)));
            pipe_.InitBuffer(gammaQueue_, 1, (colsPerLoopAlignB16_ * sizeof(float)));
        } else {
            pipe_.InitBuffer(x1Queue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
            pipe_.InitBuffer(x2Queue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
            if constexpr (IS_BIAS_ELEWISE) {
                pipe_.InitBuffer(biasQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
            } else if constexpr (IS_BIAS_BROADCAST) {
                pipe_.InitBuffer(biasQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
            }
            pipe_.InitBuffer(xQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
            pipe_.InitBuffer(yQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
            pipe_.InitBuffer(x32Queue_, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(float)));
            pipe_.InitBuffer(betaQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
            pipe_.InitBuffer(gammaQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        }

        pipe_.InitBuffer(meanQueue_, BUFFER_NUM, BLOCK_ALIGN(rowsPerLoop_ * sizeof(float), blockSize_));
        pipe_.InitBuffer(rstdQueue_, BUFFER_NUM, BLOCK_ALIGN(rowsPerLoop_ * sizeof(float), blockSize_));
        int64_t binaryAddBufSize = BLOCK_ALIGN((binaryAddNum_ / vlFp32_) * sizeof(float), blockSize_);
        if (binaryAddBufSize > 0) {
            pipe_.InitBuffer(binaryAddBuf_, binaryAddBufSize);
        }
    }

    __aicore__ inline void CopyBiasToUB(LocalTensor<BIAS_TYPE> biasLocal, int32_t copyLen)
    {
        if constexpr (isMix) {
            const int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(BIAS_TYPE), blockSize_) / sizeof(BIAS_TYPE);
            const int32_t copyLenAlignB16 = BLOCK_ALIGN(copyLen * sizeof(half), blockSize_) / sizeof(half);

            DataCopyPadExtParams<BIAS_TYPE> padParams;
            padParams.isPad = true;
            padParams.paddingValue = static_cast<BIAS_TYPE>(0.0);
            padParams.rightPadding = copyLenAlign - copyLen;

            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = copyLen * sizeof(BIAS_TYPE);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (copyLenAlignB16 != copyLenAlign) ? 1 : 0;

            DataCopyPad(biasLocal, biasGm_[0], dataCopyParams, padParams);
            biasQueue_.EnQue(biasLocal);
        } else {
            int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(BIAS_TYPE), blockSize_) / sizeof(BIAS_TYPE);

            DataCopyPadExtParams<BIAS_TYPE> padParams;
            padParams.isPad = true;
            padParams.paddingValue = static_cast<BIAS_TYPE>(0.0);
            padParams.rightPadding = copyLenAlign - copyLen;

            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = copyLen * sizeof(BIAS_TYPE);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;

            DataCopyPad(biasLocal, biasGm_[0], dataCopyParams, padParams);
            biasQueue_.EnQue(biasLocal);
        }
    }

    __aicore__ inline void CopyInputsToUB(
        LocalTensor<X1_TYPE> x1Local, LocalTensor<X2_TYPE> x2Local, LocalTensor<BIAS_TYPE> biasLocal,
        int64_t inputOffset, int32_t copyLen, int32_t rowsCount)
    {
        if constexpr (isMix) {
            {
                const int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(X1_TYPE), blockSize_) / sizeof(X1_TYPE);
                const int32_t copyLenAlignB16 = BLOCK_ALIGN(copyLen * sizeof(half), blockSize_) / sizeof(half);
                DataCopyPadExtParams<X1_TYPE> padParams;
                padParams.isPad = true;
                padParams.paddingValue = static_cast<X1_TYPE>(0.0);
                padParams.rightPadding = copyLenAlign - copyLen;
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = rowsCount;
                dataCopyParams.blockLen = copyLen * sizeof(X1_TYPE);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = (copyLenAlignB16 != copyLenAlign) ? 1 : 0;
                DataCopyPad(x1Local, x1Gm_[inputOffset], dataCopyParams, padParams);
                x1Queue_.EnQue(x1Local);
            }
            {
                const int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(X2_TYPE), blockSize_) / sizeof(X2_TYPE);
                const int32_t copyLenAlignB16 = BLOCK_ALIGN(copyLen * sizeof(half), blockSize_) / sizeof(half);
                DataCopyPadExtParams<X2_TYPE> padParams;
                padParams.isPad = true;
                padParams.paddingValue = static_cast<X2_TYPE>(0.0);
                padParams.rightPadding = copyLenAlign - copyLen;
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = rowsCount;
                dataCopyParams.blockLen = copyLen * sizeof(X2_TYPE);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = (copyLenAlignB16 != copyLenAlign) ? 1 : 0;
                DataCopyPad(x2Local, x2Gm_[inputOffset], dataCopyParams, padParams);
                x2Queue_.EnQue(x2Local);
            }
            {
                const int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(BIAS_TYPE), blockSize_) / sizeof(BIAS_TYPE);
                const int32_t copyLenAlignB16 = BLOCK_ALIGN(copyLen * sizeof(half), blockSize_) / sizeof(half);
                DataCopyPadExtParams<BIAS_TYPE> padParams;
                padParams.isPad = true;
                padParams.paddingValue = static_cast<BIAS_TYPE>(0.0);
                padParams.rightPadding = copyLenAlign - copyLen;
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = rowsCount;
                dataCopyParams.blockLen = copyLen * sizeof(BIAS_TYPE);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = (copyLenAlignB16 != copyLenAlign) ? 1 : 0;
                if constexpr (IS_BIAS_ELEWISE) {
                    DataCopyPad(biasLocal, biasGm_[inputOffset], dataCopyParams, padParams);
                    biasQueue_.EnQue(biasLocal);
                }
            }
        } else {
            int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(X1_TYPE), blockSize_) / sizeof(X1_TYPE);
            DataCopyPadExtParams<X1_TYPE> padParams;
            padParams.isPad = true;
            padParams.paddingValue = static_cast<X1_TYPE>(0.0);
            padParams.rightPadding = copyLenAlign - copyLen;
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = rowsCount;
            dataCopyParams.blockLen = copyLen * sizeof(X1_TYPE);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(x1Local, x1Gm_[inputOffset], dataCopyParams, padParams);
            x1Queue_.EnQue(x1Local);
            DataCopyPad(x2Local, x2Gm_[inputOffset], dataCopyParams, padParams);
            x2Queue_.EnQue(x2Local);
            if constexpr (IS_BIAS_ELEWISE) {
                DataCopyPad(biasLocal, biasGm_[inputOffset], dataCopyParams, padParams);
                biasQueue_.EnQue(biasLocal);
            }
        }
    }

    __aicore__ inline void CopyXToGm(LocalTensor<BIAS_TYPE> xLocal, int64_t xOffset, int32_t copyLen, int32_t rowsCount)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = rowsCount;
        dataCopyParams.blockLen = copyLen * sizeof(BIAS_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(xGm_[xOffset], xLocal, dataCopyParams);
    }

    __aicore__ inline void CopyMeanToGm(LocalTensor<float> meanLocal, int64_t meanOffset, int32_t rowsCount)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = rowsCount * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(meanGm_[meanOffset], meanLocal, dataCopyParams);
    }

    __aicore__ inline void CopyRstdToGm(LocalTensor<float> rstdLocal, int64_t rstdOffset, int32_t rowsCount)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = rowsCount * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(rstdGm_[rstdOffset], rstdLocal, dataCopyParams);
    }

    __aicore__ inline void CopyYToGm(LocalTensor<BIAS_TYPE> yLocal, int64_t yOffset, int32_t copyLen, int32_t rowsCount)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = rowsCount;
        dataCopyParams.blockLen = copyLen * sizeof(BIAS_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(yGm_[yOffset], yLocal, dataCopyParams);
    }

    __aicore__ inline void CopyGammaAndBetaToUB(
        LocalTensor<GAMMA_TYPE> gammaLocal, LocalTensor<BETA_TYPE> betaLocal, int64_t offset, int32_t copyLen)
    {
        if constexpr (isMix) {
            int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(float), blockSize_) / sizeof(float);
            DataCopyPadExtParams<float> padParams;
            padParams.isPad = true;
            padParams.paddingValue = static_cast<float>(0.0);
            padParams.rightPadding = copyLenAlign - copyLen;
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = copyLen * sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(betaLocal, betaGm_[offset], dataCopyParams, padParams);
            betaQueue_.EnQue(betaLocal);
            DataCopyPad(gammaLocal, gammaGm_[offset], dataCopyParams, padParams);
            gammaQueue_.EnQue(gammaLocal);
        } else {
            int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(GAMMA_TYPE), blockSize_) / sizeof(GAMMA_TYPE);
            DataCopyPadExtParams<GAMMA_TYPE> padParams;
            padParams.isPad = true;
            padParams.paddingValue = static_cast<GAMMA_TYPE>(0.0);
            padParams.rightPadding = copyLenAlign - copyLen;
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = copyLen * sizeof(GAMMA_TYPE);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(betaLocal, betaGm_[offset], dataCopyParams, padParams);
            betaQueue_.EnQue(betaLocal);
            DataCopyPad(gammaLocal, gammaGm_[offset], dataCopyParams, padParams);
            gammaQueue_.EnQue(gammaLocal);
        }
    }

    __aicore__ inline void VFCalcMeanVarFast(
        __local_mem__ X1_TYPE* x1Addr, __local_mem__ X2_TYPE* x2Addr, __local_mem__ BIAS_TYPE* biasAddr,
        __local_mem__ BIAS_TYPE* xOutAddr, __local_mem__ float* x32Addr, __local_mem__ float* meanAddr,
        __local_mem__ float* varAddr, uint16_t rowsCount)
    {
        float n = static_cast<float>(1) / static_cast<float>(powerOfTwo_);
        float nCorrectionFactor = static_cast<float>(powerOfTwo_) / static_cast<float>(colsPerLoop_);
        float eps = eps_;
        uint32_t vlFp32 = vlFp32_;
        uint16_t rowsLoopCount = CEIL_DIV(rowsCount, vlFp32);
        uint32_t colsPerLoop = colsPerLoop_;
        uint32_t colsPerLoopAlign = colsPerLoopAlign_;
        uint32_t colsPerLoopAlignB16 = colsPerLoopAlignB16_;
        uint32_t colsPerLoopAlignB32 = colsPerLoopAlignB32_;
        uint32_t colsPerLoopAlignBias = colsPerLoopAlignBias_;

        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFactor;
            RegTensor<float> mean;

            RegTensor<float> y;
            RegTensor<float> yFactor;
            RegTensor<float> var;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            uint32_t sreg0 = colsPerLoop;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t i = 0; i < rowsCount; i++) {
                if constexpr (isMix) {
                    if constexpr (IS_BIAS_BROADCAST) {
                        LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                            x1Addr, x2Addr, biasAddr, x, pregLoop, i * colsPerLoopAlignB16, i * colsPerLoopAlignB16, 0);
                    } else {
                        LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                            x1Addr, x2Addr, biasAddr, x, pregLoop, i * colsPerLoopAlignB16, i * colsPerLoopAlignB16,
                            i * colsPerLoopAlignB16);
                    }
                } else {
                    if constexpr (IS_BIAS_BROADCAST) {
                        LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                            x1Addr, x2Addr, biasAddr, x, pregLoop, i * colsPerLoopAlign, i * colsPerLoopAlign, 0);
                    } else {
                        LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                            x1Addr, x2Addr, biasAddr, x, pregLoop, i * colsPerLoopAlign, i * colsPerLoopAlign,
                            i * colsPerLoopAlign);
                    }
                }
                // save xOut
                if constexpr (isMix) {
                    StoreRegToOutput(xOutAddr, x, pregLoop, i * colsPerLoopAlignBias);
                } else {
                    StoreRegToOutput(xOutAddr, x, pregLoop, i * colsPerLoopAlign);
                }
                // save x32
                if constexpr (isMix) {
                    DataCopy((__local_mem__ float*)x32Addr + i * colsPerLoopAlignB32, x, pregLoop);
                } else {
                    DataCopy((__local_mem__ float*)x32Addr + i * colsPerLoopAlign, x, pregLoop);
                }
                Muls(xFactor, x, n, pregLoop);
                ReduceSum(mean, xFactor, pregLoop);
                Muls(mean, mean, nCorrectionFactor, pregMerge);

                // save mean
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__local_mem__ float*)meanAddr + i, mean, pregMerge);

                Duplicate(mean, mean, pregMain);
                Muls(mean, mean, (float)-1.0, pregMain);
                // xDelta = x - mean
                Add(x, x, mean, pregLoop);
                Mul(y, x, x, pregLoop);
                Muls(yFactor, y, n, pregLoop);
                ReduceSum(var, yFactor, pregLoop);
                Muls(var, var, nCorrectionFactor, pregMerge);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__local_mem__ float*)varAddr + i, var, pregMerge);
            }
        }
    }

    __aicore__ inline void VFCalcMeanVar(
        __local_mem__ X1_TYPE* x1Addr, __local_mem__ X2_TYPE* x2Addr, __local_mem__ BIAS_TYPE* biasAddr,
        __local_mem__ BIAS_TYPE* xOutAddr, __local_mem__ float* x32Addr, __local_mem__ float* meanAddr,
        __local_mem__ float* varAddr, uint16_t rowsCount)
    {
        float n = static_cast<float>(1) / static_cast<float>(powerOfTwo_);
        float nCorrectionFactor = static_cast<float>(powerOfTwo_) / static_cast<float>(colsPerLoop_);
        float eps = eps_;
        uint32_t vlFp32 = vlFp32_;
        uint32_t colsPerLoopAlign = colsPerLoopAlign_;
        uint32_t colsPerLoopAlignB16 = colsPerLoopAlignB16_;
        uint32_t colsPerLoopAlignBias = colsPerLoopAlignBias_;
        uint32_t colsPerLoopAlignB32 = colsPerLoopAlignB32_;
        uint16_t rowsLoopCount = CEIL_DIV(rowsCount, vlFp32);

        uint64_t binaryAddLastNum = binaryAddLastNum_;
        uint32_t binaryAddOffset = binaryAddNum_;
        int64_t binaryAddRemainder = colsPerLoop_ - binaryAddNum_;
        uint16_t binaryAddRemainderLoop = CEIL_DIV(binaryAddRemainder, vlFp32);
        uint16_t binaryAddQuotientLoop = CEIL_DIV(binaryAddNum_, vlFp32);
        uint16_t binaryAddKLoop = binaryAddK_;
        uint16_t binaryAddLoopMean = ((binaryAddNum_ / vlFp32_) / vlFp32);
        uint16_t binaryAddLoopVar = binaryAddLoopMean;

        LocalTensor<float> binaryAddLocal = binaryAddBuf_.Get<float>();
        __local_mem__ float* binaryAddAddr = (__local_mem__ float*)binaryAddLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> meanTemp;
            RegTensor<float> mean;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> y1Pow;
            RegTensor<float> varTemp;
            RegTensor<float> var;

            RegTensor<float> binaryAddQ;
            RegTensor<float> binaryAddR;
            RegTensor<float> vlMean;

            RegTensor<float> binaryAddQPow;
            RegTensor<float> binaryAddRPow;
            RegTensor<float> vlVar;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t k = 0; k < rowsCount; k++) {
                uint32_t sreg0 = binaryAddRemainder;
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    if constexpr (isMix) {
                        if constexpr (IS_BIAS_BROADCAST) {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlignB16,
                                i * vlFp32 + k * colsPerLoopAlignB16, i * vlFp32);
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                                i * vlFp32 + k * colsPerLoopAlignB16 + binaryAddOffset,
                                i * vlFp32 + k * colsPerLoopAlignB16 + binaryAddOffset, i * vlFp32 + binaryAddOffset);
                        } else {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlignB16,
                                i * vlFp32 + k * colsPerLoopAlignB16, i * vlFp32 + k * colsPerLoopAlignB16);
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                                i * vlFp32 + k * colsPerLoopAlignB16 + binaryAddOffset,
                                i * vlFp32 + k * colsPerLoopAlignB16 + binaryAddOffset,
                                i * vlFp32 + k * colsPerLoopAlignB16 + binaryAddOffset);
                        }
                        StoreRegToOutput(xOutAddr, binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlignBias);
                        StoreRegToOutput(
                            xOutAddr, binaryAddR, pregLoop, i * vlFp32 + k * colsPerLoopAlignBias + binaryAddOffset);
                        DataCopy(
                            (__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlignB32, binaryAddQ, pregLoop);
                        DataCopy(
                            (__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlignB32 + binaryAddOffset,
                            binaryAddR, pregLoop);
                    } else {
                        if constexpr (IS_BIAS_BROADCAST) {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlign,
                                i * vlFp32 + k * colsPerLoopAlign, i * vlFp32);
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                                i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                                i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset, i * vlFp32 + binaryAddOffset);
                        } else {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlign,
                                i * vlFp32 + k * colsPerLoopAlign, i * vlFp32 + k * colsPerLoopAlign);
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                                i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                                i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                                i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                        }
                        StoreRegToOutput(xOutAddr, binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlign);
                        StoreRegToOutput(
                            xOutAddr, binaryAddR, pregLoop, i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                        DataCopy(
                            (__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign, binaryAddQ, pregLoop);
                        DataCopy(
                            (__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                            binaryAddR, pregLoop);
                    }
                    Muls(binaryAddQ, binaryAddQ, n, pregLoop);
                    Muls(binaryAddR, binaryAddR, n, pregLoop);
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregLoop);
                    ReduceSum(vlMean, binaryAddQ, pregLoop);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddAddr + i), vlMean, pregMerge);
                }
                {
                    pregLoop = UpdateMask<float>(sreg0);
                    if constexpr (isMix) {
                        if constexpr (IS_BIAS_BROADCAST) {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddQ, pregMain,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignB16,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignB16,
                                (binaryAddRemainderLoop - 1) * vlFp32);
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignB16 + binaryAddOffset,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignB16 + binaryAddOffset,
                                (binaryAddRemainderLoop - 1) * vlFp32 + binaryAddOffset);
                        } else {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddQ, pregMain,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignB16,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignB16,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignB16);
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignB16 + binaryAddOffset,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignB16 + binaryAddOffset,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignB16 + binaryAddOffset);
                        }
                        StoreRegToOutput(
                            xOutAddr, binaryAddQ, pregMain,
                            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignBias);
                        StoreRegToOutput(
                            xOutAddr, binaryAddR, pregLoop,
                            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlignBias + binaryAddOffset);
                        DataCopy(
                            (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                                k * colsPerLoopAlignB32,
                            binaryAddQ, pregMain);
                        DataCopy(
                            (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                                k * colsPerLoopAlignB32 + binaryAddOffset,
                            binaryAddR, pregLoop);
                    } else {
                        if constexpr (IS_BIAS_BROADCAST) {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddQ, pregMain,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                                (binaryAddRemainderLoop - 1) * vlFp32);
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                                (binaryAddRemainderLoop - 1) * vlFp32 + binaryAddOffset);
                        } else {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddQ, pregMain,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign);
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                                (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                        }
                        StoreRegToOutput(
                            xOutAddr, binaryAddQ, pregMain,
                            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign);
                        StoreRegToOutput(
                            xOutAddr, binaryAddR, pregLoop,
                            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                        DataCopy(
                            (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                                k * colsPerLoopAlign,
                            binaryAddQ, pregMain);
                        DataCopy(
                            (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                                k * colsPerLoopAlign + binaryAddOffset,
                            binaryAddR, pregLoop);
                    }
                    Muls(binaryAddQ, binaryAddQ, n, pregMain);
                    Muls(binaryAddR, binaryAddR, n, pregLoop);
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                    ReduceSum(vlMean, binaryAddQ, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddAddr + binaryAddRemainderLoop - 1), vlMean, pregMerge);
                }
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                    if constexpr (isMix) {
                        if constexpr (IS_BIAS_BROADCAST) {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, x, pregMain,
                                (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlignB16,
                                (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlignB16,
                                (i + binaryAddRemainderLoop) * vlFp32);
                        } else {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, x, pregMain,
                                (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlignB16,
                                (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlignB16,
                                (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlignB16);
                        }
                        StoreRegToOutput(
                            xOutAddr, x, pregMain, (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlignBias);
                        DataCopy(
                            (__local_mem__ float*)x32Addr + (i + binaryAddRemainderLoop) * vlFp32 +
                                k * colsPerLoopAlignB32,
                            x, pregMain);
                    } else {
                        if constexpr (IS_BIAS_BROADCAST) {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, x, pregMain,
                                (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                                (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                                (i + binaryAddRemainderLoop) * vlFp32);
                        } else {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, x, pregMain,
                                (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                                (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                                (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign);
                        }
                        StoreRegToOutput(
                            xOutAddr, x, pregMain, (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign);
                        DataCopy(
                            (__local_mem__ float*)x32Addr + (i + binaryAddRemainderLoop) * vlFp32 +
                                k * colsPerLoopAlign,
                            x, pregMain);
                    }
                    Muls(x, x, n, pregMain);
                    ReduceSum(vlMean, x, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddAddr + binaryAddRemainderLoop + i), vlMean, pregMerge);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                uint16_t curBinaryAddLoopMean = binaryAddLoopMean;
                for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                    curBinaryAddLoopMean = curBinaryAddLoopMean / 2;
                    for (uint16_t j = 0; j < curBinaryAddLoopMean; j++) {
                        DataCopy(binaryAddQ, ((__local_mem__ float*)binaryAddAddr + j * vlFp32));
                        DataCopy(
                            binaryAddR, ((__local_mem__ float*)binaryAddAddr + (j + curBinaryAddLoopMean) * vlFp32));
                        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                        DataCopy(((__local_mem__ float*)binaryAddAddr + j * vlFp32), binaryAddQ, pregMain);
                    }
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                }
                {
                    uint32_t sreg2 = binaryAddLastNum;
                    pregLoop = UpdateMask<float>(sreg2);
                    DataCopy(meanTemp, ((__local_mem__ float*)binaryAddAddr));
                    ReduceSum(mean, meanTemp, pregLoop);
                    Muls(mean, mean, nCorrectionFactor, pregMerge);
                }

                // batch mean
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)meanAddr + k), mean, pregMerge);
                Duplicate(mean, mean, pregMain);
                LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

                uint32_t sreg1 = binaryAddRemainder;
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
                    pregLoop = UpdateMask<float>(sreg1);
                    if constexpr (isMix) {
                        DataCopy(binaryAddQ, (__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlignB32);
                        DataCopy(
                            binaryAddR,
                            (__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlignB32 + binaryAddOffset);
                    } else {
                        DataCopy(binaryAddQ, (__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign);
                        DataCopy(
                            binaryAddR,
                            (__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                    }
                    Sub(binaryAddQ, binaryAddQ, mean, pregLoop);
                    Sub(binaryAddR, binaryAddR, mean, pregLoop);
                    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregLoop);
                    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);
                    Muls(binaryAddQPow, binaryAddQPow, n, pregLoop);
                    Muls(binaryAddRPow, binaryAddRPow, n, pregLoop);
                    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregLoop);
                    ReduceSum(vlVar, binaryAddQPow, pregLoop);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddAddr + i), vlVar, pregMerge);
                }
                {
                    pregLoop = UpdateMask<float>(sreg1);
                    if constexpr (isMix) {
                        DataCopy(
                            binaryAddQ, (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                                            k * colsPerLoopAlignB32);
                        DataCopy(
                            binaryAddR, (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                                            k * colsPerLoopAlignB32 + binaryAddOffset);
                    } else {
                        DataCopy(
                            binaryAddQ, (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                                            k * colsPerLoopAlign);
                        DataCopy(
                            binaryAddR, (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                                            k * colsPerLoopAlign + binaryAddOffset);
                    }
                    Sub(binaryAddQ, binaryAddQ, mean, pregMain);
                    Sub(binaryAddR, binaryAddR, mean, pregLoop);
                    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregMain);
                    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);
                    Muls(binaryAddQPow, binaryAddQPow, n, pregMain);
                    Muls(binaryAddRPow, binaryAddRPow, n, pregLoop);
                    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregMain);
                    ReduceSum(vlVar, binaryAddQPow, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddAddr + binaryAddRemainderLoop - 1), vlVar, pregMerge);
                }
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                    if constexpr (isMix) {
                        DataCopy(
                            x1, (__local_mem__ float*)x32Addr + (i + binaryAddRemainderLoop) * vlFp32 +
                                    k * colsPerLoopAlignB32);
                    } else {
                        DataCopy(
                            x1, (__local_mem__ float*)x32Addr + (i + binaryAddRemainderLoop) * vlFp32 +
                                    k * colsPerLoopAlign);
                    }
                    Sub(y1, x1, mean, pregMain);
                    Mul(y1Pow, y1, y1, pregMain);
                    Muls(y1Pow, y1Pow, n, pregMain);
                    ReduceSum(vlVar, y1Pow, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddAddr + binaryAddRemainderLoop + i), vlVar, pregMerge);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                uint16_t curBinaryAddLoopVar = binaryAddLoopVar;
                for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                    curBinaryAddLoopVar = curBinaryAddLoopVar / 2;
                    for (uint16_t j = 0; j < curBinaryAddLoopVar; j++) {
                        DataCopy(binaryAddQ, ((__local_mem__ float*)binaryAddAddr + j * vlFp32));
                        DataCopy(
                            binaryAddR, ((__local_mem__ float*)binaryAddAddr + (j + curBinaryAddLoopVar) * vlFp32));
                        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                        DataCopy(((__local_mem__ float*)binaryAddAddr + j * vlFp32), binaryAddQ, pregMain);
                    }
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                }
                {
                    uint32_t sreg2 = binaryAddLastNum;
                    pregLoop = UpdateMask<float>(sreg2);
                    DataCopy(varTemp, ((__local_mem__ float*)binaryAddAddr));
                    ReduceSum(var, varTemp, pregLoop);
                    Muls(var, var, nCorrectionFactor, pregMerge);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)varAddr + k), var, pregMerge);
                }
                LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
            }
        }
    }

    __aicore__ inline void VFCalcRstd(__local_mem__ float* varAddr, uint32_t rowsCount)
    {
        uint32_t vlFp32 = vlFp32_;
        uint16_t rowsLoopCount = CEIL_DIV(rowsCount, vlFp32);
        float eps = eps_;

        __VEC_SCOPE__
        {
            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> e;
            RegTensor<float> scalar1;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> var;
            RegTensor<float> rstd;

            RegTensor<float> one;
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            Duplicate(one, (float)1.0, pregMain);

            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregLoop;
            uint32_t sreg0 = rowsCount;
            for (uint16_t j = 0; j < rowsLoopCount; j++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(var, ((__local_mem__ float*)varAddr + j * vlFp32));
                Duplicate(scalar1, float(SCALAR3), pregLoop);
                Duplicate(scalarInf, POS_INF, pregLoop);
                Duplicate(scalarZero, float(0.0), pregLoop);
                Duplicate(t1, float(SCALAR2), pregLoop);
                Duplicate(s, float(1.0), pregLoop);

                // rstd
                Adds(var, var, eps, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, float(SCALAR1), pregLoop);
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
                DataCopy(((__local_mem__ float*)varAddr + j * vlFp32), rstd, pregLoop);
            }
        }
    }

    __aicore__ inline void VFCalcY(
        __local_mem__ float* x32Addr, __local_mem__ BETA_TYPE* betaAddr, __local_mem__ GAMMA_TYPE* gammaAddr,
        __local_mem__ float* meanAddr, __local_mem__ float* rstdAddr, __local_mem__ BIAS_TYPE* yOutAddr,
        uint32_t rowsCount, uint32_t colsCount)
    {
        uint32_t vlFp32 = vlFp32_;
        uint16_t colsLoopCount = CEIL_DIV(colsCount, vlFp32);
        uint32_t colsPerLoopAlign = colsPerLoopAlign_;
        uint32_t colsPerLoopAlignB32 = colsPerLoopAlignB32_;
        uint32_t colsPerLoopAlignBias = colsPerLoopAlignBias_;

        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> beta;
            RegTensor<float> gamma;
            MaskReg pregLoop;

            for (uint16_t k = 0; k < static_cast<uint16_t>(rowsCount); k++) {
                uint32_t sreg0 = colsCount;
                for (uint16_t i = 0; i < colsLoopCount; i++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadGammaBeta(gammaAddr, betaAddr, gamma, beta, pregLoop, i * vlFp32);
                    if constexpr (isMix) {
                        DataCopy(x, ((__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlignB32));
                    } else {
                        DataCopy(x, ((__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign));
                    }
                    DataCopy<float, LoadDist::DIST_BRC_B32>(mean, ((__local_mem__ float*)meanAddr + k));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(rstd, ((__local_mem__ float*)rstdAddr + k));
                    Sub(x, x, mean, pregLoop);
                    Mul(y, x, rstd, pregLoop);
                    Mul(y, y, gamma, pregLoop);
                    Add(y, y, beta, pregLoop);
                    if constexpr (isMix) {
                        StoreRegToOutput(yOutAddr, y, pregLoop, i * vlFp32 + k * colsPerLoopAlignBias);
                    } else {
                        StoreRegToOutput(yOutAddr, y, pregLoop, i * vlFp32 + k * colsPerLoopAlign);
                    }
                }
            }
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= tiling_->usedCoreNum) {
            return;
        }

        int64_t inputOffset = 0;
        int64_t outputOffset = 0;
        int64_t xOffset = 0;
        int64_t meanOffset = 0;

        LocalTensor<GAMMA_TYPE> gammaLocal = gammaQueue_.template AllocTensor<GAMMA_TYPE>();
        LocalTensor<BETA_TYPE> betaLocal = betaQueue_.template AllocTensor<BETA_TYPE>();
        LocalTensor<BIAS_TYPE> biasLocal;
        if constexpr (IS_BIAS_BROADCAST) {
            biasLocal = biasQueue_.template AllocTensor<BIAS_TYPE>();
            CopyBiasToUB(biasLocal, colsPerLoop_);
        }

        for (int64_t i = 0; i < rowsLoopCount_; i++) {
            int32_t rowsCount = i < rowsLoopCount_ - 1 ? rowsPerLoop_ : rowsTail_;

            LocalTensor<X1_TYPE> x1Local = x1Queue_.template AllocTensor<X1_TYPE>();
            LocalTensor<X2_TYPE> x2Local = x2Queue_.template AllocTensor<X2_TYPE>();
            if constexpr (IS_BIAS_ELEWISE) {
                biasLocal = biasQueue_.template AllocTensor<BIAS_TYPE>();
            }
            // copy in x1, x2, bias
            CopyInputsToUB(x1Local, x2Local, biasLocal, inputOffset, colsPerLoop_, rowsCount);

            x1Local = x1Queue_.template DeQue<X1_TYPE>();
            x2Local = x2Queue_.template DeQue<X2_TYPE>();
            __local_mem__ X1_TYPE* x1Addr = (__local_mem__ X1_TYPE*)x1Local[0].GetPhyAddr();
            __local_mem__ X2_TYPE* x2Addr = (__local_mem__ X2_TYPE*)x2Local[0].GetPhyAddr();

            __local_mem__ BIAS_TYPE* biasAddr;
            if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                biasLocal = biasQueue_.template DeQue<BIAS_TYPE>();
                biasAddr = (__local_mem__ BIAS_TYPE*)biasLocal[0].GetPhyAddr();
            }

            LocalTensor<BIAS_TYPE> xOutLocal = xQueue_.template AllocTensor<BIAS_TYPE>();
            LocalTensor<float> x32Local = x32Queue_.Get<float>();
            LocalTensor<float> meanLocal = meanQueue_.template AllocTensor<float>();
            LocalTensor<float> rstdLocal = rstdQueue_.template AllocTensor<float>();

            __local_mem__ BIAS_TYPE* xOutAddr = (__local_mem__ BIAS_TYPE*)xOutLocal[0].GetPhyAddr();
            __local_mem__ float* x32Addr = (__local_mem__ float*)x32Local[0].GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal[0].GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal[0].GetPhyAddr();

            if (colsPerLoop_ <= vlFp32_) {
                VFCalcMeanVarFast(x1Addr, x2Addr, biasAddr, xOutAddr, x32Addr, meanAddr, rstdAddr, rowsCount);
            } else {
                VFCalcMeanVar(x1Addr, x2Addr, biasAddr, xOutAddr, x32Addr, meanAddr, rstdAddr, rowsCount);
            }

            x1Queue_.FreeTensor(x1Local);
            x2Queue_.FreeTensor(x2Local);
            if constexpr (IS_BIAS_ELEWISE) {
                biasQueue_.FreeTensor(biasLocal);
            }
            // copy out x
            if (tiling_->outputX) {
                xQueue_.EnQue(xOutLocal);
                xOutLocal = xQueue_.template DeQue<BIAS_TYPE>();
                CopyXToGm(xOutLocal, inputOffset, colsPerLoop_, rowsCount);
            }
            xQueue_.FreeTensor(xOutLocal);

            // calc rstd
            VFCalcRstd(rstdAddr, rowsCount);

            // copy out mean
            meanQueue_.EnQue(meanLocal);
            meanLocal = meanQueue_.template DeQue<float>();
            CopyMeanToGm(meanLocal, meanOffset, rowsCount);

            // copy out rstd
            rstdQueue_.EnQue(rstdLocal);
            rstdLocal = rstdQueue_.template DeQue<float>();
            CopyRstdToGm(rstdLocal, meanOffset, rowsCount);

            // copy in gamma, beta
            if (i == 0) {
                CopyGammaAndBetaToUB(gammaLocal, betaLocal, 0, colsPerLoop_);
                gammaLocal = gammaQueue_.template DeQue<GAMMA_TYPE>();
                betaLocal = betaQueue_.template DeQue<BETA_TYPE>();
            }
            LocalTensor<BIAS_TYPE> yLocal = yQueue_.template AllocTensor<BIAS_TYPE>();

            // calc y with VF
            x32Addr = (__local_mem__ float*)x32Local[0].GetPhyAddr();
            meanAddr = (__local_mem__ float*)meanLocal[0].GetPhyAddr();
            rstdAddr = (__local_mem__ float*)rstdLocal[0].GetPhyAddr();
            __local_mem__ BETA_TYPE* betaAddr = (__local_mem__ BETA_TYPE*)betaLocal[0].GetPhyAddr();
            __local_mem__ GAMMA_TYPE* gammaAddr = (__local_mem__ GAMMA_TYPE*)gammaLocal[0].GetPhyAddr();
            __local_mem__ BIAS_TYPE* yOutAddr = (__local_mem__ BIAS_TYPE*)yLocal[0].GetPhyAddr();

            VFCalcY(x32Addr, betaAddr, gammaAddr, meanAddr, rstdAddr, yOutAddr, rowsCount, colsPerLoop_);

            // copy out y
            yQueue_.EnQue(yLocal);
            yLocal = yQueue_.template DeQue<BIAS_TYPE>();
            CopyYToGm(yLocal, outputOffset, colsPerLoop_, rowsCount);

            inputOffset += rowsCount * colsPerLoop_;
            outputOffset = inputOffset;
            meanOffset += rowsCount;

            meanQueue_.FreeTensor(meanLocal);
            rstdQueue_.FreeTensor(rstdLocal);
            yQueue_.FreeTensor(yLocal);
        }
        if constexpr (IS_BIAS_BROADCAST) {
            biasQueue_.FreeTensor(biasLocal);
        }
        gammaQueue_.FreeTensor(gammaLocal);
        betaQueue_.FreeTensor(betaLocal);
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasQueue_;
    TQue<QuePosition::VECIN, 1> gammaQueue_;
    TQue<QuePosition::VECIN, 1> betaQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> meanQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstdQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xQueue_;
    TBuf<TPosition::VECCALC> binaryAddBuf_;
    TBuf<QuePosition::VECCALC> x32Queue_;

    GlobalTensor<X1_TYPE> x1Gm_;
    GlobalTensor<X2_TYPE> x2Gm_;
    GlobalTensor<BIAS_TYPE> biasGm_;
    GlobalTensor<GAMMA_TYPE> gammaGm_;
    GlobalTensor<BETA_TYPE> betaGm_;
    GlobalTensor<BIAS_TYPE> yGm_;
    GlobalTensor<BIAS_TYPE> xGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;

    uint32_t blockSize_;
    uint32_t vlFp32_;
    uint32_t tailCoreStartIndex_;
    int64_t colsPerLoop_;
    int64_t colsPerLoopAlign_;
    int64_t colsPerLoopAlignB16_;
    int64_t colsPerLoopAlignB32_;
    int64_t colsPerLoopAlignBias_;
    int64_t rowsPerCore_;
    int64_t rowsPerLoop_;
    int64_t rowsTail_;
    int64_t rowsLoopCount_;
    int64_t binaryAddNum_;
    int64_t binaryAddK_;
    int64_t binaryAddLastNum_;
    int64_t powerOfTwo_;
    float eps_;

    TPipe pipe_;
    const AddLayerNormRegbaseTilingData* tiling_;
};
} // namespace AddLayerNorm
#endif // ADD_LAYER_NORM_REGBASE_FULL_LOAD_H