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
#include "../../norm_common/reduce_common_regbase.h"

namespace AddLayerNorm {
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

template <typename X1_TYPE, typename X2_TYPE, typename GAMMA_TYPE, typename BETA_TYPE, typename BIAS_TYPE,
          int TILING_KEY, int BUFFER_NUM = DOUBLE_BUFFER_NUM>
class RegbaseFullLoad {
public:
    static constexpr bool isMix = !(IsSameType<X1_TYPE, X2_TYPE>::value && IsSameType<X1_TYPE, GAMMA_TYPE>::value &&
                                    IsSameType<X1_TYPE, BETA_TYPE>::value && IsSameType<X1_TYPE, BIAS_TYPE>::value);

    __aicore__ inline RegbaseFullLoad(const AddLayerNormRegbaseTilingData* tilingData) : tiling_(tilingData) {}

    __aicore__ inline void Init(__gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* gamma, __gm__ uint8_t* beta,
                                __gm__ uint8_t* bias, __gm__ uint8_t* y, __gm__ uint8_t* mean, __gm__ uint8_t* rstd,
                                __gm__ uint8_t* x)
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
            powerOfTwo_ *= NUM_TWO;
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

        int64_t binaryAddBufSize = 0;
        if (colsPerLoop_ > NUM_TWO * static_cast<int64_t>(vlFp32_)) {
            int64_t perRowScratch = BLOCK_ALIGN((binaryAddNum_ / vlFp32_) * sizeof(float), blockSize_);
            binaryAddBufSize = perRowScratch * rowsPerLoop_;
        }
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

    __aicore__ inline void CopyInputsToUB(LocalTensor<X1_TYPE> x1Local, LocalTensor<X2_TYPE> x2Local,
                                          LocalTensor<BIAS_TYPE> biasLocal, int64_t inputOffset, int32_t copyLen,
                                          int32_t rowsCount)
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

    __aicore__ inline void CopyYToGm(LocalTensor<BIAS_TYPE> yLocal, int64_t yOffset, int32_t copyLen, int32_t rowsCount)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = rowsCount;
        dataCopyParams.blockLen = copyLen * sizeof(BIAS_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(yGm_[yOffset], yLocal, dataCopyParams);
    }

    __aicore__ inline void VFAddFrontend(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X2_TYPE* x2Addr,
                                         __ubuf__ BIAS_TYPE* biasAddr, __ubuf__ BIAS_TYPE* xOutAddr,
                                         __ubuf__ float* x32Addr, uint16_t rowsCount)
    {
        uint32_t vlFp32 = vlFp32_;
        uint32_t colsPerLoop = colsPerLoop_;
        uint16_t colsLoopCount = CEIL_DIV(colsPerLoop, vlFp32);
        uint32_t colsPerLoopAlign = colsPerLoopAlign_;
        uint32_t colsPerLoopAlignB16 = colsPerLoopAlignB16_;
        uint32_t colsPerLoopAlignB32 = colsPerLoopAlignB32_;
        uint32_t colsPerLoopAlignBias = colsPerLoopAlignBias_;

        __VEC_SCOPE__
        {
            RegTensor<float> x;
            MaskReg pregLoop;
            for (uint16_t k = 0; k < rowsCount; k++) {
                uint32_t sreg0 = colsPerLoop;
                for (uint16_t i = 0; i < colsLoopCount; i++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    if constexpr (isMix) {
                        if constexpr (IS_BIAS_BROADCAST) {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, x, pregLoop, i * vlFp32 + k * colsPerLoopAlignB16,
                                i * vlFp32 + k * colsPerLoopAlignB16, i * vlFp32);
                        } else {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, x, pregLoop, i * vlFp32 + k * colsPerLoopAlignB16,
                                i * vlFp32 + k * colsPerLoopAlignB16, i * vlFp32 + k * colsPerLoopAlignB16);
                        }
                        StoreRegToOutput(xOutAddr, x, pregLoop, i * vlFp32 + k * colsPerLoopAlignBias);
                        StoreAlign((__ubuf__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlignB32, x, pregLoop);
                    } else {
                        if constexpr (IS_BIAS_BROADCAST) {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, x, pregLoop, i * vlFp32 + k * colsPerLoopAlign,
                                i * vlFp32 + k * colsPerLoopAlign, i * vlFp32);
                        } else {
                            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                                x1Addr, x2Addr, biasAddr, x, pregLoop, i * vlFp32 + k * colsPerLoopAlign,
                                i * vlFp32 + k * colsPerLoopAlign, i * vlFp32 + k * colsPerLoopAlign);
                        }
                        StoreRegToOutput(xOutAddr, x, pregLoop, i * vlFp32 + k * colsPerLoopAlign);
                        StoreAlign((__ubuf__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign, x, pregLoop);
                    }
                }
            }
        }
    }

    __aicore__ inline void CalculateMeanVarRLessThanVL(__ubuf__ float* xInUb, __ubuf__ float* meanInUb,
                                                       __ubuf__ float* rstdInUb, __ubuf__ float* xSubMeanUb,
                                                       uint16_t currentANum)
    {
        uint32_t reduceNum = static_cast<uint32_t>(colsPerLoop_);
        float n = static_cast<float>(1.0) / static_cast<float>(powerOfTwo_);
        float nCorrectionFactor = static_cast<float>(powerOfTwo_) / static_cast<float>(reduceNum);
        uint32_t aStride = isMix ? static_cast<uint32_t>(colsPerLoopAlignB32_) :
                                   static_cast<uint32_t>(colsPerLoopAlign_);
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> meanSum;
            RegTensor<float> mean;
            RegTensor<float> meanBrc;
            RegTensor<float> xMeanSub;
            RegTensor<float> square;
            RegTensor<float> varSum;
            RegTensor<float> var;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            uint32_t sreg0 = reduceNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t aIdx = 0; aIdx < currentANum; aIdx++) {
                LoadRegForDtype(xInUb, x, pregLoop, (aIdx * aStride));
                Muls(meanSum, x, n, pregLoop);
                Reduce<ReduceType::SUM>(mean, meanSum, pregLoop);
                Muls(mean, mean, nCorrectionFactor, pregOne);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanInUb + aIdx, mean, pregOne);

                Duplicate(meanBrc, mean, pregFull);
                Sub(xMeanSub, x, meanBrc, pregLoop);
                Mul(square, xMeanSub, xMeanSub, pregLoop);
                Muls(varSum, square, n, pregLoop);
                Reduce<ReduceType::SUM>(var, varSum, pregLoop);
                Muls(var, var, nCorrectionFactor, pregOne);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdInUb + aIdx, var, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateMeanVarRLessThanTwoVL(__ubuf__ float* xInUb, __ubuf__ float* meanInUb,
                                                          __ubuf__ float* rstdInUb, __ubuf__ float* xSubMeanUb,
                                                          uint16_t currentANum)
    {
        uint32_t VL_B32 = vlFp32_;
        uint32_t reduceNum = static_cast<uint32_t>(colsPerLoop_);
        float n = static_cast<float>(1.0) / static_cast<float>(powerOfTwo_);
        float nCorrectionFactor = static_cast<float>(powerOfTwo_) / static_cast<float>(reduceNum);
        uint32_t aStride = isMix ? static_cast<uint32_t>(colsPerLoopAlignB32_) :
                                   static_cast<uint32_t>(colsPerLoopAlign_);
        uint32_t aTail = reduceNum - VL_B32;

        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> meanSum1;
            RegTensor<float> meanSum2;
            RegTensor<float> meanSum;
            RegTensor<float> mean;
            RegTensor<float> meanBrc;
            RegTensor<float> xMeanSub1;
            RegTensor<float> xSubMeanHi;
            RegTensor<float> square1;
            RegTensor<float> square2;
            RegTensor<float> varSum1;
            RegTensor<float> varSum2;
            RegTensor<float> varSum;
            RegTensor<float> var;

            MaskReg pregTail = UpdateMask<float>(aTail);
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            for (uint16_t aIdx = 0; aIdx < currentANum; aIdx++) {
                LoadRegForDtype(xInUb, x1, pregFull, (aIdx * aStride));
                LoadRegForDtype(xInUb + VL_B32, x2, pregTail, (aIdx * aStride));
                Muls(meanSum1, x1, n, pregFull);
                Muls(meanSum2, x2, n, pregTail);
                Add(meanSum, meanSum1, meanSum2, pregFull);
                Reduce<ReduceType::SUM>(mean, meanSum, pregFull);
                Muls(mean, mean, nCorrectionFactor, pregOne);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanInUb + aIdx, mean, pregOne);

                Duplicate(meanBrc, mean, pregFull);
                Sub(xMeanSub1, x1, meanBrc, pregFull);
                Sub(xSubMeanHi, x2, meanBrc, pregTail);
                Mul(square1, xMeanSub1, xMeanSub1, pregFull);
                Mul(square2, xSubMeanHi, xSubMeanHi, pregTail);
                Muls(varSum1, square1, n, pregFull);
                Muls(varSum2, square2, n, pregTail);
                Add(varSum, varSum1, varSum2, pregFull);
                Reduce<ReduceType::SUM>(var, varSum, pregFull);
                Muls(var, var, nCorrectionFactor, pregOne);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdInUb + aIdx, var, pregOne);
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateMeanVarRCommon(__ubuf__ float* xInUb, __ubuf__ float* meanInUb,
                                                   __ubuf__ float* rstdInUb, __ubuf__ float* xSubMeanUb,
                                                   uint16_t currentANum)
    {
        __ubuf__ float* tmpUb = (__ubuf__ float*)binaryAddBuf_.Get<float>().GetPhyAddr();
        uint32_t VL_B32 = vlFp32_;
        uint32_t BLK_B32 = static_cast<uint32_t>(blockSize_) / static_cast<uint32_t>(sizeof(float));
        uint32_t reduceNum = static_cast<uint32_t>(colsPerLoop_);
        float n = static_cast<float>(1.0) / static_cast<float>(powerOfTwo_);
        float nCorrectionFactor = static_cast<float>(powerOfTwo_) / static_cast<float>(reduceNum);
        uint32_t aStride = isMix ? static_cast<uint32_t>(colsPerLoopAlignB32_) :
                                   static_cast<uint32_t>(colsPerLoopAlign_);

        uint32_t binaryAddQuotient = powerOfTwo_ >= colsPerLoop_ ? powerOfTwo_ / NUM_TWO : powerOfTwo_;
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + VL_B32 - 1) / VL_B32;

        uint32_t lastBinaryAddNum = binaryAddQuotient / VL_B32;
        uint32_t lastBinaryAddNumTmp = lastBinaryAddNum;
        uint32_t lastBinaryAddNumAlign = (binaryAddQuotient / VL_B32 + BLK_B32 - 1) / BLK_B32 * BLK_B32;

        uint32_t binaryAddTailNum = reduceNum - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddTailNum + VL_B32 - 1) / VL_B32;
        uint16_t binaryAddRemainderFloorLoop = binaryAddTailNum / VL_B32;

        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> meanSum;
            RegTensor<float> mean;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t aIdx = 0; aIdx < currentANum; aIdx++) {
                uint32_t sregRemainder = binaryAddTailNum;
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; r++) {
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadRegForDtype(xInUb, x1, pregFull, (r * VL_B32 + aIdx * aStride));
                    LoadRegForDtype(xInUb + binaryAddQuotient, x2, pregFull, (r * VL_B32 + aIdx * aStride));
                    Muls(x1, x1, n, pregFull);
                    Muls(x2, x2, n, pregFull);
                    Add(meanSum, x1, x2, pregFull);
                    Reduce<ReduceType::SUM>(mean, meanSum, pregFull);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign + r), mean, pregOne);
                }
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadRegForDtype(xInUb + binaryAddRemainderFloorLoop * VL_B32, x1, pregFull,
                                    (r * VL_B32 + aIdx * aStride));
                    LoadRegForDtype(xInUb + binaryAddRemainderFloorLoop * VL_B32 + binaryAddQuotient, x2, pregLoop,
                                    (r * VL_B32 + aIdx * aStride));
                    Muls(x1, x1, n, pregFull);
                    Muls(x2, x2, n, pregLoop);
                    Add(meanSum, x1, x2, pregFull);
                    Reduce<ReduceType::SUM>(mean, meanSum, pregFull);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), mean,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    LoadRegForDtype(xInUb + binaryAddRemainderCeilLoop * VL_B32, x1, pregFull,
                                    (r * VL_B32 + aIdx * aStride));
                    Muls(x1, x1, n, pregFull);
                    Reduce<ReduceType::SUM>(mean, x1, pregFull);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r),
                        mean, pregOne);
                }
            }

            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t aIdx = 0; aIdx < currentANum; aIdx++) {
                    LoadAlign(x1, tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign));
                    Reduce<ReduceType::SUM>(mean, x1, pregLast);
                    Muls(mean, mean, nCorrectionFactor, pregOne);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanInUb + aIdx, mean, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                for (uint16_t aIdx = 0; aIdx < currentANum; aIdx++) {
                    uint32_t lastTailNum = lastBinaryAddNum - VL_B32;
                    MaskReg pregLast = UpdateMask<float>(lastTailNum);
                    RegTensor<float> shlReg;
                    LoadAlign(x1, tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign));
                    LoadAlign(x2, tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign + VL_B32));
                    ShiftLefts((RegTensor<uint32_t>&)shlReg, (RegTensor<uint32_t>&)x2, static_cast<int16_t>(0),
                               pregLast);
                    Add(x1, x1, shlReg, pregFull);
                    Reduce<ReduceType::SUM>(mean, x1, pregFull);
                    Muls(mean, mean, nCorrectionFactor, pregOne);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanInUb + aIdx, mean, pregOne);
                }
            }
        }
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> mean;
            RegTensor<float> xMeanSub1;
            RegTensor<float> xSubMeanHi;
            RegTensor<float> square1;
            RegTensor<float> square2;
            RegTensor<float> varSum;
            RegTensor<float> var;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t aIdx = 0; aIdx < currentANum; aIdx++) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(mean, meanInUb + aIdx);
                uint32_t sregRemainder = binaryAddTailNum;
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; r++) {
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadRegForDtype(xInUb, x1, pregFull, (r * VL_B32 + aIdx * aStride));
                    LoadRegForDtype(xInUb + binaryAddQuotient, x2, pregFull, (r * VL_B32 + aIdx * aStride));
                    Sub(xMeanSub1, x1, mean, pregFull);
                    Sub(xSubMeanHi, x2, mean, pregFull);
                    Mul(square1, xMeanSub1, xMeanSub1, pregFull);
                    Mul(square2, xSubMeanHi, xSubMeanHi, pregFull);
                    Muls(square1, square1, n, pregFull);
                    Muls(square2, square2, n, pregFull);
                    Add(varSum, square1, square2, pregFull);
                    Reduce<ReduceType::SUM>(var, varSum, pregFull);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign + r), var, pregOne);
                }
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadRegForDtype(xInUb + binaryAddRemainderFloorLoop * VL_B32, x1, pregFull,
                                    (r * VL_B32 + aIdx * aStride));
                    LoadRegForDtype(xInUb + binaryAddRemainderFloorLoop * VL_B32 + binaryAddQuotient, x2, pregLoop,
                                    (r * VL_B32 + aIdx * aStride));
                    Sub(xMeanSub1, x1, mean, pregFull);
                    Sub(xSubMeanHi, x2, mean, pregLoop);
                    Mul(square1, xMeanSub1, xMeanSub1, pregFull);
                    Mul(square2, xSubMeanHi, xSubMeanHi, pregLoop);
                    Muls(square1, square1, n, pregFull);
                    Muls(square2, square2, n, pregLoop);
                    Add(varSum, square1, square2, pregFull);
                    Reduce<ReduceType::SUM>(var, varSum, pregFull);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), var,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    LoadRegForDtype(xInUb + binaryAddRemainderCeilLoop * VL_B32, x1, pregFull,
                                    (r * VL_B32 + aIdx * aStride));
                    Sub(xMeanSub1, x1, mean, pregFull);
                    Mul(square1, xMeanSub1, xMeanSub1, pregFull);
                    Muls(square1, square1, n, pregFull);
                    Reduce<ReduceType::SUM>(var, square1, pregFull);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r),
                        var, pregOne);
                }
            }

            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNumTmp);
                for (uint16_t aIdx = 0; aIdx < currentANum; aIdx++) {
                    LoadAlign(x1, tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign));
                    Reduce<ReduceType::SUM>(var, x1, pregLast);
                    Muls(var, var, nCorrectionFactor, pregOne);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdInUb + aIdx, var, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                uint32_t lastTailNum = lastBinaryAddNum - VL_B32;
                MaskReg pregLast = UpdateMask<float>(lastTailNum);
                RegTensor<float> shlReg;
                for (uint16_t aIdx = 0; aIdx < currentANum; aIdx++) {
                    LoadAlign(x1, tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign));
                    LoadAlign(x2, tmpUb + static_cast<uint32_t>(aIdx * lastBinaryAddNumAlign + VL_B32));
                    ShiftLefts((RegTensor<uint32_t>&)shlReg, (RegTensor<uint32_t>&)x2, static_cast<int16_t>(0),
                               pregLast);
                    Add(x1, x1, shlReg, pregFull);
                    Reduce<ReduceType::SUM>(var, x1, pregFull);
                    Muls(var, var, nCorrectionFactor, pregOne);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdInUb + aIdx, var, pregOne);
                }
            }
        }
    }

    __aicore__ inline void VFCalcY(__ubuf__ float* x32Addr, __ubuf__ BETA_TYPE* betaAddr,
                                   __ubuf__ GAMMA_TYPE* gammaAddr, __ubuf__ float* meanAddr, __ubuf__ float* rstdAddr,
                                   __ubuf__ BIAS_TYPE* yOutAddr, uint32_t rowsCount, uint32_t colsCount)
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
                        LoadAlign(x, ((__ubuf__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlignB32));
                    } else {
                        LoadAlign(x, ((__ubuf__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign));
                    }
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(mean, ((__ubuf__ float*)meanAddr + k));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd, ((__ubuf__ float*)rstdAddr + k));
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
            __ubuf__ X1_TYPE* x1Addr = (__ubuf__ X1_TYPE*)x1Local[0].GetPhyAddr();
            __ubuf__ X2_TYPE* x2Addr = (__ubuf__ X2_TYPE*)x2Local[0].GetPhyAddr();

            __ubuf__ BIAS_TYPE* biasAddr;
            if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                biasLocal = biasQueue_.template DeQue<BIAS_TYPE>();
                biasAddr = (__ubuf__ BIAS_TYPE*)biasLocal[0].GetPhyAddr();
            }

            LocalTensor<BIAS_TYPE> xOutLocal = xQueue_.template AllocTensor<BIAS_TYPE>();
            LocalTensor<float> x32Local = x32Queue_.Get<float>();
            LocalTensor<float> meanLocal = meanQueue_.template AllocTensor<float>();
            LocalTensor<float> rstdLocal = rstdQueue_.template AllocTensor<float>();

            __ubuf__ BIAS_TYPE* xOutAddr = (__ubuf__ BIAS_TYPE*)xOutLocal[0].GetPhyAddr();
            __ubuf__ float* x32Addr = (__ubuf__ float*)x32Local[0].GetPhyAddr();
            __ubuf__ float* meanAddr = (__ubuf__ float*)meanLocal[0].GetPhyAddr();
            __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal[0].GetPhyAddr();

            VFAddFrontend(x1Addr, x2Addr, biasAddr, xOutAddr, x32Addr, rowsCount);

            uint32_t rAlign = isMix ? static_cast<uint32_t>(colsPerLoopAlignB32_) :
                                      static_cast<uint32_t>(colsPerLoopAlign_);
            uint32_t vlB32 = vlFp32_;
            if (rAlign <= vlB32) {
                CalculateMeanVarRLessThanVL(x32Addr, meanAddr, rstdAddr, x32Addr, rowsCount);
            } else if (rAlign <= vlB32 + vlB32) {
                CalculateMeanVarRLessThanTwoVL(x32Addr, meanAddr, rstdAddr, x32Addr, rowsCount);
            } else if (rAlign <= vlB32 * vlB32 * NUM_TWO) {
                CalculateMeanVarRCommon<1>(x32Addr, meanAddr, rstdAddr, x32Addr, rowsCount);
            } else {
                CalculateMeanVarRCommon<NUM_TWO>(x32Addr, meanAddr, rstdAddr, x32Addr, rowsCount);
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
            NormCommon::ComputeRstdNewtonRaphson<false>(rstdAddr, rstdAddr, rowsCount, eps_, 1.0f, vlFp32_);

            // copy out mean
            meanQueue_.EnQue(meanLocal);
            meanLocal = meanQueue_.template DeQue<float>();
            CopyLocalToGm(meanGm_, meanLocal, meanOffset, rowsCount);

            // copy out rstd
            rstdQueue_.EnQue(rstdLocal);
            rstdLocal = rstdQueue_.template DeQue<float>();
            CopyLocalToGm(rstdGm_, rstdLocal, meanOffset, rowsCount);

            // copy in gamma, beta
            if (i == 0) {
                CopyGammaAndBetaToUBCommon<isMix>(gammaLocal, betaLocal, gammaGm_, betaGm_, gammaQueue_, betaQueue_, 0,
                                                  colsPerLoop_, blockSize_);
                gammaLocal = gammaQueue_.template DeQue<GAMMA_TYPE>();
                betaLocal = betaQueue_.template DeQue<BETA_TYPE>();
            }
            LocalTensor<BIAS_TYPE> yLocal = yQueue_.template AllocTensor<BIAS_TYPE>();

            x32Addr = (__ubuf__ float*)x32Local[0].GetPhyAddr();
            meanAddr = (__ubuf__ float*)meanLocal[0].GetPhyAddr();
            rstdAddr = (__ubuf__ float*)rstdLocal[0].GetPhyAddr();
            __ubuf__ BETA_TYPE* betaAddr = (__ubuf__ BETA_TYPE*)betaLocal[0].GetPhyAddr();
            __ubuf__ GAMMA_TYPE* gammaAddr = (__ubuf__ GAMMA_TYPE*)gammaLocal[0].GetPhyAddr();
            __ubuf__ BIAS_TYPE* yOutAddr = (__ubuf__ BIAS_TYPE*)yLocal[0].GetPhyAddr();

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
    static constexpr int BIAS_BUFFER_NUM = IS_BIAS_BROADCAST ? SINGLE_BUFFER_NUM : BUFFER_NUM;
    TQue<QuePosition::VECIN, BIAS_BUFFER_NUM> biasQueue_;
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
    float eps_;
    int64_t binaryAddNum_;
    int64_t binaryAddK_;
    int64_t binaryAddLastNum_;
    int64_t powerOfTwo_;

    TPipe pipe_;
    const AddLayerNormRegbaseTilingData* tiling_;
};
} // namespace AddLayerNorm
#endif // ADD_LAYER_NORM_REGBASE_FULL_LOAD_H
