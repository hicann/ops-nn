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
 * \file add_layer_norm_regbase_welford.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_REGBASE_WELFORD_H
#define ADD_LAYER_NORM_REGBASE_WELFORD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "add_layer_norm_regbase_common.h"

namespace AddLayerNorm {
template <
    typename X1_TYPE, typename X2_TYPE, typename GAMMA_TYPE, typename BETA_TYPE, typename BIAS_TYPE, int TILING_KEY,
    int BUFFER_NUM = 1>
class RegbaseWelford {
public:
    static constexpr bool isMix =
        !(IsSameType<X1_TYPE, X2_TYPE>::value && IsSameType<X1_TYPE, GAMMA_TYPE>::value &&
          IsSameType<X1_TYPE, BETA_TYPE>::value && IsSameType<X1_TYPE, BIAS_TYPE>::value);

    __aicore__ inline RegbaseWelford(const AddLayerNormRegbaseTilingData* tilingData) : tiling_(tilingData)
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
        cols_ = tiling_->cols;
        colsTail_ = tiling_->colsTail;
        colsPerLoop_ = tiling_->colsPerLoop;
        colsLoopCount_ = tiling_->colsLoopCount;
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
            gmOffset = (tiling_->rowsPerCore * cols_) * coreIdx;
            meanOffset = gmOffset / cols_;
        } else {
            // tail cores
            rowsPerCore_ = tiling_->rowsPerTailCore;
            gmOffset = tailCoreStartIndex_ * tiling_->rowsPerCore * cols_ +
                       (coreIdx - tailCoreStartIndex_) * tiling_->rowsPerTailCore * cols_;
            meanOffset = gmOffset / cols_;
        }

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

            pipe_.InitBuffer(x1Queue_, BUFFER_NUM, (colsPerLoopAlignB16_ * sizeof(float)));
            pipe_.InitBuffer(x2Queue_, BUFFER_NUM, (colsPerLoopAlignB16_ * sizeof(float)));
            if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                pipe_.InitBuffer(biasQueue_, BUFFER_NUM, (colsPerLoopAlignB16_ * sizeof(float)));
            }
            pipe_.InitBuffer(xQueue_, BUFFER_NUM, (colsPerLoopAlignB16_ * sizeof(float)));
            pipe_.InitBuffer(yQueue_, BUFFER_NUM, (colsPerLoopAlignB16_ * sizeof(float)));
            pipe_.InitBuffer(betaQueue_, BUFFER_NUM, (colsPerLoopAlignB16_ * sizeof(float)));
            pipe_.InitBuffer(gammaQueue_, BUFFER_NUM, (colsPerLoopAlignB16_ * sizeof(float)));
        } else {
            pipe_.InitBuffer(x1Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
            pipe_.InitBuffer(x2Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
            if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                pipe_.InitBuffer(biasQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
            }
            pipe_.InitBuffer(xQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
            pipe_.InitBuffer(yQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
            pipe_.InitBuffer(betaQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
            pipe_.InitBuffer(gammaQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        }
        pipe_.InitBuffer(meanQueue_, BUFFER_NUM, blockSize_);
        pipe_.InitBuffer(rstdQueue_, BUFFER_NUM, blockSize_);
        pipe_.InitBuffer(meanBuf_, colsPerLoopAlign_ * sizeof(float));
        pipe_.InitBuffer(varBuf_, colsPerLoopAlign_ * sizeof(float));

        int64_t binaryAddBufSize = BLOCK_ALIGN((binaryAddNum_ / vlFp32_) * sizeof(float), blockSize_);
        if (binaryAddBufSize > 0) {
            pipe_.InitBuffer(binaryAddBuf_, binaryAddBufSize);
        }
    }

    __aicore__ inline void CopyInputsToUB(
        LocalTensor<X1_TYPE> x1Local, LocalTensor<X2_TYPE> x2Local, LocalTensor<BIAS_TYPE> biasLocal,
        int64_t inputOffset, int64_t biasOffset, int32_t copyLen)
    {
        if constexpr (isMix) {
            {
                int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(X1_TYPE), blockSize_) / sizeof(X1_TYPE);
                DataCopyPadExtParams<X1_TYPE> padParams;
                padParams.isPad = true;
                padParams.paddingValue = static_cast<X1_TYPE>(0.0);
                padParams.rightPadding = copyLenAlign - copyLen;
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = copyLen * sizeof(X1_TYPE);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = 0;
                DataCopyPad(x1Local, x1Gm_[inputOffset], dataCopyParams, padParams);
                x1Queue_.EnQue(x1Local);
            }
            {
                int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(X2_TYPE), blockSize_) / sizeof(X2_TYPE);
                DataCopyPadExtParams<X2_TYPE> padParams;
                padParams.isPad = true;
                padParams.paddingValue = static_cast<X2_TYPE>(0.0);
                padParams.rightPadding = copyLenAlign - copyLen;
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = copyLen * sizeof(X2_TYPE);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = 0;
                DataCopyPad(x2Local, x2Gm_[inputOffset], dataCopyParams, padParams);
                x2Queue_.EnQue(x2Local);
            }
            {
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
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    DataCopyPad(biasLocal, biasGm_[biasOffset], dataCopyParams, padParams);
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
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = copyLen * sizeof(X1_TYPE);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(x1Local, x1Gm_[inputOffset], dataCopyParams, padParams);
            x1Queue_.EnQue(x1Local);
            DataCopyPad(x2Local, x2Gm_[inputOffset], dataCopyParams, padParams);
            x2Queue_.EnQue(x2Local);
            if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                DataCopyPad(biasLocal, biasGm_[biasOffset], dataCopyParams, padParams);
                biasQueue_.EnQue(biasLocal);
            }
        }
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

    __aicore__ inline void CopyXToGm(LocalTensor<BIAS_TYPE> xLocal, int64_t xOffset, int32_t copyLen)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(BIAS_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(xGm_[xOffset], xLocal, dataCopyParams);
    }

    __aicore__ inline void CopyYToGm(LocalTensor<BIAS_TYPE> yLocal, int64_t yOffset, int32_t copyLen)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
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

    __aicore__ inline void VFWelfordParallelUpdateWithInit(
        __local_mem__ X1_TYPE* x1Local, __local_mem__ X2_TYPE* x2Local, __local_mem__ BIAS_TYPE* biasLocal,
        __local_mem__ BIAS_TYPE* xOutLocal, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
        uint64_t calLen, uint16_t loopCount, float scale)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> tmpMean;
            RegTensor<float> tmpVar;
            RegTensor<float> delta1;
            RegTensor<float> delta2;
            RegTensor<float> delta3;
            RegTensor<float> delat4;
            MaskReg pregLoop;
            uint32_t sreg0 = calLen;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                    x1Local, x2Local, biasLocal, x1, pregLoop, i * VL_FP32, i * VL_FP32, i * VL_FP32);
                StoreRegToOutput(xOutLocal, x1, pregLoop, i * VL_FP32);
                Duplicate(tmpMean, 0.0, pregLoop);
                Sub(delta1, x1, tmpMean, pregLoop);
                Muls(delta2, delta1, scale, pregLoop);
                Add(tmpMean, tmpMean, delta2, pregLoop);
                DataCopy(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

                Duplicate(tmpVar, 0.0, pregLoop);
                Sub(delta3, x1, tmpMean, pregLoop);
                Mul(delat4, delta1, delta3, pregLoop);
                Add(tmpVar, tmpVar, delat4, pregLoop);
                DataCopy(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
            }
        }
    }

    /*
      Welford update 阶段计算公式如下:
      count += 1
      delta = new_value - mean
      mean += (delta / count)
      delta2 = new_value - mean
      var += delta * delta2
      return count, mean, var
    */
    __aicore__ inline void VFWelfordParallelUpdate(
        __local_mem__ X1_TYPE* x1Local, __local_mem__ X2_TYPE* x2Local, __local_mem__ BIAS_TYPE* biasLocal,
        __local_mem__ BIAS_TYPE* xOutLocal, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
        uint64_t calLen, uint16_t loopCount, float scale)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> tmpMean;
            RegTensor<float> tmpVar;
            RegTensor<float> delta1;
            RegTensor<float> delta2;
            RegTensor<float> delta3;
            RegTensor<float> delat4;
            MaskReg pregLoop;
            uint32_t sreg0 = calLen;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                    x1Local, x2Local, biasLocal, x1, pregLoop, i * VL_FP32, i * VL_FP32, i * VL_FP32);
                StoreRegToOutput(xOutLocal, x1, pregLoop, i * VL_FP32);
                DataCopy(tmpMean, tmpMeanLocal + i * VL_FP32);
                Sub(delta1, x1, tmpMean, pregLoop);
                Muls(delta2, delta1, scale, pregLoop);
                Add(tmpMean, tmpMean, delta2, pregLoop);
                DataCopy(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

                DataCopy(tmpVar, tmpVarLocal + i * VL_FP32);
                Sub(delta3, x1, tmpMean, pregLoop);
                Mul(delat4, delta1, delta3, pregLoop);
                Add(tmpVar, tmpVar, delat4, pregLoop);
                DataCopy(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
            }
        }
    }

    /*
      Welford Finalize对齐场景计算公式如下:
      finalize_mean = sum_fun(mean) / parallel_N
      finalize_delta = mean - finalize_mean
      finalize_delta_square = finalize_delta * finalize_delta
      M2_fixed = M2 + float(count) * finalize_delta_square
      finalize_std = sum_fun(M2_fixed) / float(parallel_N * count)

      welford采用二分累加计算mean和variance, 基本逻辑为:
      先将尾块折叠到整块上，整尾块vadd之后，做一次vcadd回刷到UB上，剩余整块直接vcadd回刷到UB上，最后对UB上的结果做完全二分对折
    */
    __aicore__ inline void VFWelfordParallelFinalizeAlign(
        __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
        __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
        uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
        float reduceScale, float scale, float cnt, float eps)
    {
        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
        uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
        uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
        uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> dichotomyAddMeanL;
            RegTensor<float> dichotomyAddMeanR;
            RegTensor<float> dichotomyAddVarL;
            RegTensor<float> dichotomyAddVarR;
            RegTensor<float> sumMean;
            RegTensor<float> mean;
            RegTensor<float> sumVar;
            RegTensor<float> var;
            RegTensor<float> deltaL;
            RegTensor<float> deltaR;
            RegTensor<float> one;
            RegTensor<float> rstd;
            MaskReg pregLoop;
            MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0 = dichotomyAddReminder;
            // 计算mean
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, scale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i, mean, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
                 i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            Duplicate(one, float(1.0), pregMain);
            Duplicate(mean, mean, pregMain);
            sreg0 = dichotomyAddReminder;
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);

                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i, var, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
                 i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
            }

            DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
        }
    }

    /*
      // Welford Finalize非对齐场景计算公式如下:
      counts = np.ones(len(mean), dtype=np.float32)*count
      for i in range(tail_size):
          counts[i] += 1
      finalize_mean = np.sum(mean*counts) / np.sum(counts)
      finalize_delta = mean - finalize_mean
      finalize_delta_square = finalize_delta * finalize_delta
      M2_fixed = M2 + counts * finalize_delta_square
      finalize_std = np.sum(M2_fixed) / np.sum(counts)

      // Welford Finalize非对齐场景下，二分累加存在如下几种场景:
      首先,非对齐场景下存在两种尾块类型
      1. welford自身的整块和尾块，需要注意的是，welford自身的整块也可能非对齐，整块+尾块=并行度
      2. 二分累加的整块和尾块
      3.
      3.1 welford整块小于二分累加整块，这种场景下，可以进一步细分为两种场景
      3.1.1 welford整块小于二分累加尾块向上对齐，那么welford整块处理逻辑就需要体现在二分累加整尾块折叠逻辑中
      3.1.2 welford整块大于等于二分累加尾块向上对齐，那么welford整块处理逻辑就需要体现剩余整块回刷UB逻辑中
      3.2 welford整块大于等于二分累加整块，那么welford整块处理逻辑就需要体现在二分累加整尾块折叠逻辑中
    */

    // welford整块大于等于二分累加整块
    __aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation1(
        __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
        __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
        uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
        uint32_t tailSize, float reduceScale, float cnt, float eps)
    {
        float tailCnt = cnt + float(1.0);
        float coeff = tailCnt / cnt;
        float tailCountScale = tailCnt * reduceScale;
        float countScale = cnt * reduceScale;

        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t dichotomyAddReminderRealLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
        uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
        uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
        uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

        uint32_t welfordDiff = tailSize - dichotomyAddPower;
        uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
        uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
        uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
        uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

        uint32_t dichotomyAddReminderAfterSplit =
            dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 - welfordDiffReminderAlign;
        uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminderAfterSplit, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> dichotomyAddMeanL;
            RegTensor<float> dichotomyAddMeanR;
            RegTensor<float> dichotomyAddVarL;
            RegTensor<float> dichotomyAddVarR;
            RegTensor<float> sumMean;
            RegTensor<float> mean;
            RegTensor<float> sumVar;
            RegTensor<float> var;
            RegTensor<float> deltaL;
            RegTensor<float> deltaR;
            RegTensor<float> one;
            RegTensor<float> rstd;
            RegTensor<float> tmp;

            MaskReg pregLoop;
            MaskReg pregLoop1;
            MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0;

            // 整块使用tailCountScale,尾块使用tailCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, tailCountScale, pregMain);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i, mean, pregMerge);
            }

            // 处理welford第一次非对齐点, 整块使用tailCountScale,尾块部分使用tailCountScale, 部分使用countScale
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            uint32_t sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                pregLoop1 = UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Muls(tmp, dichotomyAddMeanR, coeff, pregLoop1);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dichotomyAddMeanR, tmp, pregLoop1);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount, mean, pregMerge);
            }

            // 整块使用tailCountScale,尾块使用countScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                DataCopy(
                    dichotomyAddMeanR,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
            }
            // PART2: 整块剩余部分vcadd回刷UB,使用tailCountScale
            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
            }
            DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            // 计算rstd
            Duplicate(one, float(1.0), pregMain);
            Duplicate(mean, mean, pregMain);
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
                Mul(deltaR, deltaR, deltaR, pregMain);
                Muls(deltaR, deltaR, tailCnt, pregMain);

                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregMain);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i, var, pregMerge);
            }
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                pregLoop1 = UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                Muls(tmp, deltaR, coeff, pregLoop1);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(deltaR, tmp, pregLoop1);

                DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                DataCopy(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount, var, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(
                    dichotomyAddMeanR,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);

                DataCopy(
                    dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                DataCopy(
                    dichotomyAddVarR,
                    tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);
                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
            }
            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
            }
            DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
        }
    }

    // welford整块小于二分累加整块，并且小于等于二分累加尾块向上对齐
    __aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation2(
        __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
        __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
        uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
        uint32_t tailSize, float reduceScale, float cnt, float eps)
    {
        float tailCnt = cnt + float(1.0);
        float coeff = tailCnt / cnt;
        float tailCountScale = tailCnt * reduceScale;
        float countScale = cnt * reduceScale;

        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t welfordDiffLoopCount = tailSize / VL_FP32;
        uint32_t welfordDiffReminder = tailSize - welfordDiffLoopCount * VL_FP32;
        uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
        uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

        uint16_t dichotomyAddReminderRealLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
        uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
        uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
        uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

        uint32_t dichotomyAddReminderAfterSplit =
            dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 - welfordDiffReminderAlign;
        uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminderAfterSplit, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> dichotomyAddMeanL;
            RegTensor<float> dichotomyAddMeanR;
            RegTensor<float> dichotomyAddVarL;
            RegTensor<float> dichotomyAddVarR;
            RegTensor<float> sumMean;
            RegTensor<float> mean;
            RegTensor<float> sumVar;
            RegTensor<float> var;
            RegTensor<float> deltaL;
            RegTensor<float> deltaR;
            RegTensor<float> one;
            RegTensor<float> rstd;
            RegTensor<float> tmp;

            MaskReg pregLoop;
            MaskReg pregLoop1;
            MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0;
            uint32_t sreg1;

            // 整块使用tailCountScale,尾块使用countScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregMain);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i, mean, pregMerge);
            }

            // 处理welford第一次非对齐点, 尾块使用countScale,整块部分使用tailCountScale, 部分使用countScale
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                pregLoop1 = UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Muls(tmp, dichotomyAddMeanL, coeff, pregLoop1);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop1);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount, mean, pregMerge);
            }

            // 整块使用countScale,尾块使用countScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                DataCopy(
                    dichotomyAddMeanR,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
            }
            // PART2: 整块剩余部分vcadd回刷UB,使用countScale
            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
            }
            DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            // 计算rstd
            Duplicate(one, float(1.0), pregMain);
            Duplicate(mean, mean, pregMain);
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
                Mul(deltaR, deltaR, deltaR, pregMain);
                Muls(deltaR, deltaR, cnt, pregMain);

                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregMain);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i, var, pregMerge);
            }
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                pregLoop1 = UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                Muls(tmp, deltaL, coeff, pregLoop1);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(deltaL, tmp, pregLoop1);

                DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                DataCopy(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount, var, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(
                    dichotomyAddMeanR,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);

                DataCopy(
                    dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                DataCopy(
                    dichotomyAddVarR,
                    tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);
                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
            }

            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
            }
            DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
        }
    }

    // 场景3：welford整块小于二分累加整块，并且大于二分累加尾块向上对齐
    __aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation3(
        __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
        __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
        uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
        uint32_t tailSize, float reduceScale, float cnt, float eps)
    {
        float tailCnt = cnt + float(1.0);
        float coeff = tailCnt / cnt;
        float tailCountScale = tailCnt * reduceScale;
        float countScale = cnt * reduceScale;

        // 二分累加
        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
        uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
        uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
        uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;
        uint32_t dichotomyAddReminderRoundUp = dichotomyAddReminderLoopCount * VL_FP32;

        uint32_t welfordDiff = tailSize - dichotomyAddReminderRoundUp;
        uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
        uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
        uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
        uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;
        uint16_t dichotomyAddPowerRemainLoopCount = dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount -
                                                    welfordDiffLoopCount - welfordReminderLoopCount;
        uint32_t dichotomyAddPowerOffset =
            dichotomyAddReminderRoundUp + welfordDiffLoopCount * VL_FP32 + welfordDiffReminderAlign;

        __VEC_SCOPE__
        {
            RegTensor<float> dichotomyAddMeanL;
            RegTensor<float> dichotomyAddMeanR;
            RegTensor<float> dichotomyAddVarL;
            RegTensor<float> dichotomyAddVarR;
            RegTensor<float> sumMean;
            RegTensor<float> mean;
            RegTensor<float> sumVar;
            RegTensor<float> var;
            RegTensor<float> deltaL;
            RegTensor<float> deltaR;
            RegTensor<float> one;
            RegTensor<float> rstd;
            RegTensor<float> tmp;

            MaskReg pregLoop;
            MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0 = dichotomyAddReminder;
            // 整块使用tailCountScale, 尾块使用CountScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i, mean, pregMerge);
            }

            // 剩余整块需要拆分成多部分
            // 整块剩余部分回刷UB，整块使用tailCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
            }

            sreg0 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                Muls(tmp, dichotomyAddMeanL, coeff, pregLoop);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, mean, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount +
                        welfordReminderLoopCount + i,
                    mean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            // 计算rstd
            Duplicate(one, float(1.0), pregMain);
            Duplicate(mean, mean, pregMain);
            // 整块使用tailCountScale, 尾块使用CountScale
            sreg0 = dichotomyAddReminder;
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);

                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i, var, pregMerge);
            }

            // 整块剩余部分回刷UB，整块使用tailCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
            }

            sreg0 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                Muls(tmp, deltaL, coeff, pregLoop);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(deltaL, tmp, pregLoop);
                DataCopy(
                    dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, var, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddPowerOffset);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount +
                        welfordReminderLoopCount + i,
                    var, pregMerge);
            }

            DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
        }
    }

    __aicore__ inline void VFWelfordParallelFinalizeNonAlign(
        __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
        __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
        uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
        uint32_t tailSize, float reduceScale, float cnt, float eps)
    {
        // 非对齐Welford finalize阶段由于自身存在整尾块，二分折叠存在整尾块，会出现多种不同的场景，每个场景都有独立的VF
        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint32_t dichotomyAddReminderRoundUp = CEIL_DIV(dichotomyAddReminder, VL_FP32) * VL_FP32;
        if (tailSize >= dichotomyAddPower) {
            VFWelfordParallelFinalizeNonAlignSituation1(
                meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount, dichotomyAddPower,
                dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
            return;
        }
        if (tailSize <= dichotomyAddReminderRoundUp) {
            VFWelfordParallelFinalizeNonAlignSituation2(
                meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount, dichotomyAddPower,
                dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
            return;
        }
        VFWelfordParallelFinalizeNonAlignSituation3(
            meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount, dichotomyAddPower,
            dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
    }

    __aicore__ inline void VFCalcY(
        __local_mem__ X1_TYPE* x1Addr, __local_mem__ X2_TYPE* x2Addr, __local_mem__ BIAS_TYPE* biasAddr,
        __local_mem__ BETA_TYPE* betaAddr, __local_mem__ GAMMA_TYPE* gammaAddr, float mean, float rstd,
        __local_mem__ BIAS_TYPE* yOutAddr, uint32_t colsCount)
    {
        uint32_t vlFp32 = vlFp32_;
        uint16_t colsLoopCount = CEIL_DIV(colsCount, vlFp32);

        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            RegTensor<float> beta;
            RegTensor<float> gamma;
            MaskReg pregLoop;

            uint32_t sreg0 = colsCount;
            for (uint16_t i = 0; i < colsLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(
                    x1Addr, x2Addr, biasAddr, x, pregLoop, i * vlFp32, i * vlFp32, i * vlFp32);
                LoadGammaBeta(gammaAddr, betaAddr, gamma, beta, pregLoop, i * vlFp32);
                Adds(x, x, mean, pregLoop);
                Muls(y, x, rstd, pregLoop);
                Mul(y, y, gamma, pregLoop);
                Add(y, y, beta, pregLoop);
                StoreRegToOutput(yOutAddr, y, pregLoop, i * vlFp32);
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
        int64_t meanOffset = 0;

        LocalTensor<float> tmpMeanLocal = meanBuf_.Get<float>();
        LocalTensor<float> tmpVarLocal = varBuf_.Get<float>();
        LocalTensor<float> binaryAddLocal = binaryAddBuf_.Get<float>();
        __local_mem__ float* tmpMeanAddr = (__local_mem__ float*)tmpMeanLocal.GetPhyAddr();
        __local_mem__ float* tmpVarAddr = (__local_mem__ float*)tmpVarLocal.GetPhyAddr();
        __local_mem__ float* binaryAddAddr = (__local_mem__ float*)binaryAddLocal.GetPhyAddr();

        for (int64_t i = 0; i < rowsPerCore_; i++) {
            int64_t count = 0;
            int64_t inputOffsetTemp = inputOffset;
            int64_t outputOffsetTemp = outputOffset;
            int64_t biasOffset = 0;
            if constexpr (IS_BIAS_ELEWISE) {
                biasOffset = inputOffsetTemp;
            }

            LocalTensor<float> meanLocal = meanQueue_.template AllocTensor<float>();
            LocalTensor<float> rstdLocal = rstdQueue_.template AllocTensor<float>();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal[0].GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal[0].GetPhyAddr();

            for (int64_t j = 0; j < colsLoopCount_; j++) {
                int32_t copyLen = (j == colsLoopCount_ - 1) ? colsTail_ : colsPerLoop_;

                LocalTensor<X1_TYPE> x1Local = x1Queue_.template AllocTensor<X1_TYPE>();
                LocalTensor<X2_TYPE> x2Local = x2Queue_.template AllocTensor<X2_TYPE>();
                LocalTensor<BIAS_TYPE> biasLocal;
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasLocal = biasQueue_.template AllocTensor<BIAS_TYPE>();
                }
                // copy in x1, x2, bias
                CopyInputsToUB(x1Local, x2Local, biasLocal, inputOffsetTemp, biasOffset, copyLen);

                x1Local = x1Queue_.template DeQue<X1_TYPE>();
                x2Local = x2Queue_.template DeQue<X2_TYPE>();
                __local_mem__ X1_TYPE* x1Addr = (__local_mem__ X1_TYPE*)x1Local[0].GetPhyAddr();
                __local_mem__ X2_TYPE* x2Addr = (__local_mem__ X2_TYPE*)x2Local[0].GetPhyAddr();

                __local_mem__ BIAS_TYPE* biasAddr;
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasLocal = biasQueue_.template DeQue<BIAS_TYPE>();
                    biasAddr = (__local_mem__ BIAS_TYPE*)biasLocal[0].GetPhyAddr();
                }

                LocalTensor<BIAS_TYPE> xLocal = xQueue_.template AllocTensor<BIAS_TYPE>();
                __local_mem__ BIAS_TYPE* xAddr = (__local_mem__ BIAS_TYPE*)xLocal[0].GetPhyAddr();

                uint16_t loopCount = CEIL_DIV(copyLen, vlFp32_);
                count += 1;
                float scale = (float)1.0 / static_cast<float>(count);

                if (j == 0) {
                    VFWelfordParallelUpdateWithInit(
                        x1Addr, x2Addr, biasAddr, xAddr, tmpMeanAddr, tmpVarAddr, copyLen, loopCount, scale);
                } else {
                    VFWelfordParallelUpdate(
                        x1Addr, x2Addr, biasAddr, xAddr, tmpMeanAddr, tmpVarAddr, copyLen, loopCount, scale);
                }

                // copy out x
                if (tiling_->outputX) {
                    xQueue_.EnQue(xLocal);
                    xLocal = xQueue_.template DeQue<BIAS_TYPE>();
                    CopyXToGm(xLocal, outputOffsetTemp, copyLen);
                }

                xQueue_.FreeTensor(xLocal);
                x1Queue_.FreeTensor(x1Local);
                x2Queue_.FreeTensor(x2Local);
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasQueue_.FreeTensor(biasLocal);
                }

                inputOffsetTemp += copyLen;
                outputOffsetTemp = inputOffsetTemp;
                biasOffset += copyLen;
            }

            float reduceScale = float(1.0) / static_cast<float>(cols_);
            if (colsTail_ != colsPerLoop_) {
                VFWelfordParallelFinalizeNonAlign(
                    meanAddr, rstdAddr, tmpMeanAddr, tmpVarAddr, binaryAddAddr, colsPerLoop_, binaryAddNum_,
                    binaryAddK_, binaryAddLastNum_, 0, colsTail_, reduceScale, count - 1, eps_);
            } else {
                float scale = float(1.0) / static_cast<float>(colsPerLoop_);
                VFWelfordParallelFinalizeAlign(
                    meanAddr, rstdAddr, tmpMeanAddr, tmpVarAddr, binaryAddAddr, colsPerLoop_, binaryAddNum_,
                    binaryAddK_, binaryAddLastNum_, 0, reduceScale, scale, count, eps_);
            }

            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);

            float mean = meanLocal(0) * float(-1.0);
            float rstd = rstdLocal(0);

            // copy out mean
            meanQueue_.EnQue(meanLocal);
            meanLocal = meanQueue_.template DeQue<float>();
            CopyMeanToGm(meanLocal, meanOffset, 1);

            // copy out rstd
            rstdQueue_.EnQue(rstdLocal);
            rstdLocal = rstdQueue_.template DeQue<float>();
            CopyRstdToGm(rstdLocal, meanOffset, 1);

            // calc y with VF
            inputOffsetTemp = inputOffset;
            outputOffsetTemp = outputOffset;
            biasOffset = 0;
            if constexpr (IS_BIAS_ELEWISE) {
                biasOffset = inputOffsetTemp;
            }
            int64_t inputOffsetGamma = 0;
            for (int64_t j = 0; j < colsLoopCount_; j++) {
                int32_t copyLen = (j == colsLoopCount_ - 1) ? colsTail_ : colsPerLoop_;
                LocalTensor<X1_TYPE> x1Local = x1Queue_.template AllocTensor<X1_TYPE>();
                LocalTensor<X2_TYPE> x2Local = x2Queue_.template AllocTensor<X2_TYPE>();
                LocalTensor<BIAS_TYPE> biasLocal;
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasLocal = biasQueue_.template AllocTensor<BIAS_TYPE>();
                }
                LocalTensor<GAMMA_TYPE> gammaLocal = gammaQueue_.template AllocTensor<GAMMA_TYPE>();
                LocalTensor<BETA_TYPE> betaLocal = betaQueue_.template AllocTensor<BETA_TYPE>();
                // copy in x1, x2, bias
                CopyInputsToUB(x1Local, x2Local, biasLocal, inputOffsetTemp, biasOffset, copyLen);
                // copy in gamma, beta
                CopyGammaAndBetaToUB(gammaLocal, betaLocal, inputOffsetGamma, copyLen);

                x1Local = x1Queue_.template DeQue<X1_TYPE>();
                x2Local = x2Queue_.template DeQue<X2_TYPE>();
                __local_mem__ X1_TYPE* x1Addr = (__local_mem__ X1_TYPE*)x1Local[0].GetPhyAddr();
                __local_mem__ X2_TYPE* x2Addr = (__local_mem__ X2_TYPE*)x2Local[0].GetPhyAddr();

                __local_mem__ BIAS_TYPE* biasAddr;
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasLocal = biasQueue_.template DeQue<BIAS_TYPE>();
                    biasAddr = (__local_mem__ BIAS_TYPE*)biasLocal[0].GetPhyAddr();
                }

                gammaLocal = gammaQueue_.template DeQue<GAMMA_TYPE>();
                betaLocal = betaQueue_.template DeQue<BETA_TYPE>();
                LocalTensor<BIAS_TYPE> yLocal = yQueue_.template AllocTensor<BIAS_TYPE>();

                __local_mem__ BETA_TYPE* betaAddr = (__local_mem__ BETA_TYPE*)betaLocal[0].GetPhyAddr();
                __local_mem__ GAMMA_TYPE* gammaAddr = (__local_mem__ GAMMA_TYPE*)gammaLocal[0].GetPhyAddr();
                __local_mem__ BIAS_TYPE* yAddr = (__local_mem__ BIAS_TYPE*)yLocal[0].GetPhyAddr();

                VFCalcY(x1Addr, x2Addr, biasAddr, betaAddr, gammaAddr, mean, rstd, yAddr, copyLen);

                // copy out y
                yQueue_.EnQue(yLocal);
                yLocal = yQueue_.template DeQue<BIAS_TYPE>();
                CopyYToGm(yLocal, outputOffsetTemp, copyLen);
                yQueue_.FreeTensor(yLocal);

                x1Queue_.FreeTensor(x1Local);
                x2Queue_.FreeTensor(x2Local);
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasQueue_.FreeTensor(biasLocal);
                }

                betaQueue_.FreeTensor(betaLocal);
                gammaQueue_.FreeTensor(gammaLocal);

                inputOffsetTemp += copyLen;
                outputOffsetTemp = inputOffsetTemp;
                inputOffsetGamma += copyLen;
                biasOffset += copyLen;
            }

            meanQueue_.FreeTensor(meanLocal);
            rstdQueue_.FreeTensor(rstdLocal);
            meanOffset += 1;
            inputOffset = inputOffsetTemp;
            outputOffset = outputOffsetTemp;
        }
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> betaQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> meanQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstdQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xQueue_;
    TBuf<TPosition::VECCALC> meanBuf_;
    TBuf<TPosition::VECCALC> varBuf_;
    TBuf<TPosition::VECCALC> binaryAddBuf_;

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
    int64_t cols_;
    int64_t colsTail_;
    int64_t colsLoopCount_;
    int64_t colsPerLoop_;
    int64_t colsPerLoopAlign_;
    int64_t colsPerLoopAlignB16_;
    int64_t colsPerLoopAlignB32_;
    int64_t colsPerLoopAlignBias_;
    int64_t rowsPerCore_;
    int64_t binaryAddNum_;
    int64_t binaryAddK_;
    int64_t binaryAddLastNum_;
    int64_t powerOfTwo_;
    float eps_;

    TPipe pipe_;
    const AddLayerNormRegbaseTilingData* tiling_;
};
} // namespace AddLayerNorm
#endif // ADD_LAYER_NORM_REGBASE_WELFORD_H