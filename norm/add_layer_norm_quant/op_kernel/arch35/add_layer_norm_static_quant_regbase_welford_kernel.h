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
 * \file add_layer_norm_static_quant_regbase_welford_kernel.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_STATIC_QUANT_REGBASE_WELFORD_H
#define ADD_LAYER_NORM_STATIC_QUANT_REGBASE_WELFORD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "add_layer_norm_quant_regbase_helper.h"

namespace AddLayerNormQuantRegbase {
template <typename X1_TYPE, typename SCALE_TYPE, int32_t TILING_KEY, int32_t OPT_CODE, int32_t BUFFER_NUM = 1>
class KernelAddLayerNormStaticQuantRegbaseWelford {
public:
    __aicore__ inline KernelAddLayerNormStaticQuantRegbaseWelford(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR scale1, GM_ADDR scale2,
        GM_ADDR zeroOffset1, GM_ADDR zeroOffset2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR workspace,
        const AddLayerNormQuantRegbaseTilingData* tilingData)
    {
        uint32_t coreIdx = GetBlockIdx();

        cols_ = tilingData->cols;
        colsTail_ = tilingData->colsTail;
        colsPerLoop_ = tilingData->colsPerLoop;
        colsLoopCount_ = tilingData->colsLoopCount;
        eps_ = tilingData->eps;
        binaryAddNum_ = tilingData->binaryAddNum;
        binaryAddK_ = tilingData->binaryAddK;
        binaryAddLastNum_ = tilingData->binaryAddLastNum;
        outputX_ = tilingData->outputX;

        powerOfTwo_ = 1;
        while (powerOfTwo_ < colsPerLoop_) {
            powerOfTwo_ *= AddLayerNorm::TWO;
        }

        uint64_t gmOffset = (tilingData->rowsPerCore * cols_) * coreIdx;
        if (coreIdx < GetBlockNum() - 1) {
            // non-tail cores
            rowsPerCore_ = tilingData->rowsPerCore;
        } else {
            // tail cores
            rowsPerCore_ = tilingData->rowsPerTailCore;
        }

        x1Gm_.SetGlobalBuffer((__gm__ X1_TYPE*)(x1) + gmOffset);
        x2Gm_.SetGlobalBuffer((__gm__ X1_TYPE*)(x2) + gmOffset);
        if constexpr (IS_BIAS_ELEWISE) {
            biasGm_.SetGlobalBuffer((__gm__ X1_TYPE*)(bias) + gmOffset);
        } else if constexpr (IS_BIAS_BROADCAST) {
            biasGm_.SetGlobalBuffer((__gm__ X1_TYPE*)bias);
        }
        gammaGm_.SetGlobalBuffer((__gm__ X1_TYPE*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ X1_TYPE*)beta);
        scale1Gm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)scale1);
        CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, scale2Gm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)scale2));
        CONST_CONDITIONAL_EXPR(IS_OFFSET1_EXIST, zeroOffset1Gm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)zeroOffset1));
        CONST_CONDITIONAL_EXPR(IS_OFFSET2_EXIST, zeroOffset2Gm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)zeroOffset2));

        y1Gm_.SetGlobalBuffer((__gm__ int8_t*)(y1) + gmOffset);
        y2Gm_.SetGlobalBuffer((__gm__ int8_t*)(y2) + gmOffset);
        xGm_.SetGlobalBuffer((__gm__ X1_TYPE*)(x) + gmOffset);

        colsPerLoopAlign_ = BLOCK_ALIGN(colsPerLoop_ * sizeof(X1_TYPE), blockSize_) / sizeof(X1_TYPE);

        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(x2Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            pipe_->InitBuffer(biasQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        }
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(y1Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(int8_t)));
        pipe_->InitBuffer(betaQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(scale1Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(SCALE_TYPE)));

        CONST_CONDITIONAL_EXPR(
            IS_SCALE2_EXIST, pipe_->InitBuffer(y2Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(int8_t))));
        CONST_CONDITIONAL_EXPR(
            IS_SCALE2_EXIST, pipe_->InitBuffer(scale2Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(SCALE_TYPE))));
        CONST_CONDITIONAL_EXPR(
            IS_OFFSET1_EXIST,
            pipe_->InitBuffer(zeroOffset1Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(SCALE_TYPE))));
        CONST_CONDITIONAL_EXPR(
            IS_OFFSET2_EXIST,
            pipe_->InitBuffer(zeroOffset2Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(SCALE_TYPE))));

        pipe_->InitBuffer(meanBuf_, colsPerLoopAlign_ * sizeof(float));
        pipe_->InitBuffer(varBuf_, colsPerLoopAlign_ * sizeof(float));

        pipe_->InitBuffer(meanTmpBuf_, blockSize_);
        pipe_->InitBuffer(rstdTmpBuf_, blockSize_);

        int64_t binaryAddBufSize = BLOCK_ALIGN((binaryAddNum_ / vlFp32_) * sizeof(float), blockSize_);
        if (binaryAddBufSize > 0) {
            pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize);
        }
    }

    __aicore__ inline void CopyInputsToUB(
        LocalTensor<X1_TYPE> x1Local, LocalTensor<X1_TYPE> x2Local, LocalTensor<X1_TYPE> biasLocal, int64_t inputOffset,
        int64_t biasOffset, int32_t copyLen)
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
        DataCopyPad(x2Local, x2Gm_[inputOffset], dataCopyParams, padParams);
        x2Queue_.EnQue(x2Local);
        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            DataCopyPad(biasLocal, biasGm_[biasOffset], dataCopyParams, padParams);
            biasQueue_.EnQue(biasLocal);
        }
    }

    __aicore__ inline void CopyXToGm(LocalTensor<X1_TYPE> xLocal, int64_t xOffset, int32_t copyLen)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(X1_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(xGm_[xOffset], xLocal, dataCopyParams);
    }

    __aicore__ inline void CopyYToGm(
        LocalTensor<int8_t> y1Local, LocalTensor<int8_t> y2Local, int64_t yOffset, int32_t copyLen)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(int8_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(y1Gm_[yOffset], y1Local, dataCopyParams);
        CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, DataCopyPad(y2Gm_[yOffset], y2Local, dataCopyParams));
    }

    __aicore__ inline void CopyGammaAndBetaToUB(
        LocalTensor<X1_TYPE> gammaLocal, LocalTensor<X1_TYPE> betaLocal, int64_t offset, int32_t copyLen)
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
        DataCopyPad(betaLocal, betaGm_[offset], dataCopyParams, padParams);
        betaQueue_.EnQue(betaLocal);
        DataCopyPad(gammaLocal, gammaGm_[offset], dataCopyParams, padParams);
        gammaQueue_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyQuantParams2UB(
        LocalTensor<SCALE_TYPE> scale1Local, LocalTensor<SCALE_TYPE> scale2Local, LocalTensor<SCALE_TYPE> offset1Local,
        LocalTensor<SCALE_TYPE> offset2Local, int64_t offset, int32_t copyLen)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(SCALE_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(scale1Local, scale1Gm_[offset], dataCopyParams, {});
        scale1Queue_.EnQue(scale1Local);
        if constexpr (IS_SCALE2_EXIST) {
            DataCopyPad(scale2Local, scale2Gm_[offset], dataCopyParams, {});
            scale2Queue_.EnQue(scale2Local);
        }
        if constexpr (IS_OFFSET1_EXIST) {
            DataCopyPad(offset1Local, zeroOffset1Gm_[offset], dataCopyParams, {});
            zeroOffset1Queue_.EnQue(offset1Local);
        }
        if constexpr (IS_OFFSET2_EXIST) {
            DataCopyPad(offset2Local, zeroOffset2Gm_[offset], dataCopyParams, {});
            zeroOffset2Queue_.EnQue(offset2Local);
        }
    }

    static __aicore__ inline void VFCalcYQuant(
        LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local, LocalTensor<X1_TYPE>& biasLocal,
        LocalTensor<X1_TYPE>& betaLocal, LocalTensor<X1_TYPE>& gammaLocal, LocalTensor<SCALE_TYPE>& scale1Local,
        LocalTensor<SCALE_TYPE>& scale2Local, LocalTensor<SCALE_TYPE>& offset1Local,
        LocalTensor<SCALE_TYPE>& offset2Local, float mean, float rstd, LocalTensor<X1_TYPE>& yOutLocal,
        LocalTensor<int8_t>& y1Local, LocalTensor<int8_t>& y2Local, uint32_t colsCount, uint32_t vlFp32)
    {
        __local_mem__ X1_TYPE* x1Addr = (__local_mem__ X1_TYPE*)x1Local[0].GetPhyAddr();
        __local_mem__ X1_TYPE* x2Addr = (__local_mem__ X1_TYPE*)x2Local[0].GetPhyAddr();
        __local_mem__ X1_TYPE* betaAddr = (__local_mem__ X1_TYPE*)betaLocal[0].GetPhyAddr();
        __local_mem__ X1_TYPE* gammaAddr = (__local_mem__ X1_TYPE*)gammaLocal[0].GetPhyAddr();
        __local_mem__ SCALE_TYPE* scale1Addr = (__local_mem__ SCALE_TYPE*)scale1Local[0].GetPhyAddr();
        __local_mem__ int8_t* quant1OutAddr = (__local_mem__ int8_t*)y1Local[0].GetPhyAddr();

        __local_mem__ X1_TYPE* biasAddr;
        __local_mem__ int8_t* quant2OutAddr;
        __local_mem__ SCALE_TYPE* scale2Addr;
        __local_mem__ SCALE_TYPE* offset1Addr;
        __local_mem__ SCALE_TYPE* offset2Addr;

        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            biasAddr = (__local_mem__ X1_TYPE*)biasLocal[0].GetPhyAddr();
        }
        CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, quant2OutAddr, (__local_mem__ int8_t*)y2Local[0].GetPhyAddr());
        CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, scale2Addr, (__local_mem__ SCALE_TYPE*)scale2Local[0].GetPhyAddr());
        CONST_CONDITIONAL_ASSIGN(
            IS_OFFSET1_EXIST, offset1Addr, (__local_mem__ SCALE_TYPE*)offset1Local[0].GetPhyAddr());
        CONST_CONDITIONAL_ASSIGN(
            IS_OFFSET2_EXIST, offset2Addr, (__local_mem__ SCALE_TYPE*)offset2Local[0].GetPhyAddr());

        // uint32_t vlFp32 = vlFp32_;
        uint16_t colsLoopCount = CEIL_DIV(colsCount, vlFp32);

        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            RegTensor<float> beta;
            RegTensor<float> gamma;
            RegTensor<float> scale1;
            RegTensor<float> scale2;
            RegTensor<float> offset1;
            RegTensor<float> offset2;
            RegTensor<int8_t> quantOut1;
            RegTensor<int8_t> quantOut2;

            MaskReg pregLoop;

            uint32_t sreg0 = colsCount;
            for (uint16_t i = 0; i < colsLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                    x1Addr, x2Addr, biasAddr, x, pregLoop, i * vlFp32, i * vlFp32, i * vlFp32);
                LoadGammaBeta(gammaAddr, betaAddr, gamma, beta, pregLoop, i * vlFp32);
                LoadQuantParams(scale1Addr, scale1, pregLoop, i * vlFp32);
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, LoadQuantParams(scale2Addr, scale2, pregLoop, i * vlFp32));
                CONST_CONDITIONAL_EXPR(IS_OFFSET1_EXIST, LoadQuantParams(offset1Addr, offset1, pregLoop, i * vlFp32));
                CONST_CONDITIONAL_EXPR(IS_OFFSET2_EXIST, LoadQuantParams(offset2Addr, offset2, pregLoop, i * vlFp32));

                Adds(x, x, mean, pregLoop);
                Muls(y, x, rstd, pregLoop);
                Mul(y, y, gamma, pregLoop);
                Add(y, y, beta, pregLoop); // LayerNorm result

                // Do Quant1
                if constexpr (IS_DIV_SCALE) {
                    Div(x, y, scale1, pregLoop);
                    CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, Div(y, y, scale2, pregLoop));
                } else {
                    Mul(x, y, scale1, pregLoop);
                    CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, Mul(y, y, scale2, pregLoop));
                }
                CONST_CONDITIONAL_EXPR(IS_OFFSET1_EXIST, Add(x, x, offset1, pregLoop));
                CONST_CONDITIONAL_EXPR(IS_OFFSET2_EXIST, Add(y, y, offset2, pregLoop));

                Round2Int8(quantOut1, x, pregLoop);
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, Round2Int8(quantOut2, y, pregLoop));

                DataCopy<int8_t, StoreDist::DIST_PACK4_B32>(
                    (__local_mem__ int8_t*)quant1OutAddr + i * vlFp32, quantOut1, pregLoop);
                CONST_CONDITIONAL_EXPR(
                    IS_SCALE2_EXIST, (DataCopy<int8_t, StoreDist::DIST_PACK4_B32>(
                                         (__local_mem__ int8_t*)quant2OutAddr + i * vlFp32, quantOut2, pregLoop)));
            }
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();

        int64_t inputOffset = 0;
        int64_t outputOffset = 0;

        LocalTensor<float> tmpMeanLocal = meanBuf_.Get<float>();
        LocalTensor<float> tmpVarLocal = varBuf_.Get<float>();
        LocalTensor<float> binaryAddLocal = binaryAddBuf_.Get<float>();

        for (int64_t i = 0; i < rowsPerCore_; i++) {
            int64_t count = 0;
            int64_t inputOffsetTemp = inputOffset;
            int64_t outputOffsetTemp = outputOffset;
            int64_t biasOffset = 0;
            if constexpr (IS_BIAS_ELEWISE) {
                biasOffset = inputOffsetTemp;
            }

            LocalTensor<float> meanLocal = meanTmpBuf_.Get<float>();
            LocalTensor<float> rstdLocal = rstdTmpBuf_.Get<float>();

            for (int64_t j = 0; j < colsLoopCount_; j++) {
                int32_t copyLen = (j == colsLoopCount_ - 1) ? colsTail_ : colsPerLoop_;

                LocalTensor<X1_TYPE> x1Local = x1Queue_.template AllocTensor<X1_TYPE>();
                LocalTensor<X1_TYPE> x2Local = x2Queue_.template AllocTensor<X1_TYPE>();
                LocalTensor<X1_TYPE> biasLocal;
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasLocal = biasQueue_.template AllocTensor<X1_TYPE>();
                }
                // copy in x1, x2, bias
                CopyInputsToUB(x1Local, x2Local, biasLocal, inputOffsetTemp, biasOffset, copyLen);

                x1Local = x1Queue_.template DeQue<X1_TYPE>();
                x2Local = x2Queue_.template DeQue<X1_TYPE>();

                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasLocal = biasQueue_.template DeQue<X1_TYPE>();
                }

                LocalTensor<X1_TYPE> xLocal = xQueue_.template AllocTensor<X1_TYPE>();

                uint16_t loopCount = CEIL_DIV(copyLen, vlFp32_);
                count += 1;
                float scale = static_cast<float>(1.0) / static_cast<float>(count);

                if (j == 0) {
                    VFWelfordParallelUpdateWithInit<X1_TYPE, TILING_KEY>(
                        x1Local, x2Local, biasLocal, xLocal, tmpMeanLocal, tmpVarLocal, copyLen, loopCount, scale);
                } else {
                    VFWelfordParallelUpdate<X1_TYPE, TILING_KEY>(
                        x1Local, x2Local, biasLocal, xLocal, tmpMeanLocal, tmpVarLocal, copyLen, loopCount, scale);
                }

                // copy out x
                if (outputX_) {
                    xQueue_.EnQue(xLocal);
                    xLocal = xQueue_.template DeQue<X1_TYPE>();
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
                    meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, binaryAddLocal, colsPerLoop_, binaryAddNum_,
                    binaryAddK_, binaryAddLastNum_, 0, colsTail_, reduceScale, count - 1, eps_);
            } else {
                float scale = float(1.0) / static_cast<float>(colsPerLoop_);
                VFWelfordParallelFinalizeAlign(
                    meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, binaryAddLocal, colsPerLoop_, binaryAddNum_,
                    binaryAddK_, binaryAddLastNum_, 0, reduceScale, scale, count, eps_);
            }

            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
            float mean = meanLocal(0) * float(-1.0);
            float rstd = rstdLocal(0);

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
                LocalTensor<X1_TYPE> x2Local = x2Queue_.template AllocTensor<X1_TYPE>();
                LocalTensor<X1_TYPE> biasLocal;
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasLocal = biasQueue_.template AllocTensor<X1_TYPE>();
                }
                LocalTensor<X1_TYPE> gammaLocal = gammaQueue_.template AllocTensor<X1_TYPE>();
                LocalTensor<X1_TYPE> betaLocal = betaQueue_.template AllocTensor<X1_TYPE>();

                LocalTensor<SCALE_TYPE> scale1Local = scale1Queue_.template AllocTensor<SCALE_TYPE>();
                LocalTensor<SCALE_TYPE> scale2Local;
                LocalTensor<SCALE_TYPE> offset1Local;
                LocalTensor<SCALE_TYPE> offset2Local;
                CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, scale2Local, scale2Queue_.template AllocTensor<SCALE_TYPE>());
                CONST_CONDITIONAL_ASSIGN(
                    IS_OFFSET1_EXIST, offset1Local, zeroOffset1Queue_.template AllocTensor<SCALE_TYPE>());
                CONST_CONDITIONAL_ASSIGN(
                    IS_OFFSET2_EXIST, offset2Local, zeroOffset2Queue_.template AllocTensor<SCALE_TYPE>());

                // copy in x1, x2, bias
                CopyInputsToUB(x1Local, x2Local, biasLocal, inputOffsetTemp, biasOffset, copyLen);
                // copy in gamma, beta
                CopyGammaAndBetaToUB(gammaLocal, betaLocal, inputOffsetGamma, copyLen);
                // copy in scale/offset
                CopyQuantParams2UB(scale1Local, scale2Local, offset1Local, offset2Local, inputOffsetGamma, copyLen);

                x1Local = x1Queue_.template DeQue<X1_TYPE>();
                x2Local = x2Queue_.template DeQue<X1_TYPE>();

                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasLocal = biasQueue_.template DeQue<X1_TYPE>();
                }

                gammaLocal = gammaQueue_.template DeQue<X1_TYPE>();
                betaLocal = betaQueue_.template DeQue<X1_TYPE>();

                scale1Local = scale1Queue_.template DeQue<SCALE_TYPE>();
                CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, scale2Local, scale2Queue_.template DeQue<SCALE_TYPE>());
                CONST_CONDITIONAL_ASSIGN(
                    IS_OFFSET1_EXIST, offset1Local, zeroOffset1Queue_.template DeQue<SCALE_TYPE>());
                CONST_CONDITIONAL_ASSIGN(
                    IS_OFFSET2_EXIST, offset2Local, zeroOffset2Queue_.template DeQue<SCALE_TYPE>());

                LocalTensor<int8_t> y1Local = y1Queue_.template AllocTensor<int8_t>();
                LocalTensor<int8_t> y2Local;
                CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, y2Local, y2Queue_.template AllocTensor<int8_t>());

                VFCalcYQuant(
                    x1Local, x2Local, biasLocal, betaLocal, gammaLocal, scale1Local, scale2Local, offset1Local,
                    offset2Local, mean, rstd, x1Local, y1Local, y2Local, copyLen, vlFp32_);

                // copy out y
                y1Queue_.EnQue(y1Local);
                y1Local = y1Queue_.template DeQue<int8_t>();
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, y2Queue_.EnQue(y2Local));
                CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, y2Local, y2Queue_.template DeQue<int8_t>());
                CopyYToGm(y1Local, y2Local, outputOffsetTemp, copyLen);
                y1Queue_.FreeTensor(y1Local);
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, y2Queue_.FreeTensor(y2Local););

                x1Queue_.FreeTensor(x1Local);
                x2Queue_.FreeTensor(x2Local);
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasQueue_.FreeTensor(biasLocal);
                }

                betaQueue_.FreeTensor(betaLocal);
                gammaQueue_.FreeTensor(gammaLocal);

                scale1Queue_.FreeTensor(scale1Local);
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, scale2Queue_.FreeTensor(scale2Local));
                CONST_CONDITIONAL_EXPR(IS_OFFSET1_EXIST, zeroOffset1Queue_.FreeTensor(offset1Local));
                CONST_CONDITIONAL_EXPR(IS_OFFSET2_EXIST, zeroOffset2Queue_.FreeTensor(offset2Local));

                inputOffsetTemp += copyLen;
                outputOffsetTemp = inputOffsetTemp;
                inputOffsetGamma += copyLen;
                biasOffset += copyLen;
            }
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

    TQue<QuePosition::VECIN, BUFFER_NUM> scale1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> scale2Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> zeroOffset1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> zeroOffset2Queue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> y1Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> y2Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xQueue_;

    TBuf<TPosition::VECCALC> meanBuf_;
    TBuf<TPosition::VECCALC> varBuf_;
    TBuf<TPosition::VECCALC> binaryAddBuf_;

    TBuf<TPosition::VECCALC> meanTmpBuf_;
    TBuf<TPosition::VECCALC> rstdTmpBuf_;

    GlobalTensor<X1_TYPE> x1Gm_;
    GlobalTensor<X1_TYPE> x2Gm_;
    GlobalTensor<X1_TYPE> biasGm_;
    GlobalTensor<X1_TYPE> gammaGm_;
    GlobalTensor<X1_TYPE> betaGm_;

    GlobalTensor<SCALE_TYPE> scale1Gm_;
    GlobalTensor<SCALE_TYPE> scale2Gm_;
    GlobalTensor<SCALE_TYPE> zeroOffset1Gm_;
    GlobalTensor<SCALE_TYPE> zeroOffset2Gm_;

    GlobalTensor<int8_t> y1Gm_;
    GlobalTensor<int8_t> y2Gm_;
    GlobalTensor<X1_TYPE> xGm_;

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

    bool outputX_;
    TPipe* pipe_ = nullptr;
};
} // namespace AddLayerNormQuantRegbase
#endif // ADD_LAYER_NORM_STATIC_QUANT_REGBASE_WELFORD_H