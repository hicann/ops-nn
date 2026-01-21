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
 * \file add_layer_norm_static_quant_regbase_full_load_kernel.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_STATIC_QUANT_REGBASE_FULL_LOAD_H
#define ADD_LAYER_NORM_STATIC_QUANT_REGBASE_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "add_layer_norm_quant_regbase_helper.h"

namespace AddLayerNormQuantRegbase {
template <typename X1_TYPE, typename SCALE_TYPE, int32_t TILING_KEY, int32_t OPT_CODE, int32_t BUFFER_NUM = 1>
class KernelAddLayerNormStaticQuantRegbaseFullLoad {
public:
    __aicore__ inline KernelAddLayerNormStaticQuantRegbaseFullLoad(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR scale1, GM_ADDR scale2,
        GM_ADDR zeroOffset1, GM_ADDR zeroOffset2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR workspace,
        const AddLayerNormQuantRegbaseTilingData* tilingData)
    {
        uint32_t coreIdx = GetBlockIdx();

        colsPerLoop_ = tilingData->colsPerLoop;
        eps_ = tilingData->eps;
        binaryAddNum_ = tilingData->binaryAddNum;
        binaryAddK_ = tilingData->binaryAddK;
        binaryAddLastNum_ = tilingData->binaryAddLastNum;
        outputX_ = tilingData->outputX;

        powerOfTwo_ = 1;
        while (powerOfTwo_ < colsPerLoop_) {
            powerOfTwo_ *= AddLayerNorm::TWO;
        }

        int32_t rem = colsPerLoop_ % blockSize_;
        int32_t eleNumPerBlock = blockSize_ / sizeof(X1_TYPE); // 8 or 16
        dmaStride_ = (rem == 0) ? 0 : (blockSize_ - rem) / eleNumPerBlock;

        uint64_t gmOffset;
        uint64_t meanOffset;
        if (coreIdx < GetBlockNum() - 1) {
            // non-tail cores
            rowsPerCore_ = tilingData->rowsPerCore;
            rowsPerLoop_ = tilingData->rowsPerLoop;
            gmOffset = (tilingData->rowsPerCore * colsPerLoop_) * coreIdx;
            meanOffset = gmOffset / colsPerLoop_;
        } else {
            // tail cores
            rowsPerCore_ = tilingData->rowsPerTailCore;
            rowsPerLoop_ = tilingData->rowsPerLoop;
            gmOffset = (tilingData->rowsPerCore * colsPerLoop_) * coreIdx;
            meanOffset = gmOffset / colsPerLoop_;
        }
        rowsTail_ = (rowsPerCore_ % rowsPerLoop_ == 0) ? rowsPerLoop_ : (rowsPerCore_ % rowsPerLoop_);
        rowsLoopCount_ = CEIL_DIV(rowsPerCore_, rowsPerLoop_);

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

        colsPerLoopAlign_ = BLOCK_ALIGN(colsPerLoop_, blockSize_); // 32 element aligned

        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(x2Queue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        if constexpr (IS_BIAS_ELEWISE) {
            pipe_->InitBuffer(biasQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        } else if constexpr (IS_BIAS_BROADCAST) {
            pipe_->InitBuffer(biasQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        }
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(y1Queue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(int8_t)));
        pipe_->InitBuffer(x32Queue_, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(float)));
        pipe_->InitBuffer(betaQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(gammaQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));

        pipe_->InitBuffer(scale1Queue_, 1, (colsPerLoopAlign_ * sizeof(SCALE_TYPE)));
        CONST_CONDITIONAL_EXPR(
            IS_SCALE2_EXIST, pipe_->InitBuffer(scale2Queue_, 1, (colsPerLoopAlign_ * sizeof(SCALE_TYPE))));
        CONST_CONDITIONAL_EXPR(
            IS_SCALE2_EXIST,
            pipe_->InitBuffer(y2Queue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(int8_t))));
        CONST_CONDITIONAL_EXPR(
            IS_OFFSET1_EXIST, pipe_->InitBuffer(zeroOffset1Queue_, 1, (colsPerLoopAlign_ * sizeof(SCALE_TYPE))));
        CONST_CONDITIONAL_EXPR(
            IS_OFFSET2_EXIST, pipe_->InitBuffer(zeroOffset2Queue_, 1, (colsPerLoopAlign_ * sizeof(SCALE_TYPE))));

        int64_t binaryAddBufSize = BLOCK_ALIGN((binaryAddNum_ / vlFp32_) * sizeof(float), blockSize_);
        if (binaryAddBufSize > 0) {
            pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize);
        }
        pipe_->InitBuffer(meanBuf_, BLOCK_ALIGN(rowsPerLoop_ * sizeof(float), blockSize_));
        pipe_->InitBuffer(rstdBuf_, BLOCK_ALIGN(rowsPerLoop_ * sizeof(float), blockSize_));
    }

    __aicore__ inline void CopyBiasToUB(LocalTensor<X1_TYPE> biasLocal, int32_t copyLen)
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

        DataCopyPad(biasLocal, biasGm_[0], dataCopyParams, padParams);
        biasQueue_.EnQue(biasLocal);
    }

    __aicore__ inline void CopyInputsToUB(
        LocalTensor<X1_TYPE> x1Local, LocalTensor<X1_TYPE> x2Local, LocalTensor<X1_TYPE> biasLocal, int64_t inputOffset,
        int32_t copyLen, int32_t rowsCount)
    {
        int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(X1_TYPE), blockSize_) / sizeof(X1_TYPE);
        DataCopyPadExtParams<X1_TYPE> padParams;
        padParams.isPad = true;
        padParams.paddingValue = static_cast<X1_TYPE>(0.0);
        padParams.rightPadding = copyLenAlign - copyLen;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = rowsCount;
        dataCopyParams.blockLen = copyLen * sizeof(X1_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = dmaStride_;
        DataCopyPad(x1Local, x1Gm_[inputOffset], dataCopyParams, padParams);
        x1Queue_.EnQue(x1Local);
        DataCopyPad(x2Local, x2Gm_[inputOffset], dataCopyParams, padParams);
        x2Queue_.EnQue(x2Local);
        if constexpr (IS_BIAS_ELEWISE) {
            DataCopyPad(biasLocal, biasGm_[inputOffset], dataCopyParams, padParams);
            biasQueue_.EnQue(biasLocal);
        }
    }

    __aicore__ inline void CopyXToGm(LocalTensor<X1_TYPE> xLocal, int64_t xOffset, int32_t copyLen, int32_t rowsCount)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = rowsCount;
        dataCopyParams.blockLen = copyLen * sizeof(X1_TYPE);
        dataCopyParams.srcStride = dmaStride_;
        dataCopyParams.dstStride = 0;

        DataCopyPad(xGm_[xOffset], xLocal, dataCopyParams);
    }

    __aicore__ inline void CopyYToGm(
        LocalTensor<int8_t> y1Local, LocalTensor<int8_t> y2Local, int64_t yOffset, int32_t copyLen, int32_t rowsCount)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = rowsCount;
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
        LocalTensor<float>& x32Local, LocalTensor<X1_TYPE>& betaLocal, LocalTensor<X1_TYPE>& gammaLocal,
        LocalTensor<float>& meanLocal, LocalTensor<float>& rstdLocal, LocalTensor<SCALE_TYPE>& scale1Local,
        LocalTensor<SCALE_TYPE>& scale2Local, LocalTensor<SCALE_TYPE>& offset1Local,
        LocalTensor<SCALE_TYPE>& offset2Local, LocalTensor<X1_TYPE>& yOutLocal, LocalTensor<int8_t>& y1Local,
        LocalTensor<int8_t>& y2Local, uint32_t rowsCount, uint32_t colsCount, uint32_t colsPerLoopAlign,
        uint32_t vlFp32)
    {
        __local_mem__ float* x32Addr = (__local_mem__ float*)x32Local[0].GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal[0].GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal[0].GetPhyAddr();
        __local_mem__ X1_TYPE* betaAddr = (__local_mem__ X1_TYPE*)betaLocal[0].GetPhyAddr();
        __local_mem__ X1_TYPE* gammaAddr = (__local_mem__ X1_TYPE*)gammaLocal[0].GetPhyAddr();
        __local_mem__ int8_t* quant1OutAddr = (__local_mem__ int8_t*)y1Local[0].GetPhyAddr();
        __local_mem__ SCALE_TYPE* scale1Addr = (__local_mem__ SCALE_TYPE*)scale1Local[0].GetPhyAddr();

        __local_mem__ int8_t* quant2OutAddr;
        __local_mem__ SCALE_TYPE* scale2Addr;
        __local_mem__ SCALE_TYPE* offset1Addr;
        __local_mem__ SCALE_TYPE* offset2Addr;

        CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, quant2OutAddr, (__local_mem__ int8_t*)y2Local[0].GetPhyAddr());
        CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, scale2Addr, (__local_mem__ SCALE_TYPE*)scale2Local[0].GetPhyAddr());
        CONST_CONDITIONAL_ASSIGN(
            IS_OFFSET1_EXIST, offset1Addr, (__local_mem__ SCALE_TYPE*)offset1Local[0].GetPhyAddr());
        CONST_CONDITIONAL_ASSIGN(
            IS_OFFSET2_EXIST, offset2Addr, (__local_mem__ SCALE_TYPE*)offset2Local[0].GetPhyAddr());

        uint16_t colsLoopCount = CEIL_DIV(colsCount, vlFp32);

        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> beta;
            RegTensor<float> gamma;
            RegTensor<float> scale1;
            RegTensor<float> scale2;
            RegTensor<float> offset1;
            RegTensor<float> offset2;
            RegTensor<int8_t> quantOut1;
            RegTensor<int8_t> quantOut2;
            MaskReg pregLoop;

            for (uint16_t k = 0; k < (uint16_t)rowsCount; k++) {
                uint32_t sreg0 = colsCount;
                uint32_t sreg1 = colsCount * 2;
                for (uint16_t i = 0; i < colsLoopCount; i++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadGammaBeta(gammaAddr, betaAddr, gamma, beta, pregLoop, i * vlFp32);
                    LoadQuantParams(scale1Addr, scale1, pregLoop, i * vlFp32);
                    CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, LoadQuantParams(scale2Addr, scale2, pregLoop, i * vlFp32));
                    CONST_CONDITIONAL_EXPR(
                        IS_OFFSET1_EXIST, LoadQuantParams(offset1Addr, offset1, pregLoop, i * vlFp32));
                    CONST_CONDITIONAL_EXPR(
                        IS_OFFSET2_EXIST, LoadQuantParams(offset2Addr, offset2, pregLoop, i * vlFp32));

                    DataCopy(x, ((__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(mean, ((__local_mem__ float*)meanAddr + k));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(rstd, ((__local_mem__ float*)rstdAddr + k));
                    Sub(x, x, mean, pregLoop);
                    Mul(y, x, rstd, pregLoop);
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
                        (__local_mem__ int8_t*)quant1OutAddr + i * vlFp32 + k * colsPerLoopAlign, quantOut1, pregLoop);
                    CONST_CONDITIONAL_EXPR(
                        IS_SCALE2_EXIST, (DataCopy<int8_t, StoreDist::DIST_PACK4_B32>(
                                             (__local_mem__ int8_t*)quant2OutAddr + i * vlFp32 + k * colsPerLoopAlign,
                                             quantOut2, pregLoop)));
                }
            }
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();

        int64_t inputOffset = 0;
        int64_t outputOffset = 0;
        int64_t xOffset = 0;
        int64_t meanOffset = 0;

        LocalTensor<X1_TYPE> gammaLocal = gammaQueue_.template AllocTensor<X1_TYPE>();
        LocalTensor<X1_TYPE> betaLocal = betaQueue_.template AllocTensor<X1_TYPE>();

        LocalTensor<SCALE_TYPE> scale1Local = scale1Queue_.template AllocTensor<SCALE_TYPE>();
        LocalTensor<SCALE_TYPE> scale2Local;
        LocalTensor<SCALE_TYPE> offset1Local;
        LocalTensor<SCALE_TYPE> offset2Local;
        CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, scale2Local, scale2Queue_.template AllocTensor<SCALE_TYPE>());
        CONST_CONDITIONAL_ASSIGN(IS_OFFSET1_EXIST, offset1Local, zeroOffset1Queue_.template AllocTensor<SCALE_TYPE>());
        CONST_CONDITIONAL_ASSIGN(IS_OFFSET2_EXIST, offset2Local, zeroOffset2Queue_.template AllocTensor<SCALE_TYPE>());

        LocalTensor<X1_TYPE> biasLocal;
        if constexpr (IS_BIAS_BROADCAST) {
            biasLocal = biasQueue_.template AllocTensor<X1_TYPE>();
            CopyBiasToUB(biasLocal, colsPerLoop_);
        }
        LocalTensor<float> binaryAddLocal = binaryAddBuf_.Get<float>();

        for (int64_t i = 0; i < rowsLoopCount_; i++) {
            int32_t rowsCount = i < rowsLoopCount_ - 1 ? rowsPerLoop_ : rowsTail_;

            LocalTensor<X1_TYPE> x1Local = x1Queue_.template AllocTensor<X1_TYPE>();
            LocalTensor<X1_TYPE> x2Local = x2Queue_.template AllocTensor<X1_TYPE>();
            if constexpr (IS_BIAS_ELEWISE) {
                biasLocal = biasQueue_.template AllocTensor<X1_TYPE>();
            }
            // copy in x1, x2, bias
            CopyInputsToUB(x1Local, x2Local, biasLocal, inputOffset, colsPerLoop_, rowsCount);

            x1Local = x1Queue_.template DeQue<X1_TYPE>();
            x2Local = x2Queue_.template DeQue<X1_TYPE>();

            if constexpr (IS_BIAS_ELEWISE) {
                biasLocal = biasQueue_.template DeQue<X1_TYPE>();
            } else if constexpr (IS_BIAS_BROADCAST) {
                if (i == 0) {
                    biasLocal = biasQueue_.template DeQue<X1_TYPE>();
                }
            }

            LocalTensor<X1_TYPE> xOutLocal = xQueue_.template AllocTensor<X1_TYPE>();
            LocalTensor<float> x32Local = x32Queue_.Get<float>();
            LocalTensor<float> meanLocal = meanBuf_.Get<float>();
            LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();

            if (colsPerLoop_ <= vlFp32_) {
                VFCalcMeanVarFast<X1_TYPE, TILING_KEY>(
                    x1Local, x2Local, biasLocal, xOutLocal, x32Local, meanLocal, rstdLocal, rowsCount, powerOfTwo_,
                    colsPerLoop_, colsPerLoopAlign_, vlFp32_);
            } else {
                VFCalcMeanVar<X1_TYPE, TILING_KEY>(
                    x1Local, x2Local, biasLocal, xOutLocal, x32Local, meanLocal, rstdLocal, binaryAddLocal, rowsCount,
                    powerOfTwo_, colsPerLoop_, colsPerLoopAlign_, vlFp32_, binaryAddLastNum_, binaryAddNum_,
                    binaryAddK_);
            }

            x1Queue_.FreeTensor(x1Local);
            x2Queue_.FreeTensor(x2Local);
            if constexpr (IS_BIAS_ELEWISE) {
                biasQueue_.FreeTensor(biasLocal);
            }
            // copy out x
            if (outputX_) {
                xQueue_.EnQue(xOutLocal);
                xOutLocal = xQueue_.template DeQue<X1_TYPE>();
                CopyXToGm(xOutLocal, inputOffset, colsPerLoop_, rowsCount);
            }
            xQueue_.FreeTensor(xOutLocal);

            // // calc rstd
            VFCalcRstd(rstdLocal, rowsCount, vlFp32_, eps_);

            // copy in gamma, beta
            if (i == 0) {
                CopyGammaAndBetaToUB(gammaLocal, betaLocal, 0, colsPerLoop_);
                CopyQuantParams2UB(scale1Local, scale2Local, offset1Local, offset2Local, 0, colsPerLoop_);
                gammaLocal = gammaQueue_.template DeQue<X1_TYPE>();
                betaLocal = betaQueue_.template DeQue<X1_TYPE>();

                scale1Local = scale1Queue_.template DeQue<SCALE_TYPE>();
                CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, scale2Local, scale2Queue_.template DeQue<SCALE_TYPE>());
                CONST_CONDITIONAL_ASSIGN(
                    IS_OFFSET1_EXIST, offset1Local, zeroOffset1Queue_.template DeQue<SCALE_TYPE>());
                CONST_CONDITIONAL_ASSIGN(
                    IS_OFFSET2_EXIST, offset2Local, zeroOffset2Queue_.template DeQue<SCALE_TYPE>());
            }
            LocalTensor<int8_t> y1Local = y1Queue_.template AllocTensor<int8_t>();
            LocalTensor<int8_t> y2Local;
            CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, y2Local, y2Queue_.template AllocTensor<int8_t>());

            // calc y with VF
            VFCalcYQuant(
                x32Local, betaLocal, gammaLocal, meanLocal, rstdLocal, scale1Local, scale2Local, offset1Local,
                offset2Local, xOutLocal, y1Local, y2Local, rowsCount, colsPerLoop_, colsPerLoopAlign_, vlFp32_);

            // copy out y
            y1Queue_.EnQue(y1Local);
            y1Local = y1Queue_.template DeQue<int8_t>();
            if constexpr (IS_SCALE2_EXIST) {
                y2Queue_.EnQue(y2Local);
                y2Local = y2Queue_.template DeQue<int8_t>();
            }
            CopyYToGm(y1Local, y2Local, outputOffset, colsPerLoop_, rowsCount);

            inputOffset += rowsCount * colsPerLoop_;
            outputOffset = inputOffset;
            meanOffset += rowsCount;

            y1Queue_.FreeTensor(y1Local);
            CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, y2Queue_.FreeTensor(y2Local));
        }
        if constexpr (IS_BIAS_BROADCAST) {
            biasQueue_.FreeTensor(biasLocal);
        }
        gammaQueue_.FreeTensor(gammaLocal);
        betaQueue_.FreeTensor(betaLocal);
        scale1Queue_.FreeTensor(scale1Local);
        CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, scale2Queue_.FreeTensor(scale2Local));
        CONST_CONDITIONAL_EXPR(IS_OFFSET1_EXIST, zeroOffset1Queue_.FreeTensor(offset1Local));
        CONST_CONDITIONAL_EXPR(IS_OFFSET2_EXIST, zeroOffset2Queue_.FreeTensor(offset2Local));
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasQueue_;
    TQue<QuePosition::VECIN, 1> gammaQueue_;
    TQue<QuePosition::VECIN, 1> betaQueue_;

    TQue<QuePosition::VECIN, 1> scale1Queue_;
    TQue<QuePosition::VECIN, 1> scale2Queue_;
    TQue<QuePosition::VECIN, 1> zeroOffset1Queue_;
    TQue<QuePosition::VECIN, 1> zeroOffset2Queue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> y1Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> y2Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xQueue_;

    TBuf<TPosition::VECCALC> binaryAddBuf_;
    TBuf<QuePosition::VECCALC> x32Queue_;

    TBuf<TPosition::VECCALC> meanBuf_;
    TBuf<TPosition::VECCALC> rstdBuf_;

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
    bool outputX_;

    TPipe* pipe_ = nullptr;

    int32_t dmaStride_;
};
} // namespace AddLayerNormQuantRegbase
#endif // ADD_LAYER_NORM_STATIC_QUANT_REGBASE_FULL_LOAD_H