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
 * \file add_layer_norm_dynamic_quant_regbase_full_load_kernel.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_DYNAMIC_QUANT_REGBASE_FULL_LOAD_H
#define ADD_LAYER_NORM_DYNAMIC_QUANT_REGBASE_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "add_layer_norm_quant_regbase_helper.h"

namespace AddLayerNormQuantRegbase {
template <typename X1_TYPE, typename SMOOTH_TYPE, int32_t TILING_KEY, int32_t OPT_CODE, int32_t BUFFER_NUM = 1>
class KernelAddLayerNormDynamicQuantRegbaseFullLoad {
#define COLS_PER_LOOP_ (tilingData_->colsPerLoop)
#define EPS_ (tilingData_->eps)
#define BINARY_ADD_NUM_ (tilingData_->binaryAddNum)
#define BINARY_ADD_K_ (tilingData_->binaryAddK)
#define BINARY_ADD_LAST_NUM_ (tilingData_->binaryAddLastNum)
#define OUT_PUT_X_ (tilingData_->outputX)
#define ROWS_PER_LOOP_ (tilingData_->rowsPerLoop)

public:
    __aicore__ inline KernelAddLayerNormDynamicQuantRegbaseFullLoad(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR y1,
        GM_ADDR y2, GM_ADDR x, GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace,
        const AddLayerNormQuantRegbaseTilingData* tilingData)
    {
        uint32_t coreIdx = GetBlockIdx();
        tilingData_ = tilingData;

        powerOfTwo_ = 1;
        while (powerOfTwo_ < COLS_PER_LOOP_) {
            powerOfTwo_ *= AddLayerNorm::TWO;
        }

        int32_t rem = COLS_PER_LOOP_ % blockSize_;
        int32_t eleNumPerBlock = blockSize_ / sizeof(X1_TYPE); // 8 or 16
        dmaStride_ = (rem == 0) ? 0 : (blockSize_ - rem) / eleNumPerBlock;

        uint64_t gmOffset;
        uint64_t outScaleOffset;
        if (coreIdx < GetBlockNum() - 1) {
            // non-tail cores
            rowsPerCore_ = tilingData->rowsPerCore;
        } else {
            // tail cores
            rowsPerCore_ = tilingData->rowsPerTailCore;
        }
        gmOffset = (tilingData->rowsPerCore * COLS_PER_LOOP_) * coreIdx;
        outScaleOffset = gmOffset / COLS_PER_LOOP_;
        rowsTail_ = (rowsPerCore_ % ROWS_PER_LOOP_ == 0) ? ROWS_PER_LOOP_ : (rowsPerCore_ % ROWS_PER_LOOP_);
        rowsLoopCount_ = CEIL_DIV(rowsPerCore_, ROWS_PER_LOOP_);

        x1Gm_.SetGlobalBuffer((__gm__ X1_TYPE*)(x1) + gmOffset);
        x2Gm_.SetGlobalBuffer((__gm__ X1_TYPE*)(x2) + gmOffset);
        if constexpr (IS_BIAS_ELEWISE) {
            biasGm_.SetGlobalBuffer((__gm__ X1_TYPE*)(bias) + gmOffset);
        } else if constexpr (IS_BIAS_BROADCAST) {
            biasGm_.SetGlobalBuffer((__gm__ X1_TYPE*)bias);
        }
        gammaGm_.SetGlobalBuffer((__gm__ X1_TYPE*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ X1_TYPE*)beta);
        CONST_CONDITIONAL_EXPR(IS_SCALE1_EXIST, smooth1Gm_.SetGlobalBuffer((__gm__ SMOOTH_TYPE*)smooth1));
        CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, smooth2Gm_.SetGlobalBuffer((__gm__ SMOOTH_TYPE*)smooth2));

        this->useWs_ = ((1 == tilingData->rowsPerCore) && (GetBlockNum() > 1));

        y1Gm_.SetGlobalBuffer((__gm__ int8_t*)(y1) + gmOffset);
        xGm_.SetGlobalBuffer((__gm__ X1_TYPE*)(x) + gmOffset);
        outScale1Gm_.SetGlobalBuffer((__gm__ float*)(outScale1) + outScaleOffset);
        ws1Gm_.SetGlobalBuffer((__gm__ float*)(workspace + GetBlockIdx() * 128));
        if constexpr (IS_SCALE2_EXIST) {
            y2Gm_.SetGlobalBuffer((__gm__ int8_t*)(y2) + gmOffset);
            outScale2Gm_.SetGlobalBuffer((__gm__ float*)(outScale2) + outScaleOffset);
            ws2Gm_.SetGlobalBuffer((__gm__ float*)(workspace + (GetBlockIdx() + GetBlockNum()) * 128));
        }

        colsPerLoopAlign_ = BLOCK_ALIGN(COLS_PER_LOOP_, blockSize_); // 32 element aligned

        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, (ROWS_PER_LOOP_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(x2Queue_, BUFFER_NUM, (ROWS_PER_LOOP_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        if constexpr (IS_BIAS_ELEWISE) {
            pipe_->InitBuffer(biasQueue_, BUFFER_NUM, (ROWS_PER_LOOP_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        } else if constexpr (IS_BIAS_BROADCAST) {
            pipe_->InitBuffer(biasQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        }
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, (ROWS_PER_LOOP_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(y1Queue_, BUFFER_NUM, (ROWS_PER_LOOP_ * colsPerLoopAlign_ * sizeof(int8_t)));
        pipe_->InitBuffer(outScale1Queue_, BUFFER_NUM, (ROWS_PER_LOOP_ * sizeof(float)));
        pipe_->InitBuffer(x32Buf_, (ROWS_PER_LOOP_ * colsPerLoopAlign_ * sizeof(float)));
        pipe_->InitBuffer(betaQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(gammaQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));

        CONST_CONDITIONAL_EXPR(
            IS_SCALE1_EXIST, pipe_->InitBuffer(smooth1Queue_, 1, (colsPerLoopAlign_ * sizeof(SMOOTH_TYPE))));
        CONST_CONDITIONAL_EXPR(
            IS_SCALE2_EXIST, pipe_->InitBuffer(smooth2Queue_, 1, (colsPerLoopAlign_ * sizeof(SMOOTH_TYPE))));

        CONST_CONDITIONAL_EXPR(
            IS_SCALE2_EXIST,
            pipe_->InitBuffer(y2Queue_, BUFFER_NUM, (ROWS_PER_LOOP_ * colsPerLoopAlign_ * sizeof(int8_t))));
        CONST_CONDITIONAL_EXPR(
            IS_SCALE2_EXIST, pipe_->InitBuffer(outScale2Queue_, BUFFER_NUM, (ROWS_PER_LOOP_ * sizeof(float))));

        int64_t binaryAddBufSize = BLOCK_ALIGN((BINARY_ADD_NUM_ / vlFp32_) * sizeof(float), blockSize_);
        if (binaryAddBufSize > 0) {
            pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize);
        }
        pipe_->InitBuffer(meanBuf_, BLOCK_ALIGN(ROWS_PER_LOOP_ * sizeof(float), blockSize_));
        pipe_->InitBuffer(rstdBuf_, BLOCK_ALIGN(ROWS_PER_LOOP_ * sizeof(float), blockSize_));
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

    __aicore__ inline void CopyOutScaleToGm(
        LocalTensor<float> outScale1Local, LocalTensor<float> outScale2Local, int64_t outScaleOffset, int32_t rowsCount)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = rowsCount * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        if (useWs_) {
            DataCopyPad(ws1Gm_[outScaleOffset], outScale1Local, dataCopyParams);
            CONST_CONDITIONAL_EXPR(
                IS_SCALE2_EXIST, DataCopyPad(ws2Gm_[outScaleOffset], outScale2Local, dataCopyParams));
        } else {
            DataCopyPad(outScale1Gm_[outScaleOffset], outScale1Local, dataCopyParams);
            CONST_CONDITIONAL_EXPR(
                IS_SCALE2_EXIST, DataCopyPad(outScale2Gm_[outScaleOffset], outScale2Local, dataCopyParams));
        }
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
        LocalTensor<SMOOTH_TYPE> smooth1Local, LocalTensor<SMOOTH_TYPE> smooth2Local, int64_t offset, int32_t copyLen)
    {
        if constexpr (IS_SCALE1_EXIST) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = copyLen * sizeof(SMOOTH_TYPE);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(smooth1Local, smooth1Gm_[offset], dataCopyParams, {});
            smooth1Queue_.EnQue(smooth1Local);

            if constexpr (IS_SCALE2_EXIST) {
                DataCopyPad(smooth2Local, smooth2Gm_[offset], dataCopyParams, {});
                smooth2Queue_.EnQue(smooth2Local);
            }
        }
    }

    static __aicore__ inline void VFCalcYNormlization(
        LocalTensor<float>& x32Local, LocalTensor<X1_TYPE>& betaLocal, LocalTensor<X1_TYPE>& gammaLocal,
        LocalTensor<float>& meanLocal, LocalTensor<float>& rstdLocal, LocalTensor<SMOOTH_TYPE>& smooth1Local,
        LocalTensor<SMOOTH_TYPE>& smooth2Local, LocalTensor<X1_TYPE>& yOutLocal, LocalTensor<float>& outScale1Local,
        LocalTensor<float>& outScale2Local, uint32_t rowsCount, uint32_t colsCount, uint32_t colsPerLoopAlign,
        uint32_t vlFp32)
    {
        __local_mem__ float* x32Addr = (__local_mem__ float*)x32Local[0].GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal[0].GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal[0].GetPhyAddr();
        __local_mem__ X1_TYPE* betaAddr = (__local_mem__ X1_TYPE*)betaLocal[0].GetPhyAddr();
        __local_mem__ X1_TYPE* gammaAddr = (__local_mem__ X1_TYPE*)gammaLocal[0].GetPhyAddr();
        __local_mem__ float* outScale1Addr = (__local_mem__ float*)outScale1Local[0].GetPhyAddr();

        __local_mem__ SMOOTH_TYPE* smooth1Addr;
        __local_mem__ float* outScale2Addr;
        __local_mem__ SMOOTH_TYPE* smooth2Addr;

        CONST_CONDITIONAL_ASSIGN(
            IS_SCALE1_EXIST, smooth1Addr, (__local_mem__ SMOOTH_TYPE*)smooth1Local[0].GetPhyAddr());
        CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, outScale2Addr, (__local_mem__ float*)outScale2Local[0].GetPhyAddr());
        CONST_CONDITIONAL_ASSIGN(
            IS_SCALE2_EXIST, smooth2Addr, (__local_mem__ SMOOTH_TYPE*)smooth2Local[0].GetPhyAddr());

        uint16_t colsLoopCount = CEIL_DIV(colsCount, vlFp32);
        constexpr float quantConst = static_cast<float>(1.0 / 127.0);

        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> beta;
            RegTensor<float> gamma;
            RegTensor<float> smooth1;
            RegTensor<float> smooth2;

            RegTensor<float> tmpMax1;
            RegTensor<float> tmpMax2;

            RegTensor<float> outNorm1;
            RegTensor<float> outNorm2;

            RegTensor<float> outScale1;
            RegTensor<float> outScale2;

            MaskReg pregLoop;
            MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            for (uint16_t k = 0; k < (uint16_t)rowsCount; k++) {
                uint32_t sreg0 = colsCount;
                uint32_t sreg1 = colsCount * 2;
                Duplicate(tmpMax1, static_cast<float>(0.0));
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, Duplicate(tmpMax2, static_cast<float>(0.0)));

                for (uint16_t i = 0; i < colsLoopCount; i++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadGammaBeta(gammaAddr, betaAddr, gamma, beta, pregLoop, i * vlFp32);
                    CONST_CONDITIONAL_EXPR(
                        IS_SCALE1_EXIST, LoadQuantParams(smooth1Addr, smooth1, pregLoop, i * vlFp32));
                    CONST_CONDITIONAL_EXPR(
                        IS_SCALE2_EXIST, LoadQuantParams(smooth2Addr, smooth2, pregLoop, i * vlFp32));

                    DataCopy(x, ((__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(mean, ((__local_mem__ float*)meanAddr + k));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(rstd, ((__local_mem__ float*)rstdAddr + k));
                    Sub(x, x, mean, pregLoop);
                    Mul(y, x, rstd, pregLoop);
                    Mul(y, y, gamma, pregLoop);
                    Add(y, y, beta, pregLoop); // LayerNorm result

                    // Do LocalMax1:
                    if constexpr (IS_SCALE2_EXIST) {
                        Mul(outNorm1, y, smooth1, pregLoop);
                        Mul(outNorm2, y, smooth2, pregLoop);
                        Abs(outNorm1, outNorm1, pregLoop);
                        Abs(outNorm2, outNorm2, pregLoop);
                        Max(tmpMax1, tmpMax1, outNorm1, pregAll);
                        Max(tmpMax2, tmpMax2, outNorm2, pregAll);
                    } else {
                        CONST_CONDITIONAL_EXPR(IS_SCALE1_EXIST, Mul(y, y, smooth1, pregLoop));
                        Abs(outNorm1, y, pregLoop);
                        Max(tmpMax1, tmpMax1, outNorm1, pregAll);
                    }
                    DataCopy(((__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign), y, pregLoop);
                }
                // compute row max
                ReduceMax(outScale1, tmpMax1, pregAll);
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, ReduceMax(outScale2, tmpMax2, pregAll));
                // compute scale
                Muls(outScale1, outScale1, quantConst, pregOne);
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, Muls(outScale2, outScale2, quantConst, pregOne));

                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float*)outScale1Addr + k, outScale1, pregOne);
                CONST_CONDITIONAL_EXPR(
                    IS_SCALE2_EXIST, (DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                                         (__local_mem__ float*)outScale2Addr + k, outScale2, pregOne)));
            }
        }
    }

    static __aicore__ inline void VFCalcYQuant(
        LocalTensor<float>& x32Local, LocalTensor<SMOOTH_TYPE>& smooth1Local, LocalTensor<SMOOTH_TYPE>& smooth2Local,
        LocalTensor<X1_TYPE>& yOutLocal, LocalTensor<float>& outScale1Local, LocalTensor<float>& outScale2Local,
        LocalTensor<int8_t>& y1Local, LocalTensor<int8_t>& y2Local, uint32_t rowsCount, uint32_t colsCount,
        uint32_t colsPerLoopAlign, uint32_t vlFp32)
    {
        __local_mem__ float* x32Addr = (__local_mem__ float*)x32Local[0].GetPhyAddr();
        __local_mem__ float* outScale1Addr = (__local_mem__ float*)outScale1Local[0].GetPhyAddr();
        __local_mem__ int8_t* y1Addr = (__local_mem__ int8_t*)y1Local[0].GetPhyAddr();

        __local_mem__ SMOOTH_TYPE* smooth1Addr;
        __local_mem__ SMOOTH_TYPE* smooth2Addr;
        __local_mem__ float* outScale2Addr;
        __local_mem__ int8_t* y2Addr;

        CONST_CONDITIONAL_ASSIGN(
            IS_SCALE1_EXIST, smooth1Addr, (__local_mem__ SMOOTH_TYPE*)smooth1Local[0].GetPhyAddr());
        CONST_CONDITIONAL_ASSIGN(
            IS_SCALE2_EXIST, smooth2Addr, (__local_mem__ SMOOTH_TYPE*)smooth2Local[0].GetPhyAddr());
        CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, outScale2Addr, (__local_mem__ float*)outScale2Local[0].GetPhyAddr());
        CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, y2Addr, (__local_mem__ int8_t*)y2Local[0].GetPhyAddr());

        uint16_t colsLoopCount = CEIL_DIV(colsCount, vlFp32);
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> outScale1;
            RegTensor<float> outScale2;
            RegTensor<float> smooth1;
            RegTensor<float> smooth2;
            RegTensor<int8_t> q1;
            RegTensor<int8_t> q2;

            MaskReg pregLoop;

            for (uint16_t k = 0; k < (uint16_t)rowsCount; k++) {
                uint32_t sreg0 = colsCount;
                uint32_t sreg1 = colsCount * 2;
                for (uint16_t i = 0; i < colsLoopCount; i++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    DataCopy(x, ((__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(outScale1, ((__local_mem__ float*)outScale1Addr + k));

                    CONST_CONDITIONAL_EXPR(
                        IS_SCALE2_EXIST, (DataCopy<float, LoadDist::DIST_BRC_B32>(
                                             outScale2, ((__local_mem__ float*)outScale2Addr + k))));

                    if constexpr (IS_SCALE2_EXIST) {
                        // Dual Quant need to re-compute smooth
                        LoadQuantParams(smooth1Addr, smooth1, pregLoop, i * vlFp32);
                        LoadQuantParams(smooth2Addr, smooth2, pregLoop, i * vlFp32);
                        ComputeDynamicQuantWithSmooth(q1, x, smooth1, outScale1, pregLoop);
                        ComputeDynamicQuantWithSmooth(q2, x, smooth2, outScale2, pregLoop);
                    } else {
                        ComputeDynamicQuantWithOutSmooth(q1, x, outScale1, pregLoop);
                    }
                    DataCopy<int8_t, StoreDist::DIST_PACK4_B32>(
                        (__local_mem__ int8_t*)y1Addr + i * vlFp32 + k * colsPerLoopAlign, q1, pregLoop);
                    CONST_CONDITIONAL_EXPR(
                        IS_SCALE2_EXIST,
                        (DataCopy<int8_t, StoreDist::DIST_PACK4_B32>(
                            (__local_mem__ int8_t*)y2Addr + i * vlFp32 + k * colsPerLoopAlign, q2, pregLoop)));
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
        int64_t outScaleOffset = 0;

        LocalTensor<float> x32Local = x32Buf_.Get<float>();
        LocalTensor<float> meanLocal = meanBuf_.Get<float>();
        LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();

        LocalTensor<X1_TYPE> gammaLocal = gammaQueue_.template AllocTensor<X1_TYPE>();
        LocalTensor<X1_TYPE> betaLocal = betaQueue_.template AllocTensor<X1_TYPE>();

        LocalTensor<SMOOTH_TYPE> smooth1Local;
        LocalTensor<SMOOTH_TYPE> smooth2Local;
        CONST_CONDITIONAL_ASSIGN(IS_SCALE1_EXIST, smooth1Local, smooth1Queue_.template AllocTensor<SMOOTH_TYPE>());
        CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, smooth2Local, smooth2Queue_.template AllocTensor<SMOOTH_TYPE>());

        LocalTensor<X1_TYPE> biasLocal;
        if constexpr (IS_BIAS_BROADCAST) {
            biasLocal = biasQueue_.template AllocTensor<X1_TYPE>();
            CopyBiasToUB(biasLocal, COLS_PER_LOOP_);
        }
        LocalTensor<float> binaryAddLocal = binaryAddBuf_.Get<float>();

        for (int64_t i = 0; i < rowsLoopCount_; i++) {
            int32_t rowsCount = i < rowsLoopCount_ - 1 ? ROWS_PER_LOOP_ : rowsTail_;

            LocalTensor<X1_TYPE> x1Local = x1Queue_.template AllocTensor<X1_TYPE>();
            LocalTensor<X1_TYPE> x2Local = x2Queue_.template AllocTensor<X1_TYPE>();
            if constexpr (IS_BIAS_ELEWISE) {
                biasLocal = biasQueue_.template AllocTensor<X1_TYPE>();
            }
            // copy in x1, x2, bias
            CopyInputsToUB(x1Local, x2Local, biasLocal, inputOffset, COLS_PER_LOOP_, rowsCount);

            x1Local = x1Queue_.template DeQue<X1_TYPE>();
            x2Local = x2Queue_.template DeQue<X1_TYPE>();

            if constexpr (IS_BIAS_ELEWISE) {
                biasLocal = biasQueue_.template DeQue<X1_TYPE>();
            } else if constexpr (IS_BIAS_BROADCAST) {
                if (unlikely((i == 0))) {
                    biasLocal = biasQueue_.template DeQue<X1_TYPE>();
                }
            }

            LocalTensor<X1_TYPE> xOutLocal = xQueue_.template AllocTensor<X1_TYPE>();

            if (COLS_PER_LOOP_ <= vlFp32_) {
                VFCalcMeanVarFast<X1_TYPE, TILING_KEY>(
                    x1Local, x2Local, biasLocal, xOutLocal, x32Local, meanLocal, rstdLocal, rowsCount, powerOfTwo_,
                    COLS_PER_LOOP_, colsPerLoopAlign_, vlFp32_);
            } else {
                VFCalcMeanVar<X1_TYPE, TILING_KEY>(
                    x1Local, x2Local, biasLocal, xOutLocal, x32Local, meanLocal, rstdLocal, binaryAddLocal, rowsCount,
                    powerOfTwo_, COLS_PER_LOOP_, colsPerLoopAlign_, vlFp32_, BINARY_ADD_LAST_NUM_, BINARY_ADD_NUM_,
                    BINARY_ADD_K_);
            }

            x1Queue_.FreeTensor(x1Local);
            x2Queue_.FreeTensor(x2Local);
            if constexpr (IS_BIAS_ELEWISE) {
                biasQueue_.FreeTensor(biasLocal);
            }
            // copy out x
            if (OUT_PUT_X_) {
                xQueue_.EnQue(xOutLocal);
                xOutLocal = xQueue_.template DeQue<X1_TYPE>();
                CopyXToGm(xOutLocal, inputOffset, COLS_PER_LOOP_, rowsCount);
            }
            xQueue_.FreeTensor(xOutLocal);

            // // calc rstd
            VFCalcRstd(rstdLocal, rowsCount, vlFp32_, EPS_);

            // copy in gamma, beta
            if (unlikely((i == 0))) {
                CopyGammaAndBetaToUB(gammaLocal, betaLocal, 0, COLS_PER_LOOP_);
                CopyQuantParams2UB(smooth1Local, smooth2Local, 0, COLS_PER_LOOP_);
                gammaLocal = gammaQueue_.template DeQue<X1_TYPE>();
                betaLocal = betaQueue_.template DeQue<X1_TYPE>();
                CONST_CONDITIONAL_ASSIGN(IS_SCALE1_EXIST, smooth1Local, smooth1Queue_.template DeQue<SMOOTH_TYPE>());
                CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, smooth2Local, smooth2Queue_.template DeQue<SMOOTH_TYPE>());
            }

            LocalTensor<float> outScale1Local = outScale1Queue_.template AllocTensor<float>();
            LocalTensor<float> outScale2Local;
            CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, outScale2Local, outScale2Queue_.template AllocTensor<float>());

            // calc outScale with VF
            VFCalcYNormlization(
                x32Local, betaLocal, gammaLocal, meanLocal, rstdLocal, smooth1Local, smooth2Local, xOutLocal,
                outScale1Local, outScale2Local, rowsCount, COLS_PER_LOOP_, colsPerLoopAlign_, vlFp32_);

            // calc quant with VF
            LocalTensor<int8_t> y1Local = y1Queue_.template AllocTensor<int8_t>();
            LocalTensor<int8_t> y2Local;
            CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, y2Local, y2Queue_.template AllocTensor<int8_t>());
            VFCalcYQuant(
                x32Local, smooth1Local, smooth2Local, xOutLocal, outScale1Local, outScale2Local, y1Local, y2Local,
                rowsCount, COLS_PER_LOOP_, colsPerLoopAlign_, vlFp32_);

            // copy out y
            y1Queue_.EnQue(y1Local);
            outScale1Queue_.EnQue(outScale1Local);
            y1Local = y1Queue_.template DeQue<int8_t>();
            outScale1Local = outScale1Queue_.template DeQue<float>();
            if constexpr (IS_SCALE2_EXIST) {
                y2Queue_.EnQue(y2Local);
                outScale2Queue_.EnQue(outScale2Local);
                y2Local = y2Queue_.template DeQue<int8_t>();
                outScale2Local = outScale2Queue_.template DeQue<float>();
            }
            CopyYToGm(y1Local, y2Local, outputOffset, COLS_PER_LOOP_, rowsCount);
            CopyOutScaleToGm(outScale1Local, outScale2Local, outScaleOffset, rowsCount);
            inputOffset += rowsCount * COLS_PER_LOOP_;
            outputOffset = inputOffset;
            outScaleOffset += rowsCount;

            y1Queue_.FreeTensor(y1Local);
            outScale1Queue_.FreeTensor(outScale1Local);
            CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, y2Queue_.FreeTensor(y2Local));
            CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, outScale2Queue_.FreeTensor(outScale2Local));
        }

        gammaQueue_.FreeTensor(gammaLocal);
        betaQueue_.FreeTensor(betaLocal);
        CONST_CONDITIONAL_EXPR(IS_BIAS_BROADCAST, biasQueue_.FreeTensor(biasLocal));
        CONST_CONDITIONAL_EXPR(IS_SCALE1_EXIST, smooth1Queue_.FreeTensor(smooth1Local));
        CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, smooth2Queue_.FreeTensor(smooth2Local));

        if (useWs_) {
            SyncAll();
            LocalTensor<float> tmpLocal = x32Buf_.Get<float>();
            MergeOutScales<IS_SCALE2_EXIST>(tmpLocal, ws1Gm_, ws2Gm_, outScale1Gm_, outScale2Gm_);
        }
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasQueue_;
    TQue<QuePosition::VECIN, 1> gammaQueue_;
    TQue<QuePosition::VECIN, 1> betaQueue_;
    TQue<QuePosition::VECIN, 1> smooth1Queue_;
    TQue<QuePosition::VECIN, 1> smooth2Queue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> y1Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> y2Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outScale1Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outScale2Queue_;

    TBuf<TPosition::VECCALC> binaryAddBuf_;
    TBuf<QuePosition::VECCALC> x32Buf_;

    TBuf<TPosition::VECCALC> meanBuf_;
    TBuf<TPosition::VECCALC> rstdBuf_;

    GlobalTensor<X1_TYPE> x1Gm_;
    GlobalTensor<X1_TYPE> x2Gm_;
    GlobalTensor<X1_TYPE> biasGm_;
    GlobalTensor<X1_TYPE> gammaGm_;
    GlobalTensor<X1_TYPE> betaGm_;

    GlobalTensor<SMOOTH_TYPE> smooth1Gm_;
    GlobalTensor<SMOOTH_TYPE> smooth2Gm_;

    GlobalTensor<int8_t> y1Gm_;
    GlobalTensor<int8_t> y2Gm_;
    GlobalTensor<X1_TYPE> xGm_;
    GlobalTensor<float> outScale1Gm_;
    GlobalTensor<float> outScale2Gm_;
    GlobalTensor<float> ws1Gm_;
    GlobalTensor<float> ws2Gm_;

    int64_t colsPerLoopAlign_;
    int64_t colsPerLoopAlignB16_;
    int64_t colsPerLoopAlignB32_;
    int64_t colsPerLoopAlignBias_;
    int64_t rowsPerCore_;
    int64_t rowsTail_;
    int64_t rowsLoopCount_;
    int64_t powerOfTwo_;
    bool useWs_;
    int32_t dmaStride_;

    TPipe* pipe_ = nullptr;
    const AddLayerNormQuantRegbaseTilingData* tilingData_;
};
#undef COLS_PER_LOOP_
#undef EPS_
#undef BINARY_ADD_NUM_
#undef BINARY_ADD_K_
#undef BINARY_ADD_LAST_NUM_
#undef OUT_PUT_X_
} // namespace AddLayerNormQuantRegbase
#endif // ADD_LAYER_NORM_DYNAMIC_QUANT_REGBASE_FULL_LOAD_H