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
 * \file add_layer_norm_dynamic_quant_regbase_welford_kernel.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_DYNAMIC_QUANT_REGBASE_WELFORD_H
#define ADD_LAYER_NORM_DYNAMIC_QUANT_REGBASE_WELFORD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "add_layer_norm_quant_regbase_helper.h"

namespace AddLayerNormQuantRegbase {
#define COLS_ (tilingData_->cols)
#define COLS_TAIL_ (tilingData_->colsTail)
#define COLS_PER_LOOP_ (tilingData_->colsPerLoop)
#define COLS_LOOP_COUNT_ (tilingData_->colsLoopCount)
#define EPS_ (tilingData_->eps)
#define BINARY_ADD_NUM_ (tilingData_->binaryAddNum)
#define BINARY_ADD_K_ (tilingData_->binaryAddK)
#define BINARY_ADD_LAST_NUM_ (tilingData_->binaryAddLastNum)
#define OUTPUT_X_ (tilingData_->outputX)

template <typename X1_TYPE, typename SCALE_TYPE, int32_t TILING_KEY, int32_t OPT_CODE, int32_t BUFFER_NUM = 1>
class KernelAddLayerNormDynamicQuantRegbaseWelford {
public:
    __aicore__ inline KernelAddLayerNormDynamicQuantRegbaseWelford(TPipe* pipe)
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

        outNums_ = (IS_SCALE2_EXIST) ? AddLayerNorm::TWO : 1;
        uint64_t gmOffset = (tilingData->rowsPerCore * COLS_) * coreIdx;
        uint64_t outScaleOffset = tilingData->rowsPerCore * coreIdx;
        // each core need 2 row * buffnum ws
        uint64_t wsOffset = coreIdx * BUFFER_NUM * outNums_ * COLS_;

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
        CONST_CONDITIONAL_EXPR(IS_SCALE1_EXIST, smooth1Gm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)smooth1));
        CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, smooth2Gm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)smooth2));

        y1Gm_.SetGlobalBuffer((__gm__ int8_t*)(y1) + gmOffset);
        xGm_.SetGlobalBuffer((__gm__ X1_TYPE*)(x) + gmOffset);
        outScale1Gm_.SetGlobalBuffer((__gm__ float*)outScale1 + outScaleOffset);
        if constexpr (IS_SCALE2_EXIST) {
            y2Gm_.SetGlobalBuffer((__gm__ int8_t*)(y2) + gmOffset);
            outScale2Gm_.SetGlobalBuffer((__gm__ float*)outScale2 + outScaleOffset);
        }

        workspaceGm_.SetGlobalBuffer((__gm__ float*)workspace + wsOffset);

        colsPerLoopAlign_ = BLOCK_ALIGN(COLS_PER_LOOP_ * sizeof(X1_TYPE), blockSize_) / sizeof(X1_TYPE);

        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(x2Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            pipe_->InitBuffer(biasQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        }
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(y1Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(int8_t)));
        pipe_->InitBuffer(betaQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(outScale1Queue_, BUFFER_NUM, (blockSize_));

        CONST_CONDITIONAL_EXPR(
            IS_SCALE1_EXIST, pipe_->InitBuffer(smooth1Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(SCALE_TYPE))));
        if constexpr (IS_SCALE2_EXIST) {
            pipe_->InitBuffer(smooth2Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(SCALE_TYPE)));
            pipe_->InitBuffer(y2Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(int8_t)));
            pipe_->InitBuffer(outScale2Queue_, BUFFER_NUM, (blockSize_));
        }

        // should be 2 for mean/rstd tmp buf
        pipe_->InitBuffer(tmpBuf_, 2 * outNums_ * colsPerLoopAlign_ * sizeof(float)); // always DB
        pipe_->InitBuffer(meanTmpBuf_, blockSize_);
        pipe_->InitBuffer(rstdTmpBuf_, blockSize_);
        pipe_->InitBuffer(tmpMax1_, blockSize_);
        pipe_->InitBuffer(tmpMax2_, blockSize_);

        int64_t binaryAddBufSize = BLOCK_ALIGN((BINARY_ADD_NUM_ / vlFp32_) * sizeof(float), blockSize_);
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

    __aicore__ inline void CopyNormToWorkspace(
        LocalTensor<float> smoothedNorm1Local, LocalTensor<float> smoothedNorm2Local, int64_t wsOffset, int32_t copyLen)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(workspaceGm_[wsOffset], smoothedNorm1Local, dataCopyParams);
        if constexpr (IS_SCALE2_EXIST) {
            DataCopyPad(workspaceGm_[wsOffset + COLS_], smoothedNorm2Local, dataCopyParams);
        }
    }

    __aicore__ inline void CopyInNormFromWorkspace(
        LocalTensor<float> smoothedNorm1Local, LocalTensor<float> smoothedNorm2Local, int64_t wsOffset, int32_t copyLen)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(smoothedNorm1Local, workspaceGm_[wsOffset], dataCopyParams, {});
        if constexpr (IS_SCALE2_EXIST) {
            DataCopyPad(smoothedNorm2Local, workspaceGm_[wsOffset + COLS_], dataCopyParams, {});
        }
    }

    __aicore__ inline void CopyOutOutScale(int64_t outScaleOffset)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 1 * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        LocalTensor<float> outScale1Local = outScale1Queue_.template DeQue<float>();
        DataCopyPad(outScale1Gm_[outScaleOffset], outScale1Local, dataCopyParams);
        outScale1Queue_.FreeTensor(outScale1Local);

        if constexpr (IS_SCALE2_EXIST) {
            LocalTensor<float> outScale2Local = outScale2Queue_.template DeQue<float>();
            DataCopyPad(outScale2Gm_[outScaleOffset], outScale2Local, dataCopyParams);
            outScale2Queue_.FreeTensor(outScale2Local);
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
        LocalTensor<SCALE_TYPE> scale1Local, LocalTensor<SCALE_TYPE> scale2Local, int64_t offset, int32_t copyLen)
    {
        if constexpr (IS_SCALE1_EXIST) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = copyLen * sizeof(SCALE_TYPE);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(scale1Local, smooth1Gm_[offset], dataCopyParams, {});
            smooth1Queue_.EnQue(scale1Local);
            if constexpr (IS_SCALE2_EXIST) {
                DataCopyPad(scale2Local, smooth2Gm_[offset], dataCopyParams, {});
                smooth2Queue_.EnQue(scale2Local);
            }
        }
    }

    static __aicore__ inline void VFCalcNormlization(
        LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local, LocalTensor<X1_TYPE>& biasLocal,
        LocalTensor<X1_TYPE>& betaLocal, LocalTensor<X1_TYPE>& gammaLocal, LocalTensor<SCALE_TYPE>& smooth1Local,
        LocalTensor<SCALE_TYPE>& smooth2Local, LocalTensor<float>& meanLocal, LocalTensor<float>& rstdLocal,
        LocalTensor<float>& smoothedNorm1Local, LocalTensor<float>& smoothedNorm2Local,
        LocalTensor<float>& tmpMaxLocal1, LocalTensor<float>& tmpMaxLocal2, uint32_t colsCount, uint32_t vlFp32)
    {
        // 13 addr
        __local_mem__ X1_TYPE* x1Addr = (__local_mem__ X1_TYPE*)x1Local[0].GetPhyAddr();
        __local_mem__ X1_TYPE* x2Addr = (__local_mem__ X1_TYPE*)x2Local[0].GetPhyAddr();
        __local_mem__ X1_TYPE* biasAddr;
        __local_mem__ X1_TYPE* betaAddr = (__local_mem__ X1_TYPE*)betaLocal[0].GetPhyAddr();
        __local_mem__ X1_TYPE* gammaAddr = (__local_mem__ X1_TYPE*)gammaLocal[0].GetPhyAddr();
        __local_mem__ SCALE_TYPE* smooth1Addr;
        __local_mem__ SCALE_TYPE* smooth2Addr;
        __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal[0].GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal[0].GetPhyAddr();
        __local_mem__ float* smoothedNorm1Addr = (__local_mem__ float*)smoothedNorm1Local[0].GetPhyAddr();
        __local_mem__ float* smoothedNorm2Addr;
        __local_mem__ float* localMax1Addr = (__local_mem__ float*)tmpMaxLocal1[0].GetPhyAddr();
        __local_mem__ float* localMax2Addr;

        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            biasAddr = (__local_mem__ X1_TYPE*)biasLocal[0].GetPhyAddr();
        }
        CONST_CONDITIONAL_ASSIGN(IS_SCALE1_EXIST, smooth1Addr, (__local_mem__ SCALE_TYPE*)smooth1Local[0].GetPhyAddr());
        if constexpr (IS_SCALE2_EXIST) {
            smooth2Addr = (__local_mem__ SCALE_TYPE*)smooth2Local[0].GetPhyAddr();
            smoothedNorm2Addr = (__local_mem__ float*)smoothedNorm2Local[0].GetPhyAddr();
            localMax2Addr = (__local_mem__ float*)tmpMaxLocal2[0].GetPhyAddr();
        }

        // uint32_t vlFp32 = vlFp32_;
        uint16_t colsLoopCount = CEIL_DIV(colsCount, vlFp32);

        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            RegTensor<float> beta;
            RegTensor<float> gamma;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> smooth1;
            RegTensor<float> smooth2;

            RegTensor<float> tmp1;
            RegTensor<float> tmp2;

            RegTensor<float> tmpMax1;
            RegTensor<float> tmpMax2;

            RegTensor<float> outNorm1;
            RegTensor<float> outNorm2;

            MaskReg pregLoop;
            MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            uint32_t sreg0 = colsCount;

            // Locad mean/rstd/localMax12
            DataCopy<float, LoadDist::DIST_BRC_B32>(mean, ((__local_mem__ float*)meanAddr));
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstd, ((__local_mem__ float*)rstdAddr));
            DataCopy<float, LoadDist::DIST_BRC_B32>(tmpMax1, ((__local_mem__ float*)localMax1Addr));
            if constexpr (IS_SCALE2_EXIST) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(tmpMax2, ((__local_mem__ float*)localMax2Addr));
            }
            for (uint16_t i = 0; i < colsLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                    x1Addr, x2Addr, biasAddr, x, pregLoop, i * vlFp32, i * vlFp32, i * vlFp32);
                LoadGammaBeta(gammaAddr, betaAddr, gamma, beta, pregLoop, i * vlFp32);
                CONST_CONDITIONAL_EXPR(IS_SCALE1_EXIST, LoadQuantParams(smooth1Addr, smooth1, pregLoop, i * vlFp32));
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, LoadQuantParams(smooth2Addr, smooth2, pregLoop, i * vlFp32));

                Sub(x, x, mean, pregLoop);
                Mul(y, x, rstd, pregLoop);
                Mul(y, y, gamma, pregLoop);
                Add(y, y, beta, pregLoop); // LayerNorm result

                // Do LocalMax1:
                if constexpr (IS_SCALE2_EXIST) {
                    Mul(outNorm1, y, smooth1, pregLoop);
                    Mul(outNorm2, y, smooth2, pregLoop);
                    Abs(tmp1, outNorm1, pregLoop);
                    Abs(tmp2, outNorm2, pregLoop);
                    Max(tmpMax1, tmp1, tmpMax1, pregAll);
                    Max(tmpMax2, tmp2, tmpMax2, pregAll);
                    DataCopy(((__local_mem__ float*)smoothedNorm1Addr + i * vlFp32), outNorm1, pregLoop);
                    DataCopy(((__local_mem__ float*)smoothedNorm2Addr + i * vlFp32), outNorm2, pregLoop);
                } else {
                    CONST_CONDITIONAL_EXPR(IS_SCALE1_EXIST, Mul(y, y, smooth1, pregLoop));
                    Abs(outNorm1, y, pregLoop);
                    Max(tmpMax1, tmpMax1, outNorm1, pregAll);
                    DataCopy(((__local_mem__ float*)smoothedNorm1Addr + i * vlFp32), y, pregLoop);
                }
            }
            ReduceMax(tmpMax1, tmpMax1, pregAll);
            CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, ReduceMax(tmpMax2, tmpMax2, pregAll));

            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__local_mem__ float*)localMax1Addr, tmpMax1, pregOne);
            CONST_CONDITIONAL_EXPR(
                IS_SCALE2_EXIST, (DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                                     (__local_mem__ float*)localMax2Addr, tmpMax2, pregOne)));
        }
    }

    /**
     * tmpMaxLocal1, tmpMaxLocal2, outScale1Local, outScale1Loca2 <-- tmpMaxLocal1, tmpMaxLocal2
     */
    static __aicore__ inline void VFComputeScale(
        LocalTensor<float>& tmpMaxLocal1, LocalTensor<float>& tmpMaxLocal2, LocalTensor<float>& outScale1Local,
        LocalTensor<float>& outScale2Local)
    {
        __local_mem__ float* localMax1Addr = (__local_mem__ float*)tmpMaxLocal1[0].GetPhyAddr();
        __local_mem__ float* localMax2Addr;
        __local_mem__ float* outScale1Addr = (__local_mem__ float*)outScale1Local[0].GetPhyAddr();
        __local_mem__ float* outScale2Addr;

        if constexpr (IS_SCALE2_EXIST) {
            localMax2Addr = (__local_mem__ float*)tmpMaxLocal2[0].GetPhyAddr();
            outScale2Addr = (__local_mem__ float*)outScale2Local[0].GetPhyAddr();
        }

        constexpr float quantConst = static_cast<float>(1.0 / 127.0);

        __VEC_SCOPE__
        {
            RegTensor<float> localMax1;
            RegTensor<float> localMax2;

            RegTensor<float> outScale1;
            RegTensor<float> outScale2;

            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            // Locad mean/rstd/localMax12
            DataCopy<float, LoadDist::DIST_BRC_B32>(localMax1, ((__local_mem__ float*)localMax1Addr));
            CONST_CONDITIONAL_EXPR(
                IS_SCALE2_EXIST,
                (DataCopy<float, LoadDist::DIST_BRC_B32>(localMax2, ((__local_mem__ float*)localMax2Addr))));

            Muls(outScale1, localMax1, quantConst, pregOne);
            CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, (Muls(outScale2, localMax2, quantConst, pregOne)));

            // outScale for copyOut, LocalMax for nex compute
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__local_mem__ float*)outScale1Addr, outScale1, pregOne);
            CONST_CONDITIONAL_EXPR(
                IS_SCALE2_EXIST, (DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                                     (__local_mem__ float*)outScale2Addr, outScale2, pregOne)));
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__local_mem__ float*)localMax1Addr, outScale1, pregOne);
            CONST_CONDITIONAL_EXPR(
                IS_SCALE2_EXIST, (DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                                     (__local_mem__ float*)localMax2Addr, outScale2, pregOne)));
        }
    }

    // y1Local, y2Local <-- smoothedNorm1Local, smoothedNorm2Local, tmpMaxLocal1, tmpMaxLocal2
    static __aicore__ inline void VFCalcYQuant(
        LocalTensor<float>& smoothedNorm1Local, LocalTensor<float>& smoothedNorm2Local,
        LocalTensor<float>& tmpMaxLocal1, LocalTensor<float>& tmpMaxLocal2, LocalTensor<int8_t>& y1Local,
        LocalTensor<int8_t>& y2Local, uint32_t colsCount, uint32_t vlFp32)
    {
        __local_mem__ float* smoothedNorm1Addr = (__local_mem__ float*)smoothedNorm1Local[0].GetPhyAddr();
        __local_mem__ float* smoothedNorm2Addr;
        __local_mem__ float* scale1Addr = (__local_mem__ float*)tmpMaxLocal1[0].GetPhyAddr();
        __local_mem__ float* scale2Addr;
        __local_mem__ int8_t* y1Addr = (__local_mem__ int8_t*)y1Local[0].GetPhyAddr();
        __local_mem__ int8_t* y2Addr;

        if constexpr (IS_SCALE2_EXIST) {
            smoothedNorm2Addr = (__local_mem__ float*)smoothedNorm2Local[0].GetPhyAddr();
            scale2Addr = (__local_mem__ float*)tmpMaxLocal2[0].GetPhyAddr();
            y2Addr = (__local_mem__ int8_t*)y2Local[0].GetPhyAddr();
        }

        uint16_t colsLoopCount = CEIL_DIV(colsCount, vlFp32);

        __VEC_SCOPE__
        {
            RegTensor<float> smoothedNorm1;
            RegTensor<float> smoothedNorm2;
            RegTensor<float> scale1;
            RegTensor<float> scale2;

            RegTensor<int8_t> y1;
            RegTensor<int8_t> y2;

            MaskReg pregLoop;

            uint32_t sreg0 = colsCount;

            // Locad outScales
            DataCopy<float, LoadDist::DIST_BRC_B32>(scale1, ((__local_mem__ float*)scale1Addr));
            if constexpr (IS_SCALE2_EXIST) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(scale2, ((__local_mem__ float*)scale2Addr));
            }
            for (uint16_t i = 0; i < colsLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);

                DataCopy(smoothedNorm1, ((__local_mem__ float*)smoothedNorm1Addr + i * vlFp32));
                if constexpr (IS_SCALE2_EXIST) {
                    DataCopy(smoothedNorm2, ((__local_mem__ float*)smoothedNorm2Addr + i * vlFp32));
                }

                ComputeDynamicQuantWithOutSmooth(y1, smoothedNorm1, scale1, pregLoop);
                if constexpr (IS_SCALE2_EXIST) {
                    ComputeDynamicQuantWithOutSmooth(y2, smoothedNorm2, scale2, pregLoop);
                }

                DataCopy<int8_t, StoreDist::DIST_PACK4_B32>((__local_mem__ int8_t*)y1Addr + i * vlFp32, y1, pregLoop);
                CONST_CONDITIONAL_EXPR(
                    IS_SCALE2_EXIST, (DataCopy<int8_t, StoreDist::DIST_PACK4_B32>(
                                         (__local_mem__ int8_t*)y2Addr + i * vlFp32, y2, pregLoop)));
            }
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();

        int64_t inputOffset = 0;
        int64_t outputOffset = 0;
        int64_t outScaleOffset = 0;

        LocalTensor<float> meanLocal = meanTmpBuf_.Get<float>();
        LocalTensor<float> rstdLocal = rstdTmpBuf_.Get<float>();
        LocalTensor<float> binaryAddLocal = binaryAddBuf_.Get<float>();
        LocalTensor<float> tmpMaxLocal1 = tmpMax1_.Get<float>();
        LocalTensor<float> tmpMaxLocal2 = tmpMax2_.Get<float>();

        LocalTensor<float> tmpBuffer = tmpBuf_.Get<float>();
        LocalTensor<float> tmpBufferPing = tmpBuffer[0];
        LocalTensor<float> tmpBufferPong = tmpBuffer[outNums_ * colsPerLoopAlign_];

        // Since David have infi
        event_t eIdMte2ToVecPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        event_t eIdMte2ToVecPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        event_t eIdVecToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        event_t eIdVecToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());

        event_t eIdMte3ToVecPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        event_t eIdMte3ToVecPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        event_t eIdVecToMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        event_t eIdVecToMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());

        event_t eIdMte3ToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        event_t eIdMte3ToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());

        int64_t wsBaseOffsetPing = 0;
        int64_t wsBaseOffsetPong = outNums_ * COLS_;

        for (int64_t i = 0; i < rowsPerCore_; i++) {
            int64_t count = 0;
            int64_t inputOffsetTemp = inputOffset;
            int64_t outputOffsetTemp = outputOffset;
            int64_t biasOffset = 0;
            if constexpr (IS_BIAS_ELEWISE) {
                biasOffset = inputOffsetTemp;
            }

            LocalTensor<float> tmpMeanLocal = tmpBufferPing;
            LocalTensor<float> tmpVarLocal = tmpBufferPong;

            // Compute Mean/Rstd
            for (int64_t j = 0; j < COLS_LOOP_COUNT_; j++) {
                int32_t copyLen = (j == COLS_LOOP_COUNT_ - 1) ? COLS_TAIL_ : COLS_PER_LOOP_;

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
                if (OUTPUT_X_) {
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

            float reduceScale = float(1.0) / static_cast<float>(COLS_);
            if (COLS_TAIL_ != COLS_PER_LOOP_) {
                VFWelfordParallelFinalizeNonAlign(
                    meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, binaryAddLocal, COLS_PER_LOOP_, BINARY_ADD_NUM_,
                    BINARY_ADD_K_, BINARY_ADD_LAST_NUM_, 0, COLS_TAIL_, reduceScale, count - 1, EPS_);
            } else {
                float scale = float(1.0) / static_cast<float>(COLS_PER_LOOP_);
                VFWelfordParallelFinalizeAlign(
                    meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, binaryAddLocal, COLS_PER_LOOP_, BINARY_ADD_NUM_,
                    BINARY_ADD_K_, BINARY_ADD_LAST_NUM_, 0, reduceScale, scale, count, EPS_);
            }

            // calc y with VF
            inputOffsetTemp = inputOffset;
            outputOffsetTemp = outputOffset;
            biasOffset = 0;
            if constexpr (IS_BIAS_ELEWISE) {
                biasOffset = inputOffsetTemp;
            }
            int64_t inputOffsetGamma = 0;
            int64_t workspaceOffset = 0;
            int64_t workspaceBaseOffset = 0;

            // MEM base code, we can put that into Finilize tensor, but that's weired; OR we can make a single small VF
            {
                Duplicate(tmpMaxLocal1, ZERO_ME, FLOAT_BLOCK_ELEM);
                if constexpr (IS_SCALE2_EXIST) {
                    Duplicate(tmpMaxLocal2, ZERO_ME, FLOAT_BLOCK_ELEM);
                }
            }

            // Compute LayerNorm and LocalMax
            for (int64_t j = 0; j < COLS_LOOP_COUNT_; j++) {
                int32_t copyLen = (j == COLS_LOOP_COUNT_ - 1) ? COLS_TAIL_ : COLS_PER_LOOP_;
                LocalTensor<X1_TYPE> x1Local = x1Queue_.template AllocTensor<X1_TYPE>();
                LocalTensor<X1_TYPE> x2Local = x2Queue_.template AllocTensor<X1_TYPE>();
                LocalTensor<X1_TYPE> biasLocal;
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasLocal = biasQueue_.template AllocTensor<X1_TYPE>();
                }
                LocalTensor<X1_TYPE> gammaLocal = gammaQueue_.template AllocTensor<X1_TYPE>();
                LocalTensor<X1_TYPE> betaLocal = betaQueue_.template AllocTensor<X1_TYPE>();

                LocalTensor<SCALE_TYPE> smooth1Local;
                LocalTensor<SCALE_TYPE> smooth2Local;
                CONST_CONDITIONAL_ASSIGN(
                    IS_SCALE1_EXIST, smooth1Local, smooth1Queue_.template AllocTensor<SCALE_TYPE>());
                CONST_CONDITIONAL_ASSIGN(
                    IS_SCALE2_EXIST, smooth2Local, smooth2Queue_.template AllocTensor<SCALE_TYPE>());

                // copy in x1, x2, bias
                CopyInputsToUB(x1Local, x2Local, biasLocal, inputOffsetTemp, biasOffset, copyLen);
                // copy in gamma, beta
                CopyGammaAndBetaToUB(gammaLocal, betaLocal, inputOffsetGamma, copyLen);
                // copy in scale/offset
                CopyQuantParams2UB(smooth1Local, smooth2Local, inputOffsetGamma, copyLen);

                x1Local = x1Queue_.template DeQue<X1_TYPE>();
                x2Local = x2Queue_.template DeQue<X1_TYPE>();
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasLocal = biasQueue_.template DeQue<X1_TYPE>();
                }
                gammaLocal = gammaQueue_.template DeQue<X1_TYPE>();
                betaLocal = betaQueue_.template DeQue<X1_TYPE>();

                CONST_CONDITIONAL_ASSIGN(IS_SCALE1_EXIST, smooth1Local, smooth1Queue_.template DeQue<SCALE_TYPE>());
                CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, smooth2Local, smooth2Queue_.template DeQue<SCALE_TYPE>());

                // Possible Req: TQuePool can solve this case......
                LocalTensor<float> smoothedNorm1Local;
                LocalTensor<float> smoothedNorm2Local;
                event_t localMte3ToVecID;
                event_t localVecToMte3Id;
                event_t localMte3ToMte2Id;
                if (j % BUFFER_NUM == 0) {
                    smoothedNorm1Local = tmpBufferPing[0];
                    CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, smoothedNorm2Local, tmpBufferPing[colsPerLoopAlign_]);
                    localMte3ToVecID = eIdMte3ToVecPing;
                    localVecToMte3Id = eIdVecToMte3Ping;

                    localMte3ToMte2Id = eIdMte3ToMte2Ping;
                } else {
                    smoothedNorm1Local = tmpBufferPong[0];
                    CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, smoothedNorm2Local, tmpBufferPong[colsPerLoopAlign_]);
                    localMte3ToVecID = eIdMte3ToVecPong;
                    localVecToMte3Id = eIdVecToMte3Pong;

                    localMte3ToMte2Id = eIdMte3ToMte2Pong;
                }

                if (likely(j > 1)) {
                    WaitFlag<HardEvent::MTE3_V>(localMte3ToVecID);
                }
                VFCalcNormlization(
                    x1Local, x2Local, biasLocal, betaLocal, gammaLocal, smooth1Local, smooth2Local, meanLocal,
                    rstdLocal, smoothedNorm1Local, smoothedNorm2Local, tmpMaxLocal1, tmpMaxLocal2, copyLen, vlFp32_);

                SetFlag<HardEvent::V_MTE3>(localVecToMte3Id);
                WaitFlag<HardEvent::V_MTE3>(localVecToMte3Id);
                CopyNormToWorkspace(
                    smoothedNorm1Local, smoothedNorm2Local, (workspaceOffset + workspaceBaseOffset), copyLen);
                if (likely(j < COLS_LOOP_COUNT_ - BUFFER_NUM)) {
                    SetFlag<HardEvent::MTE3_V>(localMte3ToVecID);
                }

                if (unlikely(j >= COLS_LOOP_COUNT_ - BUFFER_NUM)) {
                    SetFlag<HardEvent::MTE3_MTE2>(localMte3ToMte2Id);
                }

                x1Queue_.FreeTensor(x1Local);
                x2Queue_.FreeTensor(x2Local);
                betaQueue_.FreeTensor(betaLocal);
                gammaQueue_.FreeTensor(gammaLocal);
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasQueue_.FreeTensor(biasLocal);
                }
                CONST_CONDITIONAL_EXPR(IS_SCALE1_EXIST, smooth1Queue_.FreeTensor(smooth1Local));
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, smooth2Queue_.FreeTensor(smooth2Local));

                inputOffsetTemp += copyLen;
                outputOffsetTemp = inputOffsetTemp;
                inputOffsetGamma += copyLen;
                biasOffset += copyLen;
                workspaceOffset += copyLen;
            }

            // Compute outScale
            {
                LocalTensor<float> outScale1Local = outScale1Queue_.template AllocTensor<float>();
                LocalTensor<float> outScale2Local;
                CONST_CONDITIONAL_ASSIGN(
                    IS_SCALE2_EXIST, outScale2Local, outScale2Queue_.template AllocTensor<float>());
                VFComputeScale(tmpMaxLocal1, tmpMaxLocal2, outScale1Local, outScale2Local);
                outScale1Queue_.EnQue(outScale1Local);
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, outScale2Queue_.EnQue(outScale2Local));
                CopyOutOutScale(outScaleOffset);
            }

            // Compute dynamicQuant
            outputOffsetTemp = outputOffset;
            workspaceOffset = 0;
            workspaceBaseOffset = 0;

            for (int64_t j = 0; j < COLS_LOOP_COUNT_; j++) {
                int32_t copyLen = (j == COLS_LOOP_COUNT_ - 1) ? COLS_TAIL_ : COLS_PER_LOOP_;

                LocalTensor<float> smoothedNorm1Local;
                LocalTensor<float> smoothedNorm2Local;
                event_t localMte2ToVecID;
                event_t localVecToMte2Id;
                event_t localMte3ToMte2Id;
                if (j % BUFFER_NUM == 0) {
                    smoothedNorm1Local = (COLS_LOOP_COUNT_ % BUFFER_NUM == 0) ? tmpBufferPing[0] : tmpBufferPong[0];
                    if (IS_SCALE2_EXIST) {
                        smoothedNorm2Local = (COLS_LOOP_COUNT_ % BUFFER_NUM == 0) ? tmpBufferPing[colsPerLoopAlign_] :
                                                                                  tmpBufferPong[colsPerLoopAlign_];
                    }
                    localMte2ToVecID = eIdMte2ToVecPing;
                    localVecToMte2Id = eIdVecToMte2Ping;

                    localMte3ToMte2Id = (COLS_LOOP_COUNT_ % BUFFER_NUM == 0) ? eIdMte3ToMte2Ping : eIdMte3ToMte2Pong;
                } else {
                    smoothedNorm1Local = (COLS_LOOP_COUNT_ % BUFFER_NUM == 0) ? tmpBufferPong[0] : tmpBufferPing[0];
                    if (IS_SCALE2_EXIST) {
                        smoothedNorm2Local = (COLS_LOOP_COUNT_ % BUFFER_NUM == 0) ? tmpBufferPong[colsPerLoopAlign_] :
                                                                                  tmpBufferPing[colsPerLoopAlign_];
                    }
                    localMte2ToVecID = eIdMte2ToVecPong;
                    localVecToMte2Id = eIdVecToMte2Pong;

                    localMte3ToMte2Id = (COLS_LOOP_COUNT_ % BUFFER_NUM == 0) ? eIdMte3ToMte2Pong : eIdMte3ToMte2Ping;
                }

                if (likely(j <= 1)) {
                    WaitFlag<HardEvent::MTE3_MTE2>(localMte3ToMte2Id);
                }

                if (likely(j > 1)) {
                    WaitFlag<HardEvent::V_MTE2>(localVecToMte2Id);
                }
                CopyInNormFromWorkspace(
                    smoothedNorm1Local, smoothedNorm2Local, (workspaceOffset + workspaceBaseOffset), copyLen);
                SetFlag<HardEvent::MTE2_V>(localMte2ToVecID);
                WaitFlag<HardEvent::MTE2_V>(localMte2ToVecID);

                LocalTensor<int8_t> y1Local = y1Queue_.template AllocTensor<int8_t>();
                LocalTensor<int8_t> y2Local;
                CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, y2Local, y2Queue_.template AllocTensor<int8_t>());

                VFCalcYQuant(
                    smoothedNorm1Local, smoothedNorm2Local, tmpMaxLocal1, tmpMaxLocal2, y1Local, y2Local, copyLen,
                    vlFp32_);

                if (likely(j < COLS_LOOP_COUNT_ - BUFFER_NUM)) {
                    SetFlag<HardEvent::V_MTE2>(localVecToMte2Id);
                }

                // copy out y
                y1Queue_.EnQue(y1Local);
                y1Local = y1Queue_.template DeQue<int8_t>();
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, y2Queue_.EnQue(y2Local));
                CONST_CONDITIONAL_ASSIGN(IS_SCALE2_EXIST, y2Local, y2Queue_.template DeQue<int8_t>());

                CopyYToGm(y1Local, y2Local, outputOffsetTemp, copyLen);
                y1Queue_.FreeTensor(y1Local);
                CONST_CONDITIONAL_EXPR(IS_SCALE2_EXIST, y2Queue_.FreeTensor(y2Local););

                outputOffsetTemp += copyLen;
                workspaceOffset += copyLen;
            }

            inputOffset = inputOffsetTemp;
            outputOffset = outputOffsetTemp;
            outScaleOffset += 1;
        }

        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eIdMte2ToVecPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eIdMte2ToVecPong);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eIdVecToMte2Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eIdVecToMte2Pong);

        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eIdMte3ToVecPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eIdMte3ToVecPong);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eIdVecToMte3Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eIdVecToMte3Pong);

        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eIdMte3ToMte2Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eIdMte3ToMte2Pong);
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> betaQueue_;

    TQue<QuePosition::VECIN, BUFFER_NUM> smooth1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> smooth2Queue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> y1Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> y2Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outScale1Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outScale2Queue_;

    TBuf<TPosition::VECCALC> binaryAddBuf_;

    TBuf<TPosition::VECCALC> meanTmpBuf_;
    TBuf<TPosition::VECCALC> rstdTmpBuf_;

    TBuf<TPosition::VECCALC> tmpMax1_;
    TBuf<TPosition::VECCALC> tmpMax2_;

    TBuf<TPosition::VECCALC> tmpBuf_;

    GlobalTensor<X1_TYPE> x1Gm_;
    GlobalTensor<X1_TYPE> x2Gm_;
    GlobalTensor<X1_TYPE> biasGm_;
    GlobalTensor<X1_TYPE> gammaGm_;
    GlobalTensor<X1_TYPE> betaGm_;
    GlobalTensor<SCALE_TYPE> smooth1Gm_;
    GlobalTensor<SCALE_TYPE> smooth2Gm_;

    GlobalTensor<int8_t> y1Gm_;
    GlobalTensor<int8_t> y2Gm_;
    GlobalTensor<X1_TYPE> xGm_;
    GlobalTensor<float> outScale1Gm_;
    GlobalTensor<float> outScale2Gm_;
    GlobalTensor<float> workspaceGm_;

    uint32_t tailCoreStartIndex_;
    int64_t colsPerLoopAlign_;
    int64_t colsPerLoopAlignB16_;
    int64_t colsPerLoopAlignB32_;
    int64_t colsPerLoopAlignBias_;
    int64_t rowsPerCore_;
    int64_t powerOfTwo_;
    int64_t outNums_;

    TPipe* pipe_ = nullptr;
    const AddLayerNormQuantRegbaseTilingData* tilingData_;
};
#undef COLS_
#undef COLS_TAIL_
#undef COLS_PER_LOOP_
#undef COLS_LOOP_COUNT_
#undef EPS_
#undef BINARY_ADD_NUM_
#undef BINARY_ADD_K_
#undef BINARY_ADD_LAST_NUM_
#undef OUTPUT_X_
} // namespace AddLayerNormQuantRegbase
#endif // ADD_LAYER_NORM_DYNAMIC_QUANT_REGBASE_WELFORD_H