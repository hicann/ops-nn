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
 * \file quantize_add_layer_norm_static_quant_regbase_full_load_kernel.h
 * \brief ascend950 (arch35/regbase) full-load static-quant kernel for QuantizeAddLayerNorm.
 *        Adapted from add_layer_norm_static_quant_regbase_full_load_kernel.h:
 *        single quant path (1 scales, 1 optional zero_points, 1 int8 output y), no dynamic quant.
 */

#ifndef QUANTIZE_ADD_LAYER_NORM_STATIC_QUANT_REGBASE_FULL_LOAD_H
#define QUANTIZE_ADD_LAYER_NORM_STATIC_QUANT_REGBASE_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "quantize_add_layer_norm_regbase_stats.h"
#include "quantize_add_layer_norm_tiling_data.h"

namespace NsQuantizeAddLayerNorm {
template <int32_t TILING_KEY, int32_t OPT_CODE>
__simd_callee__ inline void FullLoadYQuantMathChunk(RegTensor<float>& x, RegTensor<float>& y, RegTensor<float>& mean,
                                                    RegTensor<float>& rstd, RegTensor<float>& gamma,
                                                    RegTensor<float>& beta, RegTensor<float>& scale,
                                                    RegTensor<float>& offset, RegTensor<int8_t>& quantOut,
                                                    MaskReg& pregLoop, uint16_t colLoop, uint16_t row, uint32_t vlFp32,
                                                    uint32_t rowStride, __ubuf__ int8_t* quantOutAddr)
{
    Sub(x, x, mean, pregLoop);
    Mul(y, x, rstd, pregLoop);
    Mul(y, y, gamma, pregLoop);
    Add(y, y, beta, pregLoop);

    if constexpr (IS_DIV_SCALE) {
        Div(x, y, scale, pregLoop);
    } else {
        Mul(x, y, scale, pregLoop);
    }
    CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST, Add(x, x, offset, pregLoop));

    Round2Int8(quantOut, x, pregLoop);
    int32_t outputOffset = static_cast<int32_t>(colLoop) * static_cast<int32_t>(vlFp32) +
                           static_cast<int32_t>(row) * static_cast<int32_t>(rowStride);
    StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(quantOutAddr + outputOffset, quantOut, pregLoop);
}

template <typename X1_TYPE, typename SCALE_TYPE, int32_t TILING_KEY, int32_t OPT_CODE>
__simd_vf__ inline void FullLoadYQuantVf(__ubuf__ float* x32Addr, __ubuf__ X1_TYPE* betaAddr,
                                         __ubuf__ X1_TYPE* gammaAddr, __ubuf__ float* meanAddr,
                                         __ubuf__ float* rstdAddr, __ubuf__ SCALE_TYPE* scaleAddr,
                                         __ubuf__ SCALE_TYPE* offsetAddr, __ubuf__ int8_t* quantOutAddr,
                                         uint16_t rowsCount, uint32_t colsCount, uint32_t rowStride, uint32_t vlFp32,
                                         uint16_t colsLoopCount)
{
    RegTensor<float> x;
    RegTensor<float> y;
    RegTensor<float> mean;
    RegTensor<float> rstd;
    RegTensor<float> beta;
    RegTensor<float> gamma;
    RegTensor<float> scale;
    RegTensor<float> offset;
    RegTensor<int8_t> quantOut;
    MaskReg pregLoop;
    MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();

    if constexpr (IS_PER_TENSOR_SCALE) {
        LoadScalarQuantParam(scaleAddr, scale, pregMain);
        CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST, LoadScalarQuantParam(offsetAddr, offset, pregMain));
    }

    for (uint16_t row = 0; row < rowsCount; row++) {
        uint32_t remaining = colsCount;
        for (uint16_t colLoop = 0; colLoop < colsLoopCount; colLoop++) {
            uint32_t colOffset = static_cast<uint32_t>(colLoop) * vlFp32;
            pregLoop = UpdateMask<float>(remaining);
            LoadGammaBetaResidual(gammaAddr, betaAddr, gamma, beta, pregLoop, colOffset);
            if constexpr (!IS_PER_TENSOR_SCALE) {
                LoadQuantParams(scaleAddr, scale, pregLoop, colOffset);
                CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST, LoadQuantParams(offsetAddr, offset, pregLoop, colOffset));
            }

            int32_t inputOffset = static_cast<int32_t>(colOffset) +
                                  static_cast<int32_t>(row) * static_cast<int32_t>(rowStride);
            LoadAlign(x, x32Addr + inputOffset);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(mean, meanAddr + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd, rstdAddr + row);
            FullLoadYQuantMathChunk<TILING_KEY, OPT_CODE>(x, y, mean, rstd, gamma, beta, scale, offset, quantOut,
                                                          pregLoop, colLoop, row, vlFp32, rowStride, quantOutAddr);
        }
    }
}

template <typename X1_TYPE, typename SCALE_TYPE, int32_t TILING_KEY, int32_t OPT_CODE, int32_t BUFFER_NUM = 1>
class KernelQuantizeAddLayerNormStaticQuantRegbaseFullLoad {
public:
    __aicore__ inline explicit KernelQuantizeAddLayerNormStaticQuantRegbaseFullLoad(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR scales,
                                GM_ADDR zeroPoints, GM_ADDR y, GM_ADDR x, GM_ADDR workspace,
                                const QuantizeAddLayerNormRegbaseTilingData* tilingData)
    {
        uint32_t coreIdx = GetBlockIdx();

        colsPerLoop_ = tilingData->colsPerLoop;
        eps_ = tilingData->eps;
        binaryAddNum_ = tilingData->binaryAddNum;
        binaryAddK_ = tilingData->binaryAddK;
        binaryAddLastNum_ = tilingData->binaryAddLastNum;
        outputX_ = tilingData->outputX;

        colsPerLoopAlign_ = BLOCK_ALIGN(colsPerLoop_, QALN_VL_FP32);
        int32_t validRowBytes = BLOCK_ALIGN(colsPerLoop_ * sizeof(X1_TYPE), QALN_BLOCK_SIZE);
        dmaStride_ = (colsPerLoopAlign_ * sizeof(X1_TYPE) - validRowBytes) / QALN_BLOCK_SIZE;

        int64_t gmOffset;
        if (coreIdx < GetBlockNum() - 1) {
            // Non-tail cores.
            rowsPerCore_ = tilingData->rowsPerCore;
            rowsPerLoop_ = tilingData->rowsPerLoop;
            gmOffset = static_cast<int64_t>(tilingData->rowsPerCore) * colsPerLoop_ * coreIdx;
        } else {
            // Tail cores.
            rowsPerCore_ = tilingData->rowsPerTailCore;
            rowsPerLoop_ = tilingData->rowsPerLoop;
            gmOffset = static_cast<int64_t>(tilingData->rowsPerCore) * colsPerLoop_ * coreIdx;
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
        scaleGm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)scales);
        CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST, zeroOffsetGm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)zeroPoints));

        yGm_.SetGlobalBuffer((__gm__ int8_t*)(y) + gmOffset);
        xGm_.SetGlobalBuffer((__gm__ X1_TYPE*)(x) + gmOffset);

        InitQueues();
    }

    __aicore__ inline void CopyBiasToUB(LocalTensor<X1_TYPE> biasLocal, int32_t copyLen)
    {
        DataCopyPadExtParams<X1_TYPE> padParams = MakeZeroPadParams<X1_TYPE>(copyLen, QALN_BLOCK_SIZE);
        DataCopyExtParams dataCopyParams = MakeDataCopyParams<X1_TYPE>(copyLen);
        DataCopyPad(biasLocal, biasGm_[0], dataCopyParams, padParams);
        biasQueue_.EnQue(biasLocal);
    }

    __aicore__ inline void CopyInputsToUB(LocalTensor<X1_TYPE> x1Local, LocalTensor<X1_TYPE> x2Local,
                                          LocalTensor<X1_TYPE> biasLocal, int64_t inputOffset, int32_t copyLen,
                                          int32_t rowsCount)
    {
        DataCopyPadExtParams<X1_TYPE> padParams = MakeZeroPadParams<X1_TYPE>(copyLen, QALN_BLOCK_SIZE);
        DataCopyExtParams dataCopyParams = MakeDataCopyParams<X1_TYPE>(copyLen, rowsCount, dmaStride_);
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
        DataCopyExtParams xCopyParams;
        xCopyParams.blockCount = rowsCount;
        xCopyParams.blockLen = copyLen * sizeof(X1_TYPE);
        xCopyParams.srcStride = dmaStride_;
        xCopyParams.dstStride = 0;

        DataCopyPad(xGm_[xOffset], xLocal, xCopyParams);
    }

    __aicore__ inline void CopyYToGm(LocalTensor<int8_t> yLocal, int64_t yOffset, int32_t copyLen, int32_t rowsCount)
    {
        DataCopyExtParams yCopyParams;
        yCopyParams.blockCount = rowsCount;
        yCopyParams.blockLen = copyLen * sizeof(int8_t);
        int32_t validRowBytes = BLOCK_ALIGN(copyLen * sizeof(int8_t), QALN_BLOCK_SIZE);
        yCopyParams.srcStride = (colsPerLoopAlign_ * sizeof(int8_t) - validRowBytes) / QALN_BLOCK_SIZE;
        yCopyParams.dstStride = 0;

        DataCopyPad(yGm_[yOffset], yLocal, yCopyParams);
    }

    __aicore__ inline void CopyQuantParams2UB(LocalTensor<SCALE_TYPE> scaleLocal, LocalTensor<SCALE_TYPE> offsetLocal,
                                              int64_t offset, int32_t copyLen)
    {
        // Per-tensor quantization uses scalar scales/zero_points; only element 0 is valid in GM.
        int32_t realCopyLen = IS_PER_TENSOR_SCALE ? 1 : copyLen;
        int64_t realOffset = IS_PER_TENSOR_SCALE ? 0 : offset;

        DataCopyExtParams quantCopyParams;
        quantCopyParams.blockCount = 1;
        quantCopyParams.blockLen = realCopyLen * sizeof(SCALE_TYPE);
        quantCopyParams.srcStride = 0;
        quantCopyParams.dstStride = 0;
        DataCopyPad(scaleLocal, scaleGm_[realOffset], quantCopyParams, {});
        scaleQueue_.EnQue(scaleLocal);

        if constexpr (IS_OFFSET_EXIST) {
            DataCopyPad(offsetLocal, zeroOffsetGm_[realOffset], quantCopyParams, {});
            zeroOffsetQueue_.EnQue(offsetLocal);
        }
    }

    static __aicore__ inline void VFCalcYQuant(LocalTensor<float>& x32Local, LocalTensor<X1_TYPE>& betaLocal,
                                               LocalTensor<X1_TYPE>& gammaLocal, LocalTensor<float>& meanLocal,
                                               LocalTensor<float>& rstdLocal, LocalTensor<SCALE_TYPE>& scaleLocal,
                                               LocalTensor<SCALE_TYPE>& offsetLocal, LocalTensor<int8_t>& yLocal,
                                               uint32_t rowsCount, uint32_t colsCount, uint32_t colsPerLoopAlign,
                                               uint32_t vlFp32)
    {
        __ubuf__ float* x32Addr = (__ubuf__ float*)x32Local[0].GetPhyAddr();
        __ubuf__ float* meanAddr = (__ubuf__ float*)meanLocal[0].GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal[0].GetPhyAddr();
        __ubuf__ X1_TYPE* betaAddr = (__ubuf__ X1_TYPE*)betaLocal[0].GetPhyAddr();
        __ubuf__ X1_TYPE* gammaAddr = (__ubuf__ X1_TYPE*)gammaLocal[0].GetPhyAddr();
        __ubuf__ int8_t* quantOutAddr = (__ubuf__ int8_t*)yLocal[0].GetPhyAddr();
        __ubuf__ SCALE_TYPE* scaleAddr = (__ubuf__ SCALE_TYPE*)scaleLocal[0].GetPhyAddr();

        __ubuf__ SCALE_TYPE* offsetAddr = nullptr;
        CONST_CONDITIONAL_ASSIGN(IS_OFFSET_EXIST, offsetAddr, (__ubuf__ SCALE_TYPE*)offsetLocal[0].GetPhyAddr());

        uint16_t colsLoopCount = CEIL_DIV(colsCount, vlFp32);
        asc_vf_call<FullLoadYQuantVf<X1_TYPE, SCALE_TYPE, TILING_KEY, OPT_CODE>>(
            x32Addr, betaAddr, gammaAddr, meanAddr, rstdAddr, scaleAddr, offsetAddr, quantOutAddr,
            static_cast<uint16_t>(rowsCount), colsCount, colsPerLoopAlign, vlFp32, colsLoopCount);
    }

    __aicore__ inline void Process()
    {
        int64_t inputOffset = 0;
        int64_t outputOffset = 0;

        LocalTensor<X1_TYPE> gammaLocal = gammaQueue_.template AllocTensor<X1_TYPE>();
        LocalTensor<X1_TYPE> betaLocal = betaQueue_.template AllocTensor<X1_TYPE>();

        LocalTensor<SCALE_TYPE> scaleLocal = scaleQueue_.template AllocTensor<SCALE_TYPE>();
        LocalTensor<SCALE_TYPE> offsetLocal;
        CONST_CONDITIONAL_ASSIGN(IS_OFFSET_EXIST, offsetLocal, zeroOffsetQueue_.template AllocTensor<SCALE_TYPE>());

        LocalTensor<X1_TYPE> biasLocal;
        if constexpr (IS_BIAS_BROADCAST) {
            biasLocal = biasQueue_.template AllocTensor<X1_TYPE>();
            CopyBiasToUB(biasLocal, colsPerLoop_);
        }
        LocalTensor<float> binaryAddLocal = binaryAddBuf_.Get<float>();

        for (int64_t i = 0; i < rowsLoopCount_; i++) {
            int32_t rowsCount = i < rowsLoopCount_ - 1 ? rowsPerLoop_ : rowsTail_;

            StatsBlock(i, rowsCount, biasLocal, binaryAddLocal, inputOffset);

            // Copy gamma, beta, scales, and zero_points into UB.
            if (i == 0) {
                CopyGammaAndBetaToUBCommon(gammaLocal, betaLocal, gammaGm_, betaGm_, gammaQueue_, betaQueue_, 0,
                                           colsPerLoop_, QALN_BLOCK_SIZE);
                CopyQuantParams2UB(scaleLocal, offsetLocal, 0, colsPerLoop_);
                gammaLocal = gammaQueue_.template DeQue<X1_TYPE>();
                betaLocal = betaQueue_.template DeQue<X1_TYPE>();

                scaleLocal = scaleQueue_.template DeQue<SCALE_TYPE>();
                CONST_CONDITIONAL_ASSIGN(IS_OFFSET_EXIST, offsetLocal, zeroOffsetQueue_.template DeQue<SCALE_TYPE>());
            }

            QuantOutBlock(betaLocal, gammaLocal, scaleLocal, offsetLocal, inputOffset, outputOffset, rowsCount);

            inputOffset += rowsCount * colsPerLoop_;
            outputOffset = inputOffset;
        }
        if constexpr (IS_BIAS_BROADCAST) {
            biasQueue_.FreeTensor(biasLocal);
        }
        gammaQueue_.FreeTensor(gammaLocal);
        betaQueue_.FreeTensor(betaLocal);
        scaleQueue_.FreeTensor(scaleLocal);
        CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST, zeroOffsetQueue_.FreeTensor(offsetLocal));
    }

private:
    __aicore__ inline void InitQueues()
    {
        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(x2Queue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        if constexpr (IS_BIAS_ELEWISE) {
            pipe_->InitBuffer(biasQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        } else if constexpr (IS_BIAS_BROADCAST) {
            pipe_->InitBuffer(biasQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        }
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(int8_t)));
        pipe_->InitBuffer(x32Queue_, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(float)));
        pipe_->InitBuffer(betaQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(gammaQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));

        pipe_->InitBuffer(scaleQueue_, 1, (colsPerLoopAlign_ * sizeof(SCALE_TYPE)));
        CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST,
                               pipe_->InitBuffer(zeroOffsetQueue_, 1, (colsPerLoopAlign_ * sizeof(SCALE_TYPE))));

        int64_t binaryAddBufSize = BLOCK_ALIGN((binaryAddNum_ / QALN_VL_FP32) * sizeof(float), QALN_BLOCK_SIZE);
        if (binaryAddBufSize > 0) {
            pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize);
        }
        pipe_->InitBuffer(meanBuf_, BLOCK_ALIGN(rowsPerLoop_ * sizeof(float), QALN_BLOCK_SIZE));
        pipe_->InitBuffer(rstdBuf_, BLOCK_ALIGN(rowsPerLoop_ * sizeof(float), QALN_BLOCK_SIZE));
    }

    // Per rows-block statistics: copy in, mean/var (fast or binaryAdd two-pass), copy out x, rstd.
    __aicore__ inline void StatsBlock(int64_t i, int32_t rowsCount, LocalTensor<X1_TYPE>& biasLocal,
                                      LocalTensor<float>& binaryAddLocal, int64_t inputOffset)
    {
        LocalTensor<X1_TYPE> x1Local = x1Queue_.template AllocTensor<X1_TYPE>();
        LocalTensor<X1_TYPE> x2Local = x2Queue_.template AllocTensor<X1_TYPE>();
        if constexpr (IS_BIAS_ELEWISE) {
            biasLocal = biasQueue_.template AllocTensor<X1_TYPE>();
        }
        // Copy x1, x2, and bias into UB.
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

        if (colsPerLoop_ <= QALN_VL_FP32) {
            VFCalcMeanVarFast<X1_TYPE, TILING_KEY>(x1Local, x2Local, biasLocal, xOutLocal, x32Local, meanLocal,
                                                   rstdLocal, rowsCount, colsPerLoop_, colsPerLoopAlign_);
        } else {
            VFCalcMeanVar<X1_TYPE, TILING_KEY>(x1Local, x2Local, biasLocal, xOutLocal, x32Local, meanLocal, rstdLocal,
                                               binaryAddLocal, rowsCount, colsPerLoop_, colsPerLoopAlign_, QALN_VL_FP32,
                                               binaryAddLastNum_, binaryAddNum_, binaryAddK_);
        }

        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
        if constexpr (IS_BIAS_ELEWISE) {
            biasQueue_.FreeTensor(biasLocal);
        }
        // Copy x to GM.
        if (outputX_) {
            xQueue_.EnQue(xOutLocal);
            xOutLocal = xQueue_.template DeQue<X1_TYPE>();
            CopyXToGm(xOutLocal, inputOffset, colsPerLoop_, rowsCount);
        }
        xQueue_.FreeTensor(xOutLocal);

        // Denorm-min epsilon makes reciprocal-first rstd overflow before sqrt.
        if (eps_ < QALN_MIN_NORMAL_FP32) {
            ComputeRstdSqrtFirst<false>(rstdLocal, rstdLocal, rowsCount, eps_);
        } else {
            NormCommon::ComputeRstdNewtonRaphson<false>(rstdLocal, rstdLocal, rowsCount, eps_, 1.0f, QALN_VL_FP32);
        }
    }

    // Per rows-block quant output: VFCalcYQuant over the staged x32/mean/rstd, copy out y.
    __aicore__ inline void QuantOutBlock(LocalTensor<X1_TYPE>& betaLocal, LocalTensor<X1_TYPE>& gammaLocal,
                                         LocalTensor<SCALE_TYPE>& scaleLocal, LocalTensor<SCALE_TYPE>& offsetLocal,
                                         int64_t inputOffset, int64_t outputOffset, int32_t rowsCount)
    {
        LocalTensor<float> x32Local = x32Queue_.Get<float>();
        LocalTensor<float> meanLocal = meanBuf_.Get<float>();
        LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
        LocalTensor<int8_t> yLocal = yQueue_.template AllocTensor<int8_t>();

        // Calculate y with the VF implementation.
        VFCalcYQuant(x32Local, betaLocal, gammaLocal, meanLocal, rstdLocal, scaleLocal, offsetLocal, yLocal, rowsCount,
                     colsPerLoop_, colsPerLoopAlign_, QALN_VL_FP32);

        // Copy y to GM.
        yQueue_.EnQue(yLocal);
        yLocal = yQueue_.template DeQue<int8_t>();
        CopyYToGm(yLocal, outputOffset, colsPerLoop_, rowsCount);

        yQueue_.FreeTensor(yLocal);
    }

    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasQueue_;
    TQue<QuePosition::VECIN, 1> gammaQueue_;
    TQue<QuePosition::VECIN, 1> betaQueue_;

    TQue<QuePosition::VECIN, 1> scaleQueue_;
    TQue<QuePosition::VECIN, 1> zeroOffsetQueue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
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

    GlobalTensor<SCALE_TYPE> scaleGm_;
    GlobalTensor<SCALE_TYPE> zeroOffsetGm_;

    GlobalTensor<int8_t> yGm_;
    GlobalTensor<X1_TYPE> xGm_;

    int64_t colsPerLoop_;
    int64_t colsPerLoopAlign_;
    int64_t rowsPerCore_;
    int64_t rowsPerLoop_;
    int64_t rowsTail_;
    int64_t rowsLoopCount_;
    int64_t binaryAddNum_;
    int64_t binaryAddK_;
    int64_t binaryAddLastNum_;
    float eps_;
    bool outputX_;

    TPipe* pipe_ = nullptr;

    int32_t dmaStride_;
};
} // namespace NsQuantizeAddLayerNorm
#endif // QUANTIZE_ADD_LAYER_NORM_STATIC_QUANT_REGBASE_FULL_LOAD_H
