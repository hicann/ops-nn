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
#include "quantize_add_layer_norm_regbase_helper.h"

namespace QuantizeAddLayerNormRegbase {
template <typename X1_TYPE, typename SCALE_TYPE, int32_t TILING_KEY, int32_t OPT_CODE, int32_t BUFFER_NUM = 1>
class KernelQuantizeAddLayerNormStaticQuantRegbaseFullLoad {
public:
    __aicore__ inline KernelQuantizeAddLayerNormStaticQuantRegbaseFullLoad(TPipe* pipe) { pipe_ = pipe; }

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

        powerOfTwo_ = 1;
        while (powerOfTwo_ < colsPerLoop_) {
            powerOfTwo_ *= AddLayerNorm::NUM_TWO;
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
        scaleGm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)scales);
        CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST, zeroOffsetGm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)zeroPoints));

        yGm_.SetGlobalBuffer((__gm__ int8_t*)(y) + gmOffset);
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
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(int8_t)));
        pipe_->InitBuffer(x32Queue_, (rowsPerLoop_ * colsPerLoopAlign_ * sizeof(float)));
        pipe_->InitBuffer(betaQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(gammaQueue_, 1, (colsPerLoopAlign_ * sizeof(X1_TYPE)));

        pipe_->InitBuffer(scaleQueue_, 1, (colsPerLoopAlign_ * sizeof(SCALE_TYPE)));
        CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST,
                               pipe_->InitBuffer(zeroOffsetQueue_, 1, (colsPerLoopAlign_ * sizeof(SCALE_TYPE))));

        int64_t binaryAddBufSize = BLOCK_ALIGN((binaryAddNum_ / vlFp32_) * sizeof(float), blockSize_);
        if (binaryAddBufSize > 0) {
            pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize);
        }
        pipe_->InitBuffer(meanBuf_, BLOCK_ALIGN(rowsPerLoop_ * sizeof(float), blockSize_));
        pipe_->InitBuffer(rstdBuf_, BLOCK_ALIGN(rowsPerLoop_ * sizeof(float), blockSize_));
    }

    __aicore__ inline void CopyBiasToUB(LocalTensor<X1_TYPE> biasLocal, int32_t copyLen)
    {
        DataCopyPadExtParams<X1_TYPE> padParams = MakeZeroPadParams<X1_TYPE>(copyLen, blockSize_);
        DataCopyExtParams dataCopyParams = MakeDataCopyParams<X1_TYPE>(copyLen);
        DataCopyPad(biasLocal, biasGm_[0], dataCopyParams, padParams);
        biasQueue_.EnQue(biasLocal);
    }

    __aicore__ inline void CopyInputsToUB(LocalTensor<X1_TYPE> x1Local, LocalTensor<X1_TYPE> x2Local,
                                          LocalTensor<X1_TYPE> biasLocal, int64_t inputOffset, int32_t copyLen,
                                          int32_t rowsCount)
    {
        DataCopyPadExtParams<X1_TYPE> padParams = MakeZeroPadParams<X1_TYPE>(copyLen, blockSize_);
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
        yCopyParams.srcStride = 0;
        yCopyParams.dstStride = 0;

        DataCopyPad(yGm_[yOffset], yLocal, yCopyParams);
    }

    __aicore__ inline void CopyQuantParams2UB(LocalTensor<SCALE_TYPE> scaleLocal, LocalTensor<SCALE_TYPE> offsetLocal,
                                              int64_t offset, int32_t copyLen)
    {
        // per_tensor quant: scales/zero_points are scalars, only element 0 is valid in GM
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

        __ubuf__ SCALE_TYPE* offsetAddr;
        CONST_CONDITIONAL_ASSIGN(IS_OFFSET_EXIST, offsetAddr, (__ubuf__ SCALE_TYPE*)offsetLocal[0].GetPhyAddr());

        uint16_t colsLoopCount = CEIL_DIV(colsCount, vlFp32);

        __VEC_SCOPE__
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

            // per_tensor quant: scalar scales/zero_points, broadcast once and reused by all cols loops
            if constexpr (IS_PER_TENSOR_SCALE) {
                LoadScalarQuantParam(scaleAddr, scale, pregMain);
                CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST, LoadScalarQuantParam(offsetAddr, offset, pregMain));
            }

            for (uint16_t k = 0; k < (uint16_t)rowsCount; k++) {
                uint32_t sreg0 = colsCount;
                for (uint16_t i = 0; i < colsLoopCount; i++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadGammaBeta(gammaAddr, betaAddr, gamma, beta, pregLoop, i * vlFp32);
                    if constexpr (!IS_PER_TENSOR_SCALE) {
                        LoadQuantParams(scaleAddr, scale, pregLoop, i * vlFp32);
                        CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST,
                                               LoadQuantParams(offsetAddr, offset, pregLoop, i * vlFp32));
                    }

                    LoadAlign(x, ((__ubuf__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(mean, ((__ubuf__ float*)meanAddr + k));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd, ((__ubuf__ float*)rstdAddr + k));
                    Sub(x, x, mean, pregLoop);
                    Mul(y, x, rstd, pregLoop);
                    Mul(y, y, gamma, pregLoop);
                    Add(y, y, beta, pregLoop); // LayerNorm result

                    // quant: y = round(norm / scales + zero_points)  (per_channel, div mode)
                    //        y = round(norm * scales + zero_points)  (mul_mode / per_tensor)
                    if constexpr (IS_DIV_SCALE) {
                        Div(x, y, scale, pregLoop);
                    } else {
                        Mul(x, y, scale, pregLoop);
                    }
                    CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST, Add(x, x, offset, pregLoop));

                    Round2Int8(quantOut, x, pregLoop);
                    StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(
                        (__ubuf__ int8_t*)quantOutAddr + i * vlFp32 + k * colsPerLoopAlign, quantOut, pregLoop);
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
                VFCalcMeanVarFast<X1_TYPE, TILING_KEY>(x1Local, x2Local, biasLocal, xOutLocal, x32Local, meanLocal,
                                                       rstdLocal, rowsCount, powerOfTwo_, colsPerLoop_,
                                                       colsPerLoopAlign_, vlFp32_);
            } else {
                VFCalcMeanVar<X1_TYPE, TILING_KEY>(x1Local, x2Local, biasLocal, xOutLocal, x32Local, meanLocal,
                                                   rstdLocal, binaryAddLocal, rowsCount, powerOfTwo_, colsPerLoop_,
                                                   colsPerLoopAlign_, vlFp32_, binaryAddLastNum_, binaryAddNum_,
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

            // calc rstd
            NormCommon::ComputeRstdNewtonRaphson<false>(rstdLocal, rstdLocal, rowsCount, eps_, 1.0f, vlFp32_);

            // copy in gamma, beta, scales, zero_points
            if (i == 0) {
                CopyGammaAndBetaToUBCommon(gammaLocal, betaLocal, gammaGm_, betaGm_, gammaQueue_, betaQueue_, 0,
                                           colsPerLoop_, blockSize_);
                CopyQuantParams2UB(scaleLocal, offsetLocal, 0, colsPerLoop_);
                gammaLocal = gammaQueue_.template DeQue<X1_TYPE>();
                betaLocal = betaQueue_.template DeQue<X1_TYPE>();

                scaleLocal = scaleQueue_.template DeQue<SCALE_TYPE>();
                CONST_CONDITIONAL_ASSIGN(IS_OFFSET_EXIST, offsetLocal, zeroOffsetQueue_.template DeQue<SCALE_TYPE>());
            }
            LocalTensor<int8_t> yLocal = yQueue_.template AllocTensor<int8_t>();

            // calc y with VF
            VFCalcYQuant(x32Local, betaLocal, gammaLocal, meanLocal, rstdLocal, scaleLocal, offsetLocal, yLocal,
                         rowsCount, colsPerLoop_, colsPerLoopAlign_, vlFp32_);

            // copy out y
            yQueue_.EnQue(yLocal);
            yLocal = yQueue_.template DeQue<int8_t>();
            CopyYToGm(yLocal, outputOffset, colsPerLoop_, rowsCount);

            inputOffset += rowsCount * colsPerLoop_;
            outputOffset = inputOffset;
            meanOffset += rowsCount;

            yQueue_.FreeTensor(yLocal);
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
    int64_t powerOfTwo_;
    float eps_;
    bool outputX_;

    TPipe* pipe_ = nullptr;

    int32_t dmaStride_;
};
} // namespace QuantizeAddLayerNormRegbase
#endif // QUANTIZE_ADD_LAYER_NORM_STATIC_QUANT_REGBASE_FULL_LOAD_H
