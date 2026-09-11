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
 * \file quantize_add_layer_norm_static_quant_regbase_welford_kernel.h
 * \brief ascend950 (arch35/regbase) welford static-quant kernel for QuantizeAddLayerNorm.
 *        Adapted from add_layer_norm_static_quant_regbase_welford_kernel.h:
 *        single quant path (1 scales, 1 optional zero_points, 1 int8 output y), no dynamic quant.
 */

#ifndef QUANTIZE_ADD_LAYER_NORM_STATIC_QUANT_REGBASE_WELFORD_H
#define QUANTIZE_ADD_LAYER_NORM_STATIC_QUANT_REGBASE_WELFORD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "quantize_add_layer_norm_regbase_helper.h"

namespace QuantizeAddLayerNormRegbase {
template <typename X1_TYPE, typename SCALE_TYPE, int32_t TILING_KEY, int32_t OPT_CODE, int32_t BUFFER_NUM = 1>
class KernelQuantizeAddLayerNormStaticQuantRegbaseWelford {
public:
    __aicore__ inline KernelQuantizeAddLayerNormStaticQuantRegbaseWelford(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR scales,
                                GM_ADDR zeroPoints, GM_ADDR y, GM_ADDR x, GM_ADDR workspace,
                                const QuantizeAddLayerNormRegbaseTilingData* tilingData)
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
            powerOfTwo_ *= AddLayerNorm::NUM_TWO;
        }

        // split 1/cols into an exact power-of-two reciprocal (no rounding error accumulated through
        // the finalize reduction) plus a one-shot correction applied after DichotomyAdd, mirroring
        // the upstream add_layer_norm welford finalize precision fix
        uint64_t reducePowerOfTwo = 1;
        while (reducePowerOfTwo < static_cast<uint64_t>(cols_)) {
            reducePowerOfTwo *= AddLayerNorm::NUM_TWO;
        }
        reduceScale_ = 1.0f / static_cast<float>(reducePowerOfTwo);
        reduceScaleCorrection_ = static_cast<float>(reducePowerOfTwo) / static_cast<float>(cols_);

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
        scaleGm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)scales);
        CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST, zeroOffsetGm_.SetGlobalBuffer((__gm__ SCALE_TYPE*)zeroPoints));

        yGm_.SetGlobalBuffer((__gm__ int8_t*)(y) + gmOffset);
        xGm_.SetGlobalBuffer((__gm__ X1_TYPE*)(x) + gmOffset);

        colsPerLoopAlign_ = BLOCK_ALIGN(colsPerLoop_ * sizeof(X1_TYPE), blockSize_) / sizeof(X1_TYPE);

        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(x2Queue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(biasQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(int8_t)));
        pipe_->InitBuffer(betaQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(X1_TYPE)));
        pipe_->InitBuffer(scaleQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(SCALE_TYPE)));

        CONST_CONDITIONAL_EXPR(
            IS_OFFSET_EXIST, pipe_->InitBuffer(zeroOffsetQueue_, BUFFER_NUM, (colsPerLoopAlign_ * sizeof(SCALE_TYPE))));

        pipe_->InitBuffer(meanBuf_, colsPerLoopAlign_ * sizeof(float));
        pipe_->InitBuffer(varBuf_, colsPerLoopAlign_ * sizeof(float));

        pipe_->InitBuffer(meanTmpBuf_, blockSize_);
        pipe_->InitBuffer(rstdTmpBuf_, blockSize_);

        int64_t binaryAddBufSize = BLOCK_ALIGN((binaryAddNum_ / vlFp32_) * sizeof(float), blockSize_);
        if (binaryAddBufSize > 0) {
            pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize);
        }
    }

    __aicore__ inline void CopyInputsToUB(LocalTensor<X1_TYPE> x1Local, LocalTensor<X1_TYPE> x2Local,
                                          LocalTensor<X1_TYPE> biasLocal, int64_t inputOffset, int64_t biasOffset,
                                          int32_t copyLen)
    {
        DataCopyPadExtParams<X1_TYPE> padParams = MakeZeroPadParams<X1_TYPE>(copyLen, blockSize_);
        DataCopyExtParams dataCopyParams = MakeDataCopyParams<X1_TYPE>(copyLen);
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

    __aicore__ inline void CopyYToGm(LocalTensor<int8_t> yLocal, int64_t yOffset, int32_t copyLen)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(int8_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(yGm_[yOffset], yLocal, dataCopyParams);
    }

    __aicore__ inline void CopyQuantParams2UB(LocalTensor<SCALE_TYPE> scaleLocal, LocalTensor<SCALE_TYPE> offsetLocal,
                                              int64_t offset, int32_t copyLen)
    {
        // per_tensor quant: scales/zero_points are scalars, only element 0 is valid in GM
        int32_t realCopyLen = IS_PER_TENSOR_SCALE ? 1 : copyLen;
        int64_t realOffset = IS_PER_TENSOR_SCALE ? 0 : offset;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = realCopyLen * sizeof(SCALE_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(scaleLocal, scaleGm_[realOffset], dataCopyParams, {});
        scaleQueue_.EnQue(scaleLocal);
        if constexpr (IS_OFFSET_EXIST) {
            DataCopyPad(offsetLocal, zeroOffsetGm_[realOffset], dataCopyParams, {});
            zeroOffsetQueue_.EnQue(offsetLocal);
        }
    }

    static __aicore__ inline void VFCalcYQuant(LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local,
                                               LocalTensor<X1_TYPE>& biasLocal, LocalTensor<X1_TYPE>& betaLocal,
                                               LocalTensor<X1_TYPE>& gammaLocal, LocalTensor<SCALE_TYPE>& scaleLocal,
                                               LocalTensor<SCALE_TYPE>& offsetLocal, float mean, float rstd,
                                               LocalTensor<int8_t>& yLocal, uint32_t colsCount, uint32_t vlFp32)
    {
        __ubuf__ X1_TYPE* x1Addr = (__ubuf__ X1_TYPE*)x1Local[0].GetPhyAddr();
        __ubuf__ X1_TYPE* x2Addr = (__ubuf__ X1_TYPE*)x2Local[0].GetPhyAddr();
        __ubuf__ X1_TYPE* betaAddr = (__ubuf__ X1_TYPE*)betaLocal[0].GetPhyAddr();
        __ubuf__ X1_TYPE* gammaAddr = (__ubuf__ X1_TYPE*)gammaLocal[0].GetPhyAddr();
        __ubuf__ SCALE_TYPE* scaleAddr = (__ubuf__ SCALE_TYPE*)scaleLocal[0].GetPhyAddr();
        __ubuf__ int8_t* quantOutAddr = (__ubuf__ int8_t*)yLocal[0].GetPhyAddr();

        __ubuf__ X1_TYPE* biasAddr;
        __ubuf__ SCALE_TYPE* offsetAddr;

        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            biasAddr = (__ubuf__ X1_TYPE*)biasLocal[0].GetPhyAddr();
        }
        CONST_CONDITIONAL_ASSIGN(IS_OFFSET_EXIST, offsetAddr, (__ubuf__ SCALE_TYPE*)offsetLocal[0].GetPhyAddr());

        uint16_t colsLoopCount = CEIL_DIV(colsCount, vlFp32);

        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
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

            uint32_t sreg0 = colsCount;
            for (uint16_t i = 0; i < colsLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(x1Addr, x2Addr, biasAddr, x, pregLoop,
                                                                       i * vlFp32, i * vlFp32, i * vlFp32);
                LoadGammaBeta(gammaAddr, betaAddr, gamma, beta, pregLoop, i * vlFp32);
                if constexpr (!IS_PER_TENSOR_SCALE) {
                    LoadQuantParams(scaleAddr, scale, pregLoop, i * vlFp32);
                    CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST, LoadQuantParams(offsetAddr, offset, pregLoop, i * vlFp32));
                }

                Adds(x, x, mean, pregLoop);
                Muls(y, x, rstd, pregLoop);
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

                StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>((__ubuf__ int8_t*)quantOutAddr + i * vlFp32, quantOut,
                                                              pregLoop);
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

                count += 1;
                uint16_t loopCount = CEIL_DIV(copyLen, vlFp32_);
                float scale = static_cast<float>(1.0) / static_cast<float>(count);

                if (j == 0) {
                    VFWelfordParallelUpdateCommon<true, X1_TYPE, TILING_KEY>(
                        x1Local, x2Local, biasLocal, xLocal, tmpMeanLocal, tmpVarLocal, copyLen, loopCount, scale);
                } else {
                    VFWelfordParallelUpdateCommon<false, X1_TYPE, TILING_KEY>(
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

            if (colsTail_ != colsPerLoop_) {
                VFWelfordParallelFinalizeNonAlign(meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, binaryAddLocal,
                                                  colsPerLoop_, binaryAddNum_, binaryAddK_, binaryAddLastNum_, 0,
                                                  colsTail_, reduceScale_, reduceScaleCorrection_, count - 1, eps_);
            } else {
                float scale = 1.0f / static_cast<float>(powerOfTwo_);
                float scaleCorrection = static_cast<float>(powerOfTwo_) / static_cast<float>(colsPerLoop_);
                VFWelfordParallelFinalizeAlign(meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, binaryAddLocal,
                                               colsPerLoop_, binaryAddNum_, binaryAddK_, binaryAddLastNum_, 0,
                                               reduceScale_, reduceScaleCorrection_, scale, scaleCorrection, count,
                                               eps_);
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

                LocalTensor<SCALE_TYPE> scaleLocal = scaleQueue_.template AllocTensor<SCALE_TYPE>();
                LocalTensor<SCALE_TYPE> offsetLocal;
                CONST_CONDITIONAL_ASSIGN(IS_OFFSET_EXIST, offsetLocal,
                                         zeroOffsetQueue_.template AllocTensor<SCALE_TYPE>());

                // copy in x1, x2, bias
                CopyInputsToUB(x1Local, x2Local, biasLocal, inputOffsetTemp, biasOffset, copyLen);
                // copy in gamma, beta
                CopyGammaAndBetaToUBCommon(gammaLocal, betaLocal, gammaGm_, betaGm_, gammaQueue_, betaQueue_,
                                           inputOffsetGamma, copyLen, blockSize_);
                // copy in scale/offset
                CopyQuantParams2UB(scaleLocal, offsetLocal, inputOffsetGamma, copyLen);

                x1Local = x1Queue_.template DeQue<X1_TYPE>();
                x2Local = x2Queue_.template DeQue<X1_TYPE>();

                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasLocal = biasQueue_.template DeQue<X1_TYPE>();
                }

                gammaLocal = gammaQueue_.template DeQue<X1_TYPE>();
                betaLocal = betaQueue_.template DeQue<X1_TYPE>();

                scaleLocal = scaleQueue_.template DeQue<SCALE_TYPE>();
                CONST_CONDITIONAL_ASSIGN(IS_OFFSET_EXIST, offsetLocal, zeroOffsetQueue_.template DeQue<SCALE_TYPE>());

                LocalTensor<int8_t> yLocal = yQueue_.template AllocTensor<int8_t>();

                VFCalcYQuant(x1Local, x2Local, biasLocal, betaLocal, gammaLocal, scaleLocal, offsetLocal, mean, rstd,
                             yLocal, copyLen, vlFp32_);

                // copy out y
                yQueue_.EnQue(yLocal);
                yLocal = yQueue_.template DeQue<int8_t>();
                CopyYToGm(yLocal, outputOffsetTemp, copyLen);
                yQueue_.FreeTensor(yLocal);

                x1Queue_.FreeTensor(x1Local);
                x2Queue_.FreeTensor(x2Local);
                if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
                    biasQueue_.FreeTensor(biasLocal);
                }

                betaQueue_.FreeTensor(betaLocal);
                gammaQueue_.FreeTensor(gammaLocal);

                scaleQueue_.FreeTensor(scaleLocal);
                CONST_CONDITIONAL_EXPR(IS_OFFSET_EXIST, zeroOffsetQueue_.FreeTensor(offsetLocal));

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

    TQue<QuePosition::VECIN, BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> zeroOffsetQueue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
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

    GlobalTensor<SCALE_TYPE> scaleGm_;
    GlobalTensor<SCALE_TYPE> zeroOffsetGm_;

    GlobalTensor<int8_t> yGm_;
    GlobalTensor<X1_TYPE> xGm_;

    int64_t cols_;
    int64_t colsTail_;
    int64_t colsLoopCount_;
    int64_t colsPerLoop_;
    int64_t colsPerLoopAlign_;
    int64_t rowsPerCore_;
    int64_t binaryAddNum_;
    int64_t binaryAddK_;
    int64_t binaryAddLastNum_;
    int64_t powerOfTwo_;
    float reduceScale_;
    float reduceScaleCorrection_;
    float eps_;

    bool outputX_;
    TPipe* pipe_ = nullptr;
};
} // namespace QuantizeAddLayerNormRegbase
#endif // QUANTIZE_ADD_LAYER_NORM_STATIC_QUANT_REGBASE_WELFORD_H
