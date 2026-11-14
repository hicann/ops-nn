/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LAYER_NORM_V4_REGBASE_TRANSPOSE_H
#define LAYER_NORM_V4_REGBASE_TRANSPOSE_H

#include "layer_norm_v4_regbase_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace LayerNormV4 {
using namespace AscendC::Reg;
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

template <typename T, typename U>
class LayerNormV4RegbaseTranspose {
    static constexpr uint32_t TRANSPOSE_SIZE = NCHW_CONV_ADDR_LIST_SIZE;
    static constexpr uint32_t B16_PER_B32 = sizeof(float) / sizeof(uint16_t);
    static constexpr uint32_t B32_TRANSPOSE_COLS = TRANSPOSE_SIZE / B16_PER_B32;
    static constexpr uint64_t PACK_OUTPUT_MIN_A = 150000;

public:
    __aicore__ inline LayerNormV4RegbaseTranspose(const LayerNormV4TilingDataRegBaseTranspose* tiling, TPipe* pipe)
    {
        td_ = tiling;
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd)
    {
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
        gammaGm_.SetGlobalBuffer((__gm__ U*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ U*)beta);
        meanGm_.SetGlobalBuffer((__gm__ float*)mean);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd);
        tileA_ = td_->tileA;
        r_ = td_->r;
        rAlign_ = td_->rAlign;
        paramAlign_ = td_->paramAlign;
        hasGamma_ = td_->nullptrGamma == 0;
        hasBeta_ = td_->nullptrBeta == 0;
        packOutput_ = (r_ == 1 || r_ == NUM_TWO) && td_->a >= PACK_OUTPUT_MIN_A;

        pipe_->InitBuffer(xQueue_, DOUBLE_BUFFER, tileA_ * r_ * sizeof(float));
        pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, tileA_ * rAlign_ * sizeof(T));
        const uint32_t tempBytes = packOutput_ ? tileA_ * r_ * (sizeof(float) + sizeof(T)) :
                                                 tileA_ * rAlign_ * sizeof(float);
        pipe_->InitBuffer(tempBuf_, tempBytes);
        pipe_->InitBuffer(meanQueue_, DOUBLE_BUFFER, tileA_ * sizeof(float));
        pipe_->InitBuffer(rstdQueue_, DOUBLE_BUFFER, tileA_ * sizeof(float));
        pipe_->InitBuffer(paramsQueue_, 1, NUM_TWO * paramAlign_ * sizeof(U));
        uint32_t powerOfTwo = 1;
        while (powerOfTwo < r_) {
            powerOfTwo *= NUM_TWO;
        }
        scale_ = 1.0f / static_cast<float>(powerOfTwo);
        powerOfTwo_ = static_cast<float>(powerOfTwo);
    }

    __aicore__ inline void Process()
    {
        const uint64_t begin = GetBlockIdx() * td_->aBlockFactor;
        if (begin >= td_->a) {
            return;
        }
        const uint64_t rows = td_->a - begin < td_->aBlockFactor ? td_->a - begin : td_->aBlockFactor;
        const uint64_t tileCount = (rows + tileA_ - 1) / tileA_;
        const uint32_t tailA = rows - (tileCount - 1) * tileA_;
        CopyInParams();
        paramsLocal_ = paramsQueue_.template DeQue<U>();
        NdDmaDci();
        CopyInX(begin, tileCount == 1 ? tailA : tileA_);
        for (uint64_t idx = 0; idx + 1 < tileCount; ++idx) {
            const uint64_t row = begin + idx * tileA_;
            const uint32_t nextA = idx + NUM_TWO == tileCount ? tailA : tileA_;
            ComputeMeanRstd(tileA_);
            CopyInX(row + tileA_, nextA);
            ComputeOutput(tileA_);
            CopyOutY(row, tileA_);
            CopyOutMeanRstd(row, tileA_);
        }

        const uint64_t lastRow = begin + (tileCount - 1) * tileA_;
        ComputeMeanRstd(tailA);
        ComputeOutput(tailA);
        CopyOutY(lastRow, tailA);
        CopyOutMeanRstd(lastRow, tailA);
        paramsQueue_.FreeTensor(paramsLocal_);
    }

private:
    __aicore__ inline void CopyInParams()
    {
        LocalTensor<U> params = paramsQueue_.template AllocTensor<U>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(r_ * sizeof(U)), 0, 0, 0};
        DataCopyPadExtParams<U> pad{false, 0, 0, 0};
        if (hasGamma_) {
            DataCopyPad(params, gammaGm_, copy, pad);
        }
        if (hasBeta_) {
            DataCopyPad(params[paramAlign_], betaGm_, copy, pad);
        }
        paramsQueue_.EnQue(params);
    }

    __aicore__ inline void CopyInX(uint64_t row, uint32_t curA)
    {
        static constexpr NdDmaConfig config = {false};
        NdDmaLoopInfo<NUM_TWO> loop;
        loop.loopSrcStride[0] = 1;
        loop.loopSrcStride[1] = r_;
        loop.loopDstStride[0] = tileA_ * sizeof(float) / sizeof(T);
        loop.loopDstStride[1] = 1;
        loop.loopSize[0] = r_;
        loop.loopSize[1] = curA;
        NdDmaParams<T, NUM_TWO> params = {loop, 0};
        LocalTensor<float> x = xQueue_.template AllocTensor<float>();
        LocalTensor<T> input = x.template ReinterpretCast<T>();
        if constexpr (!IsSameType<T, float>::value) {
            // Place B16 input in each FP32 row's upper half for safe forward expansion.
            input = input[tileA_];
        }
        DataCopy<T, NUM_TWO, config>(input, xGm_[row * r_], params);
        xQueue_.EnQue(x);
    }

    __aicore__ inline void ComputeMeanRstd(uint32_t curA)
    {
        xLocal_ = xQueue_.template DeQue<float>();
        meanLocal_ = meanQueue_.template AllocTensor<float>();
        rstdLocal_ = rstdQueue_.template AllocTensor<float>();
        CalculateMeanRstd(xLocal_, meanLocal_, rstdLocal_, curA);
    }

    __aicore__ inline void ComputeOutput(uint32_t curA)
    {
        LocalTensor<T> tempY = tempBuf_.Get<T>();
        if (hasGamma_ && hasBeta_) {
            Normalize<true, true>(tempY, rstdLocal_, curA);
        } else if (hasGamma_) {
            Normalize<true, false>(tempY, rstdLocal_, curA);
        } else if (hasBeta_) {
            Normalize<false, true>(tempY, rstdLocal_, curA);
        } else {
            Normalize<false, false>(tempY, rstdLocal_, curA);
        }
        xQueue_.FreeTensor(xLocal_);
        meanQueue_.EnQue(meanLocal_);
        rstdQueue_.EnQue(rstdLocal_);
    }

    __aicore__ inline void CalculateMeanRstd(const LocalTensor<float>& x, const LocalTensor<float>& mean,
                                             const LocalTensor<float>& rstd, uint32_t curA)
    {
        LocalTensor<float> temp = tempBuf_.Get<float>();
        // RA source reuse needs no separate workspace or output-queue slot.
        LocalTensor<uint8_t> reduceTmp = temp.template ReinterpretCast<uint8_t>();
        uint32_t srcShape[NUM_TWO] = {r_, tileA_};
        CalcReduceInput<false>(x, temp, mean, curA);
        AscendC::ReduceSum<float, Pattern::Reduce::RA, true>(mean, temp, reduceTmp, srcShape, false);
        CalcMeanOrRstd<false>(mean, curA);
        if (r_ == NUM_TWO) {
            CalcR2Variance(x, rstd, curA);
            return;
        }
        CalcReduceInput<true>(x, temp, mean, curA);
        AscendC::ReduceSum<float, Pattern::Reduce::RA, true>(rstd, temp, reduceTmp, srcShape, false);
        CalcMeanOrRstd<true>(rstd, curA);
    }

    template <bool CALC_VARIANCE>
    __aicore__ inline void CalcReduceInput(const LocalTensor<float>& x, const LocalTensor<float>& temp,
                                           const LocalTensor<float>& mean, uint32_t curA)
    {
        __ubuf__ T* xAddr = (__ubuf__ T*)x.GetPhyAddr();
        __ubuf__ float* xSubMeanAddr = (__ubuf__ float*)x.GetPhyAddr();
        const uint32_t inputOffset = IsSameType<T, float>::value ? 0 : tileA_;
        const uint32_t inputStride = tileA_ * sizeof(float) / sizeof(T);
        __ubuf__ float* tempAddr = (__ubuf__ float*)temp.GetPhyAddr();
        __ubuf__ float* meanAddr = (__ubuf__ float*)mean.GetPhyAddr();
        const uint32_t stride = tileA_;
        const uint16_t rLoops = r_;
        const uint16_t aLoops = (curA + VL_B32 - 1) / VL_B32;
        const float scale = scale_;
        uint32_t maskCount = curA;
        __VEC_SCOPE__
        {
            MaskReg mask;
            RegTensor<float> xReg, meanReg, xSubMeanReg, reduceReg;
            for (uint16_t a = 0; a < aLoops; ++a) {
                mask = UpdateMask<float>(maskCount);
                const uint32_t aOffset = a * VL_B32;
                if constexpr (CALC_VARIANCE) {
                    LoadAlign(meanReg, meanAddr + aOffset);
                }
                for (uint16_t r = 0; r < rLoops; ++r) {
                    const uint32_t index = r * stride + aOffset;
                    LoadRegForDtype<T>(xAddr, xReg, mask, inputOffset + r * inputStride + aOffset);
                    if constexpr (CALC_VARIANCE) {
                        Sub(xSubMeanReg, xReg, meanReg, mask);
                        StoreAlign(xSubMeanAddr + index, xSubMeanReg, mask);
                        Mul(reduceReg, xSubMeanReg, xSubMeanReg, mask);
                        Muls(reduceReg, reduceReg, scale, mask);
                    } else {
                        Muls(reduceReg, xReg, scale, mask);
                    }
                    StoreAlign(tempAddr + index, reduceReg, mask);
                }
            }
        }
    }

    // R=2: h = (x0 - x1) / 2; var = h * h. This avoids the mean-rounding term
    // that the generic (x - mean32)^2 path adds for near-equal inputs.
    __aicore__ inline void CalcR2Variance(const LocalTensor<float>& x, const LocalTensor<float>& rstd, uint32_t curA)
    {
        __ubuf__ T* xAddr = (__ubuf__ T*)x.GetPhyAddr();
        // Reuse the x buffer for deviations, same as the generic path.
        __ubuf__ float* xSubMeanAddr = (__ubuf__ float*)x.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd.GetPhyAddr();
        const uint32_t inputOffset = IsSameType<T, float>::value ? 0 : tileA_;
        const uint32_t inputStride = tileA_ * sizeof(float) / sizeof(T);
        const uint16_t aLoops = (curA + VL_B32 - 1) / VL_B32;
        const float epsilon = td_->epsilon;
        uint32_t maskCount = curA;
        __VEC_SCOPE__
        {
            MaskReg mask;
            RegTensor<float> x0, x1, h, negH, var, rstdReg;
            for (uint16_t a = 0; a < aLoops; ++a) {
                mask = UpdateMask<float>(maskCount);
                const uint32_t aOffset = a * VL_B32;
                LoadRegForDtype<T>(xAddr, x0, mask, inputOffset + aOffset);
                LoadRegForDtype<T>(xAddr, x1, mask, inputOffset + inputStride + aOffset);
                Sub(h, x0, x1, mask);
                Muls(h, h, 0.5f, mask);
                Mul(var, h, h, mask);
                Muls(negH, h, -1.0f, mask);
                StoreAlign(xSubMeanAddr + aOffset, h, mask);
                StoreAlign(xSubMeanAddr + tileA_ + aOffset, negH, mask);
                NormCommon::ComputeRstdNewtonRaphsonReg<false>(var, rstdReg, mask, epsilon);
                StoreAlign(rstdAddr + aOffset, rstdReg, mask);
            }
        }
    }

    template <bool CALC_RSTD>
    __aicore__ inline void CalcMeanOrRstd(const LocalTensor<float>& result, uint32_t curA)
    {
        __ubuf__ float* resultAddr = (__ubuf__ float*)result.GetPhyAddr();
        const uint16_t aLoops = (curA + VL_B32 - 1) / VL_B32;
        const float powerOfTwo = powerOfTwo_;
        const float reduceNum = static_cast<float>(r_);
        const float epsilon = td_->epsilon;
        uint32_t maskCount = curA;
        __VEC_SCOPE__
        {
            MaskReg mask;
            RegTensor<float> resultReg, rReg, rstdReg;
            for (uint16_t a = 0; a < aLoops; ++a) {
                mask = UpdateMask<float>(maskCount);
                const uint32_t aOffset = a * VL_B32;
                LoadAlign(resultReg, resultAddr + aOffset);
                Duplicate(rReg, reduceNum, mask);
                Muls(resultReg, resultReg, powerOfTwo, mask);
                Div(resultReg, resultReg, rReg, mask);
                if constexpr (CALC_RSTD) {
                    NormCommon::ComputeRstdNewtonRaphsonReg<false>(resultReg, rstdReg, mask, epsilon);
                    StoreAlign(resultAddr + aOffset, rstdReg, mask);
                } else {
                    StoreAlign(resultAddr + aOffset, resultReg, mask);
                }
            }
        }
    }

    template <bool HAS_GAMMA, bool HAS_BETA>
    __aicore__ inline void Normalize(const LocalTensor<T>& y, const LocalTensor<float>& rstd, uint32_t curA)
    {
        __ubuf__ float* xSubMeanAddr = (__ubuf__ float*)xLocal_.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd.GetPhyAddr();
        __ubuf__ U* gamma = (__ubuf__ U*)paramsLocal_.GetPhyAddr();
        __ubuf__ U* beta = gamma + paramAlign_;
        __ubuf__ T* yAddr = (__ubuf__ T*)y.GetPhyAddr();
        const uint32_t stride = tileA_;
        const uint16_t rLoops = r_;
        const uint16_t aLoops = (curA + VL_B32 - 1) / VL_B32;
        uint32_t maskCount = curA;
        __VEC_SCOPE__
        {
            MaskReg mask;
            RegTensor<float> xSubMeanReg, yReg, rstdReg, gammaReg, betaReg;
            for (uint16_t a = 0; a < aLoops; ++a) {
                mask = UpdateMask<float>(maskCount);
                const uint32_t aOffset = a * VL_B32;
                LoadAlign(rstdReg, rstdAddr + aOffset);
                for (uint16_t r = 0; r < rLoops; ++r) {
                    const uint32_t index = r * stride + aOffset;
                    LoadAlign(xSubMeanReg, xSubMeanAddr + index);
                    Mul(yReg, xSubMeanReg, rstdReg, mask);
                    if constexpr (HAS_GAMMA) {
                        LoadRegForDtype<U, LoadDist::DIST_BRC_B32, LoadDist::DIST_BRC_B16>(gamma, gammaReg, mask, r);
                    }
                    if constexpr (HAS_BETA) {
                        LoadRegForDtype<U, LoadDist::DIST_BRC_B32, LoadDist::DIST_BRC_B16>(beta, betaReg, mask, r);
                    }
                    if constexpr (HAS_GAMMA && HAS_BETA) {
                        MulDstAdd(yReg, gammaReg, betaReg, mask);
                    } else if constexpr (HAS_GAMMA) {
                        Mul(yReg, yReg, gammaReg, mask);
                    } else if constexpr (HAS_BETA) {
                        Add(yReg, yReg, betaReg, mask);
                    }
                    StoreRegForDtype<T>(yAddr, yReg, mask, index);
                }
            }
        }
    }

    __aicore__ inline void TransposeOutput(const LocalTensor<T>& src, const LocalTensor<T>& dst, uint32_t curA)
    {
        const uint32_t aRepeats = (curA + TRANSPOSE_SIZE - 1) / TRANSPOSE_SIZE;
        constexpr uint32_t cols = IsSameType<T, float>::value ? B32_TRANSPOSE_COLS : TRANSPOSE_SIZE;
        const uint32_t rRepeats = rAlign_ / cols;
        for (uint32_t r = 0; r < rRepeats; ++r) {
            TransDataTo5HDParams params;
            params.repeatTimes = aRepeats;
            params.srcRepStride = aRepeats == 1 ? 0 : sizeof(T) / sizeof(uint16_t);
            params.dstRepStride = aRepeats == 1 ? 0 : TRANSPOSE_SIZE * rRepeats;
            LocalTensor<T> srcList[TRANSPOSE_SIZE];
            LocalTensor<T> dstList[TRANSPOSE_SIZE];
            if constexpr (IsSameType<T, float>::value) {
                for (uint32_t j = 0; j < B32_TRANSPOSE_COLS; ++j) {
                    const uint32_t srcOffset = (r * B32_TRANSPOSE_COLS + j) * tileA_;
                    srcList[j] = src[srcOffset];
                    srcList[j + B32_TRANSPOSE_COLS] = src[srcOffset + B32_TRANSPOSE_COLS];
                    const uint32_t dstOffset = r * B32_TRANSPOSE_COLS + rAlign_ * j;
                    dstList[j * B16_PER_B32] = dst[dstOffset];
                    dstList[j * B16_PER_B32 + 1] = dst[dstOffset + rAlign_ * B32_TRANSPOSE_COLS];
                }
            } else {
                for (uint32_t j = 0; j < TRANSPOSE_SIZE; ++j) {
                    srcList[j] = src[(r * TRANSPOSE_SIZE + j) * tileA_];
                    dstList[j] = dst[r * TRANSPOSE_SIZE + rAlign_ * j];
                }
            }
            TransDataTo5HD<T>(dstList, srcList, params);
        }
    }

    __aicore__ inline void PackOutput(const LocalTensor<T>& src, const LocalTensor<T>& dst)
    {
        const uint32_t groupRows = tileA_ / TRANSPOSE_SIZE;
        constexpr uint32_t cols = IsSameType<T, float>::value ? B32_TRANSPOSE_COLS : TRANSPOSE_SIZE;
        uint64_t srcList[TRANSPOSE_SIZE];
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        uint64_t dstList[TRANSPOSE_SIZE];
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        TransDataTo5HDParams params;
        params.repeatTimes = groupRows / cols;
        params.srcRepStride = params.repeatTimes == 1 ? 0 : 1;
        params.dstRepStride = params.repeatTimes == 1 ? 0 : TRANSPOSE_SIZE * r_;
        for (uint32_t r = 0; r < r_; ++r) {
            for (uint32_t group = 0; group < TRANSPOSE_SIZE; ++group) {
                srcList[group] = (uint64_t)(srcAddr + r * tileA_ + group * groupRows);
                if constexpr (IsSameType<T, float>::value) {
                    dstList[group] = (uint64_t)(dstAddr + (group / B16_PER_B32 * r_ + r) * TRANSPOSE_SIZE +
                                                group % B16_PER_B32 * B32_TRANSPOSE_COLS);
                } else {
                    dstList[group] = (uint64_t)(dstAddr + (group * r_ + r) * TRANSPOSE_SIZE);
                }
            }
            TransDataTo5HD<T>(dstList, srcList, params);
        }
    }

    __aicore__ inline void TransposePackedOutput(const LocalTensor<T>& src, const LocalTensor<T>& dst)
    {
        const uint32_t groupLen = tileA_ / TRANSPOSE_SIZE * r_;
        uint64_t srcList[TRANSPOSE_SIZE];
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        uint64_t dstList[TRANSPOSE_SIZE];
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        TransDataTo5HDParams params;
        params.repeatTimes = groupLen / TRANSPOSE_SIZE;
        params.srcRepStride = params.repeatTimes == 1 ? 0 : TRANSPOSE_SIZE * sizeof(T) / sizeof(uint16_t);
        params.dstRepStride = params.repeatTimes == 1 ? 0 : sizeof(T) / sizeof(uint16_t);
        constexpr uint32_t parts = sizeof(T) / sizeof(uint16_t);
        for (uint32_t part = 0; part < parts; ++part) {
            for (uint32_t idx = 0; idx < TRANSPOSE_SIZE; ++idx) {
                srcList[idx] = (uint64_t)(srcAddr + idx * TRANSPOSE_SIZE + part * B32_TRANSPOSE_COLS);
                if constexpr (IsSameType<T, float>::value) {
                    dstList[idx] = (uint64_t)(dstAddr + (part * B32_TRANSPOSE_COLS + idx / B16_PER_B32) * groupLen +
                                              (idx % B16_PER_B32) * B32_TRANSPOSE_COLS);
                } else {
                    dstList[idx] = (uint64_t)(dstAddr + idx * groupLen);
                }
            }
            TransDataTo5HD<T>(dstList, srcList, params);
        }
    }

    __aicore__ inline void CopyOutY(uint64_t row, uint32_t curA)
    {
        LocalTensor<T> src = tempBuf_.Get<T>();
        LocalTensor<T> dst = yQueue_.template AllocTensor<T>();
        if (packOutput_) {
            // Keep the RA source separate from the packed [B * R, 16] input to 5HD.
            LocalTensor<T> packed = src[tileA_ * r_ * sizeof(float) / sizeof(T)];
            PackOutput(src, packed);
            PipeBarrier<PIPE_V>();
            TransposePackedOutput(packed, dst);
        } else {
            TransposeOutput(src, dst, curA);
        }
        yQueue_.EnQue(dst);
        dst = yQueue_.template DeQue<T>();
        DataCopyExtParams copy;
        copy.blockCount = packOutput_ ? 1 : curA;
        copy.blockLen = (packOutput_ ? curA : 1) * r_ * sizeof(T);
        copy.srcStride = 0;
        copy.dstStride = 0;
        DataCopyPad(yGm_[row * r_], dst, copy);
        yQueue_.FreeTensor(dst);
    }

    __aicore__ inline void CopyOutMeanRstd(uint64_t row, uint32_t curA)
    {
        LocalTensor<float> mean = meanQueue_.template DeQue<float>();
        LocalTensor<float> rstd = rstdQueue_.template DeQue<float>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(curA * sizeof(float)), 0, 0, 0};
        DataCopyPad(meanGm_[row], mean, copy);
        DataCopyPad(rstdGm_[row], rstd, copy);
        meanQueue_.FreeTensor(mean);
        rstdQueue_.FreeTensor(rstd);
    }

    const LayerNormV4TilingDataRegBaseTranspose* td_;
    TPipe* pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQueue_;
    TQue<QuePosition::VECIN, 1> paramsQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> meanQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> rstdQueue_;
    TBuf<TPosition::VECCALC> tempBuf_;
    GlobalTensor<T> xGm_, yGm_;
    GlobalTensor<U> gammaGm_, betaGm_;
    GlobalTensor<float> meanGm_, rstdGm_;
    LocalTensor<U> paramsLocal_;
    LocalTensor<float> xLocal_;
    LocalTensor<float> meanLocal_, rstdLocal_;
    uint32_t tileA_, r_, rAlign_, paramAlign_;
    float scale_, powerOfTwo_;
    bool hasGamma_, hasBeta_, packOutput_;
};
} // namespace LayerNormV4
#endif // LAYER_NORM_V4_REGBASE_TRANSPOSE_H
