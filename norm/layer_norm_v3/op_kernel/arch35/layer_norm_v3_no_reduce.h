/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_v3_no_reduce.h
 * \brief
 */

#ifndef LAYER_NORM_V3_NO_REDUCE_H
#define LAYER_NORM_V3_NO_REDUCE_H

#include "layer_norm_v3_common.h"

namespace LayerNormV3 {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;
constexpr static float SCALAR1 = -0.5;
constexpr static float SCALAR2 = 1.5;
constexpr static float SCALAR3 = 0.5;
constexpr static float SCALAR0 = -99.99;

template <typename T, typename U, typename M, bool IsOutRstd>
class LayerNormV3RegbaseNoReduce {
public:
    __aicore__ inline LayerNormV3RegbaseNoReduce(const LayerNormV3TilingDataRegBaseNoReduce* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        tl_ = tilingData;
        hasGamma_ = (tilingData->nullptrGamma == 0);
        hasBeta_ = (tilingData->nullptrBeta == 0);
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd)
    {
        elemNum_ = tl_->aUbFactor;
        pipe_->InitBuffer(xQueue_, DOUBLE_BUFFER, elemNum_ * sizeof(T));
        pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, elemNum_ * sizeof(T));
        pipe_->InitBuffer(gammaBetaQueue_, 1, NUM_TWO * BLOCK_SIZE);
        pipe_->InitBuffer(meanQueue_, DOUBLE_BUFFER, elemNum_ * sizeof(M));
        pipe_->InitBuffer(rstdQueue_, DOUBLE_BUFFER, elemNum_ * sizeof(M));
        pipe_->InitBuffer(tmpBuf, elemNum_ * sizeof(float));

        gammaGm_.SetGlobalBuffer((__gm__ U*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ U*)beta);
        int64_t meanBlockOffset = GetBlockIdx() * tl_->aBlockFactor;
        int64_t xBlockOffset = meanBlockOffset;
        xGm_.SetGlobalBuffer((__gm__ T*)x + xBlockOffset);
        yGm_.SetGlobalBuffer((__gm__ T*)y + xBlockOffset);
        meanGm_.SetGlobalBuffer((__gm__ M*)mean + meanBlockOffset);
        rstdGm_.SetGlobalBuffer((__gm__ M*)rstd + meanBlockOffset);
    }
    __aicore__ inline void Process()
    {
        CopyInGammaBeta();
        gammaBetaInUb_ = gammaBetaQueue_.template DeQue<U>();
        int64_t ubLoopNum = GetBlockIdx() == GetBlockNum() - 1 ? tl_->tailBlockUbLoops : tl_->formerBlockUbLoops;
        int64_t tailA =
            GetBlockIdx() == GetBlockNum() - 1 ?
                tl_->a - tl_->aBlockFactor * (GetBlockNum() - 1) - (tl_->tailBlockUbLoops - 1) * tl_->aUbFactor :
                tl_->aBlockFactor - (tl_->formerBlockUbLoops - 1) * tl_->aUbFactor;
        // ub循环
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx++) {
            int64_t currentA = ubLoopIdx == ubLoopNum - 1 ? tailA : tl_->aUbFactor;
            int64_t offset = ubLoopIdx * tl_->aUbFactor;
            CopyInX(offset, currentA);
            Calculate(currentA);
            CopyOutY(offset, currentA);
            CopyOutMeanRstd(offset, currentA);
        }
    }

private:
    __aicore__ inline void CopyInGammaBeta()
    {
        DataCopyPadExtParams<U> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = sizeof(U);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        gammaBetaInUb_ = gammaBetaQueue_.AllocTensor<U>();
        if (hasGamma_) {
            DataCopyPad(gammaBetaInUb_, gammaGm_, copyInParams, dataCopyPadExtParams);
        }
        if (hasBeta_) {
            DataCopyPad(gammaBetaInUb_[BLOCK_SIZE / sizeof(U)], betaGm_, copyInParams, dataCopyPadExtParams);
        }
        gammaBetaQueue_.EnQue(gammaBetaInUb_);
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T> xInUb = xQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(xInUb, xGm_[offset], copyInParams, dataCopyPadExtParams);
        xQueue_.EnQue(xInUb);
    }
    __aicore__ inline void Calculate(int64_t currentANum)
    {
        LocalTensor<T> xInUb = xQueue_.template DeQue<T>();
        meanOutUb_ = meanQueue_.AllocTensor<M>();
        rstdOutUb_ = rstdQueue_.AllocTensor<M>();
        LocalTensor<float> tmpTensor = tmpBuf.Get<float>();

        __local_mem__ T* xInUbAddr = (__local_mem__ T*)xInUb.GetPhyAddr();
        __local_mem__ M* meanOutUbAddr = (__local_mem__ M*)meanOutUb_.GetPhyAddr();
        __local_mem__ M* rstdOutUbAddr = (__local_mem__ M*)rstdOutUb_.GetPhyAddr();
        __local_mem__ float* tmpUbAddr = (__local_mem__ float*)tmpTensor.GetPhyAddr();
        CalculateMeanVar(xInUbAddr, meanOutUbAddr, tmpUbAddr, currentANum);
        CalculateRstd(rstdOutUbAddr, tmpUbAddr, currentANum);

        LocalTensor<T> yOutUb = yQueue_.AllocTensor<T>();
        __local_mem__ U* gammaInUbAddr = (__local_mem__ U*)gammaBetaInUb_.GetPhyAddr();
        __local_mem__ U* betaInUbAddr = (__local_mem__ U*)gammaBetaInUb_.GetPhyAddr() + BLOCK_SIZE / sizeof(U);
        __local_mem__ T* yOutUbAddr = (__local_mem__ T*)yOutUb.GetPhyAddr();

        if (hasGamma_ && hasBeta_) {
            CalculateY<true, true>(xInUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, tmpUbAddr, currentANum);
        } else if (!hasGamma_ && hasBeta_) {
            CalculateY<false, true>(xInUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, tmpUbAddr, currentANum);
        } else if (hasGamma_ && !hasBeta_) {
            CalculateY<true, false>(xInUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, tmpUbAddr, currentANum);
        } else {
            CalculateY<false, false>(xInUbAddr, betaInUbAddr, gammaInUbAddr, yOutUbAddr, tmpUbAddr, currentANum);
        }
        xQueue_.FreeTensor(xInUb);
        yQueue_.EnQue(yOutUb);
    }

    __aicore__ inline void CalculateMeanVar(
        __local_mem__ T* xInUb, __local_mem__ M* meanInUb, __local_mem__ float* tmpUb, uint64_t currentANum)
    {
        uint16_t aLoop = static_cast<uint16_t>((currentANum + VL_B32 - 1) / VL_B32);
        uint32_t sreg = static_cast<uint32_t>(currentANum);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> subReg;
            RegTensor<float> varReg;
            for (uint16_t a = 0; a < aLoop; a++) {
                MaskReg pregLoop = UpdateMask<float>(sreg);
                LoadTensorForDtypeTIn(xInUb, xReg, pregLoop, (a * VL_B32));
                StoreTensorForDtypeTOut(meanInUb, xReg, pregLoop, (a * VL_B32));
                Sub(subReg, xReg, xReg, pregLoop);
                Mul(varReg, subReg, subReg, pregLoop);
                StoreTensorForDtypeTOut(tmpUb, varReg, pregLoop, (a * VL_B32));
            }
        }
    }

    __aicore__ inline void CopyOutMeanRstd(int64_t offset, int64_t currentANum)
    {
        meanQueue_.EnQue(meanOutUb_);
        rstdQueue_.EnQue(rstdOutUb_);
        LocalTensor<M> meanInUb = meanQueue_.template DeQue<M>();
        LocalTensor<M> rstdInUb = rstdQueue_.template DeQue<M>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(M);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(meanGm_[offset], meanInUb, copyInParams);
        DataCopyPad(rstdGm_[offset], rstdInUb, copyInParams);
        meanQueue_.FreeTensor(meanInUb);
        rstdQueue_.FreeTensor(rstdInUb);
    }
    __aicore__ inline void CalculateRstd(__local_mem__ M* rstdOutUb, __local_mem__ float* tmpUb, int64_t currentANum)
    {
        uint16_t aLoop = static_cast<uint16_t>((currentANum + VL_B32 - 1) / VL_B32);
        uint32_t sreg = static_cast<uint32_t>(currentANum);
        float epsilon = tl_->epsilon;
        __VEC_SCOPE__
        {
            RegTensor<float> varReg;
            RegTensor<float> rstdReg;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            for (uint16_t a = 0; a < aLoop; a++) {
                LoadTensorForDtypeTIn(tmpUb, varReg, pregLoop, (a * VL_B32));
                if constexpr (!IsOutRstd) {
                    StoreTensorForDtypeTOut(rstdOutUb, varReg, pregLoop, (a * VL_B32));
                }
                CalRstdByHighPrecision(varReg, rstdReg, epsilon);
                if constexpr (IsOutRstd) {
                    StoreTensorForDtypeTOut(rstdOutUb, rstdReg, pregLoop, (a * VL_B32));
                }
                StoreTensorForDtypeTOut(tmpUb, rstdReg, pregLoop, (a * VL_B32));
            }
        }
    }

    __aicore__ inline void CalRstdByHighPrecision(RegTensor<float>& var, RegTensor<float>& rstd, const float epsilon)
    {
        RegTensor<float> r;
        RegTensor<float> y;
        RegTensor<float> s;
        RegTensor<float> t;
        RegTensor<float> e;
        RegTensor<float> one;
        RegTensor<float> scalar1;
        RegTensor<float> scalar2;
        RegTensor<float> t1;
        RegTensor<float> t2;
        RegTensor<float> t3;
        RegTensor<float> t4;
        RegTensor<float> scalarInf;
        RegTensor<float> scalarZero;
        MaskReg cmpRegZero;
        MaskReg cmpRegInf;
        MaskReg pregMerge = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

        Duplicate(scalarInf, POS_INF, pregMerge);
        Duplicate(scalarZero, float(0.0), pregMerge);
        Duplicate(one, float(1.0), pregMerge);
        Duplicate(scalar1, SCALAR3, pregMerge);
        Duplicate(t1, SCALAR2, pregMerge);
        Duplicate(s, float(1.0), pregMerge);

        Adds(var, var, epsilon, pregMerge);
        // we need sqrt(1/var) = nan, when var < 0.
        // But div donot support subnormal(when var is less -1e38, 1/var will be 0), then sqrt(1/var) is 0.
        // So we do maxs to avoid the subnormal problem, sqrt(1/var) = nan
        Maxs(var, var, SCALAR0, pregMerge);
        Div(r, one, var, pregMerge);
        Sqrt(y, r, pregMerge);
        Muls(t, var, SCALAR1, pregMerge);
        Mul(t, t, y, pregMerge);                // -0.5 * x * y
        Mula(t1, t, y, pregMerge);              // 1.5 + (-0.5 * x * y) * y
        Mul(rstd, y, t1, pregMerge);            // y = y * (1.5 - 0.5 * x * y)
        Muls(t3, var, float(-1.0), pregMerge);  // -1 * x
        Mula(s, t3, r, pregMerge);              // 1 + (-1) * x * r
        Muls(t4, rstd, float(-1.0), pregMerge); // (-1) * y
        Mula(r, t4, rstd, pregMerge);           // r + (-1) * y * y
        Mula(s, var, r, pregMerge);             // s + x * t
        Mul(s, s, rstd, pregMerge);             // e * y
        Mula(rstd, s, scalar1, pregMerge);      // y + y * e * 0.5
        CompareScalar(cmpRegZero, var, POS_INF, pregMerge);
        Select(rstd, scalarZero, rstd, cmpRegZero);
        CompareScalar(cmpRegInf, var, float(0.0), pregMerge);
        Select(rstd, scalarInf, rstd, cmpRegInf);
    }
    template <bool hasGammaFlag, bool hasBetaFlag>
    __aicore__ inline void CalculateY(
        __local_mem__ T* xInUb, __local_mem__ U* betaInUb, __local_mem__ U* gammaInUb, __local_mem__ T* yOutUb,
        __local_mem__ float* tmpUb, int64_t currentANum)
    {
        uint16_t aLoop = static_cast<uint16_t>((currentANum + VL_B32 - 1) / VL_B32);
        uint32_t sreg = static_cast<uint32_t>(currentANum);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> yReg;
            RegTensor<float> varReg;
            RegTensor<float> rstdReg;
            RegTensor<float> gammaReg;
            RegTensor<float> betaReg;
            RegTensor<float> subReg;
            RegTensor<float> mulReg;
            MaskReg pregMask = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            if constexpr (hasGammaFlag) {
                LoadsTensorForDtypeT<U>(gammaInUb, gammaReg, pregMask, 1);
            }
            if constexpr (hasBetaFlag) {
                LoadsTensorForDtypeT<U>(betaInUb, betaReg, pregMask, 1);
            }
            MaskReg pregLoop = UpdateMask<float>(sreg);
            for (uint16_t a = 0; a < aLoop; a++) {
                LoadTensorForDtypeTIn(xInUb, xReg, pregLoop, (a * VL_B32));
                Sub(subReg, xReg, xReg, pregLoop);
                LoadTensorForDtypeTIn(tmpUb, rstdReg, pregLoop, (a * VL_B32));
                Mul(yReg, subReg, rstdReg, pregLoop);
                if constexpr (hasGammaFlag && hasBetaFlag) {
                    FusedMulDstAdd(yReg, gammaReg, betaReg, pregLoop);
                } else {
                    if constexpr (hasGammaFlag) {
                        Mul(yReg, yReg, gammaReg, pregLoop);
                    }
                    if constexpr (hasBetaFlag) {
                        Add(yReg, yReg, betaReg, pregLoop);
                    }
                }
                StoreTensorForDtypeTOut(yOutUb, yReg, pregLoop, (a * VL_B32));
            }
        }
    }
    __aicore__ inline void CopyOutY(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T> yOutUb = yQueue_.template DeQue<T>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(yGm_[offset], yOutUb, copyInParams);
        yQueue_.FreeTensor(yOutUb);
    }

    template <typename T_IN>
    __aicore__ inline void LoadTensorForDtypeTIn(
        __local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_IN, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
        } else {
            RegTensor<T_IN> xIn;
            DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
            Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
        }
    }

    template <typename T_OUT>
    __aicore__ inline void StoreTensorForDtypeTOut(
        __local_mem__ T_OUT* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_OUT, float>::value) {
            DataCopy<T_OUT, StoreDist::DIST_NORM>(dst + offset, src, preg);
        } else {
            RegTensor<T_OUT> xOut;
            Cast<T_OUT, float, castTraitB322B16>(xOut, src, preg);
            DataCopy<T_OUT, StoreDist::DIST_PACK_B32>(dst + offset, xOut, preg);
        }
    }

    template <typename H>
    __aicore__ inline void LoadsTensorForDtypeT(
        const __local_mem__ void* src, MicroAPI::RegTensor<float>& dst, MicroAPI::MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<H, float>::value) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(dst, (__local_mem__ float*)src + offset);
        } else { // fp16、bf16
            RegTensor<H> xFp16;
            DataCopy<H, LoadDist::DIST_BRC_B16>(xFp16, ((__local_mem__ H*)src + offset));
            Cast<float, H, castTraitB162B32>(dst, xFp16, preg);
        }
    }
    /* global memory address */
    GlobalTensor<T> xGm_;
    GlobalTensor<U> betaGm_;
    GlobalTensor<U> gammaGm_;

    GlobalTensor<T> yGm_;
    GlobalTensor<M> meanGm_;
    GlobalTensor<M> rstdGm_;

    /* variable */
    bool hasGamma_ = true;
    bool hasBeta_ = true;
    int64_t elemNum_ = -1;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;

    const LayerNormV3TilingDataRegBaseNoReduce* tl_ = nullptr;

    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> gammaBetaQueue_;

    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECOUT, 1> meanQueue_;
    TQue<QuePosition::VECOUT, 1> rstdQueue_;

    TBuf<TPosition::VECCALC> tmpBuf;
    LocalTensor<U> gammaBetaInUb_;
    LocalTensor<M> meanOutUb_;
    LocalTensor<M> rstdOutUb_;
};
} // namespace LayerNormV3

#endif // LAYER_NORM_V3_NO_REDUCE_H