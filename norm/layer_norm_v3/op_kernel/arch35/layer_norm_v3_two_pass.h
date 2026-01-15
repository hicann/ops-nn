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
 * \file layer_norm_v3_two_pass.h
 * \brief
 */

#ifndef LAYER_NORM_V3_TWO_PASS_H
#define LAYER_NORM_V3_TWO_PASS_H

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

constexpr static LayerNormConfig hasGammaBetaConfig = {
    false,
    false,
    false,
};

constexpr static LayerNormConfig hasGammaNoBetaConfig = {
    true,
    false,
    false,
};

constexpr static LayerNormConfig noGammaHasBetaConfig = {
    false,
    true,
    false,
};

constexpr static LayerNormConfig noGammaNoBetaConfig = {
    true,
    true,
    false,
};

constexpr static LayerNormConfig hasGammaBetaOutVarConfig = {
    false,
    false,
    false,
    false,
};

constexpr static LayerNormConfig hasGammaNoBetaOutVarConfig = {
    true,
    false,
    false,
    false,
};

constexpr static LayerNormConfig noGammaHasBetaOutVarConfig = {
    false,
    true,
    false,
    false,
};

constexpr static LayerNormConfig noGammaNoBetaOutVarConfig = {
    true,
    true,
    false,
    false,
};

template <typename T, typename U, typename M, bool IsOutRstd>
class LayerNormV3RegbaseTwoPass {
public:
    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y > 0) {
            return (x + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline LayerNormV3RegbaseTwoPass(const LayerNormV3TilingDataRegBaseTwoPass* tilingData)
    {
        this->r = tilingData->r;
        this->aFactor = tilingData->aFactor;
        this->a = tilingData->a;
        this->rAlign = tilingData->rAlign;
        this->blockNum = tilingData->blockNum;
        this->aBlockFactor = tilingData->aBlockFactor;
        this->epsilon = tilingData->epsilon;

        this->tmpBufferSize = tilingData->tmpBufferSize;

        this->powerOfTwoForR = tilingData->powerOfTwoForR;
        this->binaryAddQuotient = tilingData->binaryAddQuotient;
        this->binaryAddK = tilingData->binaryAddK;
        this->binaryAddLastNum = tilingData->binaryAddLastNum;

        this->hasGamma = (tilingData->nullptrGamma == 0);
        this->hasBeta = (tilingData->nullptrBeta == 0);

        this->layerNormTiling = tilingData->layerNormTiling;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR lastout)
    {
        auto blockIdx = GetBlockIdx();

        this->singleA = (blockIdx == this->blockNum - 1) ? (this->a - this->aBlockFactor * (this->blockNum - 1)) :
                                                           this->aBlockFactor;

        int64_t meanBlockOffset = this->aBlockFactor * blockIdx;
        int64_t xBlockOffset = meanBlockOffset * this->r;
        xGm.SetGlobalBuffer((__gm__ T*)x + xBlockOffset);
        gammaGm.SetGlobalBuffer((__gm__ U*)gamma);
        betaGm.SetGlobalBuffer((__gm__ U*)beta);

        yGm.SetGlobalBuffer((__gm__ T*)y + xBlockOffset);
        batchMeanGm.SetGlobalBuffer((__gm__ M*)mean + meanBlockOffset);
        batchLastoutGm.SetGlobalBuffer((__gm__ M*)lastout + meanBlockOffset);

        if (this->hasGamma) {
            pipe.InitBuffer(gammaQueue, 1, this->rAlign * sizeof(U));
        }
        if (this->hasBeta) {
            pipe.InitBuffer(betaQueue, 1, this->rAlign * sizeof(U));
        }
        int64_t aFactorAlignF32 =
            (((this->aFactor * FLOAT_BYTES + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / FLOAT_BYTES;
        pipe.InitBuffer(batchMeanQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        pipe.InitBuffer(batchLastoutQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));

        int64_t xBufferSize = this->aFactor * this->rAlign;
        pipe.InitBuffer(xQueue, DOUBLE_BUFFER, xBufferSize * sizeof(T));
        pipe.InitBuffer(yQueue, DOUBLE_BUFFER, xBufferSize * sizeof(T));

        if (this->tmpBufferSize > 0) {
            pipe.InitBuffer(tmpBuf, this->tmpBufferSize);
        }
    }

    __aicore__ inline void Process()
    {
        CopyInGammaBeta();
        LocalTensor<U> gammaInUb;
        LocalTensor<U> betaInUb;
        if (hasGamma) {
            gammaInUb = gammaQueue.template DeQue<U>();
        }
        if (hasBeta) {
            betaInUb = betaQueue.template DeQue<U>();
        }
        int64_t quotient = (this->singleA + this->aFactor - 1) / this->aFactor;
        for (int64_t ubLoopIdx = 0; ubLoopIdx < quotient; ubLoopIdx++) {
            int64_t offset = ubLoopIdx * this->aFactor * this->r;
            int64_t aOffset = ubLoopIdx * this->aFactor;
            int64_t currentA =
                (ubLoopIdx == (quotient - 1)) ? (this->singleA - (quotient - 1) * this->aFactor) : this->aFactor;
            CopyInX(offset, currentA);
            CaculateWithHighLevelApi(gammaInUb, betaInUb, currentA);
            CopyOutY(offset, currentA);
            CopyOutBatchMeanLastout(aOffset, currentA);
        }
        if (hasGamma) {
            gammaQueue.FreeTensor(gammaInUb);
        }
        if (hasBeta) {
            betaQueue.FreeTensor(betaInUb);
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
        copyInParams.blockLen = this->r * sizeof(U);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        if (hasGamma) {
            LocalTensor<U> gammaInUb = gammaQueue.AllocTensor<U>();
            DataCopyPad(gammaInUb, gammaGm, copyInParams, dataCopyPadExtParams);
            gammaQueue.EnQue(gammaInUb);
        }
        if (hasBeta) {
            LocalTensor<U> betaInUb = betaQueue.AllocTensor<U>();
            DataCopyPad(betaInUb, betaGm, copyInParams, dataCopyPadExtParams);
            betaQueue.EnQue(betaInUb);
        }
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T> xInUb = xQueue.AllocTensor<T>();
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = (this->r != this->rAlign);
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = (this->rAlign - this->r);
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentANum;
        copyInParams.blockLen = this->r * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
        xQueue.EnQue(xInUb);
    }

    __aicore__ inline void Caculate(LocalTensor<U> gammaInUb, LocalTensor<U> betaInUb, int64_t currentANum)
    {
        LocalTensor<T> xInUb = xQueue.template DeQue<T>();
        LocalTensor<T> yInUb = yQueue.AllocTensor<T>();
        batchMeanOutUb = batchMeanQueue.AllocTensor<float>();
        batchLastoutOutUb = batchLastoutQueue.AllocTensor<float>();

        __local_mem__ T* xInUbAddr = (__local_mem__ T*)xInUb.GetPhyAddr();
        __local_mem__ U* gammaInUbAddr = nullptr;
        __local_mem__ U* betaInUbAddr = nullptr;
        if (hasGamma) {
            gammaInUbAddr = (__local_mem__ U*)gammaInUb.GetPhyAddr();
        }
        if (hasBeta) {
            betaInUbAddr = (__local_mem__ U*)betaInUb.GetPhyAddr();
        }
        __local_mem__ T* yInUbAddr = (__local_mem__ T*)yInUb.GetPhyAddr();
        __local_mem__ float* batchMeanInUbAddr = (__local_mem__ float*)batchMeanOutUb.GetPhyAddr();
        __local_mem__ float* batchLastoutInUbAddr = (__local_mem__ float*)batchLastoutOutUb.GetPhyAddr();
        if (this->r <= VL_F32) {
            CalculateMeanVarRLessThanVL64VF(xInUbAddr, batchMeanInUbAddr, batchLastoutInUbAddr, currentANum);
        } else {
            CalculateMeanVarVF(xInUbAddr, batchMeanInUbAddr, batchLastoutInUbAddr, currentANum);
        }

        CalculateRuningMeanVarVF(batchMeanInUbAddr, batchLastoutInUbAddr, currentANum);
        if (this->hasGamma && this->hasBeta) {
            CalculateNormalizeVF<true, true>(
                xInUbAddr, yInUbAddr, betaInUbAddr, gammaInUbAddr, batchMeanInUbAddr, batchLastoutInUbAddr,
                currentANum);
        } else if (!this->hasGamma && this->hasBeta) {
            CalculateNormalizeVF<false, true>(
                xInUbAddr, yInUbAddr, betaInUbAddr, gammaInUbAddr, batchMeanInUbAddr, batchLastoutInUbAddr,
                currentANum);
        } else if (this->hasGamma && !this->hasBeta) {
            CalculateNormalizeVF<true, false>(
                xInUbAddr, yInUbAddr, betaInUbAddr, gammaInUbAddr, batchMeanInUbAddr, batchLastoutInUbAddr,
                currentANum);
        } else {
            CalculateNormalizeVF<false, false>(
                xInUbAddr, yInUbAddr, betaInUbAddr, gammaInUbAddr, batchMeanInUbAddr, batchLastoutInUbAddr,
                currentANum);
        }
        xQueue.FreeTensor(xInUb);
        yQueue.EnQue(yInUb);
    }

    __aicore__ inline void CaculateWithHighLevelApi(
        LocalTensor<U> gammaInUb, LocalTensor<U> betaInUb, int64_t currentANum)
    {
        LocalTensor<T> xInUb = xQueue.template DeQue<T>();
        LocalTensor<T> yInUb = yQueue.AllocTensor<T>();
        batchMeanOutUb = batchMeanQueue.AllocTensor<float>();
        batchLastoutOutUb = batchLastoutQueue.AllocTensor<float>();
        LocalTensor<uint8_t> binaryAddTensor = tmpBuf.Get<uint8_t>();

        LayerNormPara para;
        para.aLength = currentANum;
        para.rLength = this->layerNormTiling.rLength;
        para.rLengthWithPadding = this->rAlign;
        if constexpr (IsOutRstd) {
            if (this->hasGamma && this->hasBeta) {
                AscendC::LayerNorm<U, T, true, hasGammaBetaConfig>(
                    yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb, gammaInUb, betaInUb, this->epsilon,
                    binaryAddTensor, para, this->layerNormTiling);
            } else if (!this->hasGamma && this->hasBeta) {
                AscendC::LayerNorm<U, T, true, noGammaHasBetaConfig>(
                    yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb, gammaInUb, betaInUb, this->epsilon,
                    binaryAddTensor, para, this->layerNormTiling);
            } else if (this->hasGamma && !this->hasBeta) {
                AscendC::LayerNorm<U, T, true, hasGammaNoBetaConfig>(
                    yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb, gammaInUb, betaInUb, this->epsilon,
                    binaryAddTensor, para, this->layerNormTiling);
            } else {
                AscendC::LayerNorm<U, T, true, noGammaNoBetaConfig>(
                    yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb, gammaInUb, betaInUb, this->epsilon,
                    binaryAddTensor, para, this->layerNormTiling);
            }
        } else {
            if (this->hasGamma && this->hasBeta) {
                AscendC::LayerNorm<U, T, true, hasGammaBetaOutVarConfig>(
                    yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb, gammaInUb, betaInUb, this->epsilon,
                    binaryAddTensor, para, this->layerNormTiling);
            } else if (!this->hasGamma && this->hasBeta) {
                AscendC::LayerNorm<U, T, true, noGammaHasBetaOutVarConfig>(
                    yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb, gammaInUb, betaInUb, this->epsilon,
                    binaryAddTensor, para, this->layerNormTiling);
            } else if (this->hasGamma && !this->hasBeta) {
                AscendC::LayerNorm<U, T, true, hasGammaNoBetaOutVarConfig>(
                    yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb, gammaInUb, betaInUb, this->epsilon,
                    binaryAddTensor, para, this->layerNormTiling);
            } else {
                AscendC::LayerNorm<U, T, true, noGammaNoBetaOutVarConfig>(
                    yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb, gammaInUb, betaInUb, this->epsilon,
                    binaryAddTensor, para, this->layerNormTiling);
            }
        }

        xQueue.FreeTensor(xInUb);
        yQueue.EnQue(yInUb);
    }

    __aicore__ inline void CopyOutY(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T> yOutUb = yQueue.template DeQue<T>();

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentANum;
        copyInParams.blockLen = this->r * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(yGm[offset], yOutUb, copyInParams);
        yQueue.FreeTensor(yOutUb);
    }

    __aicore__ inline void CastBatchMeanLastout(uint64_t currentANum)
    {
        __local_mem__ float* batchMeanInAddr = (__local_mem__ float*)batchMeanOutUb.GetPhyAddr();
        __local_mem__ float* batchLastoutInAddr = (__local_mem__ float*)batchLastoutOutUb.GetPhyAddr();
        __local_mem__ M* batchMeanOutAddr = (__local_mem__ M*)batchMeanOutUb.GetPhyAddr();
        __local_mem__ M* batchLastoutOutAddr = (__local_mem__ M*)batchLastoutOutUb.GetPhyAddr();

        uint32_t castCount = static_cast<uint32_t>(currentANum);
        uint16_t castLoops = static_cast<uint32_t>((castCount + VL_F32 - 1) / VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> input_mean;
            RegTensor<float> input_lastout;
            RegTensor<M> output_mean;
            RegTensor<M> output_lastout;
            MicroAPI::MaskReg pregLoop;
            for (uint16_t i = 0; i < castLoops; i++) {
                pregLoop = MicroAPI::UpdateMask<float>(castCount);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_mean, batchMeanInAddr + VL_F32 * i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                    input_lastout, batchLastoutInAddr + VL_F32 * i);
                Cast<M, float, castTraitB322B16>(output_mean, input_mean, pregLoop);
                Cast<M, float, castTraitB322B16>(output_lastout, input_lastout, pregLoop);
                DataCopy<M, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ M*)batchMeanOutAddr + i * VL_MEAN), output_mean, pregLoop);
                DataCopy<M, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ M*)batchLastoutOutAddr + i * VL_MEAN), output_lastout, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyOutBatchMeanLastout(int64_t offset, int64_t currentANum)
    {
        if constexpr (!IsSameType<M, float>::value) {
            // float to bfloat16 or float16, input continue and output each repeat have only half value
            CastBatchMeanLastout(currentANum);
            batchMeanQueue.EnQue(batchMeanOutUb);
            batchLastoutQueue.EnQue(batchLastoutOutUb);
            LocalTensor<M> batchMeanInUb = batchMeanQueue.template DeQue<M>();
            LocalTensor<M> batchLastoutInUb = batchLastoutQueue.template DeQue<M>();
            // VL_F32
            uint32_t castDmaCount = static_cast<uint32_t>(currentANum);
            uint32_t castDmaLoops = static_cast<uint32_t>(castDmaCount / VL_F32);
            if (castDmaLoops > 0) {
                DataCopyExtParams copyInParams;
                copyInParams.blockCount = castDmaLoops;
                copyInParams.blockLen = VL_F32 * sizeof(M);
                copyInParams.srcStride = (VECTOR_REG_WIDTH - VL_F32 * sizeof(M)) / BLOCK_SIZE;
                copyInParams.dstStride = 0;
                DataCopyPad(batchMeanGm[offset], batchMeanInUb, copyInParams);
                DataCopyPad(batchLastoutGm[offset], batchLastoutInUb, copyInParams);
            }

            // tail
            uint32_t tailSize = static_cast<uint32_t>(castDmaCount % VL_F32);
            if (tailSize > 0) {
                DataCopyExtParams copyInParamsTail;
                copyInParamsTail.blockCount = 1;
                copyInParamsTail.blockLen = tailSize * sizeof(M);
                copyInParamsTail.srcStride = 0;
                copyInParamsTail.dstStride = 0;
                DataCopyPad(
                    batchMeanGm[offset + castDmaLoops * VL_F32], batchMeanInUb[castDmaLoops * VL_MEAN],
                    copyInParamsTail);
                DataCopyPad(
                    batchLastoutGm[offset + castDmaLoops * VL_F32], batchLastoutInUb[castDmaLoops * VL_MEAN],
                    copyInParamsTail);
            }
            batchMeanQueue.FreeTensor(batchMeanInUb);
            batchLastoutQueue.FreeTensor(batchLastoutInUb);
        } else {
            batchMeanQueue.EnQue(batchMeanOutUb);
            batchLastoutQueue.EnQue(batchLastoutOutUb);
            LocalTensor<float> batchMeanInUb = batchMeanQueue.template DeQue<float>();
            LocalTensor<float> batchLastoutInUb = batchLastoutQueue.template DeQue<float>();
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = currentANum * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            DataCopyPad(batchMeanGm[offset], batchMeanInUb, copyInParams);
            DataCopyPad(batchLastoutGm[offset], batchLastoutInUb, copyInParams);
            batchMeanQueue.FreeTensor(batchMeanInUb);
            batchLastoutQueue.FreeTensor(batchLastoutInUb);
        }
    }

    __aicore__ inline void LoadTwoTensorForDtypeT(
        __local_mem__ T* src1, __local_mem__ T* src2, RegTensor<float>& dst1, RegTensor<float>& dst2, MaskReg& dst1Preg,
        MaskReg& dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
    {
        if constexpr (IsSameType<T, half>::value) {
            RegTensor<half> xFp16Q;
            RegTensor<half> xFp16R;
            DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ half*)(src1) + (src1Offset)));
            DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ half*)(src2) + (src2Offset)));
            Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
            Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            RegTensor<bfloat16_t> xFp16Q;
            RegTensor<bfloat16_t> xFp16R;
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ bfloat16_t*)(src1) + (src1Offset)));
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ bfloat16_t*)(src2) + (src2Offset)));
            Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
            Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
        } else {
            DataCopy(dst1, ((__local_mem__ float*)(src1) + (src1Offset)));
            DataCopy(dst2, ((__local_mem__ float*)(src2) + (src2Offset)));
        }
    }

    template <typename DT>
    __aicore__ inline void LoadOneTensorForDtypeT(
        __local_mem__ DT* input, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<DT, half>::value) {
            RegTensor<half> xFp16;
            DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
            Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
        } else if constexpr (IsSameType<DT, bfloat16_t>::value) {
            RegTensor<bfloat16_t> xBf16;
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
            Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
        } else {
            DataCopy(dst, ((__local_mem__ float*)(input) + (offset)));
        }
    }

    __aicore__ inline void CalculateMeanVarRLessThanVL64VF(
        __local_mem__ T* xInUb, __local_mem__ float* batchMeanInUb, __local_mem__ float* batchLastoutInUb,
        uint16_t currentANum)
    {
        int64_t calcNum = this->r;
        float n = static_cast<float>(1) / static_cast<float>(this->powerOfTwoForR);
        float nCorrectionFactor = static_cast<float>(this->powerOfTwoForR) / static_cast<float>(calcNum);
        uint32_t xyUbOffset = this->rAlign;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            RegTensor<float> mean_sum;
            RegTensor<float> mean;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> y1Pow;
            RegTensor<float> var_sum;
            RegTensor<float> var;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            uint32_t sreg0 = calcNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t k = 0; k < currentANum; k++) {
                LoadOneTensorForDtypeT(xInUb, x, pregLoop, (k * xyUbOffset));
                Muls(mean_sum, x, n, pregLoop);
                ReduceSum(mean, mean_sum, pregLoop);
                Muls(mean, mean, nCorrectionFactor, pregMerge);

                // save mean
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)batchMeanInUb + k), mean, pregMerge);
                Duplicate(mean, mean, pregMain);
                Muls(mean, mean, (float)-1.0, pregMain);

                LoadOneTensorForDtypeT(xInUb, x1, pregLoop, (k * xyUbOffset));
                Add(y1, x1, mean, pregLoop);
                Mul(y1Pow, y1, y1, pregLoop);
                Muls(var_sum, y1Pow, n, pregLoop);
                ReduceSum(var, var_sum, pregLoop);
                Muls(var, var, nCorrectionFactor, pregMerge);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)batchLastoutInUb + k), var, pregMerge);
            }
        }
    }

    __aicore__ inline void CalculateMeanVarVF(
        __local_mem__ T* xInUb, __local_mem__ float* batchMeanInUb, __local_mem__ float* batchLastoutInUb,
        uint16_t currentANum)
    {
        int64_t reduceNum = this->r;
        float n = static_cast<float>(1) / static_cast<float>(this->powerOfTwoForR);
        float nCorrectionFactor = static_cast<float>(this->powerOfTwoForR) / static_cast<float>(reduceNum);
        uint32_t xyUbOffset = this->rAlign;

        uint32_t binaryAddQuotientOffset = this->binaryAddQuotient;
        int64_t binaryAddRemainder = reduceNum - this->binaryAddQuotient;
        uint16_t binaryAddRemainderLoop = CEIL_DIV(binaryAddRemainder, VL_F32);
        uint16_t binaryAddQuotientLoop = CEIL_DIV(this->binaryAddQuotient, VL_F32);

        uint16_t binaryAddKLoop = this->binaryAddK;
        uint16_t binaryAddLoopMean = ((this->binaryAddQuotient / VL_F32) / VL_F32);
        uint16_t binaryAddLoopVar = binaryAddLoopMean;
        LocalTensor<float> binaryAddTensor = tmpBuf.Get<float>();
        __local_mem__ float* binaryAddTensorAddr = (__local_mem__ float*)binaryAddTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> mean_sum;
            RegTensor<float> mean;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> y1Pow;
            RegTensor<float> var_sum;
            RegTensor<float> var;

            RegTensor<float> binaryAddQ;
            RegTensor<float> binaryAddR;
            RegTensor<float> vlMean;

            RegTensor<float> binaryAddQPow;
            RegTensor<float> binaryAddRPow;
            RegTensor<float> vlVar;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t k = 0; k < currentANum; k++) {
                uint32_t sreg0 = binaryAddRemainder;
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadTwoTensorForDtypeT(
                        xInUb, xInUb, binaryAddQ, binaryAddR, pregLoop, pregLoop, (i * VL_F32 + k * xyUbOffset),
                        (i * VL_F32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Muls(binaryAddQ, binaryAddQ, n, pregLoop);
                    Muls(binaryAddR, binaryAddR, n, pregLoop);
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregLoop);
                    ReduceSum(vlMean, binaryAddQ, pregLoop);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + i), vlMean, pregMerge);
                }
                {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadTwoTensorForDtypeT(
                        xInUb, xInUb, binaryAddQ, binaryAddR, pregMain, pregLoop,
                        ((binaryAddRemainderLoop - 1) * VL_F32 + k * xyUbOffset),
                        ((binaryAddRemainderLoop - 1) * VL_F32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Muls(binaryAddQ, binaryAddQ, n, pregMain);
                    Muls(binaryAddR, binaryAddR, n, pregLoop);
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                    ReduceSum(vlMean, binaryAddQ, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + binaryAddRemainderLoop - 1), vlMean, pregMerge);
                }
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                    LoadOneTensorForDtypeT(
                        xInUb, x, pregMain, ((i + binaryAddRemainderLoop) * VL_F32 + k * xyUbOffset));
                    Muls(x, x, n, pregMain);
                    ReduceSum(vlMean, x, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + binaryAddRemainderLoop + i), vlMean, pregMerge);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                uint16_t curBinaryAddLoopMean = binaryAddLoopMean;
                for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                    curBinaryAddLoopMean = curBinaryAddLoopMean / 2;
                    for (uint16_t j = 0; j < curBinaryAddLoopMean; j++) {
                        DataCopy(binaryAddQ, ((__local_mem__ float*)binaryAddTensorAddr + j * VL_F32));
                        DataCopy(
                            binaryAddR,
                            ((__local_mem__ float*)binaryAddTensorAddr + (j + curBinaryAddLoopMean) * VL_F32));
                        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                        DataCopy(((__local_mem__ float*)binaryAddTensorAddr + j * VL_F32), binaryAddQ, pregMain);
                    }
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                }
                {
                    uint32_t sreg2 = this->binaryAddLastNum;
                    pregLoop = UpdateMask<float>(sreg2);
                    DataCopy(mean_sum, ((__local_mem__ float*)binaryAddTensorAddr));
                    ReduceSum(mean, mean_sum, pregLoop);
                    Muls(mean, mean, nCorrectionFactor, pregMerge);
                }

                // batch mean
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)batchMeanInUb + k), mean, pregMerge);
                Duplicate(mean, mean, pregMain);
                LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

                uint32_t sreg1 = binaryAddRemainder;
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
                    pregLoop = UpdateMask<float>(sreg1);
                    LoadTwoTensorForDtypeT(
                        xInUb, xInUb, binaryAddQ, binaryAddR, pregLoop, pregLoop, (i * VL_F32 + k * xyUbOffset),
                        (i * VL_F32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Sub(binaryAddQ, binaryAddQ, mean, pregLoop);
                    Sub(binaryAddR, binaryAddR, mean, pregLoop);
                    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregLoop);
                    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);
                    Muls(binaryAddQPow, binaryAddQPow, n, pregLoop);
                    Muls(binaryAddRPow, binaryAddRPow, n, pregLoop);
                    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregLoop);
                    ReduceSum(vlVar, binaryAddQPow, pregLoop);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + i), vlVar, pregMerge);
                }
                {
                    pregLoop = UpdateMask<float>(sreg1);
                    LoadTwoTensorForDtypeT(
                        xInUb, xInUb, binaryAddQ, binaryAddR, pregMain, pregLoop,
                        ((binaryAddRemainderLoop - 1) * VL_F32 + k * xyUbOffset),
                        ((binaryAddRemainderLoop - 1) * VL_F32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Sub(binaryAddQ, binaryAddQ, mean, pregMain);
                    Sub(binaryAddR, binaryAddR, mean, pregLoop);
                    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregMain);
                    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);
                    Muls(binaryAddQPow, binaryAddQPow, n, pregMain);
                    Muls(binaryAddRPow, binaryAddRPow, n, pregLoop);
                    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregMain);
                    ReduceSum(vlVar, binaryAddQPow, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + binaryAddRemainderLoop - 1), vlVar, pregMerge);
                }
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                    LoadOneTensorForDtypeT(
                        xInUb, x1, pregMain, ((i + binaryAddRemainderLoop) * VL_F32 + k * xyUbOffset));
                    Sub(y1, x1, mean, pregMain);
                    Mul(y1Pow, y1, y1, pregMain);
                    Muls(y1Pow, y1Pow, n, pregMain);
                    ReduceSum(vlVar, y1Pow, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + binaryAddRemainderLoop + i), vlVar, pregMerge);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                uint16_t curBinaryAddLoopVar = binaryAddLoopVar;
                for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                    curBinaryAddLoopVar = curBinaryAddLoopVar / 2;
                    for (uint16_t j = 0; j < curBinaryAddLoopVar; j++) {
                        DataCopy(binaryAddQ, ((__local_mem__ float*)binaryAddTensorAddr + j * VL_F32));
                        DataCopy(
                            binaryAddR,
                            ((__local_mem__ float*)binaryAddTensorAddr + (j + curBinaryAddLoopVar) * VL_F32));
                        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                        DataCopy(((__local_mem__ float*)binaryAddTensorAddr + j * VL_F32), binaryAddQ, pregMain);
                    }
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                }
                {
                    uint32_t sreg2 = this->binaryAddLastNum;
                    pregLoop = UpdateMask<float>(sreg2);
                    DataCopy(var_sum, ((__local_mem__ float*)binaryAddTensorAddr));
                    ReduceSum(var, var_sum, pregLoop);
                    Muls(var, var, nCorrectionFactor, pregMerge);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)batchLastoutInUb + k), var, pregMerge);
                }
                LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
            }
        }
    }

    __aicore__ inline void CalculateRuningMeanVarVF(
        __local_mem__ float* batchMeanInUb, __local_mem__ float* batchLastoutInUb, uint16_t currentANum)
    {
        uint16_t aLoop = CEIL_DIV(currentANum, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> mean;
            RegTensor<float> var;

            RegTensor<float> sqrtVar;
            RegTensor<float> one;
            RegTensor<float> rsqrtVar;

            RegTensor<float> runningMean;
            RegTensor<float> saveMean;
            RegTensor<float> runningVar;
            RegTensor<float> saveVar;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();

            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> e;
            RegTensor<float> scalar1;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> lastout;

            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregLoop;

            Duplicate(one, 1.0, pregMain);
            uint32_t sreg2 = currentANum;
            for (uint16_t k = 0; k < aLoop; k++) {
                pregLoop = UpdateMask<float>(sreg2);
                Duplicate(scalar1, float(0.5), pregLoop);
                Duplicate(scalarInf, POS_INF, pregLoop);
                Duplicate(scalarZero, zero, pregLoop);
                Duplicate(t1, float(1.5), pregLoop);
                Duplicate(s, float(1.0), pregLoop);

                // lastout
                DataCopy(var, ((__local_mem__ float*)batchLastoutInUb + k * VL_F32));
                Adds(var, var, epsilon, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, float(-0.5), pregLoop);
                Mul(t, t, y, pregLoop);                   // -0.5 * x * y
                Mula(t1, t, y, pregLoop);                 // 1.5 + (-0.5 * x * y) * y
                Mul(lastout, y, t1, pregLoop);            // y = y * (1.5 - 0.5 * x * y)
                Muls(t3, var, float(-1.0), pregLoop);     // -1 * x
                Mula(s, t3, r, pregLoop);                 // 1 + (-1) * x * r
                Muls(t4, lastout, float(-1.0), pregLoop); // (-1) * y
                Mula(r, t4, lastout, pregLoop);           // r + (-1) * y * y
                Mula(s, var, r, pregLoop);                // s + x * t
                Mul(s, s, lastout, pregLoop);             // e * y
                Mula(lastout, s, scalar1, pregLoop);      // y + y * e * 0.5
                CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
                Select(lastout, scalarZero, lastout, cmpRegZero);
                CompareScalar(cmpRegInf, var, zero, pregLoop);
                Select(lastout, scalarInf, lastout, cmpRegInf);
                DataCopy(((__local_mem__ float*)batchLastoutInUb + k * VL_F32), lastout, pregLoop);
            }
        }
    }

    template <bool hasGammaFlag, bool hasBetaFlag>
    __aicore__ inline void CalculateNormalizeVF(
        __local_mem__ T* xInUb, __local_mem__ T* yInUb, __local_mem__ U* betaInUb, __local_mem__ U* gammaInUb,
        __local_mem__ float* batchMeanInUb, __local_mem__ float* batchLastoutInUb, uint16_t currentANum)
    {
        int64_t calcNum = this->r;
        uint32_t xyUbOffset = this->rAlign;
        uint16_t loopCount = CEIL_DIV(calcNum, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> mean;

            RegTensor<float> x2;
            RegTensor<float> y2;
            RegTensor<float> rsqrtVar;

            RegTensor<float> beta;
            RegTensor<float> gamma;

            MaskReg pregLoop;

            for (uint16_t k = 0; k < currentANum; k++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(mean, ((__local_mem__ float*)batchMeanInUb + k));
                DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrtVar, ((__local_mem__ float*)batchLastoutInUb + k));
                uint32_t sreg3 = calcNum;
                for (uint16_t i = 0; i < loopCount; i++) {
                    pregLoop = UpdateMask<float>(sreg3);
                    if constexpr (hasGammaFlag) {
                        LoadOneTensorForDtypeT(gammaInUb, gamma, pregLoop, (i * VL_F32));
                    }
                    if constexpr (hasBetaFlag) {
                        LoadOneTensorForDtypeT(betaInUb, beta, pregLoop, (i * VL_F32));
                    }
                    LoadOneTensorForDtypeT(xInUb, x2, pregLoop, (i * VL_F32 + k * xyUbOffset));
                    Sub(x2, x2, mean, pregLoop);
                    Mul(y2, x2, rsqrtVar, pregLoop);
                    if constexpr (hasGammaFlag) {
                        Mul(y2, y2, gamma, pregLoop);
                    }
                    if constexpr (hasBetaFlag) {
                        Add(y2, y2, beta, pregLoop);
                    }
                    if constexpr (IsSameType<T, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, castTraitB322B16>(yFp16, y2, pregLoop);
                        DataCopy<half, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ half*)yInUb + i * VL_F32 + k * xyUbOffset), yFp16, pregLoop);
                    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y2, pregLoop);
                        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ bfloat16_t*)yInUb + i * VL_F32 + k * xyUbOffset), xBf16, pregLoop);
                    } else {
                        DataCopy(((__local_mem__ float*)yInUb + i * VL_F32 + k * xyUbOffset), y2, pregLoop);
                    }
                }
            }
        }
    }

    /* global memory address */
    GlobalTensor<T> xGm;
    GlobalTensor<U> betaGm;
    GlobalTensor<U> gammaGm;

    GlobalTensor<T> yGm;
    GlobalTensor<M> batchMeanGm;
    GlobalTensor<M> batchLastoutGm;

    /* variable */
    int64_t powerOfTwoForR;
    int64_t r;
    int64_t aFactor;
    int64_t a;
    int64_t rAlign;

    int64_t blockNum;
    int64_t aBlockFactor;
    int64_t singleA;

    int64_t binaryAddQuotient;
    int64_t binaryAddK;
    int64_t binaryAddLastNum;

    int64_t tmpBufferSize;

    bool hasGamma = true;
    bool hasBeta = true;

    static constexpr uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr uint32_t VL_MEAN = VECTOR_REG_WIDTH / sizeof(M);

    float epsilon = 1e-5;

    LayerNormSeparateTiling layerNormTiling;

    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> gammaQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> betaQueue;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> batchMeanQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> batchLastoutQueue;

    TBuf<TPosition::VECCALC> tmpBuf;

    LocalTensor<float> batchMeanOutUb;
    LocalTensor<float> batchLastoutOutUb;
};
} // namespace LayerNormV3

#endif // LAYER_NORM_V3_TWO_PASS_H
