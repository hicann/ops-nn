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
        int64_t aFactorAlignF32 = (((this->aFactor * FLOAT_BYTES + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) /
                                  FLOAT_BYTES;
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
            int64_t currentA = (ubLoopIdx == (quotient - 1)) ? (this->singleA - (quotient - 1) * this->aFactor) :
                                                               this->aFactor;
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
        DataCopyExtParams copyInParams;
        copyInParams.blockLen = this->r * sizeof(U);
        copyInParams.blockCount = 1;
        copyInParams.dstStride = 0;
        copyInParams.srcStride = 0;
        DataCopyPadExtParams<U> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.paddingValue = 0;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;

        if (hasGamma) {
            LocalTensor<U> gammaLocalTensor = gammaQueue.AllocTensor<U>();
            DataCopyPad(gammaLocalTensor, gammaGm, copyInParams, dataCopyPadExtParams);
            gammaQueue.EnQue(gammaLocalTensor);
        }
        if (hasBeta) {
            LocalTensor<U> betaLocalTensor = betaQueue.AllocTensor<U>();
            DataCopyPad(betaLocalTensor, betaGm, copyInParams, dataCopyPadExtParams);
            betaQueue.EnQue(betaLocalTensor);
        }
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t currentANum)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockLen = this->r * sizeof(T);
        copyInParams.blockCount = currentANum;
        copyInParams.dstStride = 0;
        copyInParams.srcStride = 0;
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.rightPadding = (this->rAlign - this->r);
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        dataCopyPadExtParams.isPad = (this->r != this->rAlign);
        LocalTensor<T> xInUb = xQueue.AllocTensor<T>();
        DataCopyPad(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
        xQueue.EnQue(xInUb);
    }

    __aicore__ inline void CaculateWithHighLevelApi(LocalTensor<U> gammaInUb, LocalTensor<U> betaInUb,
                                                    int64_t currentANum)
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
                AscendC::LayerNorm<U, T, true, hasGammaBetaConfig>(yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb,
                                                                   gammaInUb, betaInUb, this->epsilon, binaryAddTensor,
                                                                   para, this->layerNormTiling);
            } else if (!this->hasGamma && this->hasBeta) {
                AscendC::LayerNorm<U, T, true, noGammaHasBetaConfig>(yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb,
                                                                     gammaInUb, betaInUb, this->epsilon,
                                                                     binaryAddTensor, para, this->layerNormTiling);
            } else if (this->hasGamma && !this->hasBeta) {
                AscendC::LayerNorm<U, T, true, hasGammaNoBetaConfig>(yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb,
                                                                     gammaInUb, betaInUb, this->epsilon,
                                                                     binaryAddTensor, para, this->layerNormTiling);
            } else {
                AscendC::LayerNorm<U, T, true, noGammaNoBetaConfig>(yInUb, batchMeanOutUb, batchLastoutOutUb, xInUb,
                                                                    gammaInUb, betaInUb, this->epsilon, binaryAddTensor,
                                                                    para, this->layerNormTiling);
            }
        } else {
            if (this->hasGamma && this->hasBeta) {
                AscendC::LayerNorm<U, T, true, hasGammaBetaOutVarConfig>(yInUb, batchMeanOutUb, batchLastoutOutUb,
                                                                         xInUb, gammaInUb, betaInUb, this->epsilon,
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
                AscendC::LayerNorm<U, T, true, noGammaNoBetaOutVarConfig>(yInUb, batchMeanOutUb, batchLastoutOutUb,
                                                                          xInUb, gammaInUb, betaInUb, this->epsilon,
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
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_lastout,
                                                                         batchLastoutInAddr + VL_F32 * i);
                Cast<M, float, castTraitB322B16>(output_mean, input_mean, pregLoop);
                Cast<M, float, castTraitB322B16>(output_lastout, input_lastout, pregLoop);
                DataCopy<M, StoreDist::DIST_PACK_B32>(((__local_mem__ M*)batchMeanOutAddr + i * VL_MEAN), output_mean,
                                                      pregLoop);
                DataCopy<M, StoreDist::DIST_PACK_B32>(((__local_mem__ M*)batchLastoutOutAddr + i * VL_MEAN),
                                                      output_lastout, pregLoop);
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
                DataCopyPad(batchMeanGm[offset + castDmaLoops * VL_F32], batchMeanInUb[castDmaLoops * VL_MEAN],
                            copyInParamsTail);
                DataCopyPad(batchLastoutGm[offset + castDmaLoops * VL_F32], batchLastoutInUb[castDmaLoops * VL_MEAN],
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
