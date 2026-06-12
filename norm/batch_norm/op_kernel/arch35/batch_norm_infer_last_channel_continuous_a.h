/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_infer_last_channel_continuous_a.h
 * \brief
 */

#ifndef BATCH_NORM_INFER_LAST_CHANNEL_CONTINUOUS_A_H
#define BATCH_NORM_INFER_LAST_CHANNEL_CONTINUOUS_A_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "batch_norm_base.h"

namespace BatchNormOps {
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

template <typename T, typename T_GAMMA, typename T_RUNNING_MEAN>
class BatchNormInferLastChannelContinuousA {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = BatchNormOps::VECTOR_LENGTH;
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = BatchNormOps::BLOCK_SIZE;
    static constexpr int32_t MEAN_VAR_OUTPUT_COUNT = 2;  // mean, var

    constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};

    constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

public:
    __aicore__ inline BatchNormInferLastChannelContinuousA(){};

    __aicore__ inline BatchNormInferLastChannelContinuousA(const BatchNormInferLastChannelTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                GM_ADDR batch_mean, GM_ADDR batch_variance, GM_ADDR reserve_space_1,
                                GM_ADDR reserve_space_2, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        xGm_.SetGlobalBuffer((__gm__ T*)x);
        betaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)beta);
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        meanGm_.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean);
        varGm_.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
        batchMeanGm_.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)batch_mean);
        batchVarGm_.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)batch_variance);
        reserveSpace1Gm_.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)reserve_space_1);
        reserveSpace2Gm_.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)reserve_space_2);

        int64_t paramAlignLen = AlignUp(tilingData_->tileBlockALen, static_cast<int64_t>(VL_FP32));
        pipe_->InitBuffer(betaQueue_, BUFFER_NUM, paramAlignLen * sizeof(T_GAMMA));
        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, paramAlignLen * sizeof(T_GAMMA));
        pipe_->InitBuffer(meanQueue_, BUFFER_NUM, paramAlignLen * sizeof(float));
        pipe_->InitBuffer(varQueue_, BUFFER_NUM, paramAlignLen * sizeof(float));
        pipe_->InitBuffer(betaFp32Buf_, paramAlignLen * sizeof(float));
        pipe_->InitBuffer(gammaFp32Buf_, paramAlignLen * sizeof(float));
        pipe_->InitBuffer(meanFp32Buf_, paramAlignLen * sizeof(float));
        pipe_->InitBuffer(rstdFp32Buf_, paramAlignLen * sizeof(float));

        int64_t xShapeLen = tilingData_->tileBlockBLen * tilingData_->tileBlockALen;
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(T));
        int64_t varOffset = AlignUp(tilingData_->tileBlockALen, BLOCK_SIZE / sizeof(T_RUNNING_MEAN));
        pipe_->InitBuffer(meanVarOutQueue_, BUFFER_DEPTH, MEAN_VAR_OUTPUT_COUNT * varOffset * sizeof(T_RUNNING_MEAN));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        if (beginIdx >= endIdx) {
            return;
        }

        CopyInBetaGammaMeanVar(0, tilingData_->tileBlockALen);
        PrepareParamCache();

        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curTileBLen =
                curIdx == (tilingData_->bOuter - 1) ? tilingData_->tileBlockBTail : tilingData_->tileBlockBLen;
            int64_t xOffset = curIdx * tilingData_->totalALen * tilingData_->tileBlockBLen;

            CopyInX(xOffset, curTileBLen);
            Compute(curTileBLen);
            CopyOutY(xOffset, curTileBLen);
        }
    }

private:
    __aicore__ inline int64_t AlignUp(int64_t value, int64_t base) const
    {
        return (value + base - 1) / base * base;
    }

    __aicore__ inline void CopyInX(int64_t xGmOffset, int64_t curTileBLen)
    {
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();

        DataCopyExtParams extParam;
        extParam.blockLen = curTileBLen * tilingData_->tileBlockALen * sizeof(T);
        extParam.srcStride = 0;
        extParam.dstStride = 0;
        extParam.blockCount = 1;

        DataCopyPadExtParams<T> padExtParam;
        padExtParam.isPad = false;

        DataCopyPad(xLocal, xGm_[xGmOffset], extParam, padExtParam);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInBetaGammaMeanVar(int64_t offset, int64_t curTileALen)
    {
        LocalTensor<T_GAMMA> betaLocal = betaQueue_.AllocTensor<T_GAMMA>();
        LocalTensor<T_GAMMA> gammaLocal = gammaQueue_.AllocTensor<T_GAMMA>();
        LocalTensor<T_RUNNING_MEAN> meanLocal = meanQueue_.AllocTensor<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> varLocal = varQueue_.AllocTensor<T_RUNNING_MEAN>();

        DataCopyExtParams extParam;
        extParam.blockCount = 1;
        extParam.srcStride = 0;
        extParam.dstStride = 0;

        extParam.blockLen = curTileALen * sizeof(T_GAMMA);
        DataCopyPadExtParams<T_GAMMA> padExtParam;
        padExtParam.isPad = false;
        DataCopyPad(betaLocal, betaGm_[offset], extParam, padExtParam);
        DataCopyPad(gammaLocal, gammaGm_[offset], extParam, padExtParam);

        extParam.blockLen = curTileALen * sizeof(T_RUNNING_MEAN);
        DataCopyPadExtParams<T_RUNNING_MEAN> padExtParams1;
        padExtParams1.isPad = false;
        DataCopyPad(meanLocal, meanGm_[offset], extParam, padExtParams1);
        DataCopyPad(varLocal, varGm_[offset], extParam, padExtParams1);

        betaQueue_.EnQue(betaLocal);
        gammaQueue_.EnQue(gammaLocal);
        meanQueue_.EnQue(meanLocal);
        varQueue_.EnQue(varLocal);
    }

    __aicore__ inline void PrepareParamCache()
    {
        LocalTensor<T_GAMMA> beta = betaQueue_.DeQue<T_GAMMA>();
        LocalTensor<T_GAMMA> gamma = gammaQueue_.DeQue<T_GAMMA>();
        LocalTensor<T_RUNNING_MEAN> mean = meanQueue_.DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> var = varQueue_.DeQue<T_RUNNING_MEAN>();
        LocalTensor<float> betaFp32 = betaFp32Buf_.Get<float>();
        LocalTensor<float> gammaFp32 = gammaFp32Buf_.Get<float>();
        LocalTensor<float> meanFp32 = meanFp32Buf_.Get<float>();
        LocalTensor<float> rstdFp32 = rstdFp32Buf_.Get<float>();

        __local_mem__ T_GAMMA* betaLocal = (__local_mem__ T_GAMMA*)beta.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaLocal = (__local_mem__ T_GAMMA*)gamma.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* meanLocal = (__local_mem__ T_RUNNING_MEAN*)mean.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* varLocal = (__local_mem__ T_RUNNING_MEAN*)var.GetPhyAddr();
        __local_mem__ float* betaFp32Local = (__local_mem__ float*)betaFp32.GetPhyAddr();
        __local_mem__ float* gammaFp32Local = (__local_mem__ float*)gammaFp32.GetPhyAddr();
        __local_mem__ float* meanFp32Local = (__local_mem__ float*)meanFp32.GetPhyAddr();
        __local_mem__ float* rstdFp32Local = (__local_mem__ float*)rstdFp32.GetPhyAddr();

        VFPrepareParamCache(
            gammaLocal, betaLocal, meanLocal, varLocal, gammaFp32Local, betaFp32Local, meanFp32Local, rstdFp32Local);
        CopyOutMeanVar(tilingData_->tileBlockALen);

        betaQueue_.FreeTensor<T_GAMMA>(beta);
        gammaQueue_.FreeTensor<T_GAMMA>(gamma);
        meanQueue_.FreeTensor<T_RUNNING_MEAN>(mean);
        varQueue_.FreeTensor<T_RUNNING_MEAN>(var);
    }

    __aicore__ inline void CopyOutMeanVar(int64_t curTileALen)
    {
        if (GetBlockIdx() != 0) { return; }

        DataCopyExtParams extParams;
        extParams.blockLen = curTileALen * sizeof(T_RUNNING_MEAN);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        extParams.blockCount = 1;

        int64_t varOffset = AlignUp(curTileALen, BLOCK_SIZE / sizeof(T_RUNNING_MEAN));
        LocalTensor<T_RUNNING_MEAN> meanVarBuf = meanVarOutQueue_.AllocTensor<T_RUNNING_MEAN>();

        DataCopyPadExtParams<T_RUNNING_MEAN> padExtParams;
        padExtParams.isPad = false;
        DataCopyPad(meanVarBuf, meanGm_[0], extParams, padExtParams);
        DataCopyPad(meanVarBuf[varOffset], varGm_[0], extParams, padExtParams);

        meanVarOutQueue_.EnQue<QuePosition::GM, QuePosition::VECIN, T_RUNNING_MEAN>(meanVarBuf);

        LocalTensor<T_RUNNING_MEAN> meanVarOut =
            meanVarOutQueue_.DeQue<QuePosition::VECOUT, QuePosition::GM, T_RUNNING_MEAN>();

        DataCopyPad(batchMeanGm_[0], meanVarOut, extParams);
        DataCopyPad(reserveSpace1Gm_[0], meanVarOut, extParams);
        DataCopyPad(batchVarGm_[0], meanVarOut[varOffset], extParams);
        DataCopyPad(reserveSpace2Gm_[0], meanVarOut[varOffset], extParams);

        meanVarOutQueue_.FreeTensor(meanVarOut);
    }

    __aicore__ inline void Compute(int64_t curTileBLen)
    {
        LocalTensor<T> x = xQueue_.DeQue<T>();
        LocalTensor<T> y = yQueue_.AllocTensor<T>();
        LocalTensor<float> betaFp32 = betaFp32Buf_.Get<float>();
        LocalTensor<float> gammaFp32 = gammaFp32Buf_.Get<float>();
        LocalTensor<float> meanFp32 = meanFp32Buf_.Get<float>();
        LocalTensor<float> rstdFp32 = rstdFp32Buf_.Get<float>();

        __local_mem__ T* xLocal = (__local_mem__ T*)x.GetPhyAddr();
        __local_mem__ T* yLocal = (__local_mem__ T*)y.GetPhyAddr();
        __local_mem__ float* betaFp32Local = (__local_mem__ float*)betaFp32.GetPhyAddr();
        __local_mem__ float* gammaFp32Local = (__local_mem__ float*)gammaFp32.GetPhyAddr();
        __local_mem__ float* meanFp32Local = (__local_mem__ float*)meanFp32.GetPhyAddr();
        __local_mem__ float* rstdFp32Local = (__local_mem__ float*)rstdFp32.GetPhyAddr();

        VFNormalize(xLocal, gammaFp32Local, betaFp32Local, meanFp32Local, rstdFp32Local, yLocal, curTileBLen);

        yQueue_.EnQue(y);

        xQueue_.FreeTensor<T>(x);
    }

    __aicore__ inline void VFPrepareParamCache(__local_mem__ T_GAMMA* gammaLocal, __local_mem__ T_GAMMA* betaLocal,
        __local_mem__ T_RUNNING_MEAN* meanLocal, __local_mem__ T_RUNNING_MEAN* varLocal,
        __local_mem__ float* gammaFp32Local, __local_mem__ float* betaFp32Local, __local_mem__ float* meanFp32Local,
        __local_mem__ float* rstdFp32Local)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> mean;
            RegTensor<float> var;
            RegTensor<float> rstd;

            uint32_t totalALen = static_cast<uint32_t>(tilingData_->tileBlockALen);
            uint16_t aLoopNum = ops::CeilDiv(static_cast<uint16_t>(totalALen), VL_FP32);
            for (uint16_t aLoop = 0; aLoop < aLoopNum; aLoop++) {
                uint32_t aOffset = static_cast<uint32_t>(aLoop) * VL_FP32;
                uint32_t activeLen = totalALen - aOffset > VL_FP32 ? VL_FP32 : totalALen - aOffset;
                MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(activeLen);

                LoadParamForDtypeT(gammaLocal, gamma, preg, aOffset);
                LoadParamForDtypeT(betaLocal, beta, preg, aOffset);
                LoadRunningParamForDtypeT(varLocal, var, preg, aOffset);
                NormCommon::ComputeRstdNewtonRaphsonReg(var, rstd, preg, tilingData_->epsilon);
                LoadRunningParamForDtypeT(meanLocal, mean, preg, aOffset);

                DataCopy(gammaFp32Local + aOffset, gamma, preg);
                DataCopy(betaFp32Local + aOffset, beta, preg);
                DataCopy(meanFp32Local + aOffset, mean, preg);
                DataCopy(rstdFp32Local + aOffset, rstd, preg);
            }
        }
    }

    __aicore__ inline void VFNormalize(__local_mem__ T* xLocal, __local_mem__ float* gammaFp32Local,
        __local_mem__ float* betaFp32Local, __local_mem__ float* meanFp32Local, __local_mem__ float* rstdFp32Local,
        __local_mem__ T* yLocal, int64_t curTileBLen)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> y;

            uint32_t totalALen = static_cast<uint32_t>(tilingData_->tileBlockALen);
            uint16_t aLoopNum = ops::CeilDiv(static_cast<uint16_t>(totalALen), VL_FP32);
            uint16_t curTileBLoop = static_cast<uint16_t>(curTileBLen);
            for (uint16_t aLoop = 0; aLoop < aLoopNum; aLoop++) {
                uint32_t aOffset = static_cast<uint32_t>(aLoop) * VL_FP32;
                uint32_t activeLen = totalALen - aOffset > VL_FP32 ? VL_FP32 : totalALen - aOffset;
                uint32_t maskLen = activeLen;
                MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(maskLen);

                DataCopy<float, LoadDist::DIST_NORM>(gamma, gammaFp32Local + aOffset);
                DataCopy<float, LoadDist::DIST_NORM>(beta, betaFp32Local + aOffset);
                DataCopy<float, LoadDist::DIST_NORM>(mean, meanFp32Local + aOffset);
                DataCopy<float, LoadDist::DIST_NORM>(rstd, rstdFp32Local + aOffset);

                for (uint16_t bIdx = 0; bIdx < curTileBLoop; bIdx++) {
                    uint32_t elemOffset = static_cast<uint32_t>(bIdx) * totalALen + aOffset;
                    NormCommon::LoadTensorUnAlignForDtypeT(xLocal + elemOffset, x, preg, activeLen);
                    NormCommon::NormalizeWithScaleBiasReg(x, gamma, beta, mean, rstd, y, preg);
                    NormCommon::StoreTensorUnAlignForDtypeT(yLocal + elemOffset, y, preg, activeLen);
                }
            }
        }
    }

    template <typename T_SRC>
    __aicore__ inline void LoadParamForDtypeT(
        __local_mem__ T_SRC* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_SRC, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
        } else {
            RegTensor<T_SRC> srcB16;
            DataCopy<T_SRC, LoadDist::DIST_UNPACK_B16>(srcB16, ((__local_mem__ T_SRC*)src + offset));
            Cast<float, T_SRC, castTraitB162B32>(dst, srcB16, preg);
        }
    }

    __aicore__ inline void LoadRunningParamForDtypeT(
        __local_mem__ T_RUNNING_MEAN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_RUNNING_MEAN, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
        } else {
            RegTensor<T_RUNNING_MEAN> srcB16;
            DataCopy<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                srcB16, ((__local_mem__ T_RUNNING_MEAN*)src + offset));
            Cast<float, T_RUNNING_MEAN, castTraitB162B32>(dst, srcB16, preg);
        }
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t curTileBLen)
    {
        LocalTensor<T> y = yQueue_.DeQue<T>();

        DataCopyExtParams extParams;
        extParams.blockLen = curTileBLen * tilingData_->tileBlockALen * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        extParams.blockCount = 1;

        DataCopyPad(yGm_[yGmOffset], y, extParams);

        yQueue_.FreeTensor(y);
    }

private:
    const BatchNormInferLastChannelTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> betaQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> gammaQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> meanQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> varQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> yQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_DEPTH> meanVarOutQueue_;
    TBuf<TPosition::VECCALC> betaFp32Buf_;
    TBuf<TPosition::VECCALC> gammaFp32Buf_;
    TBuf<TPosition::VECCALC> meanFp32Buf_;
    TBuf<TPosition::VECCALC> rstdFp32Buf_;

    GlobalTensor<T> yGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T_GAMMA> betaGm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<T_RUNNING_MEAN> meanGm_;
    GlobalTensor<T_RUNNING_MEAN> varGm_;
    GlobalTensor<T_RUNNING_MEAN> batchMeanGm_;
    GlobalTensor<T_RUNNING_MEAN> batchVarGm_;
    GlobalTensor<T_RUNNING_MEAN> reserveSpace1Gm_;
    GlobalTensor<T_RUNNING_MEAN> reserveSpace2Gm_;
};
} // namespace BatchNormOps

#endif
