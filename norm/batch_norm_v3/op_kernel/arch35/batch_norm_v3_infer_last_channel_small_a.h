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
 * \file batch_norm_v3_infer_last_channel_small_a.h
 * \brief
 */

#ifndef BATCH_NORM_V3_INFER_LAST_CHANNEL_SMALL_A_H
#define BATCH_NORM_V3_INFER_LAST_CHANNEL_SMALL_A_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "batch_norm_v3.h"
namespace BatchNormV3Ops {
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

template <typename T, typename T_GAMMA, typename T_RUNNING_MEAN>
class BatchNormV3InferLastChannelSmallA {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = GetUbBlockSize();

    constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};

    constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

public:
    __aicore__ inline BatchNormV3InferLastChannelSmallA(){};

    __aicore__ inline BatchNormV3InferLastChannelSmallA(const BatchNormV3InferLastChannelTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        xGm_.SetGlobalBuffer((__gm__ T*)x);
        betaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)beta);
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        meanGm_.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean);
        varGm_.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var);

        yGm_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(betaQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T_GAMMA));
        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T_GAMMA));
        pipe_->InitBuffer(meanQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(float));
        pipe_->InitBuffer(varQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(float));

        int64_t xShapeLen = tilingData_->tileBlockBLen * tilingData_->tileBlockALen;
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(T));

        int64_t paramCacheElemLen = GetSmallLastChannelParamCacheElemLen();
        int64_t offsetLen = AlignUp(paramCacheElemLen, BLOCK_SIZE / sizeof(uint32_t));
        int64_t paramCacheLen = AlignUp(paramCacheElemLen, BLOCK_SIZE / sizeof(float));
        pipe_->InitBuffer(offsetBuf_, offsetLen * sizeof(uint32_t));
        pipe_->InitBuffer(betaFp32Buf_, paramCacheLen * sizeof(float));
        pipe_->InitBuffer(gammaFp32Buf_, paramCacheLen * sizeof(float));
        pipe_->InitBuffer(meanFp32Buf_, paramCacheLen * sizeof(float));
        pipe_->InitBuffer(rstdFp32Buf_, paramCacheLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        InitGatherOffset();
        CopyInBetaGammaMeanVar(0, tilingData_->tileBlockALen);
        PrepareSmallLastChannelParamCache();

        int64_t curTileALen = tilingData_->tileBlockALen;
        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curTileBLen = curIdx == (tilingData_->bOuter - 1) ? tilingData_->tileBlockBTail :
                                                                        tilingData_->tileBlockBLen;
            int64_t xOffset = curIdx * tilingData_->totalALen * tilingData_->tileBlockBLen;

            CopyInX(xOffset, curTileBLen, curTileALen);
            Compute(curTileBLen, curTileALen);
            CopyOutY(xOffset, curTileBLen, curTileALen);
        }
    }

private:
    __aicore__ inline int64_t AlignUp(int64_t value, int64_t base) const { return (value + base - 1) / base * base; }

    __aicore__ inline uint32_t GetSmallLastChannelParamCacheElemLen() const
    {
        uint32_t aLen = static_cast<uint32_t>(tilingData_->tileBlockALen);
        return static_cast<uint32_t>(VL_FP32 / aLen) * aLen;
    }

    __aicore__ inline void InitGatherOffset()
    {
        LocalTensor<uint32_t> offset = offsetBuf_.Get<uint32_t>();
        uint32_t aLen = static_cast<uint32_t>(tilingData_->tileBlockALen);
        uint32_t paramCacheElemLen = GetSmallLastChannelParamCacheElemLen();
        for (uint32_t i = 0; i < paramCacheElemLen; i++) {
            offset.SetValue(i, i % aLen);
        }
    }

    __aicore__ inline void CopyInX(int64_t xGmOffset, int64_t curTileBLen, int64_t curTileALen)
    {
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();

        DataCopyExtParams extParam;
        extParam.blockLen = curTileBLen * curTileALen * sizeof(T);
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

    __aicore__ inline void PrepareSmallLastChannelParamCache()
    {
        LocalTensor<T_GAMMA> beta = betaQueue_.DeQue<T_GAMMA>();
        LocalTensor<T_GAMMA> gamma = gammaQueue_.DeQue<T_GAMMA>();
        LocalTensor<T_RUNNING_MEAN> mean = meanQueue_.DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> var = varQueue_.DeQue<T_RUNNING_MEAN>();
        LocalTensor<uint32_t> offset = offsetBuf_.Get<uint32_t>();
        LocalTensor<float> betaFp32 = betaFp32Buf_.Get<float>();
        LocalTensor<float> gammaFp32 = gammaFp32Buf_.Get<float>();
        LocalTensor<float> meanFp32 = meanFp32Buf_.Get<float>();
        LocalTensor<float> rstdFp32 = rstdFp32Buf_.Get<float>();

        __local_mem__ T_GAMMA* betaLocal = (__local_mem__ T_GAMMA*)beta.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaLocal = (__local_mem__ T_GAMMA*)gamma.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* meanLocal = (__local_mem__ T_RUNNING_MEAN*)mean.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* varLocal = (__local_mem__ T_RUNNING_MEAN*)var.GetPhyAddr();
        __ubuf__ uint32_t* offsetLocal = (__ubuf__ uint32_t*)offset.GetPhyAddr();
        __local_mem__ float* betaFp32Local = (__local_mem__ float*)betaFp32.GetPhyAddr();
        __local_mem__ float* gammaFp32Local = (__local_mem__ float*)gammaFp32.GetPhyAddr();
        __local_mem__ float* meanFp32Local = (__local_mem__ float*)meanFp32.GetPhyAddr();
        __local_mem__ float* rstdFp32Local = (__local_mem__ float*)rstdFp32.GetPhyAddr();

        VFPrepareSmallLastChannelParamCache(gammaLocal, betaLocal, meanLocal, varLocal, offsetLocal, gammaFp32Local,
                                            betaFp32Local, meanFp32Local, rstdFp32Local);

        betaQueue_.FreeTensor<T_GAMMA>(beta);
        gammaQueue_.FreeTensor<T_GAMMA>(gamma);
        meanQueue_.FreeTensor<T_RUNNING_MEAN>(mean);
        varQueue_.FreeTensor<T_RUNNING_MEAN>(var);
    }

    __aicore__ inline void Compute(int64_t curTileBLen, int64_t curTileALen)
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

        VFNormalize(xLocal, gammaFp32Local, betaFp32Local, meanFp32Local, rstdFp32Local, yLocal,
                    curTileBLen * curTileALen);

        yQueue_.EnQue(y);

        xQueue_.FreeTensor<T>(x);
    }

    __aicore__ inline void VFPrepareSmallLastChannelParamCache(
        __local_mem__ T_GAMMA* gammaLocal, __local_mem__ T_GAMMA* betaLocal, __local_mem__ T_RUNNING_MEAN* meanLocal,
        __local_mem__ T_RUNNING_MEAN* varLocal, __ubuf__ uint32_t* offsetLocal, __local_mem__ float* gammaFp32Local,
        __local_mem__ float* betaFp32Local, __local_mem__ float* meanFp32Local, __local_mem__ float* rstdFp32Local)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> mean;
            RegTensor<float> var;
            RegTensor<float> rstd;
            RegTensor<uint32_t> paramOffset;
            uint32_t paramCacheElemLen = GetSmallLastChannelParamCacheElemLen();
            uint32_t maskLen = paramCacheElemLen;
            MaskReg pregMaskFp32 = AscendC::MicroAPI::UpdateMask<float>(maskLen);

            AscendC::MicroAPI::DataCopy<uint32_t, LoadDist::DIST_NORM>(paramOffset, offsetLocal);
            GatherParamForDtypeT(gammaLocal, gamma, paramOffset, pregMaskFp32, paramCacheElemLen);
            GatherParamForDtypeT(betaLocal, beta, paramOffset, pregMaskFp32, paramCacheElemLen);
            GatherRunningParamForDtypeT(varLocal, var, paramOffset, pregMaskFp32, paramCacheElemLen);
            NormCommon::ComputeRstdNewtonRaphsonReg(var, rstd, pregMaskFp32, tilingData_->epsilon);
            GatherRunningParamForDtypeT(meanLocal, mean, paramOffset, pregMaskFp32, paramCacheElemLen);

            DataCopy(gammaFp32Local, gamma, pregMaskFp32);
            DataCopy(betaFp32Local, beta, pregMaskFp32);
            DataCopy(meanFp32Local, mean, pregMaskFp32);
            DataCopy(rstdFp32Local, rstd, pregMaskFp32);
        }
    }

    __aicore__ inline void VFNormalize(__local_mem__ T* xLocal, __local_mem__ float* gammaFp32Local,
                                       __local_mem__ float* betaFp32Local, __local_mem__ float* meanFp32Local,
                                       __local_mem__ float* rstdFp32Local, __local_mem__ T* yLocal, uint32_t curElemLen)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> y;

            uint32_t paramCacheElemLen = GetSmallLastChannelParamCacheElemLen();
            uint16_t loopNum = CeilDiv(curElemLen, paramCacheElemLen);
            __local_mem__ T* xLocalTmp = xLocal;
            __local_mem__ T* yLocalTmp = yLocal;
            AscendC::MicroAPI::UnalignReg uX;
            AscendC::MicroAPI::UnalignReg uY;
            AscendC::MicroAPI::DataCopyUnAlignPre(uX, xLocalTmp);
            DataCopy<float, LoadDist::DIST_NORM>(gamma, gammaFp32Local);
            DataCopy<float, LoadDist::DIST_NORM>(beta, betaFp32Local);
            DataCopy<float, LoadDist::DIST_NORM>(mean, meanFp32Local);
            DataCopy<float, LoadDist::DIST_NORM>(rstd, rstdFp32Local);
            for (uint16_t i = 0; i < loopNum; i++) {
                uint32_t elemOffset = i * paramCacheElemLen;
                uint32_t activeLen = curElemLen - elemOffset > paramCacheElemLen ? paramCacheElemLen :
                                                                                   curElemLen - elemOffset;
                uint32_t maskLen = activeLen;
                MaskReg pregMaskFp32 = AscendC::MicroAPI::UpdateMask<float>(maskLen);

                NormCommon::LoadTensorUnAlignForDtypeT(xLocalTmp, x, uX, pregMaskFp32, activeLen);
                NormCommon::NormalizeWithScaleBiasReg(x, gamma, beta, mean, rstd, y, pregMaskFp32);
                NormCommon::StoreTensorUnAlignForDtypeT(yLocalTmp, y, uY, pregMaskFp32, activeLen);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(yLocalTmp, uY, 0);
        }
    }

    template <typename T_SRC>
    __aicore__ inline void GatherParamForDtypeT(__local_mem__ T_SRC* src, RegTensor<float>& dst,
                                                RegTensor<uint32_t>& paramOffset, MaskReg& preg, uint32_t calcLen)
    {
        if constexpr (IsSameType<T_SRC, float>::value) {
            AscendC::MicroAPI::DataCopyGather(dst, (__local_mem__ float*)src, paramOffset, preg);
        } else {
            MaskReg pregSrc = AscendC::MicroAPI::UpdateMask<T_SRC>(calcLen);
            RegTensor<uint16_t> paramOffsetB16;
            RegTensor<T_SRC> srcB16;
            RegTensor<T_SRC> srcB16Unpack;
            AscendC::MicroAPI::Pack(paramOffsetB16, paramOffset);
            AscendC::MicroAPI::DataCopyGather(srcB16, ((__local_mem__ T_SRC*)src), paramOffsetB16, pregSrc);
            AscendC::MicroAPI::UnPack((RegTensor<uint32_t>&)srcB16Unpack, (RegTensor<uint16_t>&)srcB16);
            AscendC::MicroAPI::Cast<float, T_SRC, castTraitB162B32>(dst, srcB16Unpack, preg);
        }
    }

    __aicore__ inline void GatherRunningParamForDtypeT(__local_mem__ T_RUNNING_MEAN* src, RegTensor<float>& dst,
                                                       RegTensor<uint32_t>& paramOffset, MaskReg& preg,
                                                       uint32_t calcLen)
    {
        if constexpr (IsSameType<T_RUNNING_MEAN, float>::value) {
            AscendC::MicroAPI::DataCopyGather(dst, (__local_mem__ float*)src, paramOffset, preg);
        } else {
            MaskReg pregSrc = AscendC::MicroAPI::UpdateMask<T_RUNNING_MEAN>(calcLen);
            RegTensor<uint16_t> paramOffsetB16;
            RegTensor<T_RUNNING_MEAN> srcB16;
            RegTensor<T_RUNNING_MEAN> srcB16Unpack;
            AscendC::MicroAPI::Pack(paramOffsetB16, paramOffset);
            AscendC::MicroAPI::DataCopyGather(srcB16, ((__local_mem__ T_RUNNING_MEAN*)src), paramOffsetB16, pregSrc);
            AscendC::MicroAPI::UnPack((RegTensor<uint32_t>&)srcB16Unpack, (RegTensor<uint16_t>&)srcB16);
            AscendC::MicroAPI::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(dst, srcB16Unpack, preg);
        }
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t curTileBLen, int64_t curTileALen)
    {
        LocalTensor<T> y = yQueue_.DeQue<T>();

        DataCopyExtParams extParams;
        extParams.blockLen = curTileBLen * curTileALen * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        extParams.blockCount = 1;

        DataCopyPad(yGm_[yGmOffset], y, extParams);

        yQueue_.FreeTensor(y);
    }

private:
    const BatchNormV3InferLastChannelTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> betaQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> gammaQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> meanQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> varQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> yQueue_;
    TBuf<TPosition::VECCALC> offsetBuf_;
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
};
} // namespace BatchNormV3Ops

#endif
