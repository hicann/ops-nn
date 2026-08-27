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
 * \file batch_norm_infer_small_ab1.h
 * \brief
 */

#ifndef BATCH_NORM_INFER_SMALL_AB1_H
#define BATCH_NORM_INFER_SMALL_AB1_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "batch_norm_base.h"

namespace BatchNormOps {
using namespace AscendC;

using AscendC::Reg::LoadAlign;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskMergeMode;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreAlign;

template <typename T, typename T_GAMMA, typename T_RUNNING_MEAN>
class BatchNormInferSmallAB1 {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = BatchNormOps::VECTOR_LENGTH;
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = BatchNormOps::BLOCK_SIZE;
    static constexpr int32_t MEAN_VAR_OUTPUT_COUNT = 2; // mean, var

    constexpr static AscendC::Reg::CastTrait castTraitB162B32 = {AscendC::Reg::RegLayout::ZERO,
                                                                 AscendC::Reg::SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};

public:
    __aicore__ inline BatchNormInferSmallAB1(){};

    __aicore__ inline BatchNormInferSmallAB1(const BatchNormInferTilingData* tilingDataIn)
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

        pipe_->InitBuffer(betaQueue_, BUFFER_NUM, tilingData_->totalALen * sizeof(T_GAMMA));
        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, tilingData_->totalALen * sizeof(T_GAMMA));
        pipe_->InitBuffer(meanQueue_, BUFFER_NUM, tilingData_->totalALen * sizeof(float));
        pipe_->InitBuffer(varQueue_, BUFFER_NUM, tilingData_->totalALen * sizeof(float));

        int64_t xShapeLen = tilingData_->tileBlockB0Len * tilingData_->totalALen * tilingData_->totalB1Len;
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(T));
        int64_t varOffset = AlignUp(tilingData_->totalALen, BLOCK_SIZE / sizeof(T_RUNNING_MEAN));
        pipe_->InitBuffer(meanVarOutQueue_, BUFFER_DEPTH, MEAN_VAR_OUTPUT_COUNT * varOffset * sizeof(T_RUNNING_MEAN));

        int64_t paramCacheElemLen = GetSmallAB1ParamCacheElemLen();
        int64_t alignedParamCacheLen = AlignUp(paramCacheElemLen, VL_FP32);
        pipe_->InitBuffer(offsetBuf_, alignedParamCacheLen * sizeof(uint32_t));
        pipe_->InitBuffer(betaFp32Buf_, alignedParamCacheLen * sizeof(float));
        pipe_->InitBuffer(gammaFp32Buf_, alignedParamCacheLen * sizeof(float));
        pipe_->InitBuffer(meanFp32Buf_, alignedParamCacheLen * sizeof(float));
        pipe_->InitBuffer(rstdFp32Buf_, alignedParamCacheLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        InitSmallAB1ParamOffset();
        CopyInBetaGammaMeanVar(true, 0, tilingData_->totalALen);
        PrepareSmallAB1ParamCache();

        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curTileB0Len = curIdx == (tilingData_->b0Outer - 1) ? tilingData_->tileBlockB0Tail :
                                                                          tilingData_->tileBlockB0Len;
            int64_t xOffset = curIdx * tilingData_->tileBlockB0Len * tilingData_->totalALen * tilingData_->totalB1Len;

            CopyInX(xOffset, curTileB0Len);
            Compute(curTileB0Len);
            CopyOutY(xOffset, curTileB0Len);
        }
    }

private:
    __aicore__ inline int64_t AlignUp(int64_t value, int64_t base) const { return (value + base - 1) / base * base; }

    __aicore__ inline void InitSmallAB1ParamOffset()
    {
        LocalTensor<uint32_t> offset = offsetBuf_.Get<uint32_t>();
        uint32_t abLen = static_cast<uint32_t>(tilingData_->totalALen * tilingData_->totalB1Len);
        uint32_t b1Len = static_cast<uint32_t>(tilingData_->totalB1Len);
        uint32_t paramCacheElemLen = GetSmallAB1ParamCacheElemLen();
        for (uint32_t i = 0; i < paramCacheElemLen; i++) {
            offset.SetValue(i, (i % abLen) / b1Len);
        }
    }

    __aicore__ inline void CopyInX(int64_t xGmOffset, int64_t curTileB0Len)
    {
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = curTileB0Len * tilingData_->totalALen * tilingData_->totalB1Len * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyPad<T, PaddingMode::Normal>(xLocal, xGm_[xGmOffset], copyInParams, dataCopyPadExtParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInBetaGammaMeanVar(bool needCopy, int64_t offset, int64_t curTileALen)
    {
        LocalTensor<T_GAMMA> betaLocal = betaQueue_.AllocTensor<T_GAMMA>();
        LocalTensor<T_GAMMA> gammaLocal = gammaQueue_.AllocTensor<T_GAMMA>();
        LocalTensor<T_RUNNING_MEAN> meanLocal = meanQueue_.AllocTensor<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> varLocal = varQueue_.AllocTensor<T_RUNNING_MEAN>();

        if (needCopy) {
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
        }

        betaQueue_.EnQue(betaLocal);
        gammaQueue_.EnQue(gammaLocal);
        meanQueue_.EnQue(meanLocal);
        varQueue_.EnQue(varLocal);
    }

    __aicore__ inline void PrepareSmallAB1ParamCache()
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

        __ubuf__ T_GAMMA* betaLocal = (__ubuf__ T_GAMMA*)beta.GetPhyAddr();
        __ubuf__ T_GAMMA* gammaLocal = (__ubuf__ T_GAMMA*)gamma.GetPhyAddr();
        __ubuf__ T_RUNNING_MEAN* meanLocal = (__ubuf__ T_RUNNING_MEAN*)mean.GetPhyAddr();
        __ubuf__ T_RUNNING_MEAN* varLocal = (__ubuf__ T_RUNNING_MEAN*)var.GetPhyAddr();
        __ubuf__ uint32_t* offsetLocal = (__ubuf__ uint32_t*)offset.GetPhyAddr();
        __ubuf__ float* betaFp32Local = (__ubuf__ float*)betaFp32.GetPhyAddr();
        __ubuf__ float* gammaFp32Local = (__ubuf__ float*)gammaFp32.GetPhyAddr();
        __ubuf__ float* meanFp32Local = (__ubuf__ float*)meanFp32.GetPhyAddr();
        __ubuf__ float* rstdFp32Local = (__ubuf__ float*)rstdFp32.GetPhyAddr();

        VFPrepareSmallAB1ParamCache(gammaLocal, betaLocal, meanLocal, varLocal, offsetLocal, gammaFp32Local,
                                    betaFp32Local, meanFp32Local, rstdFp32Local);
        CopyOutMeanVar(tilingData_->totalALen);

        betaQueue_.FreeTensor<T_GAMMA>(beta);
        gammaQueue_.FreeTensor<T_GAMMA>(gamma);
        meanQueue_.FreeTensor<T_RUNNING_MEAN>(mean);
        varQueue_.FreeTensor<T_RUNNING_MEAN>(var);
    }

    __aicore__ inline void CopyOutMeanVar(int64_t curTileALen)
    {
        if (GetBlockIdx() != 0) {
            return;
        }

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

        LocalTensor<T_RUNNING_MEAN> meanVarOut = meanVarOutQueue_
                                                     .DeQue<QuePosition::VECOUT, QuePosition::GM, T_RUNNING_MEAN>();

        DataCopyPad(batchMeanGm_[0], meanVarOut, extParams);
        DataCopyPad(reserveSpace1Gm_[0], meanVarOut, extParams);
        DataCopyPad(batchVarGm_[0], meanVarOut[varOffset], extParams);
        DataCopyPad(reserveSpace2Gm_[0], meanVarOut[varOffset], extParams);

        meanVarOutQueue_.FreeTensor(meanVarOut);
    }

    __aicore__ inline void Compute(int64_t curTileB0Len)
    {
        LocalTensor<T> x = xQueue_.DeQue<T>();
        LocalTensor<T> y = yQueue_.AllocTensor<T>();
        LocalTensor<float> betaFp32 = betaFp32Buf_.Get<float>();
        LocalTensor<float> gammaFp32 = gammaFp32Buf_.Get<float>();
        LocalTensor<float> meanFp32 = meanFp32Buf_.Get<float>();
        LocalTensor<float> rstdFp32 = rstdFp32Buf_.Get<float>();

        __ubuf__ T* xLocal = (__ubuf__ T*)x.GetPhyAddr();
        __ubuf__ T* yLocal = (__ubuf__ T*)y.GetPhyAddr();
        __ubuf__ float* betaFp32Local = (__ubuf__ float*)betaFp32.GetPhyAddr();
        __ubuf__ float* gammaFp32Local = (__ubuf__ float*)gammaFp32.GetPhyAddr();
        __ubuf__ float* meanFp32Local = (__ubuf__ float*)meanFp32.GetPhyAddr();
        __ubuf__ float* rstdFp32Local = (__ubuf__ float*)rstdFp32.GetPhyAddr();

        VFNormalize(xLocal, gammaFp32Local, betaFp32Local, meanFp32Local, rstdFp32Local, yLocal, curTileB0Len);

        yQueue_.EnQue(y);

        xQueue_.FreeTensor<T>(x);
    }

    __aicore__ inline void VFPrepareSmallAB1ParamCache(__ubuf__ T_GAMMA* gammaLocal, __ubuf__ T_GAMMA* betaLocal,
                                                       __ubuf__ T_RUNNING_MEAN* meanLocal,
                                                       __ubuf__ T_RUNNING_MEAN* varLocal,
                                                       __ubuf__ uint32_t* offsetLocal, __ubuf__ float* gammaFp32Local,
                                                       __ubuf__ float* betaFp32Local, __ubuf__ float* meanFp32Local,
                                                       __ubuf__ float* rstdFp32Local)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> mean;
            RegTensor<float> var;
            RegTensor<float> rstd;
            RegTensor<uint32_t> paramOffset;

            uint32_t paramCacheElemLen = GetSmallAB1ParamCacheElemLen();
            uint32_t maskLen = paramCacheElemLen;
            MaskReg pregMask = AscendC::Reg::UpdateMask<float>(maskLen);

            AscendC::Reg::LoadAlign<uint32_t, LoadDist::DIST_NORM>(paramOffset, offsetLocal);
            GatherParamForDtypeT(gammaLocal, gamma, paramOffset, pregMask, paramCacheElemLen);
            GatherParamForDtypeT(betaLocal, beta, paramOffset, pregMask, paramCacheElemLen);
            GatherRunningParamForDtypeT(varLocal, var, paramOffset, pregMask, paramCacheElemLen);
            NormCommon::ComputeRstdNewtonRaphsonReg(var, rstd, pregMask, tilingData_->epsilon);
            GatherRunningParamForDtypeT(meanLocal, mean, paramOffset, pregMask, paramCacheElemLen);

            StoreAlign(gammaFp32Local, gamma, pregMask);
            StoreAlign(betaFp32Local, beta, pregMask);
            StoreAlign(meanFp32Local, mean, pregMask);
            StoreAlign(rstdFp32Local, rstd, pregMask);
        }
    }

    __aicore__ inline void VFNormalize(__ubuf__ T* xLocal, __ubuf__ float* gammaFp32Local,
                                       __ubuf__ float* betaFp32Local, __ubuf__ float* meanFp32Local,
                                       __ubuf__ float* rstdFp32Local, __ubuf__ T* yLocal, uint16_t curTileB0Len)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> y;

            uint32_t elemLen = static_cast<uint32_t>(curTileB0Len * tilingData_->totalALen * tilingData_->totalB1Len);
            uint32_t paramCacheElemLen = GetSmallAB1ParamCacheElemLen();
            uint16_t loopNum = ops::CeilDiv(elemLen, paramCacheElemLen);
            __ubuf__ T* xLocalTmp = xLocal;
            __ubuf__ T* yLocalTmp = yLocal;
            AscendC::Reg::UnalignRegForLoad uX;
            AscendC::Reg::UnalignRegForStore uY;
            AscendC::Reg::LoadUnAlignPre(uX, xLocalTmp);
            LoadAlign<float, LoadDist::DIST_NORM>(gamma, gammaFp32Local);
            LoadAlign<float, LoadDist::DIST_NORM>(beta, betaFp32Local);
            LoadAlign<float, LoadDist::DIST_NORM>(mean, meanFp32Local);
            LoadAlign<float, LoadDist::DIST_NORM>(rstd, rstdFp32Local);
            for (uint16_t i = 0; i < loopNum; i++) {
                uint32_t elemOffset = i * paramCacheElemLen;
                uint32_t activeLen = elemLen - elemOffset > paramCacheElemLen ? paramCacheElemLen :
                                                                                elemLen - elemOffset;
                uint32_t maskLen = activeLen;
                MaskReg pregMask = AscendC::Reg::UpdateMask<float>(maskLen);

                NormCommon::LoadTensorUnAlignForDtypeT(xLocalTmp, x, uX, pregMask, activeLen);
                NormCommon::NormalizeWithScaleBiasReg(x, gamma, beta, mean, rstd, y, pregMask);
                NormCommon::StoreTensorUnAlignForDtypeT(yLocalTmp, y, uY, pregMask, activeLen);
            }
            AscendC::Reg::StoreUnAlignPost(yLocalTmp, uY, 0);
        }
    }

    __aicore__ inline uint32_t GetSmallAB1ParamCacheElemLen() const
    {
        uint32_t abLen = static_cast<uint32_t>(tilingData_->totalALen * tilingData_->totalB1Len);
        return static_cast<uint32_t>(VL_FP32 / abLen) * abLen;
    }

    template <typename T_SRC>
    __aicore__ inline void GatherParamForDtypeT(__ubuf__ T_SRC* src, RegTensor<float>& dst,
                                                RegTensor<uint32_t>& paramOffset, MaskReg& preg, uint32_t calcLen)
    {
        if constexpr (IsSameType<T_SRC, float>::value) {
            AscendC::Reg::Gather(dst, (__ubuf__ float*)src, paramOffset, preg);
        } else {
            MaskReg pregSrc = AscendC::Reg::UpdateMask<T_SRC>(calcLen);
            RegTensor<uint16_t> paramOffsetB16;
            RegTensor<T_SRC> srcB16;
            RegTensor<T_SRC> srcB16Unpack;
            AscendC::Reg::Pack(paramOffsetB16, paramOffset);
            AscendC::Reg::Gather(srcB16, ((__ubuf__ T_SRC*)src), paramOffsetB16, pregSrc);
            AscendC::Reg::UnPack((RegTensor<uint32_t>&)srcB16Unpack, (RegTensor<uint16_t>&)srcB16);
            AscendC::Reg::Cast<float, T_SRC, castTraitB162B32>(dst, srcB16Unpack, preg);
        }
    }

    __aicore__ inline void GatherRunningParamForDtypeT(__ubuf__ T_RUNNING_MEAN* src, RegTensor<float>& dst,
                                                       RegTensor<uint32_t>& paramOffset, MaskReg& preg,
                                                       uint32_t calcLen)
    {
        if constexpr (IsSameType<T_RUNNING_MEAN, float>::value) {
            AscendC::Reg::Gather(dst, (__ubuf__ float*)src, paramOffset, preg);
        } else {
            MaskReg pregSrc = AscendC::Reg::UpdateMask<T_RUNNING_MEAN>(calcLen);
            RegTensor<uint16_t> paramOffsetB16;
            RegTensor<T_RUNNING_MEAN> srcB16;
            RegTensor<T_RUNNING_MEAN> srcB16Unpack;
            AscendC::Reg::Pack(paramOffsetB16, paramOffset);
            AscendC::Reg::Gather(srcB16, ((__ubuf__ T_RUNNING_MEAN*)src), paramOffsetB16, pregSrc);
            AscendC::Reg::UnPack((RegTensor<uint32_t>&)srcB16Unpack, (RegTensor<uint16_t>&)srcB16);
            AscendC::Reg::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(dst, srcB16Unpack, preg);
        }
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t curTileB0Len)
    {
        LocalTensor<T> y = yQueue_.DeQue<T>();
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curTileB0Len * tilingData_->totalALen * tilingData_->totalB1Len * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad<T, PaddingMode::Normal>(yGm_[yGmOffset], y, copyOutParams);
        yQueue_.FreeTensor(y);
    }

private:
    const BatchNormInferTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> betaQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> gammaQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> meanQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> varQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> yQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_DEPTH> meanVarOutQueue_;
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
    GlobalTensor<T_RUNNING_MEAN> batchMeanGm_;
    GlobalTensor<T_RUNNING_MEAN> batchVarGm_;
    GlobalTensor<T_RUNNING_MEAN> reserveSpace1Gm_;
    GlobalTensor<T_RUNNING_MEAN> reserveSpace2Gm_;
};
} // namespace BatchNormOps

#endif
