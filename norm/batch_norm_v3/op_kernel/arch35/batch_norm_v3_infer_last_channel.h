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
 * \file batch_norm_v3_infer_last_channel.h
 * \brief
 */

#ifndef BATCH_NORM_V3_INFER_LAST_CHANNEL_H
#define BATCH_NORM_V3_INFER_LAST_CHANNEL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "batch_norm_v3.h"
namespace BatchNormV3Ops {
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;

template <typename T, typename T_GAMMA, typename T_RUNNING_MEAN>
class BatchNormV3InferLastChannel {
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
    __aicore__ inline BatchNormV3InferLastChannel(){};

    __aicore__ inline BatchNormV3InferLastChannel(const BatchNormV3InferLastChannelTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var, GM_ADDR y, TPipe* pipeIn)
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
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        int64_t paddingANumSizeT = tilingData_->tileBlockAPaddingNum * sizeof(T) / BLOCK_SIZE;

        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curColumIdx = FloorDiv(curIdx, tilingData_->bOuter);
            int64_t curRowIdx = curIdx % tilingData_->bOuter;

            // ping、pang搬运首次或者tile块沿B轴换列时需要拷贝mean、var、beta、gamma
            bool needCopy = (curIdx <= beginIdx + 1) || (curRowIdx <= 1);

            int64_t curTileBLen =
                curRowIdx == (tilingData_->bOuter - 1) ? tilingData_->tileBlockBTail : tilingData_->tileBlockBLen;

            int64_t curTileALen = tilingData_->tileBlockALen;
            int64_t ubStrideT = 0;
            int64_t ubStrideFloat = 0;

            if (curColumIdx == (tilingData_->aOuter - 1)) {
                curTileALen = tilingData_->tileBlockATail;
                ubStrideT = paddingANumSizeT;
            }

            // x、y偏移一致，beta、gamma、mean、var偏移一致
            int64_t betaOffset = curColumIdx * tilingData_->tileBlockALen;
            int64_t xOffset = curRowIdx * tilingData_->totalALen * tilingData_->tileBlockBLen + betaOffset;

            CopyInX(xOffset, curTileBLen, curTileALen, ubStrideT);
            CopyInBetaGammaMeanVar(needCopy, betaOffset, curTileALen);
            Compute(curTileBLen, curTileALen);
            CopyOutY(xOffset, curTileBLen, curTileALen, ubStrideT);
        }
    }

private:
    __aicore__ inline void CopyInX(int64_t xGmOffset, int64_t curTileBLen, int64_t curTileALen, int64_t ubStrideT)
    {
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();

        DataCopyExtParams extParam;
        extParam.blockLen = curTileALen * sizeof(T);
        extParam.srcStride = (tilingData_->totalALen - curTileALen) * sizeof(T);
        extParam.dstStride = ubStrideT;
        extParam.blockCount = curTileBLen;

        DataCopyPadExtParams<T> padExtParam;
        padExtParam.isPad = false;

        DataCopyPad(xLocal, xGm_[xGmOffset], extParam, padExtParam);
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

            // beta、gamma
            extParam.blockLen = curTileALen * sizeof(T_GAMMA);

            DataCopyPadExtParams<T_GAMMA> padExtParam;
            padExtParam.isPad = false;

            DataCopyPad(betaLocal, betaGm_[offset], extParam, padExtParam);
            DataCopyPad(gammaLocal, gammaGm_[offset], extParam, padExtParam);

            // mean、var
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

    __aicore__ inline void Compute(int64_t curTileBLen, int64_t curTileALen)
    {
        LocalTensor<T> x = xQueue_.DeQue<T>();
        LocalTensor<T_GAMMA> beta = betaQueue_.DeQue<T_GAMMA>();
        LocalTensor<T_GAMMA> gamma = gammaQueue_.DeQue<T_GAMMA>();
        LocalTensor<T_RUNNING_MEAN> mean = meanQueue_.DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> var = varQueue_.DeQue<T_RUNNING_MEAN>();
        LocalTensor<T> y = yQueue_.AllocTensor<T>();

        __local_mem__ T* xLocal = (__local_mem__ T*)x.GetPhyAddr();
        __local_mem__ T_GAMMA* betaLocal = (__local_mem__ T_GAMMA*)beta.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaLocal = (__local_mem__ T_GAMMA*)gamma.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* meanLocal = (__local_mem__ T_RUNNING_MEAN*)mean.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* varLocal = (__local_mem__ T_RUNNING_MEAN*)var.GetPhyAddr();
        __local_mem__ T* yLocal = (__local_mem__ T*)y.GetPhyAddr();

        VFNormalize(xLocal, gammaLocal, betaLocal, meanLocal, varLocal, yLocal, curTileBLen, curTileALen);

        yQueue_.EnQue(y);

        xQueue_.FreeTensor<T>(x);
        betaQueue_.FreeTensor<T_GAMMA>(beta);
        gammaQueue_.FreeTensor<T_GAMMA>(gamma);
        meanQueue_.FreeTensor<T_RUNNING_MEAN>(mean);
        varQueue_.FreeTensor<T_RUNNING_MEAN>(var);
    }

    __aicore__ inline void VFNormalize(
        __local_mem__ T* xLocal, __local_mem__ T_GAMMA* gammaLocal, __local_mem__ T_GAMMA* betaLocal,
        __local_mem__ T_RUNNING_MEAN* meanLocal, __local_mem__ T_RUNNING_MEAN* varLocal, __local_mem__ T* yLocal,
        uint16_t curTileBLen, uint16_t curTileALen)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> mean;
            RegTensor<float> var;
            RegTensor<float> y;

            RegTensor<float> rstd;

            MaskReg pregMaskFp32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

            uint16_t loopNum = CeilDiv(curTileALen, VL_FP32);
            for (uint16_t i = 0; i < loopNum; i++) {
                uint32_t offset = i * VL_FP32;

                // load var
                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把T_RUNNING_MEAN的输入cast到float
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> runningVarTmp;
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        runningVarTmp, ((__local_mem__ T_RUNNING_MEAN*)varLocal + offset));
                    AscendC::MicroAPI::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(var, runningVarTmp, pregMaskFp32);
                } else {
                    AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_NORM>(var, varLocal + offset);
                }

                CalRstdByHighPrecision(var, rstd, tilingData_->epsilon);

                // load mean
                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把T_RUNNING_MEAN的输入cast到float
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> runningMeanTmp;
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        runningMeanTmp, ((__local_mem__ T_RUNNING_MEAN*)meanLocal + offset));
                    AscendC::MicroAPI::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(
                        mean, runningMeanTmp, pregMaskFp32);
                } else {
                    AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_NORM>(mean, meanLocal + offset);
                }
                // load gamma、beta
                LoadTensorForDtypeT(gammaLocal, gamma, pregMaskFp32, offset);
                LoadTensorForDtypeT(betaLocal, beta, pregMaskFp32, offset);

                uint32_t tileBlockALenTmp = static_cast<uint32_t>(tilingData_->tileBlockALen);
                for (uint16_t j = 0; j < curTileBLen; j++) {
                    uint32_t xOffset = j * tileBlockALenTmp + offset;
                    // load x
                    LoadTensorForDtypeT(xLocal, x, pregMaskFp32, xOffset);

                    // compute
                    Sub(x, x, mean, pregMaskFp32);
                    Mul(x, x, gamma, pregMaskFp32);
                    Mul(x, x, rstd, pregMaskFp32);
                    Add(y, x, beta, pregMaskFp32);

                    // copy out
                    if constexpr (IsSameType<T, float>::value) {
                        DataCopy(((__local_mem__ float*)yLocal) + xOffset, y, pregMaskFp32);
                    } else { // fp16、bf16
                        RegTensor<T> xFp16;
                        Cast<T, float, castTraitB322B16>(xFp16, y, pregMaskFp32);
                        DataCopy<T, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ T*)yLocal) + xOffset, xFp16, pregMaskFp32);
                    }
                }
            }
        }
    }

    template <typename T_SRC>
    __aicore__ inline void LoadTensorForDtypeT(
        __local_mem__ T_SRC* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_SRC, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
        } else { // fp16、bf16
            RegTensor<T> xFp16;
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
            Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
        }
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t curTileBLen, int64_t curTileALen, int64_t ubStrideT)
    {
        LocalTensor<T> y = yQueue_.DeQue<T>();

        DataCopyExtParams extParams;
        extParams.blockLen = curTileALen * sizeof(T);
        extParams.srcStride = ubStrideT;
        extParams.dstStride = (tilingData_->totalALen - curTileALen) * sizeof(T);
        extParams.blockCount = curTileBLen;

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

    GlobalTensor<T> yGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T_GAMMA> betaGm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<T_RUNNING_MEAN> meanGm_;
    GlobalTensor<T_RUNNING_MEAN> varGm_;
};
} // namespace BatchNormV3Ops

#endif
