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
 * \file batch_norm_infer_last_channel.h
 * \brief
 */

#ifndef NORM_BATCH_NORM_INFER_LAST_CHANNEL_H
#define NORM_BATCH_NORM_INFER_LAST_CHANNEL_H

#include "batch_norm_base.h"
namespace BatchNormOps
{
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;

template <typename T1, typename T2>
class BatchNormInferLastChannel
{
public:
    __aicore__ inline BatchNormInferLastChannel(){};

    __aicore__ inline BatchNormInferLastChannel(const BatchNormInferLastChannelTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                GM_ADDR batch_mean, GM_ADDR batch_variance, GM_ADDR reserve_space_1,
                                GM_ADDR reserve_space_2, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        xGm_.SetGlobalBuffer((__gm__ T1*)x);
        betaGm_.SetGlobalBuffer((__gm__ T2*)beta);
        gammaGm_.SetGlobalBuffer((__gm__ T2*)gamma);
        meanGm_.SetGlobalBuffer((__gm__ T2*)mean);
        varGm_.SetGlobalBuffer((__gm__ T2*)var);

        yGm_.SetGlobalBuffer((__gm__ T1*)y);
        batchMeanGm_.SetGlobalBuffer((__gm__ T2*)batch_mean);
        batchVarGm_.SetGlobalBuffer((__gm__ T2*)batch_variance);
        reserveSpace1Gm_.SetGlobalBuffer((__gm__ T2*)reserve_space_1);
        reserveSpace2Gm_.SetGlobalBuffer((__gm__ T2*)reserve_space_2);

        pipe_->InitBuffer(betaQueue_, DOUBLE_BUFFER, tilingData_->tileBlockALen * sizeof(T2));
        pipe_->InitBuffer(gammaQueue_, DOUBLE_BUFFER, tilingData_->tileBlockALen * sizeof(T2));
        pipe_->InitBuffer(meanQueue_, DOUBLE_BUFFER, tilingData_->tileBlockALen * sizeof(T2));
        pipe_->InitBuffer(varQueue_, DOUBLE_BUFFER, tilingData_->tileBlockALen * sizeof(T2));
        pipe_->InitBuffer(meanOutQueue_, DOUBLE_BUFFER, tilingData_->tileBlockALen * sizeof(T2));
        pipe_->InitBuffer(varOutQueue_, DOUBLE_BUFFER, tilingData_->tileBlockALen * sizeof(T2));

        int64_t xShapeLen = tilingData_->tileBlockBLen * tilingData_->tileBlockALen;
        pipe_->InitBuffer(xQueue_, DOUBLE_BUFFER, xShapeLen * sizeof(T1));
        pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, xShapeLen * sizeof(T1));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        int64_t paddingANumSizeT = tilingData_->tileBlockAPaddingNum * sizeof(T1) / BLOCK_SIZE;

        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curColumIdx = ops::FloorDiv(curIdx, tilingData_->bOuter);
            int64_t curRowIdx = curIdx % tilingData_->bOuter;

            // ping、pang搬运首次或者tile块沿B轴换列时需要拷贝mean、var、beta、gamma
            bool needCopyIn = (curIdx <= beginIdx + 1) || (curRowIdx <= 1);

            // mean、var只在沿B轴每列首行时搬出
            bool needCopyOut = curRowIdx == 0;

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
            CopyInBetaGammaMeanVar(needCopyIn, betaOffset, curTileALen);
            Compute(curTileBLen, curTileALen, needCopyOut);
            CopyOutY(xOffset, curTileBLen, curTileALen, ubStrideT);
            if (needCopyOut) {
                CopyOutMeanVar(betaOffset, curTileALen);
            }
        }
    }

private:
    __aicore__ inline void CopyInX(int64_t xGmOffset, int64_t curTileBLen, int64_t curTileALen, int64_t ubStrideT)
    {
        LocalTensor<T1> xLocal = xQueue_.AllocTensor<T1>();

        DataCopyExtParams extParam;
        extParam.blockLen = curTileALen * sizeof(T1);
        extParam.srcStride = (tilingData_->totalALen - curTileALen) * sizeof(T1);
        extParam.dstStride = ubStrideT;
        extParam.blockCount = curTileBLen;

        DataCopyPadExtParams<T1> padExtParam;
        padExtParam.isPad = false;

        DataCopyPad(xLocal, xGm_[xGmOffset], extParam, padExtParam);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInBetaGammaMeanVar(bool needCopyIn, int64_t offset, int64_t curTileALen)
    {
        LocalTensor<T2> betaLocal = betaQueue_.AllocTensor<T2>();
        LocalTensor<T2> gammaLocal = gammaQueue_.AllocTensor<T2>();
        LocalTensor<T2> meanLocal = meanQueue_.AllocTensor<T2>();
        LocalTensor<T2> varLocal = varQueue_.AllocTensor<T2>();

        if (needCopyIn) {
            DataCopyExtParams extParam;
            extParam.blockCount = 1;
            extParam.blockLen = curTileALen * sizeof(T2);

            DataCopyPadExtParams<T2> padExtParam;
            padExtParam.isPad = false;

            // beta、gamma、mean、var
            DataCopyPad(betaLocal, betaGm_[offset], extParam, padExtParam);
            DataCopyPad(gammaLocal, gammaGm_[offset], extParam, padExtParam);
            DataCopyPad(meanLocal, meanGm_[offset], extParam, padExtParam);
            DataCopyPad(varLocal, varGm_[offset], extParam, padExtParam);
        }

        betaQueue_.EnQue(betaLocal);
        gammaQueue_.EnQue(gammaLocal);
        meanQueue_.EnQue(meanLocal);
        varQueue_.EnQue(varLocal);
    }

    __aicore__ inline void Compute(int64_t curTileBLen, int64_t curTileALen, bool needCopyOut)
    {
        LocalTensor<T1> x = xQueue_.DeQue<T1>();
        LocalTensor<T2> beta = betaQueue_.DeQue<T2>();
        LocalTensor<T2> gamma = gammaQueue_.DeQue<T2>();
        LocalTensor<T2> mean = meanQueue_.DeQue<T2>();
        LocalTensor<T2> var = varQueue_.DeQue<T2>();
        LocalTensor<T1> y = yQueue_.AllocTensor<T1>();

        __local_mem__ T1* xLocal = (__local_mem__ T1*)x.GetPhyAddr();
        __local_mem__ T2* betaLocal = (__local_mem__ T2*)beta.GetPhyAddr();
        __local_mem__ T2* gammaLocal = (__local_mem__ T2*)gamma.GetPhyAddr();
        __local_mem__ T2* meanLocal = (__local_mem__ T2*)mean.GetPhyAddr();
        __local_mem__ T2* varLocal = (__local_mem__ T2*)var.GetPhyAddr();
        __local_mem__ T1* yLocal = (__local_mem__ T1*)y.GetPhyAddr();

        VFNormalize(xLocal, gammaLocal, betaLocal, meanLocal, varLocal, yLocal, curTileBLen, curTileALen);

        yQueue_.EnQue(y);

        if (needCopyOut) {
            DataCopyMeanVar(meanLocal, varLocal, curTileALen);
        }

        xQueue_.FreeTensor<T1>(x);
        betaQueue_.FreeTensor<T2>(beta);
        gammaQueue_.FreeTensor<T2>(gamma);
        meanQueue_.FreeTensor<T2>(mean);
        varQueue_.FreeTensor<T2>(var);
    }

    __aicore__ inline void DataCopyMeanVar(__local_mem__ T2* meanLocal, __local_mem__ T2* varLocal, int64_t curTileALen)
    {
        LocalTensor<T2> meanOut = meanOutQueue_.AllocTensor<T2>();
        __local_mem__ T2* meanOutLocal = (__local_mem__ T2*)meanOut.GetPhyAddr();

        LocalTensor<T2> varOut = varOutQueue_.AllocTensor<T2>();
        __local_mem__ T2* varOutLocal = (__local_mem__ T2*)varOut.GetPhyAddr();

        uint16_t loopNum = ops::CeilDiv(curTileALen, static_cast<int64_t>(VL_FP32));
        __VEC_SCOPE__
        {
            RegTensor<float> meanReg;
            RegTensor<float> varReg;

            MaskReg pregMask;
            uint32_t sreg = curTileALen;

            for (uint16_t k = 0; k < loopNum; k++) {
                pregMask = UpdateMask<float>(sreg);
                int64_t offset = k * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(meanReg, (__local_mem__ float*)meanLocal + offset);
                DataCopy<float, LoadDist::DIST_NORM>(varReg, (__local_mem__ float*)varLocal + offset);
                LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
                DataCopy(((__local_mem__ float*)meanOutLocal) + offset, meanReg, pregMask);
                DataCopy(((__local_mem__ float*)varOutLocal) + offset, varReg, pregMask);
            }
        }
        meanOutQueue_.EnQue(meanOut);
        varOutQueue_.EnQue(varOut);
    }

    __aicore__ inline void VFNormalize(__local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal,
                                       __local_mem__ T2* betaLocal, __local_mem__ T2* meanLocal,
                                       __local_mem__ T2* varLocal, __local_mem__ T1* yLocal, uint16_t curTileBLen,
                                       uint16_t curTileALen)
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

            uint16_t loopNum = ops::CeilDiv(curTileALen, VL_FP32);
            for (uint16_t i = 0; i < loopNum; i++) {
                uint32_t offset = i * VL_FP32;

                // load var mean
                LoadTwoTensorForDtypeT<T2>(var, mean, varLocal, meanLocal, pregMaskFp32, pregMaskFp32, offset, offset);

                CalRstdByHighPrecision(var, rstd, tilingData_->epsilon);

                // load gamma、beta
                LoadTwoTensorForDtypeT<T2>(gamma, beta, gammaLocal, betaLocal, pregMaskFp32, pregMaskFp32, offset,
                                           offset);

                uint32_t tileBlockALenTmp = static_cast<uint32_t>(tilingData_->tileBlockALen);
                for (uint16_t j = 0; j < curTileBLen; j++) {
                    uint32_t xOffset = j * tileBlockALenTmp + offset;
                    // load x
                    LoadTensorForDtypeT<T1>(x, xLocal, pregMaskFp32, xOffset);

                    // compute
                    Sub(x, x, mean, pregMaskFp32);
                    Mul(x, x, gamma, pregMaskFp32);
                    Mul(x, x, rstd, pregMaskFp32);
                    Add(y, x, beta, pregMaskFp32);

                    // store y
                    StoreTensorForDtypeT<T1>(yLocal, y, pregMaskFp32, xOffset);
                }
            }
        }
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t curTileBLen, int64_t curTileALen, int64_t ubStrideT)
    {
        LocalTensor<T1> y = yQueue_.DeQue<T1>();

        DataCopyExtParams extParams;
        extParams.blockLen = curTileALen * sizeof(T1);
        extParams.srcStride = ubStrideT;
        extParams.dstStride = (tilingData_->totalALen - curTileALen) * sizeof(T1);
        extParams.blockCount = curTileBLen;

        DataCopyPad(yGm_[yGmOffset], y, extParams);

        yQueue_.FreeTensor(y);
    }

    __aicore__ inline void CopyOutMeanVar(int64_t betaOffset, int64_t curTileALen)
    {
        DataCopyExtParams extParams;
        extParams.blockLen = curTileALen * sizeof(T2);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        extParams.blockCount = 1;

        LocalTensor<T2> mean = meanOutQueue_.DeQue<T2>();
        DataCopyPad(batchMeanGm_[betaOffset], mean, extParams);
        DataCopyPad(reserveSpace1Gm_[betaOffset], mean, extParams);
        meanOutQueue_.FreeTensor(mean);

        LocalTensor<T2> var = varOutQueue_.DeQue<T2>();
        DataCopyPad(batchVarGm_[betaOffset], var, extParams);
        DataCopyPad(reserveSpace2Gm_[betaOffset], var, extParams);
        varOutQueue_.FreeTensor(var);
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
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> meanOutQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> varOutQueue_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> betaGm_;
    GlobalTensor<T2> gammaGm_;
    GlobalTensor<T2> meanGm_;
    GlobalTensor<T2> varGm_;

    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> batchMeanGm_;
    GlobalTensor<T2> batchVarGm_;
    GlobalTensor<T2> reserveSpace1Gm_;
    GlobalTensor<T2> reserveSpace2Gm_;
};
}  // namespace BatchNormOps

#endif // NORM_BATCH_NORM_INFER_LAST_CHANNEL_H
