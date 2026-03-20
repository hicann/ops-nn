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
 * \file batch_norm_infer.h
 * \brief
 */

#ifndef NORM_BATCH_NORM_INFER_H
#define NORM_BATCH_NORM_INFER_H

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
class BatchNormInfer
{
public:
    __aicore__ inline BatchNormInfer(){};

    __aicore__ inline BatchNormInfer(const BatchNormInferTilingData* tilingDataIn)
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

        int64_t xShapeLen = tilingData_->tileBlockB1Len * tilingData_->tileBlockALen * tilingData_->tileBlockB0Len;
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

        // pattern is [B0, A, B1]
        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curB1Idx = curIdx % tilingData_->b1Outer;
            int64_t curAIdx = (curIdx / tilingData_->b1Outer) / tilingData_->b0Outer;
            int64_t curB0Idx = (curIdx / tilingData_->b1Outer) % tilingData_->b0Outer;

            // ping、pang搬运首次或者tile块沿B轴换列时需要拷贝mean、var、beta、gamma
            // curIdx 整数倍(tilingData_->b0Outer * tilingData_->b1Outer) 表示一轮循环, 出现换行
            int64_t modFusedBOuter = curIdx % (tilingData_->b0Outer * tilingData_->b1Outer);
            bool needCopyIn = (curIdx <= beginIdx + 1) || (modFusedBOuter <= 1);

            // mean、var只在一轮首次时搬出
            bool needCopyOut = modFusedBOuter == 0;

            // Tile整尾块
            int64_t curTileB0Len =
                curB0Idx == (tilingData_->b0Outer - 1) ? tilingData_->tileBlockB0Tail : tilingData_->tileBlockB0Len;
            int64_t curTileALen =
                curAIdx == (tilingData_->aOuter - 1) ? tilingData_->tileBlockATail : tilingData_->tileBlockALen;
            int64_t curTileB1Len =
                curB1Idx == (tilingData_->b1Outer - 1) ? tilingData_->tileBlockB1Tail : tilingData_->tileBlockB1Len;

            int64_t ubStrideT = 0;
            int64_t ubStrideFloat = 0;

            if (curAIdx == (tilingData_->aOuter - 1)) {
                ubStrideT = paddingANumSizeT;
            }

            // x、y偏移一致，beta、gamma、mean、var偏移一致
            int64_t betaOffset = curAIdx * tilingData_->tileBlockALen;
            int64_t xOffset =
                // b0 offset
                curB0Idx * tilingData_->tileBlockB0Len * tilingData_->totalALen * tilingData_->totalB1Len +
                // a offset
                curAIdx * tilingData_->tileBlockALen * tilingData_->totalB1Len +
                // b1 offset
                curB1Idx * tilingData_->tileBlockB1Len;

            CopyInX(xOffset, curTileB0Len, curTileALen, curTileB1Len, ubStrideT);
            CopyInBetaGammaMeanVar(needCopyIn, betaOffset, curTileALen);
            Compute(curTileB0Len, curTileALen, curTileB1Len, needCopyOut);
            CopyOutY(xOffset, curTileB0Len, curTileALen, curTileB1Len, ubStrideT);
            if (needCopyOut) {
                CopyOutMeanVar(betaOffset, curTileALen);
            }
        }
    }

private:
    __aicore__ inline void CopyInX(int64_t xGmOffset, int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len,
                                   int64_t ubStrideT)
    {
        LocalTensor<T1> xLocal = xQueue_.AllocTensor<T1>();
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curTileB0Len;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->totalALen * tilingData_->totalB1Len * sizeof(T1);
        loopParams.loop1DstStride = tilingData_->tileBlockB1Len * tilingData_->tileBlockALen * sizeof(T1);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPadExtParams<T1> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileALen;
        copyInParams.blockLen = curTileB1Len * sizeof(T1);
        copyInParams.srcStride = (tilingData_->totalB1Len - curTileB1Len) * sizeof(T1);
        copyInParams.dstStride = (tilingData_->tileBlockB1Len - curTileB1Len) * sizeof(T1) / BLOCK_SIZE;
        DataCopyPad<T1, PaddingMode::Normal>(xLocal, xGm_[xGmOffset], copyInParams, dataCopyPadExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInBetaGammaMeanVar(bool needCopy, int64_t offset, int64_t curTileALen)
    {
        LocalTensor<T2> betaLocal = betaQueue_.AllocTensor<T2>();
        LocalTensor<T2> gammaLocal = gammaQueue_.AllocTensor<T2>();
        LocalTensor<T2> meanLocal = meanQueue_.AllocTensor<T2>();
        LocalTensor<T2> varLocal = varQueue_.AllocTensor<T2>();

        if (needCopy) {
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

    __aicore__ inline void Compute(int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len, bool needCopyOut)
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

        VFNormalize(xLocal, gammaLocal, betaLocal, meanLocal, varLocal, yLocal, curTileB0Len, curTileALen,
                    curTileB1Len);

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
                                       __local_mem__ T2* varLocal, __local_mem__ T1* yLocal, uint16_t curTileB0Len,
                                       uint16_t curTileALen, uint16_t curTileB1Len)
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

            MaskReg pregMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

            uint16_t loopNum = (curTileB1Len + VL_FP32 - 1) / VL_FP32;
            for (uint16_t i = 0; i < curTileALen; i++) {
                // loads var  1->64
                LoadTwoTensorForDtypeTBrc<T2>(var, mean, varLocal, meanLocal, pregMask, pregMask, i, i);
                CalRstdByHighPrecision(var, rstd, tilingData_->epsilon);

                // load gamma、beta  1->64
                LoadTwoTensorForDtypeTBrc<T2>(gamma, beta, gammaLocal, betaLocal, pregMask, pregMask, i, i);

                uint32_t tileBlockALenTmp = static_cast<uint32_t>(tilingData_->tileBlockALen);
                uint32_t tileBlockB1LenTmp = static_cast<uint32_t>(tilingData_->tileBlockB1Len);
                for (uint16_t j = 0; j < curTileB0Len; j++) {
                    for (uint16_t k = 0; k < loopNum; k++) {
                        uint32_t xOffset = (j * tileBlockALenTmp + i) * tileBlockB1LenTmp + k * VL_FP32;

                        // load x
                        LoadTensorForDtypeT<T1>(x, xLocal, pregMask, xOffset);

                        // compute
                        Sub(x, x, mean, pregMask);
                        Mul(x, x, gamma, pregMask);
                        Mul(x, x, rstd, pregMask);
                        Add(y, x, beta, pregMask);

                        // store y
                        StoreTensorForDtypeT<T1>(yLocal, y, pregMask, xOffset);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len,
                                    int64_t ubStrideT)
    {
        LocalTensor<T1> y = yQueue_.DeQue<T1>();
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curTileB0Len;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->tileBlockB1Len * tilingData_->tileBlockALen * sizeof(T1);
        loopParams.loop1DstStride = tilingData_->totalALen * tilingData_->totalB1Len * sizeof(T1);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileALen;
        copyInParams.blockLen = curTileB1Len * sizeof(T1);
        copyInParams.srcStride = (tilingData_->tileBlockB1Len - curTileB1Len) * sizeof(T1) / BLOCK_SIZE;
        copyInParams.dstStride = (tilingData_->totalB1Len - curTileB1Len) * sizeof(T1);
        DataCopyPad<T1, PaddingMode::Normal>(yGm_[yGmOffset], y, copyInParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
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
    const BatchNormInferTilingData* tilingData_;

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

#endif // NORM_BATCH_NORM_INFER_H
