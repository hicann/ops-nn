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
 * \file batch_norm_grad_infer_channel_last.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_INFER_CHANNEL_LAST_H
#define BATCH_NORM_GRAD_INFER_CHANNEL_LAST_H

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "batch_norm_grad_common.h"

namespace BatchNormGrad {
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MemType;

using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::StoreDist;

template <typename T1, typename T2>
class BatchNormGradInferChannelLastDx
{
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradInferChannelLastDx(){};

    __aicore__ inline BatchNormGradInferChannelLastDx(const BatchNormGradInferChannelLastDxTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }
    __aicore__ inline void InitStg0(GM_ADDR dy, GM_ADDR weight, GM_ADDR runningVar, GM_ADDR dx, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        dyGm_.SetGlobalBuffer((__gm__ T1*)dy);
        gammaGm_.SetGlobalBuffer((__gm__ T2*)weight);
        runningVarGm_.SetGlobalBuffer((__gm__ T2*)runningVar);
        dxGm_.SetGlobalBuffer((__gm__ T1*)dx);

        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T2));
        pipe_->InitBuffer(runningVarQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T2));

        int64_t xShapeLen = tilingData_->tileBlockBLen * tilingData_->tileBlockALen;
        pipe_->InitBuffer(dxQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
        pipe_->InitBuffer(dyQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
    }

    __aicore__ inline void ProcessStg0()
    {
        if (GetBlockIdx() >= tilingData_->usedCoreNums) {
            return;
        }
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        int64_t paddingANumSizeT1 = tilingData_->tileBlockAPaddingNum * sizeof(T1) / BLOCK_SIZE;

        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curColumIdx = ops::FloorDiv(curIdx, tilingData_->bOuter);
            int64_t curRowIdx = curIdx % tilingData_->bOuter;

            // ping、pang搬运首次或者tile块沿B轴换列时需要拷贝gamma、runningVar
            bool needCopy = (curIdx <= beginIdx + 1) || (curRowIdx <= 1);

            int64_t curTileBLen =
                curRowIdx == (tilingData_->bOuter - 1) ? tilingData_->tileBlockBTail : tilingData_->tileBlockBLen;

            int64_t curTileALen = tilingData_->tileBlockALen;
            int64_t ubStrideT1 = 0;
            int64_t ubStrideFloat = 0;

            if (curColumIdx == (tilingData_->aOuter - 1)) {
                curTileALen = tilingData_->tileBlockATail;
                ubStrideT1 = paddingANumSizeT1;
            }

            // x、y偏移一致，gamma、runningVar偏移一致
            int64_t weightOffset = curColumIdx * tilingData_->tileBlockALen;
            int64_t dyOffset = curRowIdx * tilingData_->aDim * tilingData_->tileBlockBLen + weightOffset;

            CopyInDy(dyOffset, curTileBLen, curTileALen, ubStrideT1);
            CopyInGammaVar(needCopy, weightOffset, curTileALen);
            Compute(curTileBLen, curTileALen);
            CopyOutDx(dyOffset, curTileBLen, curTileALen, ubStrideT1);
        }
    }

private:
    __aicore__ inline void CopyInDy(int64_t dyGmOffset, int64_t curTileBLen, int64_t curTileALen, int64_t ubStrideT1)
    {
        LocalTensor<T1> dyLocal = dyQueue_.AllocTensor<T1>();

        DataCopyExtParams extParam;
        extParam.blockLen = curTileALen * sizeof(T1);
        extParam.srcStride = (tilingData_->aDim - curTileALen) * sizeof(T1);
        extParam.dstStride = ubStrideT1;
        extParam.blockCount = curTileBLen;

        DataCopyPadExtParams<T1> padExtParam;
        padExtParam.isPad = false;

        DataCopyPad(dyLocal, dyGm_[dyGmOffset], extParam, padExtParam);
        dyQueue_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInGammaVar(bool needCopy, int64_t offset, int64_t curTileALen)
    {
        LocalTensor<T2> gammaLocal = gammaQueue_.AllocTensor<T2>();
        LocalTensor<T2> varLocal = runningVarQueue_.AllocTensor<T2>();

        if (needCopy) {
            DataCopyExtParams extParam;
            extParam.blockCount = 1;

            // gamma
            extParam.blockLen = curTileALen * sizeof(T2);

            DataCopyPadExtParams<T2> padExtParam;
            padExtParam.isPad = false;

            DataCopyPad(gammaLocal, gammaGm_[offset], extParam, padExtParam);

            // runningVar
            extParam.blockLen = curTileALen * sizeof(T2);

            DataCopyPadExtParams<T2> padExtParams1;
            padExtParams1.isPad = false;

            DataCopyPad(varLocal, runningVarGm_[offset], extParam, padExtParams1);
        }

        gammaQueue_.EnQue(gammaLocal);
        runningVarQueue_.EnQue(varLocal);
    }

    __aicore__ inline void Compute(int64_t curTileBLen, int64_t curTileALen)
    {
        LocalTensor<T1> dy = dyQueue_.DeQue<T1>();
        LocalTensor<T2> gamma = gammaQueue_.DeQue<T2>();
        LocalTensor<T2> runningVar = runningVarQueue_.DeQue<T2>();
        LocalTensor<T1> dx = dxQueue_.AllocTensor<T1>();

        __local_mem__ T1* dyLocal = (__local_mem__ T1*)dy.GetPhyAddr();
        __local_mem__ T2* gammaLocal = (__local_mem__ T2*)gamma.GetPhyAddr();
        __local_mem__ T2* varLocal = (__local_mem__ T2*)runningVar.GetPhyAddr();
        __local_mem__ T1* dxLocal = (__local_mem__ T1*)dx.GetPhyAddr();

        VFNormalize(dyLocal, gammaLocal, varLocal, dxLocal, curTileBLen, curTileALen);

        dxQueue_.EnQue(dx);

        dyQueue_.FreeTensor<T1>(dy);
        gammaQueue_.FreeTensor<T2>(gamma);
        runningVarQueue_.FreeTensor<T2>(runningVar);
    }

    __aicore__ inline void VFNormalize(
        __local_mem__ T1* dyLocal, __local_mem__ T2* gammaLocal, __local_mem__ T2* varLocal, __local_mem__ T1* dxLocal,
        uint16_t curTileBLen, uint16_t curTileALen)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> dy;
            RegTensor<float> gamma;
            RegTensor<float> runningVar;
            RegTensor<float> dx;
            RegTensor<float> invstd;

            MaskReg pregMaskFp32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            uint32_t tileBlockALenTmp = static_cast<uint32_t>(tilingData_->tileBlockALen);
            float epsilonTmp = tilingData_->epsilon;
            uint16_t loopNum = ops::CeilDiv(curTileALen, VL_FP32);
            for (uint16_t i = 0; i < loopNum; i++) {
                uint32_t offset = i * VL_FP32;

                // load runningVar, gamma
                LoadOneTensor<T2>(varLocal, runningVar, pregMaskFp32, offset);
                CalRstdByHighPrecision(runningVar, invstd, epsilonTmp);

                LoadOneTensor<T2>(gammaLocal, gamma, pregMaskFp32, offset);

                for (uint16_t j = 0; j < curTileBLen; j++) {
                    uint32_t dyOffset = j * tileBlockALenTmp + offset;
                    // load dy
                    LoadOneTensor<T1>(dyLocal, dy, pregMaskFp32, dyOffset);

                    // compute
                    Mul(dx, dy, invstd, pregMaskFp32);
                    Mul(dx, dx, gamma, pregMaskFp32);

                    // copy out
                    StoreOneTensor<T1>(dxLocal, dx, pregMaskFp32, dyOffset);
                }
            }
        }
    }

    __aicore__ inline void CopyOutDx(int64_t dxGmOffset, int64_t curTileBLen, int64_t curTileALen, int64_t ubStrideT1)
    {
        LocalTensor<T1> dx = dxQueue_.DeQue<T1>();

        DataCopyExtParams extParams;
        extParams.blockLen = curTileALen * sizeof(T1);
        extParams.srcStride = ubStrideT1;
        extParams.dstStride = (tilingData_->aDim - curTileALen) * sizeof(T1);
        extParams.blockCount = curTileBLen;

        DataCopyPad(dxGm_[dxGmOffset], dx, extParams);

        dxQueue_.FreeTensor(dx);
    }

private:
    const BatchNormGradInferChannelLastDxTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> gammaQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> runningVarQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dxQueue_;

    GlobalTensor<T1> dyGm_;
    GlobalTensor<T2> gammaGm_;
    GlobalTensor<T2> runningVarGm_;

    GlobalTensor<T1> dxGm_;
};

template <typename T1, typename T2, int BUFFER_NUM = 1, bool EN_DGAMMA = true, bool EN_DBETA = true>
class BatchNormGradInferChannelLastRecompute
{
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradInferChannelLastRecompute(){};

    __aicore__ inline BatchNormGradInferChannelLastRecompute(
        TPipe* pipeIn, const BatchNormGradInferChannelLastTilingData* tilingDataIn)
    {
        pipe_ = pipeIn;
        epsilon = tilingDataIn->dxTilingData.epsilon;
        tilingData_ = tilingDataIn;
        enableDx = tilingData_->enableDx;
        enableDgamma = tilingData_->enableDgamma;
        enableDbeta = tilingData_->enableDbeta;
    }

    __aicore__ inline void Process(
        GM_ADDR dy, GM_ADDR x, GM_ADDR save_mean, GM_ADDR runningVar, GM_ADDR dgamma, GM_ADDR dbeta,
        GM_ADDR usrWorkspace)
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= tilingData_->usedCoreNumsStg1) {
            return;
        }
        InitStage1(dy, x, save_mean, runningVar, dgamma, dbeta);
        int64_t beginIdx = blockIdx * tilingData_->aOuterPerCoreStg1;
        int64_t endIdx = beginIdx + tilingData_->aOuterPerCoreStg1;
        endIdx = endIdx > tilingData_->aOuterStg1 ? tilingData_->aOuterStg1 : endIdx;

        // [R, A]
        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curALen =
                curIdx == (tilingData_->aOuterStg1 - 1) ? tilingData_->aTailStg1 : tilingData_->aInnerStg1;

            int64_t meanOffset = curIdx * tilingData_->aInnerStg1;
            uint16_t vlLoopNum = ops::CeilDiv(curALen, static_cast<int64_t>(VL_FP32));

            ComputeDbetaAndDgamma(curALen, meanOffset, vlLoopNum);
        }
    }

    __aicore__ inline void InitStage1(
        __gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean, __gm__ uint8_t* runningVar, __gm__ uint8_t* dgamma,
        __gm__ uint8_t* dbeta)
    {
        dyGm_.SetGlobalBuffer((__gm__ T1*)(dy));
        xGm_.SetGlobalBuffer((__gm__ T1*)(x));
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean));
        varGm_.SetGlobalBuffer((__gm__ float*)(runningVar));
        dgammaGm_.SetGlobalBuffer((__gm__ float*)(dgamma));
        dbetaGm_.SetGlobalBuffer((__gm__ float*)(dbeta));

        int64_t dyLen = tilingData_->binAddRFactorStg1 * tilingData_->aInnerStg1;
        pipe_->InitBuffer(dyQueue_, BUFFER_NUM, dyLen * sizeof(T1));
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, dyLen * sizeof(T1));

        if constexpr (!IsSameType<T1, float>::value) {
            pipe_->InitBuffer(cacheBufferDyAndX_, dyLen * 2 * sizeof(float));
            cacheBufferXOffset_ = dyLen;
        }

        int64_t meanLen = tilingData_->aInnerStg1 * sizeof(float);
        pipe_->InitBuffer(meanQueue_, BUFFER_NUM, meanLen);
        pipe_->InitBuffer(varQueue_, BUFFER_NUM, meanLen);
        pipe_->InitBuffer(dbetaQueue_, BUFFER_NUM, meanLen);
        pipe_->InitBuffer(dgammaQueue_, BUFFER_NUM, meanLen);
        pipe_->InitBuffer(tempBufferDbeta_, meanLen);
        pipe_->InitBuffer(tempBufferDgamma_, meanLen);

        int64_t cacheBufSize = tilingData_->binAddCacheBufferCountStg1 * tilingData_->aInnerStg1 * sizeof(float);
        pipe_->InitBuffer(cacheBufferDbeta_, cacheBufSize);
        pipe_->InitBuffer(cacheBufferDgamma_, cacheBufSize);
    }

    __aicore__ inline void ComputeDbetaAndDgamma(uint32_t curALen, int64_t meanOffset, uint16_t vlLoopNum)
    {
        if constexpr (EN_DBETA) {
            tempTensorDbeta_ = tempBufferDbeta_.Get<float>();
            cacheTensorDbeta_ = cacheBufferDbeta_.Get<float>();
        }
        if constexpr (EN_DGAMMA) {
            CopyInMeanAndVar(meanOffset, curALen);
            tempTensorDgamma_ = tempBufferDgamma_.Get<float>();
            cacheTensorDgamma_ = cacheBufferDgamma_.Get<float>();
            mean_ = meanQueue_.DeQue<float>();
            var_ = varQueue_.DeQue<float>();
            ComputeRstd(curALen, meanOffset, vlLoopNum);
        }

        if constexpr (!IsSameType<T1, float>::value) {
            tempTensorDy_ = cacheBufferDyAndX_.Get<float>();
            tempTensorX_ = tempTensorDy_[cacheBufferXOffset_];
        }

        int64_t blockOffset = tilingData_->aDimStg1 * tilingData_->binAddRFactorStg1;
        for (int64_t basicBlockIdx = 0; basicBlockIdx < tilingData_->binAddBasicBlockLoopStg1; basicBlockIdx++) {
            ProcessMainBlock(
                tilingData_->binAddRFactorStg1, curALen, meanOffset + basicBlockIdx * blockOffset, vlLoopNum);
            if (basicBlockIdx < tilingData_->binAddMainFoldCountStg1) {
                ProcessFoldBlock(
                    tilingData_->binAddRFactorStg1, curALen,
                    meanOffset + (basicBlockIdx + tilingData_->binAddBasicBlockLoopStg1) * blockOffset, vlLoopNum);
            } else if (
                basicBlockIdx == tilingData_->binAddMainFoldCountStg1 && tilingData_->binAddRTailStg1 > 0 &&
                tilingData_->binAddRTailStg1 != tilingData_->binAddRFactorStg1) {
                ProcessFoldBlock(
                    tilingData_->binAddRTailStg1, curALen,
                    meanOffset + (basicBlockIdx + tilingData_->binAddBasicBlockLoopStg1) * blockOffset, vlLoopNum);
            }
            ProcessSummation(basicBlockIdx, tilingData_->binAddRFactorStg1);
        }

        if (tilingData_->binAddBasicBlockLoopStg1 == 0) {
            ProcessMainBlock(tilingData_->binAddRTailStg1, curALen, meanOffset, vlLoopNum);
            ProcessSummation(0, tilingData_->binAddRTailStg1);
        }
        if constexpr (EN_DGAMMA) {
            meanQueue_.FreeTensor(mean_);
            varQueue_.FreeTensor(var_);
        }
        if constexpr (EN_DBETA) {
            LocalTensor<T2> dbeta = dbetaQueue_.AllocTensor<T2>();
            CopyUB2UB(
                dbeta, cacheTensorDbeta_[tilingData_->binAddResultCacheIDStg1 * tilingData_->aInnerStg1],
                tilingData_->aInnerStg1);
            dbetaQueue_.EnQue(dbeta);
        }
        if constexpr (EN_DGAMMA) {
            LocalTensor<T2> dgamma = dgammaQueue_.AllocTensor<T2>();
            CopyUB2UB(
                dgamma, cacheTensorDgamma_[tilingData_->binAddResultCacheIDStg1 * tilingData_->aInnerStg1],
                tilingData_->aInnerStg1);
            dgammaQueue_.EnQue(dgamma);
        }
        CopyOutDbetaAndDgamma(meanOffset, curALen);
    }

    __aicore__ inline void ComputeRstd(const int64_t curALen, const int64_t gmOffset, uint16_t vlLoopNum)
    {
        __local_mem__ float* var = (__local_mem__ float*)var_.GetPhyAddr();
        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = curALen;
            uint16_t loopNum = vlLoopNum;

            RegTensor<float> varReg;
            RegTensor<float> rstdReg;

            for (uint16_t j = 0; j < loopNum; ++j) {
                pregMask = UpdateMask<float>(sreg);
                uint32_t varOffset = j * VL_FP32;
                LoadOneTensor<float>(var, varReg, pregMask, varOffset);

                CalRstdByHighPrecision(varReg, rstdReg, epsilon);

                StoreOneTensor<float>(var, rstdReg, pregMask, varOffset);
            }
        }
    }

    // copy to main
    __aicore__ inline void ProcessMainBlock(
        const int64_t curRLen, const int64_t curALen, const int64_t gmOffset, uint16_t vlLoopNum)
    {
        CopyInDyAndX(gmOffset, curRLen, curALen);

        dyMain_ = dyQueue_.DeQue<T1>();
        if constexpr (EN_DGAMMA) {
            xMain_ = xQueue_.DeQue<T1>();
        }

        __local_mem__ T1* dy = (__local_mem__ T1*)dyMain_.GetPhyAddr();
        __local_mem__ T1* x = (__local_mem__ T1*)xMain_.GetPhyAddr();

        __local_mem__ float* dyDst = nullptr;
        __local_mem__ float* xDst = nullptr;

        if constexpr (IsSameType<T1, float>::value) {
            dyDst = (__local_mem__ float*)dyMain_.GetPhyAddr();
            xDst = (__local_mem__ float*)xMain_.GetPhyAddr();
        } else {
            dyDst = (__local_mem__ float*)tempTensorDy_.GetPhyAddr();
            xDst = (__local_mem__ float*)tempTensorX_.GetPhyAddr();
        }

        __local_mem__ float* mean = (__local_mem__ float*)mean_.GetPhyAddr();
        __local_mem__ float* rstd = (__local_mem__ float*)var_.GetPhyAddr();

        uint16_t rLoopTimes = static_cast<uint16_t>(curRLen);
        uint32_t rLoopStride = tilingData_->aInnerStg1;

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = curALen;
            uint16_t loopNum = vlLoopNum;

            RegTensor<float> dyReg;
            RegTensor<float> xReg;
            RegTensor<float> meanReg;
            RegTensor<float> rstdReg;

            for (uint16_t j = 0; j < loopNum; ++j) {
                pregMask = UpdateMask<float>(sreg);
                uint32_t meanOffset = j * VL_FP32;
                if constexpr (EN_DGAMMA) {
                    LoadOneTensor<float>(mean, meanReg, pregMask, meanOffset);
                    LoadOneTensor<float>(rstd, rstdReg, pregMask, meanOffset);
                }

                for (uint16_t i = 0; i < rLoopTimes; ++i) {
                    uint32_t dyOffset = i * rLoopStride + j * VL_FP32;
                    LoadOneTensor<T1>(dy, dyReg, pregMask, dyOffset);
                    if constexpr (EN_DBETA && !IsSameType<T1, float>::value) {
                        StoreOneTensor<float>(dyDst, dyReg, pregMask, dyOffset);
                    }
                    if constexpr (EN_DGAMMA) {
                        LoadOneTensor<T1>(x, xReg, pregMask, dyOffset);
                        Sub(xReg, xReg, meanReg, pregMask);
                        Mul(xReg, xReg, dyReg, pregMask);
                        Mul(xReg, xReg, rstdReg, pregMask);
                        StoreOneTensor<float>(xDst, xReg, pregMask, dyOffset);
                    }
                }
            }
        }

        if constexpr (!IsSameType<T1, float>::value && EN_DGAMMA) {
            xQueue_.FreeTensor(xMain_);
        }

        if constexpr (!IsSameType<T1, float>::value || !EN_DBETA) {
            dyQueue_.FreeTensor(dyMain_);
        }
    }

    // fold cast fp32 加到main
    __aicore__ inline void ProcessFoldBlock(
        const int64_t curRLen, const int64_t curALen, const int64_t gmOffset, uint16_t vlLoopNum)
    {
        CopyInDyAndX(gmOffset, curRLen, curALen);

        LocalTensor<T1> dyFold = dyQueue_.DeQue<T1>();
        LocalTensor<T1> xFold;
        if constexpr (EN_DGAMMA) {
            xFold = xQueue_.DeQue<T1>();
        }

        __local_mem__ float* dyDst = nullptr;
        __local_mem__ float* xDst = nullptr;
        if constexpr (IsSameType<T1, float>::value) {
            dyDst = (__local_mem__ float*)dyMain_.GetPhyAddr();
            xDst = (__local_mem__ float*)xMain_.GetPhyAddr();
        } else {
            dyDst = (__local_mem__ float*)tempTensorDy_.GetPhyAddr();
            xDst = (__local_mem__ float*)tempTensorX_.GetPhyAddr();
        }

        __local_mem__ T1* dyFoldLocal = (__local_mem__ T1*)dyFold.GetPhyAddr();
        __local_mem__ T1* xFoldLocal = (__local_mem__ T1*)xFold.GetPhyAddr();

        __local_mem__ float* mean = (__local_mem__ float*)mean_.GetPhyAddr();
        __local_mem__ float* rstd = (__local_mem__ float*)var_.GetPhyAddr();

        uint16_t rLoopTimes = static_cast<uint16_t>(curRLen);
        uint32_t rLoopStride = tilingData_->aInnerStg1;

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = curALen;
            uint16_t loopNum = vlLoopNum;

            RegTensor<float> dyRegMain;
            RegTensor<float> xRegMain;
            RegTensor<float> dyRegFold;
            RegTensor<float> xRegFold;
            RegTensor<float> meanReg;
            RegTensor<float> rstdReg;

            for (uint16_t j = 0; j < loopNum; ++j) {
                pregMask = UpdateMask<float>(sreg);
                uint32_t meanOffset = j * VL_FP32;
                if constexpr (EN_DGAMMA) {
                    LoadOneTensor<float>(mean, meanReg, pregMask, meanOffset);
                    LoadOneTensor<float>(rstd, rstdReg, pregMask, meanOffset);
                }

                for (uint16_t i = 0; i < rLoopTimes; ++i) {
                    uint32_t dyOffset = i * rLoopStride + j * VL_FP32;

                    LoadOneTensor<T1>(dyFoldLocal, dyRegFold, pregMask, dyOffset);
                    if constexpr (EN_DBETA) {
                        LoadOneTensor<float>(dyDst, dyRegMain, pregMask, dyOffset);
                        Add(dyRegMain, dyRegMain, dyRegFold, pregMask);
                        StoreOneTensor<float>(dyDst, dyRegMain, pregMask, dyOffset);
                    }
                    if constexpr (EN_DGAMMA) {
                        LoadOneTensor<float>(xDst, xRegMain, pregMask, dyOffset);
                        LoadOneTensor<T1>(xFoldLocal, xRegFold, pregMask, dyOffset);
                        Sub(xRegFold, xRegFold, meanReg, pregMask);
                        Mul(xRegFold, xRegFold, dyRegFold, pregMask);
                        Mul(xRegFold, xRegFold, rstdReg, pregMask);
                        Add(xRegMain, xRegMain, xRegFold, pregMask);
                        StoreOneTensor<float>(xDst, xRegMain, pregMask, dyOffset);
                    }
                }
            }
        }

        if constexpr (EN_DGAMMA) {
            xQueue_.FreeTensor(xFold);
        }
        dyQueue_.FreeTensor(dyFold);
    }

    __aicore__ inline void UpdateCache(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const int64_t cacheId,
        const int64_t stride, const int64_t count)
    {
        uint16_t outerLoopTimes =
            ops::CeilDiv(static_cast<int64_t>(count * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
        uint16_t innerLoopTimes = cacheId;
        uint32_t outerLoopStride = VL_FP32;
        uint32_t innerLoopStride = stride;

        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* cache = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheId * stride;
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t sreg = static_cast<uint32_t>(count);
            MicroAPI::RegTensor<float> aReg, bReg;
            MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(sreg);
                MicroAPI::DataCopy(aReg, (__local_mem__ float*)src + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    MicroAPI::DataCopy(bReg, (__local_mem__ float*)dst + i * outerLoopStride + j * innerLoopStride);
                    MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                }
                MicroAPI::DataCopy((__local_mem__ float*)cache + i * outerLoopStride, aReg, pMask);
            }
        }
    }

    __aicore__ inline void ProcessSummation(const int64_t basicBlockIdx, int64_t binAddRLen)
    {
        int64_t cacheID = GetCacheID(basicBlockIdx);
        uint32_t srcShape[2] = {uint32_t(binAddRLen), uint32_t(tilingData_->aInnerStg1)};

        if constexpr (EN_DBETA) {
            if constexpr (IsSameType<T1, float>::value) {
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                    tempTensorDbeta_, dyMain_, srcShape, false);
                dyQueue_.FreeTensor(dyMain_);
            } else {
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                    tempTensorDbeta_, tempTensorDy_, srcShape, false);
            }

            UpdateCache(cacheTensorDbeta_, tempTensorDbeta_, cacheID, tilingData_->aInnerStg1, tilingData_->aInnerStg1);
        }

        if constexpr (EN_DGAMMA) {
            if constexpr (IsSameType<T1, float>::value) {
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                    tempTensorDgamma_, xMain_, srcShape, false);
                xQueue_.FreeTensor(xMain_);
            } else {
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                    tempTensorDgamma_, tempTensorX_, srcShape, false);
            }

            UpdateCache(
                cacheTensorDgamma_, tempTensorDgamma_, cacheID, tilingData_->aInnerStg1, tilingData_->aInnerStg1);
        }
    }

    __aicore__ inline void CopyInDyAndX(int64_t gmOffset, int64_t curRLen, uint32_t curALen)
    {
        DataCopyExtParams params;
        params.blockCount = curRLen;
        params.blockLen = curALen * sizeof(T1);
        params.srcStride = (tilingData_->aDimStg1 - curALen) * sizeof(T1);
        params.dstStride = (tilingData_->aInnerStg1 * sizeof(T1) - params.blockLen) / BLOCK_SIZE;
        DataCopyPadExtParams<T1> padParams;
        padParams.isPad = false;

        LocalTensor<T1> dy = dyQueue_.AllocTensor<T1>();
        DataCopyPad(dy, dyGm_[gmOffset], params, padParams);
        dyQueue_.EnQue(dy);

        if constexpr (EN_DGAMMA) {
            LocalTensor<T1> x = xQueue_.AllocTensor<T1>();
            DataCopyPad(x, xGm_[gmOffset], params, padParams);
            xQueue_.EnQue(x);
        }
    }

    __aicore__ inline void CopyInMeanAndVar(int64_t gmOffset, uint32_t curALen)
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = curALen * sizeof(T2);
        params.srcStride = (tilingData_->aDimStg1 - curALen) * sizeof(T2);
        params.dstStride = (tilingData_->aInnerStg1 * sizeof(T2) - params.blockLen) / BLOCK_SIZE;
        DataCopyPadExtParams<T2> padParams;
        padParams.isPad = false;

        LocalTensor<T2> mean = meanQueue_.AllocTensor<T2>();
        DataCopyPad(mean, meanGm_[gmOffset], params, padParams);
        meanQueue_.EnQue(mean);

        LocalTensor<T2> var = varQueue_.AllocTensor<T2>();
        DataCopyPad(var, varGm_[gmOffset], params, padParams);
        varQueue_.EnQue(var);
    }

    __aicore__ inline void CopyUB2UB(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const int64_t count)
    {
        DataCopy(
            dstTensor, srcTensor,
            ops::Aligned(
                static_cast<int64_t>(count), static_cast<int64_t>(platform::GetUbBlockSize() / sizeof(float))));
    }

    __aicore__ inline void CopyOutDbetaAndDgamma(int64_t gmOffset, uint32_t curALen)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curALen * sizeof(T2);
        copyOutParams.srcStride = (tilingData_->aInnerStg1 - curALen) * sizeof(T2) / BLOCK_SIZE;
        copyOutParams.dstStride = 0;
        if constexpr (EN_DBETA) {
            LocalTensor<T2> dbeta = dbetaQueue_.DeQue<T2>();
            DataCopyPad<T2, PaddingMode::Normal>(dbetaGm_[gmOffset], dbeta, copyOutParams);
            dbetaQueue_.FreeTensor(dbeta);
        }
        if constexpr (EN_DGAMMA) {
            LocalTensor<T2> dgamma = dgammaQueue_.DeQue<T2>();
            DataCopyPad<T2, PaddingMode::Normal>(dgammaGm_[gmOffset], dgamma, copyOutParams);
            dgammaQueue_.FreeTensor(dgamma);
        }
    }

private:
    const BatchNormGradInferChannelLastTilingData* tilingData_;
    TPipe* pipe_;

    float epsilon;
    bool enableDx;
    bool enableDgamma;
    bool enableDbeta;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> meanQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> varQueue_;

    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dbetaQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dgammaQueue_;

    // b16
    TBuf<TPosition::VECCALC> cacheBufferDyAndX_;
    LocalTensor<float> tempTensorDy_;
    LocalTensor<float> tempTensorX_;
    int64_t cacheBufferXOffset_{0};

    // bin add
    TBuf<TPosition::VECCALC> cacheBufferDbeta_;
    TBuf<TPosition::VECCALC> tempBufferDbeta_;

    TBuf<TPosition::VECCALC> cacheBufferDgamma_;
    TBuf<TPosition::VECCALC> tempBufferDgamma_;

    LocalTensor<float> tempTensorDbeta_;
    LocalTensor<float> cacheTensorDbeta_;

    LocalTensor<float> tempTensorDgamma_;
    LocalTensor<float> cacheTensorDgamma_;

    LocalTensor<T1> xMain_;
    LocalTensor<T1> dyMain_;

    LocalTensor<float> mean_;
    LocalTensor<float> var_;

    GlobalTensor<T1> dyGm_;
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> meanGm_;
    GlobalTensor<T2> varGm_;
    GlobalTensor<T2> dbetaGm_;
    GlobalTensor<T2> dgammaGm_;
};

template <typename T1, typename T2>
class BatchNormGradInferChannelLast
{
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradInferChannelLast(){};

    __aicore__ inline BatchNormGradInferChannelLast(
        TPipe* pipeIn, const BatchNormGradInferChannelLastTilingData* tilingDataIn)
    {
        pipe_ = pipeIn;
        tilingData_ = tilingDataIn;
        enableDx = tilingData_->enableDx;
        enableDgamma = tilingData_->enableDgamma;
        enableDbeta = tilingData_->enableDbeta;
    }

    __aicore__ inline void Process(
        GM_ADDR dy, GM_ADDR x, GM_ADDR gamma, GM_ADDR save_mean, GM_ADDR runningVar, GM_ADDR dx, GM_ADDR dgamma,
        GM_ADDR dbeta, GM_ADDR usrWorkspace)
    {
        if (enableDx) {
            BatchNormGradInferChannelLastDx<T1, T2> opDx(&tilingData_->dxTilingData);
            opDx.InitStg0(dy, gamma, runningVar, dx, pipe_);
            opDx.ProcessStg0();
            PipeBarrier<PIPE_ALL>();
            pipe_->Reset();
        }
        if (enableDgamma && enableDbeta) {
            BatchNormGradInferChannelLastRecompute<T1, T2, BUFFER_NUM, true, true> opDgammaDbeta(pipe_, tilingData_);
            opDgammaDbeta.Process(dy, x, save_mean, runningVar, dgamma, dbeta, usrWorkspace);
        } else if (enableDgamma && !enableDbeta) {
            BatchNormGradInferChannelLastRecompute<T1, T2, BUFFER_NUM, true, false> opDgammaDbeta(pipe_, tilingData_);
            opDgammaDbeta.Process(dy, x, save_mean, runningVar, dgamma, dbeta, usrWorkspace);
        } else if (!enableDgamma && enableDbeta) {
            BatchNormGradInferChannelLastRecompute<T1, T2, BUFFER_NUM, false, true> opDgammaDbeta(pipe_, tilingData_);
            opDgammaDbeta.Process(dy, x, save_mean, runningVar, dgamma, dbeta, usrWorkspace);
        }
    }

private:
    const BatchNormGradInferChannelLastTilingData* tilingData_;
    TPipe* pipe_;

    bool enableDx;
    bool enableDgamma;
    bool enableDbeta;
};

} // namespace BatchNormGrad

#endif
