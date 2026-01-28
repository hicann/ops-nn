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
 * \file batch_norm_v3_infer.h
 * \brief
 */

#ifndef BATCH_NORM_V3_INFER_H
#define BATCH_NORM_V3_INFER_H

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
class BatchNormV3Infer {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = GetVRegSize();
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = GetUbBlockSize();

    constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};

    constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

public:
    __aicore__ inline BatchNormV3Infer(){};

    __aicore__ inline BatchNormV3Infer(const BatchNormV3InferTilingData* tilingDataIn)
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

        int64_t xShapeLen = tilingData_->tileBlockB1Len * tilingData_->tileBlockALen * tilingData_->tileBlockB0Len;
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

        // pattern is [B0, A, B1]
        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curB1Idx = curIdx % tilingData_->b1Outer;
            int64_t curAIdx = (curIdx / tilingData_->b1Outer) / tilingData_->b0Outer;
            int64_t curB0Idx = (curIdx / tilingData_->b1Outer) % tilingData_->b0Outer;

            // ping、pang搬运首次或者tile块沿B轴换列时需要拷贝mean、var、beta、gamma
            // curIdx 整数倍(tilingData_->b0Outer * tilingData_->b1Outer) 表示一轮循环, 出现换行
            bool needCopy = (curIdx <= beginIdx + 1) || (curIdx % (tilingData_->b0Outer * tilingData_->b1Outer) <= 1);

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
            CopyInBetaGammaMeanVar(needCopy, betaOffset, curTileALen);
            Compute(curTileB0Len, curTileALen, curTileB1Len);
            CopyOutY(xOffset, curTileB0Len, curTileALen, curTileB1Len, ubStrideT);
        }
    }

private:
    __aicore__ inline void CopyInX(
        int64_t xGmOffset, int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len, int64_t ubStrideT)
    {
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curTileB0Len;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->totalALen * tilingData_->totalB1Len * sizeof(T);
        loopParams.loop1DstStride = tilingData_->tileBlockB1Len * tilingData_->tileBlockALen * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileALen;
        copyInParams.blockLen = curTileB1Len * sizeof(T);
        copyInParams.srcStride = (tilingData_->totalB1Len - curTileB1Len) * sizeof(T);
        copyInParams.dstStride = (tilingData_->tileBlockB1Len - curTileB1Len) * sizeof(T) / BLOCK_SIZE;
        DataCopyPad<T, PaddingMode::Normal>(xLocal, xGm_[xGmOffset], copyInParams, dataCopyPadExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
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

    __aicore__ inline void Compute(int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len)
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

        VFNormalize(
            xLocal, gammaLocal, betaLocal, meanLocal, varLocal, yLocal, curTileB0Len, curTileALen, curTileB1Len);

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
        uint16_t curTileB0Len, uint16_t curTileALen, uint16_t curTileB1Len)
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
                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把T_RUNNING_MEAN的输入cast到float
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> runningVarTmp;
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(
                        runningVarTmp, ((__local_mem__ T_RUNNING_MEAN*)(varLocal) + i));
                    AscendC::MicroAPI::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(var, runningVarTmp, pregMask);
                } else {
                    AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(
                        var, ((__local_mem__ float*)(varLocal) + i));
                }
                CalRstdByHighPrecision(var, rstd, tilingData_->epsilon);
                // load mean
                DataCopy<float, LoadDist::DIST_BRC_B32>(mean, ((__local_mem__ float*)(meanLocal) + i));
                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把T_RUNNING_MEAN的输入cast到float
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> runningMeanTmp;
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(
                        runningMeanTmp, ((__local_mem__ T_RUNNING_MEAN*)(meanLocal) + i));
                    AscendC::MicroAPI::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(mean, runningMeanTmp, pregMask);
                } else {
                    AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(
                        mean, ((__local_mem__ float*)(meanLocal) + i));
                }
                // load gamma、beta  1->64
                LoadsTensorForDtypeT(gammaLocal, gamma, pregMask, i);
                LoadsTensorForDtypeT(betaLocal, beta, pregMask, i);

                uint32_t tileBlockALenTmp = static_cast<uint32_t>(tilingData_->tileBlockALen);
                uint32_t tileBlockB1LenTmp = static_cast<uint32_t>(tilingData_->tileBlockB1Len);
                for (uint16_t j = 0; j < curTileB0Len; j++) {
                    for (uint16_t k = 0; k < loopNum; k++) {
                        uint32_t xOffset = (j * tileBlockALenTmp + i) * tileBlockB1LenTmp + k * VL_FP32;

                        // load x
                        LoadTensorForDtypeT(xLocal, x, pregMask, xOffset);

                        // compute
                        Sub(x, x, mean, pregMask);
                        Mul(x, x, gamma, pregMask);
                        Mul(x, x, rstd, pregMask);
                        Add(y, x, beta, pregMask);

                        // copy out
                        if constexpr (IsSameType<T, float>::value) {
                            DataCopy(((__local_mem__ float*)yLocal) + xOffset, y, pregMask);
                        } else { // fp16、bf16
                            RegTensor<T> xFp16;
                            Cast<T, float, castTraitB322B16>(xFp16, y, pregMask);
                            DataCopy<T, StoreDist::DIST_PACK_B32>(
                                ((__local_mem__ T*)yLocal) + xOffset, xFp16, pregMask);
                        }
                    }
                }
            }
        }
    }

    template <typename T_SRC>
    __aicore__ inline void LoadsTensorForDtypeT(
        __local_mem__ T_SRC* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_SRC, float>::value) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(dst, (__local_mem__ float*)src + offset);
        } else { // fp16、bf16
            RegTensor<T> xFp16;
            DataCopy<T, LoadDist::DIST_BRC_B16>(xFp16, ((__local_mem__ T*)src + offset));
            Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
        }
    }

    __aicore__ inline void LoadTensorForDtypeT(
        __local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
        } else { // fp16、bf16
            RegTensor<T> xFp16;
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
            Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
        }
    }

    __aicore__ inline void CopyOutY(
        int64_t yGmOffset, int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len, int64_t ubStrideT)
    {
        LocalTensor<T> y = yQueue_.DeQue<T>();
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curTileB0Len;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->tileBlockB1Len * tilingData_->tileBlockALen * sizeof(T);
        loopParams.loop1DstStride = tilingData_->totalALen * tilingData_->totalB1Len * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileALen;
        copyInParams.blockLen = curTileB1Len * sizeof(T);
        copyInParams.srcStride = (tilingData_->tileBlockB1Len - curTileB1Len) * sizeof(T) / BLOCK_SIZE;
        copyInParams.dstStride = (tilingData_->totalB1Len - curTileB1Len) * sizeof(T);
        DataCopyPad<T, PaddingMode::Normal>(yGm_[yGmOffset], y, copyInParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        yQueue_.FreeTensor(y);
    }

private:
    const BatchNormV3InferTilingData* tilingData_;

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
