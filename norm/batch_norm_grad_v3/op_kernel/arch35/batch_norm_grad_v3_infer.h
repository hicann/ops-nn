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
 * \file batch_norm_grad_v3_infer.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_V3_INFER_H
#define BATCH_NORM_GRAD_V3_INFER_H

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "batch_norm_grad_v3_common.h"

namespace BatchNormGradV3 {
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

template <typename T1, typename T2, typename T3>
class BatchNormGradV3Infer {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradV3Infer(){};

    __aicore__ inline BatchNormGradV3Infer(const BatchNormGradV3InferTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR gamma, GM_ADDR runningVar, GM_ADDR dx, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        dyGm_.SetGlobalBuffer((__gm__ T1*)dy);
        gammaGm_.SetGlobalBuffer((__gm__ T2*)gamma);
        runningVarGm_.SetGlobalBuffer((__gm__ T3*)runningVar);
        dxGm_.SetGlobalBuffer((__gm__ T1*)dx);

        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T2));
        pipe_->InitBuffer(runningVarQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T3));

        int64_t xShapeLen = tilingData_->tileBlockB1Len * tilingData_->tileBlockALen * tilingData_->tileBlockB0Len;
        pipe_->InitBuffer(dxQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
        pipe_->InitBuffer(dyQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        int64_t paddingANumSizeT1 = tilingData_->tileBlockAPaddingNum * sizeof(T1) / BLOCK_SIZE;

        // pattern is [B0, A, B1]
        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curB1Idx = curIdx % tilingData_->b1Outer;
            int64_t curAIdx = (curIdx / tilingData_->b1Outer) / tilingData_->b0Outer;
            int64_t curB0Idx = (curIdx / tilingData_->b1Outer) % tilingData_->b0Outer;

            // ping、pang搬运首次或者tile块沿B轴换列时需要拷贝gamma、invstd
            // curIdx 整数倍(tilingData_->b0Outer * tilingData_->b1Outer) 表示一轮循环, 出现换行
            bool needCopy = (curIdx <= beginIdx + 1) || (curIdx % (tilingData_->b0Outer * tilingData_->b1Outer) <= 1);

            // Tile整尾块
            int64_t curTileB0Len =
                curB0Idx == (tilingData_->b0Outer - 1) ? tilingData_->tileBlockB0Tail : tilingData_->tileBlockB0Len;
            int64_t curTileALen =
                curAIdx == (tilingData_->aOuter - 1) ? tilingData_->tileBlockATail : tilingData_->tileBlockALen;
            int64_t curTileB1Len =
                curB1Idx == (tilingData_->b1Outer - 1) ? tilingData_->tileBlockB1Tail : tilingData_->tileBlockB1Len;

            int64_t ubStrideT1 = curAIdx == (tilingData_->aOuter - 1) ? paddingANumSizeT1 : 0;

            // x、y偏移一致，gamma、runningVar偏移一致
            int64_t weightOffset = curAIdx * tilingData_->tileBlockALen;
            int64_t dyOffset =
                // b0 offset
                curB0Idx * tilingData_->tileBlockB0Len * tilingData_->totalALen * tilingData_->totalB1Len +
                // a offset
                curAIdx * tilingData_->tileBlockALen * tilingData_->totalB1Len +
                // b1 offset
                curB1Idx * tilingData_->tileBlockB1Len;

            CopyInDy(dyOffset, curTileB0Len, curTileALen, curTileB1Len, ubStrideT1);
            CopyInGammaVar(needCopy, weightOffset, curTileALen);
            Compute(curTileB0Len, curTileALen, curTileB1Len);
            CopyOutDx(dyOffset, curTileB0Len, curTileALen, curTileB1Len, ubStrideT1);
        }
    }

private:
    __aicore__ inline void CopyInDy(
        int64_t dyGmOffset, int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len, int64_t ubStrideT1)
    {
        LocalTensor<T1> dyLocal = dyQueue_.AllocTensor<T1>();
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
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileALen;
        copyInParams.blockLen = curTileB1Len * sizeof(T1);
        copyInParams.srcStride = (tilingData_->totalB1Len - curTileB1Len) * sizeof(T1);
        copyInParams.dstStride = (tilingData_->tileBlockB1Len - curTileB1Len) * sizeof(T1) / BLOCK_SIZE;
        DataCopyPad<T1, PaddingMode::Normal>(dyLocal, dyGm_[dyGmOffset], copyInParams, dataCopyPadExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        dyQueue_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInGammaVar(bool needCopy, int64_t offset, int64_t curTileALen)
    {
        LocalTensor<T2> gammaLocal = gammaQueue_.AllocTensor<T2>();
        LocalTensor<T3> varLocal = runningVarQueue_.AllocTensor<T3>();

        if (needCopy) {
            DataCopyExtParams extParam;
            extParam.blockCount = 1;

            // gamma
            extParam.blockLen = curTileALen * sizeof(T2);
            DataCopyPadExtParams<T2> padExtParam;
            padExtParam.isPad = false;
            DataCopyPad(gammaLocal, gammaGm_[offset], extParam, padExtParam);

            // runningVar
            extParam.blockLen = curTileALen * sizeof(T3);
            DataCopyPadExtParams<T3> padExtParams1;
            padExtParams1.isPad = false;
            DataCopyPad(varLocal, runningVarGm_[offset], extParam, padExtParams1);
        }

        gammaQueue_.EnQue(gammaLocal);
        runningVarQueue_.EnQue(varLocal);
    }

    __aicore__ inline void Compute(int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len)
    {
        LocalTensor<T1> dy = dyQueue_.DeQue<T1>();
        LocalTensor<T2> gamma = gammaQueue_.DeQue<T2>();
        LocalTensor<T3> runningVar = runningVarQueue_.DeQue<T3>();
        LocalTensor<T1> dx = dxQueue_.AllocTensor<T1>();

        __local_mem__ T1* dyLocal = (__local_mem__ T1*)dy.GetPhyAddr();
        __local_mem__ T2* gammaLocal = (__local_mem__ T2*)gamma.GetPhyAddr();
        __local_mem__ T3* varLocal = (__local_mem__ T3*)runningVar.GetPhyAddr();
        __local_mem__ T1* dxLocal = (__local_mem__ T1*)dx.GetPhyAddr();

        VFNormalize(dyLocal, gammaLocal, varLocal, dxLocal, curTileB0Len, curTileALen, curTileB1Len);

        dxQueue_.EnQue(dx);

        dyQueue_.FreeTensor<T1>(dy);
        gammaQueue_.FreeTensor<T2>(gamma);
        runningVarQueue_.FreeTensor<T3>(runningVar);
    }

    __aicore__ inline void VFNormalize(
        __local_mem__ T1* dyLocal, __local_mem__ T2* gammaLocal, __local_mem__ T3* varLocal, __local_mem__ T1* dxLocal,
        uint16_t curTileB0Len, uint16_t curTileALen, uint16_t curTileB1Len)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> dy;
            RegTensor<float> gamma;
            RegTensor<float> runningVar;
            RegTensor<float> dx;
            RegTensor<float> invstd;

            MaskReg pregMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

            uint16_t loopNum = ops::CeilDiv(curTileB1Len, VL_FP32);
            uint32_t tileBlockALenTmp = static_cast<uint32_t>(tilingData_->tileBlockALen);
            uint32_t tileBlockB1LenTmp = static_cast<uint32_t>(tilingData_->tileBlockB1Len);
            float epsilonTmp = tilingData_->epsilon;
            for (uint16_t i = 0; i < curTileALen; i++) {
                // loads runningVar  1->64
                LoadsTensorForDtypeT<T3>(varLocal, runningVar, pregMask, i);
                CalRstdByHighPrecision(runningVar, invstd, epsilonTmp);

                // load gamma 1->64
                LoadsTensorForDtypeT<T2>(gammaLocal, gamma, pregMask, i);
                for (uint16_t j = 0; j < curTileB0Len; j++) {
                    for (uint16_t k = 0; k < loopNum; k++) {
                        uint32_t dyOffset = (j * tileBlockALenTmp + i) * tileBlockB1LenTmp + k * VL_FP32;

                        // load dy
                        LoadOneTensor<T1>(dyLocal, dy, pregMask, dyOffset);

                        // compute
                        Mul(dx, dy, gamma, pregMask);
                        Mul(dx, dx, invstd, pregMask);

                        // copy out
                        StoreOneTensor<T1>(dxLocal, dx, pregMask, dyOffset);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutDx(
        int64_t dxGmOffset, int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len, int64_t ubStrideT1)
    {
        LocalTensor<T1> dx = dxQueue_.DeQue<T1>();
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
        DataCopyPad<T1, PaddingMode::Normal>(dxGm_[dxGmOffset], dx, copyInParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        dxQueue_.FreeTensor(dx);
    }

private:
    const BatchNormGradV3InferTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> gammaQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> runningVarQueue_;

    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dxQueue_;

    GlobalTensor<T1> dyGm_;
    GlobalTensor<T2> gammaGm_;
    GlobalTensor<T3> runningVarGm_;

    GlobalTensor<T1> dxGm_;
};
} // namespace BatchNormGradV3

#endif
