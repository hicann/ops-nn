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
 * \file batch_norm_grad_v3_infer_channel_last.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_H
#define BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_H

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
class BatchNormGradV3InferChannelLast {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradV3InferChannelLast(){};

    __aicore__ inline BatchNormGradV3InferChannelLast(const BatchNormGradV3InferChannelLastTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR weight, GM_ADDR runningVar, GM_ADDR dx, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        dyGm_.SetGlobalBuffer((__gm__ T1*)dy);
        gammaGm_.SetGlobalBuffer((__gm__ T2*)weight);
        runningVarGm_.SetGlobalBuffer((__gm__ T3*)runningVar);
        dxGm_.SetGlobalBuffer((__gm__ T1*)dx);

        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T2));
        pipe_->InitBuffer(runningVarQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T3));

        int64_t xShapeLen = tilingData_->tileBlockBLen * tilingData_->tileBlockALen;
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
            int64_t dyOffset = curRowIdx * tilingData_->totalALen * tilingData_->tileBlockBLen + weightOffset;

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
        extParam.srcStride = (tilingData_->totalALen - curTileALen) * sizeof(T1);
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

    __aicore__ inline void Compute(int64_t curTileBLen, int64_t curTileALen)
    {
        LocalTensor<T1> dy = dyQueue_.DeQue<T1>();
        LocalTensor<T2> gamma = gammaQueue_.DeQue<T2>();
        LocalTensor<T3> runningVar = runningVarQueue_.DeQue<T3>();
        LocalTensor<T1> dx = dxQueue_.AllocTensor<T1>();

        __local_mem__ T1* dyLocal = (__local_mem__ T1*)dy.GetPhyAddr();
        __local_mem__ T2* gammaLocal = (__local_mem__ T2*)gamma.GetPhyAddr();
        __local_mem__ T3* varLocal = (__local_mem__ T3*)runningVar.GetPhyAddr();
        __local_mem__ T1* dxLocal = (__local_mem__ T1*)dx.GetPhyAddr();

        VFNormalize(dyLocal, gammaLocal, varLocal, dxLocal, curTileBLen, curTileALen);

        dxQueue_.EnQue(dx);

        dyQueue_.FreeTensor<T1>(dy);
        gammaQueue_.FreeTensor<T2>(gamma);
        runningVarQueue_.FreeTensor<T3>(runningVar);
    }

    __aicore__ inline void VFNormalize(
        __local_mem__ T1* dyLocal, __local_mem__ T2* gammaLocal, __local_mem__ T3* varLocal, __local_mem__ T1* dxLocal,
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
                LoadOneTensor<T3>(varLocal, runningVar, pregMaskFp32, offset);
                CalRstdByHighPrecision(runningVar, invstd, epsilonTmp);

                LoadOneTensor<T2>(gammaLocal, gamma, pregMaskFp32, offset);

                for (uint16_t j = 0; j < curTileBLen; j++) {
                    uint32_t dyOffset = j * tileBlockALenTmp + offset;
                    // load dy
                    LoadOneTensor<T1>(dyLocal, dy, pregMaskFp32, dyOffset);

                    // compute
                    Mul(dx, dy, gamma, pregMaskFp32);
                    Mul(dx, dx, invstd, pregMaskFp32);

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
        extParams.dstStride = (tilingData_->totalALen - curTileALen) * sizeof(T1);
        extParams.blockCount = curTileBLen;

        DataCopyPad(dxGm_[dxGmOffset], dx, extParams);

        dxQueue_.FreeTensor(dx);
    }

private:
    const BatchNormGradV3InferChannelLastTilingData* tilingData_;

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
