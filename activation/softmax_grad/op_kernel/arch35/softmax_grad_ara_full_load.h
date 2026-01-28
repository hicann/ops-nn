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
 * \file softmax_grad_ara.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_ARA_H
#define SOFTMAX_GRAD_ARA_H

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "softmax_grad_base.h"

namespace SoftmaxGradOps
{
using namespace AscendC;
using namespace AscendC::MicroAPI;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;

template <typename T>
class SoftmaxGradARA
{
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline SoftmaxGradARA(){};

    __aicore__ inline SoftmaxGradARA(const SoftmaxGradARATilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR x0, GM_ADDR x1, GM_ADDR y, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        x0Gm_.SetGlobalBuffer((__gm__ T*)x0);
        x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
        yGm_.SetGlobalBuffer((__gm__ T*)y);

        int64_t xShapeLen = tilingData_->a1Inner * tilingData_->tileA0Len * tilingData_->totalRLen;
        pipe_->InitBuffer(x0Queue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(float));

        pipe_->InitBuffer(xSumBuffer_, tilingData_->tileA0Len * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        // pattern is [A1, R, A0]
        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curA0Idx = curIdx % tilingData_->tileA0Outer;
            int64_t curA1Idx = curIdx / tilingData_->tileA0Outer;

            uint32_t curTileA0Len =
                curA0Idx == (tilingData_->tileA0Outer - 1) ? tilingData_->tileA0Tail : tilingData_->tileA0Len;
            int64_t curTileA1Len = curA1Idx == (tilingData_->a1Outer - 1) ? tilingData_->a1Tail : tilingData_->a1Inner;

            int64_t xOffset =
                // a1 offset
                curA1Idx * tilingData_->totalRLen * tilingData_->totalA0Len * tilingData_->a1Inner +
                // a0 offset
                curA0Idx * tilingData_->tileA0Len;

            CopyInX(xOffset, curTileA0Len, curTileA1Len);

            LocalTensor<T> x0Tensor = x0Queue_.DeQue<T>();
            LocalTensor<T> x1Tensor = x1Queue_.DeQue<T>();

            __local_mem__ T* x0Local = (__local_mem__ T*)x0Tensor.GetPhyAddr();
            __local_mem__ T* x1Local = (__local_mem__ T*)x1Tensor.GetPhyAddr();

            yMain_ = yQueue_.AllocTensor<float>();

            for (int64_t j = 0; j < curTileA1Len; j++) {
                int64_t offset = j * tilingData_->tileA0Len * tilingData_->totalRLen;
                CalcReduceSum(x0Local + offset, x1Local + offset, curTileA0Len, offset);
                CalcOutput(x0Local + offset, x1Local + offset, curTileA0Len, offset);
            }

            yQueue_.EnQue(yMain_);

            x0Queue_.FreeTensor(x0Tensor);
            x1Queue_.FreeTensor(x1Tensor);

            CopyOutY(xOffset, curTileA0Len, curTileA1Len);
        }
    }

private:
    __aicore__ inline void CalcReduceSum(const __local_mem__ T* x0Local, const __local_mem__ T* x1Local,
                                         uint32_t curTileA0Len, int64_t a0BlockOffset)
    {
        __local_mem__ float* yLocal = (__local_mem__ float*)yMain_.GetPhyAddr() + a0BlockOffset;

        uint32_t tileA0Len = tilingData_->tileA0Len;
        uint16_t curTileRLenVl = static_cast<uint16_t>(tilingData_->totalRLen);
        uint16_t loopA0Num = ops::CeilDiv(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x0Reg;
            RegTensor<float> x1Reg;

            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = UpdateMask<float>(sreg);
                for (uint16_t i = 0; i < curTileRLenVl; i++) {
                    uint32_t xOffset = i * tileA0Len + k * VL_FP32;
                    LoadTensorForDtypeT(x0Local, x0Reg, pregMask, xOffset);
                    LoadTensorForDtypeT(x1Local, x1Reg, pregMask, xOffset);

                    Mul(x0Reg, x0Reg, x1Reg, pregMask);
                    DataCopy(((__local_mem__ float*)yLocal) + xOffset, x0Reg, pregMask);
                }
            }
        }

        xSumTensor_ = xSumBuffer_.Get<float>();
        uint32_t srcShape[2] = {uint32_t(tilingData_->totalRLen), uint32_t(tilingData_->tileA0Len)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(xSumTensor_, yMain_[a0BlockOffset], srcShape,
                                                                      false);
    }

    __aicore__ inline void CalcOutput(const __local_mem__ T* x0Local, const __local_mem__ T* x1Local,
                                      uint32_t curTileA0Len, int64_t a0BlockOffset)
    {
        __local_mem__ T* yLocal = (__local_mem__ T*)yMain_.GetPhyAddr() + a0BlockOffset;
        __local_mem__ float* xSumLocal = (__local_mem__ float*)xSumTensor_.GetPhyAddr();

        uint32_t tileA0Len = tilingData_->tileA0Len;
        uint16_t curTileRLenVl = static_cast<uint16_t>(tilingData_->totalRLen);
        uint16_t loopA0Num = ops::CeilDiv(curTileA0Len, VL_FP32);

        __VEC_SCOPE__
        {
            RegTensor<float> x0Reg;
            RegTensor<float> x1Reg;
            RegTensor<float> sumReg;

            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(sumReg, (__local_mem__ float*)xSumLocal + k * VL_FP32);
                for (uint16_t i = 0; i < curTileRLenVl; i++) {
                    uint32_t xOffset = i * tileA0Len + k * VL_FP32;
                    LoadTensorForDtypeT(x0Local, x0Reg, pregMask, xOffset);
                    LoadTensorForDtypeT(x1Local, x1Reg, pregMask, xOffset);

                    Mul(x1Reg, x0Reg, x1Reg, pregMask);
                    Mul(x0Reg, x0Reg, sumReg, pregMask);
                    Sub(x0Reg, x1Reg, x0Reg, pregMask);

                    if constexpr (IsSameType<T, float>::value) {
                        DataCopy(((__local_mem__ float*)yLocal) + xOffset, x0Reg, pregMask);
                    } else {  // fp16、bf16
                        RegTensor<T> xFp16;
                        Cast<T, float, castTraitFp32ToFp16>(xFp16, x0Reg, pregMask);
                        DataCopy<T, StoreDist::DIST_PACK_B32>(((__local_mem__ T*)yLocal) + xOffset, xFp16, pregMask);
                    }
                }
            }
        }
    }

    __aicore__ inline void LoadTensorForDtypeT(const __local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg,
                                               uint32_t offset)
    {
        if constexpr (IsSameType<T, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
        } else {  // fp16、bf16
            RegTensor<T> xFp16;
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
            Cast<float, T, castTraitFp16ToFp32>(dst, xFp16, preg);
        }
    }

    __aicore__ inline void CopyInX(int64_t xGmOffset, uint32_t curTileA0Len, uint32_t curTileA1Len)
    {
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curTileA1Len;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->totalA0Len * tilingData_->totalRLen * sizeof(T);
        loopParams.loop1DstStride = tilingData_->tileA0Len * tilingData_->totalRLen * sizeof(T);

        DataCopyExtParams params;
        params.blockCount = tilingData_->totalRLen;
        params.blockLen = curTileA0Len * sizeof(T);
        params.srcStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T);
        params.dstStride = (tilingData_->tileA0Len * sizeof(T) - params.blockLen) / BLOCK_SIZE;
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;

        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

        LocalTensor<T> x0 = x0Queue_.AllocTensor<T>();
        DataCopyPad(x0, x0Gm_[xGmOffset], params, padParams);
        x0Queue_.EnQue(x0);

        LocalTensor<T> x1 = x1Queue_.AllocTensor<T>();
        DataCopyPad(x1, x1Gm_[xGmOffset], params, padParams);
        x1Queue_.EnQue(x1);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, uint32_t curTileA0Len, uint32_t curTileA1Len)
    {
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curTileA1Len;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->tileA0Len * tilingData_->totalRLen * sizeof(T);
        loopParams.loop1DstStride = tilingData_->totalA0Len * tilingData_->totalRLen * sizeof(T);

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = tilingData_->totalRLen;
        copyOutParams.blockLen = curTileA0Len * sizeof(T);
        copyOutParams.srcStride = (tilingData_->tileA0Len - curTileA0Len) * sizeof(T) / BLOCK_SIZE;
        copyOutParams.dstStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T);

        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        LocalTensor<T> y = yQueue_.DeQue<T>();
        DataCopyPad<T, PaddingMode::Normal>(yGm_[yGmOffset], y, copyOutParams);
        yQueue_.FreeTensor(y);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }

private:
    const SoftmaxGradARATilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> x0Queue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> x1Queue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> yQueue_;

    GlobalTensor<T> yGm_;
    GlobalTensor<T> x0Gm_;
    GlobalTensor<T> x1Gm_;

    TBuf<TPosition::VECCALC> xSumBuffer_;

    LocalTensor<float> xSumTensor_;

    LocalTensor<float> yMain_;
};
}  // namespace SoftmaxGradOps

#endif