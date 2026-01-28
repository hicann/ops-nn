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
 * \file softmax_grad_ara_recompute.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_ARA_RECOMPUTE_H
#define SOFTMAX_GRAD_ARA_RECOMPUTE_H

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
class SoftmaxGradARARecompute : public SoftmaxGradOpsBase
{
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline SoftmaxGradARARecompute(){};

    __aicore__ inline SoftmaxGradARARecompute(const SoftmaxGradARARecomputeTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR x0, GM_ADDR x1, GM_ADDR y, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        x0Gm_.SetGlobalBuffer((__gm__ T*)x0);
        x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
        yGm_.SetGlobalBuffer((__gm__ T*)y);

        int64_t xShapeLen = tilingData_->tileA0Len * tilingData_->binAddRFactor;
        pipe_->InitBuffer(x0Queue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(float));

        pipe_->InitBuffer(cacheBuffer_, tilingData_->binAddCacheBufferCount * tilingData_->tileA0Len * sizeof(float));
        pipe_->InitBuffer(tempBuffer_, tilingData_->tileA0Len * sizeof(float));
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

            int64_t xOffset =
                // a1 offset
                curA1Idx * tilingData_->totalRLen * tilingData_->totalA0Len +
                // a0 offset
                curA0Idx * tilingData_->tileA0Len;

            uint16_t loopA0Num = ops::CeilDiv(curTileA0Len, VL_FP32);

            CalcReduceSum(curTileA0Len, xOffset, loopA0Num);

            for (int64_t idx = 0; idx < tilingData_->binAddRTotalLoop; idx++) {
                int64_t curTileRLen =
                    idx == (tilingData_->binAddRTotalLoop - 1) ? tilingData_->binAddRTail : tilingData_->binAddRFactor;

                int64_t offset = xOffset + idx * tilingData_->binAddRFactor * tilingData_->totalA0Len;
                CopyInX(offset, curTileRLen, curTileA0Len);
                CalcOutput(curTileRLen, curTileA0Len, loopA0Num);
                CopyOutY(offset, curTileRLen, curTileA0Len);
            }
        }
    }

private:
    __aicore__ inline void CalcReduceSum(uint32_t curTileA0Len, int64_t xOffsetStart, uint16_t loopA0Num)
    {
        tempTensor_ = tempBuffer_.Get<float>();
        cacheTensor_ = cacheBuffer_.Get<float>();

        int64_t blockOffset = tilingData_->totalA0Len * tilingData_->binAddRFactor;

        for (int64_t basicBlockIdx = 0; basicBlockIdx < tilingData_->binAddBasicBlockLoop; basicBlockIdx++) {
            ProcessMainBlock(tilingData_->binAddRFactor, curTileA0Len, xOffsetStart + basicBlockIdx * blockOffset,
                             loopA0Num);
            if (basicBlockIdx < tilingData_->binAddMainFoldCount) {
                ProcessFoldBlock(tilingData_->binAddRFactor, curTileA0Len,
                                 xOffsetStart + (basicBlockIdx + tilingData_->binAddBasicBlockLoop) * blockOffset,
                                 loopA0Num);
            } else if (basicBlockIdx == tilingData_->binAddMainFoldCount && tilingData_->binAddRTail > 0 &&
                       tilingData_->binAddRTail != tilingData_->binAddRFactor) {
                ProcessFoldBlock(tilingData_->binAddRTail, curTileA0Len,
                                 xOffsetStart + (basicBlockIdx + tilingData_->binAddBasicBlockLoop) * blockOffset,
                                 loopA0Num);
            }
            ProcessSummation(basicBlockIdx, tilingData_->binAddRFactor);
        }

        if (tilingData_->binAddBasicBlockLoop == 0) {
            ProcessMainBlock(tilingData_->binAddRTail, curTileA0Len, xOffsetStart, loopA0Num);
            ProcessSummation(0, tilingData_->binAddRTail);
        }

        xSumTensor_ = cacheTensor_[tilingData_->binAddResultCacheID * tilingData_->tileA0Len];
    }

    // copy to main
    __aicore__ inline void ProcessMainBlock(const int64_t curTileRLen, const int64_t curTileA0Len,
                                            const int64_t gmOffset, uint16_t loopA0Num)
    {
        CopyInX(gmOffset, curTileRLen, curTileA0Len);

        LocalTensor<T> x0Main_ = x0Queue_.DeQue<T>();
        LocalTensor<T> x1Main_ = x1Queue_.DeQue<T>();
        yMain_ = yQueue_.AllocTensor<float>();

        uint16_t outerLoopTimes = static_cast<uint16_t>(curTileRLen);
        uint32_t outerLoopSrcStride = tilingData_->tileA0Len;

        __local_mem__ float* dst = (__local_mem__ float*)yMain_.GetPhyAddr();
        __local_mem__ T* x0 = (__local_mem__ T*)x0Main_.GetPhyAddr();
        __local_mem__ T* x1 = (__local_mem__ T*)x1Main_.GetPhyAddr();

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            RegTensor<float> x0Reg;
            RegTensor<float> x1Reg;

            for (uint16_t j = 0; j < loopA0Num; ++j) {
                pregMask = UpdateMask<float>(sreg);
                for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                    uint32_t xOffset = i * outerLoopSrcStride + j * VL_FP32;
                    LoadTensorForDtypeT(x0, x0Reg, pregMask, xOffset);
                    LoadTensorForDtypeT(x1, x1Reg, pregMask, xOffset);
                    Mul(x0Reg, x0Reg, x1Reg, pregMask);
                    DataCopy((__local_mem__ float*)dst + xOffset, x0Reg, pregMask);
                }
            }
        }

        x0Queue_.FreeTensor(x0Main_);
        x1Queue_.FreeTensor(x1Main_);
    }

    // fold cast fp32 加到main
    __aicore__ inline void ProcessFoldBlock(const int64_t curTileRLen, const int64_t curTileA0Len,
                                            const int64_t gmOffset, uint16_t loopA0Num)
    {
        CopyInX(gmOffset, curTileRLen, curTileA0Len);

        LocalTensor<T> x0Fold = x0Queue_.DeQue<T>();
        LocalTensor<T> x1Fold = x1Queue_.DeQue<T>();

        uint16_t outerLoopTimes = static_cast<uint16_t>(curTileRLen);
        uint32_t outerLoopSrcStride = tilingData_->tileA0Len;

        __local_mem__ float* dst = (__local_mem__ float*)yMain_.GetPhyAddr();
        __local_mem__ T* x0 = (__local_mem__ T*)x0Fold.GetPhyAddr();
        __local_mem__ T* x1 = (__local_mem__ T*)x1Fold.GetPhyAddr();

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            RegTensor<float> x0Reg;
            RegTensor<float> x1Reg;

            for (uint16_t j = 0; j < loopA0Num; ++j) {
                pregMask = UpdateMask<float>(sreg);
                for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                    uint32_t xOffset = i * outerLoopSrcStride + j * VL_FP32;
                    LoadTensorForDtypeT(x0, x0Reg, pregMask, i * outerLoopSrcStride + j * VL_FP32);
                    LoadTensorForDtypeT(x1, x1Reg, pregMask, i * outerLoopSrcStride + j * VL_FP32);
                    Mul(x0Reg, x0Reg, x1Reg, pregMask);
                    DataCopy(x1Reg, (__local_mem__ float*)dst + xOffset);
                    Add(x1Reg, x1Reg, x0Reg, pregMask);
                    DataCopy((__local_mem__ float*)dst + xOffset, x1Reg, pregMask);
                }
            }
        }

        x0Queue_.FreeTensor(x0Fold);
        x1Queue_.FreeTensor(x1Fold);
    }

    __aicore__ inline void ProcessSummation(const int64_t basicBlockIdx, int64_t binAddRLen)
    {
        int64_t cacheID = GetCacheID(basicBlockIdx);
        uint32_t srcShape[2] = {uint32_t(binAddRLen), uint32_t(tilingData_->tileA0Len)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tempTensor_, yMain_, srcShape, false);
        yQueue_.FreeTensor(yMain_);
        UpdateCache(cacheTensor_, tempTensor_, cacheID, tilingData_->tileA0Len, tilingData_->tileA0Len);
    }

    __aicore__ inline void CopyInX(int64_t xGmOffset, int64_t curTileRLen, uint32_t curTileA0Len)
    {
        DataCopyExtParams params;
        params.blockCount = curTileRLen;
        params.blockLen = curTileA0Len * sizeof(T);
        params.srcStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T);
        params.dstStride = (tilingData_->tileA0Len * sizeof(T) - params.blockLen) / BLOCK_SIZE;
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;

        LocalTensor<T> x0 = x0Queue_.AllocTensor<T>();
        DataCopyPad(x0, x0Gm_[xGmOffset], params, padParams);
        x0Queue_.EnQue(x0);

        LocalTensor<T> x1 = x1Queue_.AllocTensor<T>();
        DataCopyPad(x1, x1Gm_[xGmOffset], params, padParams);
        x1Queue_.EnQue(x1);
    }

    __aicore__ inline void CalcOutput(int64_t curTileRLen, uint32_t curTileA0Len, uint16_t loopA0Num)
    {
        LocalTensor<T> x0 = x0Queue_.DeQue<T>();
        __local_mem__ T* x0Local = (__local_mem__ T*)x0.GetPhyAddr();
        LocalTensor<T> x1 = x1Queue_.DeQue<T>();
        __local_mem__ T* x1Local = (__local_mem__ T*)x1.GetPhyAddr();

        LocalTensor<T> y = yQueue_.template AllocTensor<T>();
        __local_mem__ T* yLocal = (__local_mem__ T*)y.GetPhyAddr();

        __local_mem__ float* xSumLocal = (__local_mem__ float*)xSumTensor_.GetPhyAddr();

        uint32_t tileA0Len = tilingData_->tileA0Len;
        uint16_t curTileRLenVl = static_cast<uint16_t>(curTileRLen);
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

                    // copy out
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

        yQueue_.EnQue(y);

        x0Queue_.FreeTensor<T>(x0);
        x1Queue_.FreeTensor<T>(x1);
    }

    __aicore__ inline void LoadTensorForDtypeT(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg,
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

    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t curTileRLen, uint32_t curTileA0Len)
    {
        LocalTensor<T> y = yQueue_.DeQue<T>();
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = curTileRLen;
        copyOutParams.blockLen = curTileA0Len * sizeof(T);
        copyOutParams.srcStride = (tilingData_->tileA0Len - curTileA0Len) * sizeof(T) / BLOCK_SIZE;
        copyOutParams.dstStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T);
        DataCopyPad<T, PaddingMode::Normal>(yGm_[yGmOffset], y, copyOutParams);
        yQueue_.FreeTensor(y);
    }

private:
    const SoftmaxGradARARecomputeTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> x0Queue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> x1Queue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> yQueue_;

    GlobalTensor<T> yGm_;
    GlobalTensor<T> x0Gm_;
    GlobalTensor<T> x1Gm_;

    // bin add
    TBuf<TPosition::VECCALC> cacheBuffer_;
    TBuf<TPosition::VECCALC> tempBuffer_;

    LocalTensor<float> tempTensor_;
    LocalTensor<float> cacheTensor_;

    LocalTensor<float> xSumTensor_;

    LocalTensor<float> yMain_;
};
}  // namespace SoftmaxGradOps

#endif
