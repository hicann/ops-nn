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
 * \file log_softmax_grad_base.h
 * \brief
 */
#ifndef __LOG_SOFTMAX_GRAD_BASE_H__
#define __LOG_SOFTMAX_GRAD_BASE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "log_softmax_grad_tiling_data.h"
#include "log_softmax_grad_tiling_key.h"

namespace NsLogSoftmaxGrad {

using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t REPEAT_SIZE = 256;
constexpr uint32_t FP32_ELEMS_PER_BLOCK = BLOCK_SIZE / sizeof(float);
constexpr uint32_t FP32_ELEMS_PER_REPEAT = REPEAT_SIZE / sizeof(float);
constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr uint32_t MAX_WIDTH = MAX_REPEAT_TIME * (BLOCK_SIZE / sizeof(float));

template <typename T1, typename T2>
__aicore__ inline T1 GetMin(T1 a, T2 b)
{
    T1 tempB = static_cast<T1>(b);
    return a <= tempB ? a : tempB;
}

template <typename T, int BUF_NUM>
class LogSoftmaxGradBase {
protected:
    __aicore__ inline LogSoftmaxGradBase() = default;

    __aicore__ inline void BaseInit(LogSoftmaxGradTilingData& tiling, TPipe* pipePtr)
    {
        singleBufElems_ = tiling.singleBufElems;
        mergedDim0_ = tiling.mergedDim0;
        mergedDim1_ = tiling.mergedDim1;
        mergedDim2_ = tiling.mergedDim2;
        dim0Tile_ = tiling.dim0Tile;
        dim1Tile_ = tiling.dim1Tile;
        dim2Tile_ = tiling.dim2Tile;
        totalElems_ = tiling.totalElems;
        dim0LoopTime_ = tiling.dim0LoopTime;
        dim0Remained_ = tiling.dim0Remained;
        dim1LoopTime_ = tiling.dim1LoopTime;
        dim1Remained_ = tiling.dim1Remained;
        dim2LoopTime_ = tiling.dim2LoopTime;
        dim2Remained_ = tiling.dim2Remained;
        pipePtr_ = pipePtr;
    }

    template <bool IS_CONTIGUOUS>
    __aicore__ inline void CopyInDy(uint64_t offset, uint64_t count)
    {
        constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
        LocalTensor<float> dyFloatIn = inQueDy_.template AllocTensor<float>();
        if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
            LocalTensor<T> dyIn = dyFloatIn.ReinterpretCast<T>()[singleBufElems_];
            if constexpr (IS_CONTIGUOUS) {
                DataCopy(dyIn, dyGM_[offset], AlignUp(count, elemsPerBlock));
            } else {
                DataCopyPad(dyIn, dyGM_[offset], inParams_, {false, 0, 0, T(0)});
            }
            int32_t evtId = static_cast<int32_t>(pipePtr_->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evtId);
            WaitFlag<HardEvent::MTE2_V>(evtId);
            Cast(dyFloatIn, dyIn, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        } else {
            if constexpr (IS_CONTIGUOUS) {
                DataCopy(dyFloatIn, dyGM_[offset], AlignUp(count, elemsPerBlock));
            } else {
                DataCopyPad(dyFloatIn, dyGM_[offset], inParams_, {false, 0, 0, T(0)});
            }
        }
        inQueDy_.EnQue(dyFloatIn);
    }

    template <bool IS_CONTIGUOUS>
    __aicore__ inline void CopyInX(uint64_t offset, uint64_t count)
    {
        constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
        LocalTensor<float> xFloatIn = inQueX_.template AllocTensor<float>();
        if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
            LocalTensor<T> xIn = xFloatIn.ReinterpretCast<T>()[singleBufElems_];
            if constexpr (IS_CONTIGUOUS) {
                DataCopy(xIn, xGM_[offset], AlignUp(count, elemsPerBlock));
            } else {
                DataCopyPad(xIn, xGM_[offset], inParams_, {false, 0, 0, T(0)});
            }
            int32_t evtId = static_cast<int32_t>(pipePtr_->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evtId);
            WaitFlag<HardEvent::MTE2_V>(evtId);
            Cast(xFloatIn, xIn, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        } else {
            if constexpr (IS_CONTIGUOUS) {
                DataCopy(xFloatIn, xGM_[offset], AlignUp(count, elemsPerBlock));
            } else {
                DataCopyPad(xFloatIn, xGM_[offset], inParams_, {false, 0, 0, T(0)});
            }
        }
        inQueX_.EnQue(xFloatIn);
    }

    template <bool IS_CONTIGUOUS>
    __aicore__ inline void CopyOutZ(uint64_t offset, uint64_t count)
    {
        constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
        uint64_t countFloorAlign = 0, countRemained = 0;
        if constexpr (IS_CONTIGUOUS) {
            countFloorAlign = count / elemsPerBlock * elemsPerBlock;
            countRemained = count - countFloorAlign;
        }
        LocalTensor<float> zFloatOut = outQueZ_.template DeQue<float>();
        if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
            LocalTensor<T> zOut = zFloatOut.ReinterpretCast<T>();
            PipeBarrier<PIPE_V>();
            Cast(zOut, zFloatOut, RoundMode::CAST_RINT, count);
            int32_t evtId = static_cast<int32_t>(pipePtr_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(evtId);
            WaitFlag<HardEvent::V_MTE3>(evtId);
            if constexpr (IS_CONTIGUOUS) {
                DataCopy(zGM_[offset], zOut, countFloorAlign);
                if (countRemained) {
                    auto len = static_cast<uint16_t>(countRemained * sizeof(T));
                    DataCopyPad(zGM_[offset + countFloorAlign], zOut[countFloorAlign], {1, len, 0, 0});
                }
            } else {
                DataCopyPad(zGM_[offset], zOut, outParams_);
            }
        } else {
            if constexpr (IS_CONTIGUOUS) {
                DataCopy(zGM_[offset], zFloatOut, countFloorAlign);
                if (countRemained) {
                    auto len = static_cast<uint16_t>(countRemained * sizeof(T));
                    DataCopyPad(zGM_[offset + countFloorAlign], zFloatOut[countFloorAlign], {1, len, 0, 0});
                }
            } else {
                DataCopyPad(zGM_[offset], zFloatOut, outParams_);
            }
        }
        outQueZ_.FreeTensor(zFloatOut);
    }

protected:
    TPipe* pipePtr_;
    GlobalTensor<T> dyGM_, xGM_, zGM_;
    TQue<QuePosition::VECIN, BUF_NUM> inQueDy_;
    TQue<QuePosition::VECIN, BUF_NUM> inQueX_;
    TQue<QuePosition::VECOUT, BUF_NUM> outQueZ_;
    DataCopyExtParams inParams_;
    DataCopyExtParams outParams_;

    // tiling参数
    uint64_t singleBufElems_;
    uint64_t mergedDim0_;
    uint64_t mergedDim1_;
    uint64_t mergedDim2_;
    uint64_t dim0Tile_;
    uint64_t dim1Tile_;
    uint64_t dim2Tile_;
    uint64_t totalElems_;
    uint64_t dim0LoopTime_;
    uint64_t dim0Remained_;
    uint64_t dim1LoopTime_;
    uint64_t dim1Remained_;
    uint64_t dim2LoopTime_;
    uint64_t dim2Remained_;
};

} // namespace NsLogSoftmaxGrad
#endif // __LOG_SOFTMAX_GRAD_BASE_H__
