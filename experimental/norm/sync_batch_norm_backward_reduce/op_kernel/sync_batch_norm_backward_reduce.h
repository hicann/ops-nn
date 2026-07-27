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
 * \file sync_batch_norm_backward_reduce.h
 * \brief SyncBatchNormBackwardReduce kernel class.
 *
 * element-wise（per-channel）算子，4 输入 2 输出，多核切分。各 dtype 在 UB 上
 * 统一提升到 float 计算：
 *   sum_dy_xmu = sum_dy_dx_pad - mean * sum_dy
 *   y          = sum_dy_xmu * invert_std
 */

#ifndef SYNCBNBR_H
#define SYNCBNBR_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sync_batch_norm_backward_reduce_tiling_data.h"

namespace NsSyncBatchNormBackwardReduce {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class SyncBatchNormBackwardReduceKernel {
public:
    __aicore__ inline SyncBatchNormBackwardReduceKernel(const SyncBatchNormBackwardReduceTilingData* tilingData)
        : bufferNum(tilingData->bufferNum),
          epochs(tilingData->epochs),
          tileLength(tilingData->tileLength),
          tailTileLength(tilingData->tailTileLength)
    {
        this->globalOffset = tilingData->coreLength * AscendC::GetBlockIdx();
        this->isLastCore = (AscendC::GetBlockIdx() == tilingData->coreNum - 1u);
        if (this->isLastCore) {
            this->epochs = tilingData->epochsForLastCore;
            this->tailTileLength = tilingData->tailTileLengthForLastCore;
            this->tailElems = tilingData->tailElems;
        }
        this->pipe.InitBuffer(this->inQue0, this->bufferNum, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(this->inQue1, this->bufferNum, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(this->inQue2, this->bufferNum, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(this->inQue3, this->bufferNum, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(this->outQue0, this->bufferNum, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(this->outQue1, this->bufferNum, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(this->calcBuf0, this->tileLength * sizeof(float));
        this->pipe.InitBuffer(this->calcBuf1, this->tileLength * sizeof(float));
        this->pipe.InitBuffer(this->calcBuf2, this->tileLength * sizeof(float));
        this->pipe.InitBuffer(this->calcBuf3, this->tileLength * sizeof(float));
    }

    // 参数顺序：sum_dy, sum_dy_dx_pad, mean, invert_std, sum_dy_xmu(out), y(out)
    __aicore__ inline void Init(GM_ADDR sumDy, GM_ADDR sumDyDxPad, GM_ADDR mean, GM_ADDR invertStd, GM_ADDR sumDyXmu,
                                GM_ADDR y)
    {
        this->sumDyGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sumDy) + this->globalOffset);
        this->sumDyDxPadGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sumDyDxPad) + this->globalOffset);
        this->meanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(mean) + this->globalOffset);
        this->invertStdGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(invertStd) + this->globalOffset);
        this->sumDyXmuGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sumDyXmu) + this->globalOffset);
        this->yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + this->globalOffset);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> calc0 = this->calcBuf0.template Get<float>();
        AscendC::LocalTensor<float> calc1 = this->calcBuf1.template Get<float>();
        AscendC::LocalTensor<float> calc2 = this->calcBuf2.template Get<float>();
        AscendC::LocalTensor<float> calc3 = this->calcBuf3.template Get<float>();

        for (uint64_t i = 0u; i < this->epochs; i++) {
            this->CopyIn(calc0, calc1, calc2, calc3, i * this->tileLength, this->tileLength, this->tileLength);
            this->Compute(calc0, calc1, calc2, calc3, this->tileLength);
            this->CopyOut(calc1, calc3, i * this->tileLength, this->tileLength, this->tileLength);
        }

        if (this->tailTileLength || (this->isLastCore && this->tailElems)) {
            uint64_t tailLength = this->tailTileLength;
            if (this->isLastCore && this->tailElems) {
                tailLength += this->tailElems;
            }
            uint64_t tailAligned = (tailLength + ELEM_PER_BLOCK - 1u) & ~(ELEM_PER_BLOCK - 1u);
            this->CopyIn(calc0, calc1, calc2, calc3, this->epochs * this->tileLength, tailLength, tailAligned);
            this->Compute(calc0, calc1, calc2, calc3, tailAligned);
            this->CopyOut(calc1, calc3, this->epochs * this->tileLength, tailLength, tailAligned);
        }
    }

private:
    __aicore__ inline void LoadAndCast(AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM>& que,
                                       const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<float>& dst,
                                       uint64_t offset, uint64_t copyLength, uint64_t ubLength)
    {
        AscendC::LocalTensor<T> local = que.template AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1u, static_cast<uint32_t>(copyLength * sizeof(T)), 0u, 0u, 0u};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0u, 0u, static_cast<T>(0)};
        AscendC::DataCopyPad(local, src[offset], copyParams, padParams);
        que.template EnQue<T>(local);
        local = que.template DeQue<T>();
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopy<float>(dst, local.template ReinterpretCast<float>(), ubLength);
        } else {
            AscendC::Cast<float, T>(dst, local, AscendC::RoundMode::CAST_NONE, ubLength);
        }
        que.template FreeTensor<T>(local);
    }

    __aicore__ inline void CopyIn(const AscendC::LocalTensor<float>& calc0, const AscendC::LocalTensor<float>& calc1,
                                  const AscendC::LocalTensor<float>& calc2, const AscendC::LocalTensor<float>& calc3,
                                  uint64_t offset, uint64_t copyLength, uint64_t ubLength)
    {
        LoadAndCast(this->inQue0, this->sumDyGlobal, calc0, offset, copyLength, ubLength);
        LoadAndCast(this->inQue1, this->sumDyDxPadGlobal, calc1, offset, copyLength, ubLength);
        LoadAndCast(this->inQue2, this->meanGlobal, calc2, offset, copyLength, ubLength);
        LoadAndCast(this->inQue3, this->invertStdGlobal, calc3, offset, copyLength, ubLength);
    }

    // 输入：calc0=sum_dy, calc1=sum_dy_dx_pad, calc2=mean, calc3=invert_std
    // 输出：calc1=sum_dy_xmu, calc3=y
    __aicore__ inline void Compute(const AscendC::LocalTensor<float>& calc0, const AscendC::LocalTensor<float>& calc1,
                                   const AscendC::LocalTensor<float>& calc2, const AscendC::LocalTensor<float>& calc3,
                                   const uint64_t length)
    {
        AscendC::Mul<float>(calc2, calc2, calc0, length); // mean * sum_dy
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(calc1, calc1, calc2, length); // sum_dy_xmu = sum_dy_dx_pad - mean*sum_dy
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul<float>(calc3, calc1, calc3, length); // y = sum_dy_xmu * invert_std
    }

    __aicore__ inline void StoreFromFloat(AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM>& que,
                                          const AscendC::GlobalTensor<T>& dst, const AscendC::LocalTensor<float>& srcF,
                                          uint64_t offset, uint64_t copyLength, uint64_t ubLength)
    {
        AscendC::LocalTensor<T> local = que.template AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopy<float>(local.template ReinterpretCast<float>(), srcF, ubLength);
        } else {
#if __CCE_AICORE__ == 200
            AscendC::Cast<T, float>(local, srcF, AscendC::RoundMode::CAST_NONE, ubLength);
#else
            AscendC::Cast<T, float>(local, srcF, AscendC::RoundMode::CAST_RINT, ubLength);
#endif
        }
        que.template EnQue<T>(local);
        local = que.template DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1u, static_cast<uint32_t>(copyLength * sizeof(T)), 0u, 0u, 0u};
        AscendC::DataCopyPad(dst[offset], local, copyParams);
        que.template FreeTensor<T>(local);
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<float>& sumDyXmuF, const AscendC::LocalTensor<float>& yF,
                                   uint64_t offset, uint64_t copyLength, uint64_t ubLength)
    {
        StoreFromFloat(this->outQue0, this->sumDyXmuGlobal, sumDyXmuF, offset, copyLength, ubLength);
        StoreFromFloat(this->outQue1, this->yGlobal, yF, offset, copyLength, ubLength);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQue0;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQue1;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQue2;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQue3;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQue0;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQue1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf3;

    AscendC::GlobalTensor<T> sumDyGlobal;
    AscendC::GlobalTensor<T> sumDyDxPadGlobal;
    AscendC::GlobalTensor<T> meanGlobal;
    AscendC::GlobalTensor<T> invertStdGlobal;
    AscendC::GlobalTensor<T> sumDyXmuGlobal;
    AscendC::GlobalTensor<T> yGlobal;

    bool isLastCore = false;
    uint64_t tailElems = 0u;
    uint64_t bufferNum = 1u;
    uint64_t epochs = 0u;
    uint64_t globalOffset = 0u;
    uint64_t tileLength = 0u;
    uint64_t tailTileLength = 0u;
    constexpr static uint64_t ELEM_PER_BLOCK = 32u / sizeof(T);
};

template <typename T>
__aicore__ inline void Run(GM_ADDR sumDy, GM_ADDR sumDyDxPad, GM_ADDR mean, GM_ADDR invertStd, GM_ADDR sumDyXmu,
                           GM_ADDR y, GM_ADDR /*workspace*/, const SyncBatchNormBackwardReduceTilingData* tilingData)
{
    SyncBatchNormBackwardReduceKernel<T> op(tilingData);
    op.Init(sumDy, sumDyDxPad, mean, invertStd, sumDyXmu, y);
    op.Process();
}

} // namespace NsSyncBatchNormBackwardReduce

#endif
