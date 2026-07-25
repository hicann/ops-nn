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
 * \file hard_sigmoid.cpp
 * \brief HardSigmoid kernel（arch35 / DAV_3510，非模板 regbase 手写实现）
 *
 * y = clamp(alpha * x + beta, 0, 1)
 *
 * 采用 EnQue/DeQue 三级流水（CopyIn / Compute / CopyOut），跨管道同步由 Queue 自动管理：
 *   - FLOAT           : 原生 fp32 计算
 *   - FLOAT16/BFLOAT16: Cast 升 fp32 计算，再按目标类型舍入转回
 *   - INT32           : CAST_RINT 升 fp32 计算，再 CAST_TRUNC 转回
 */

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_sigmoid_tiling_data.h"
#include "hard_sigmoid_tiling_key.h"

using namespace AscendC;

// 缓冲深度取自 Host/Kernel 共享头，确保与 tiling 的 UB 预算推导一致。
static constexpr int32_t BUFFER_NUM = static_cast<int32_t>(HARD_SIGMOID_BUFFER_NUM);

template <typename T>
class HardSigmoidKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const HardSigmoidTilingData* tilingData)
    {
        totalElements_ = tilingData->totalElements;
        blockFactor_ = tilingData->blockFactor;
        ubFactor_ = tilingData->ubFactor;
        alpha_ = tilingData->alpha;
        beta_ = tilingData->beta;

        const int64_t coreIdx = static_cast<int64_t>(GetBlockIdx());
        startIdx_ = coreIdx * blockFactor_;
        int64_t remainderLength = totalElements_ - startIdx_;
        blockLength_ = (remainderLength > blockFactor_) ? blockFactor_ : remainderLength;
        if (startIdx_ >= totalElements_ || blockLength_ <= 0 || ubFactor_ <= 0) {
            blockLength_ = 0;
            return;
        }

        xGM_.SetGlobalBuffer((__gm__ T*)x + startIdx_, blockLength_);
        yGM_.SetGlobalBuffer((__gm__ T*)y + startIdx_, blockLength_);

        pipe_.InitBuffer(inQue_, BUFFER_NUM, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(outQue_, BUFFER_NUM, ubFactor_ * sizeof(T));
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(f32Buf_, ubFactor_ * sizeof(float));
        }

        loopCount_ = (blockLength_ + ubFactor_ - 1) / ubFactor_;
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= 0) {
            return;
        }
        for (int64_t ci = 0; ci < loopCount_; ci++) {
            uint32_t currentChunk = static_cast<uint32_t>((ci == (loopCount_ - 1)) ? (blockLength_ - ubFactor_ * ci) :
                                                                                     ubFactor_);
            CopyIn(ci, currentChunk);
            Compute(currentChunk);
            CopyOut(ci, currentChunk);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t ci, uint32_t currentChunk)
    {
        LocalTensor<T> xLocal = inQue_.template AllocTensor<T>();
        DataCopyExtParams cpIn;
        cpIn.blockCount = 1;
        cpIn.blockLen = currentChunk * sizeof(T);
        cpIn.srcStride = 0;
        cpIn.dstStride = 0;
        DataCopyPad(xLocal, xGM_[ci * ubFactor_], cpIn, {false, 0, 0, 0});
        inQue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t currentChunk)
    {
        LocalTensor<T> xLocal = inQue_.template DeQue<T>();
        LocalTensor<T> yLocal = outQue_.template AllocTensor<T>();

        if constexpr (std::is_same_v<T, float>) {
            // FLOAT: 原生 fp32，affine + clamp 全程 fp32
            Muls(yLocal, xLocal, alpha_, currentChunk);
            Adds(yLocal, yLocal, beta_, currentChunk);
            Mins(yLocal, yLocal, 1.0f, currentChunk);
            Maxs(yLocal, yLocal, 0.0f, currentChunk);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            // INT32: RINT 升 fp32 -> affine+clamp(fp32) -> TRUNC 转回
            LocalTensor<float> f32 = f32Buf_.template Get<float>();
            Cast(f32, xLocal, RoundMode::CAST_RINT, currentChunk);
            Muls(f32, f32, alpha_, currentChunk);
            Adds(f32, f32, beta_, currentChunk);
            Mins(f32, f32, 1.0f, currentChunk);
            Maxs(f32, f32, 0.0f, currentChunk);
            Cast(yLocal, f32, RoundMode::CAST_TRUNC, currentChunk);
        } else {
            // FLOAT16/BFLOAT16: 升 fp32 -> affine(fp32) -> 按目标类型舍入降回 -> clamp 在原生 dtype
            // 0.0/1.0 在 fp16/bf16 均可精确表示，先降后 clamp 与 golden(fp32内clamp后降) 等价，
            // 且省去 2 个 fp32 域 clamp 算子（原生 dtype 向量吞吐更高）。
            LocalTensor<float> f32 = f32Buf_.template Get<float>();
            Cast(f32, xLocal, RoundMode::CAST_NONE, currentChunk);
            Muls(f32, f32, alpha_, currentChunk);
            Adds(f32, f32, beta_, currentChunk);
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                Cast(yLocal, f32, RoundMode::CAST_ROUND, currentChunk);
            } else {
                Cast(yLocal, f32, RoundMode::CAST_RINT, currentChunk);
            }
            Mins(yLocal, yLocal, static_cast<T>(1.0f), currentChunk);
            Maxs(yLocal, yLocal, static_cast<T>(0.0f), currentChunk);
        }

        outQue_.template EnQue<T>(yLocal);
        inQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t ci, uint32_t currentChunk)
    {
        LocalTensor<T> yLocal = outQue_.template DeQue<T>();
        DataCopyExtParams cpOut;
        cpOut.blockCount = 1;
        cpOut.blockLen = currentChunk * sizeof(T);
        cpOut.srcStride = 0;
        cpOut.dstStride = 0;
        DataCopyPad(yGM_[ci * ubFactor_], yLocal, cpOut);
        outQue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    TBuf<TPosition::VECCALC> f32Buf_; // 非 fp32 dtype 的 fp32 中间计算缓冲

    GlobalTensor<T> xGM_, yGM_;

    int64_t totalElements_ = 0;
    int64_t blockFactor_ = 0;
    int64_t ubFactor_ = 0;
    int64_t startIdx_ = 0;
    int64_t blockLength_ = 0;
    int64_t loopCount_ = 0;
    float alpha_ = 0.0f;
    float beta_ = 0.0f;
};

// D_T_X（input0 dtype tiling-key）分发：每个 dtype 计算路径独立实例化，无运行时 dtype 分支。
template <typename D_T_X>
__global__ __aicore__ void hard_sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HardSigmoidTilingData);
    GET_TILING_DATA_WITH_STRUCT(HardSigmoidTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    HardSigmoidKernel<D_T_X> kernel;
    kernel.Init(x, y, &tilingData);
    kernel.Process();
}
