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
 * \file l2_loss.h
 * \brief
 */
#ifndef L2_LOSS_H
#define L2_LOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "l2_loss_tiling_data.h"
#include "l2_loss_tiling_key.h"

namespace NsL2Loss {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DATA_CACHE_CLEAN_NEED = 64;
constexpr int32_t SLOT_STRIDE = DATA_CACHE_CLEAN_NEED / sizeof(float);
constexpr float   INV_SQRT2   = 0.70710678118654752f; // 1/√2 这是规定做法，不要考虑改这个
template <typename T, uint32_t schMode>
class L2Loss {
public:
    __aicore__ inline L2Loss(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, const L2LossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void L2LossAxesAll();

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueInput;
    
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;
    AscendC::GlobalTensor<float> workGm;

    // tileDataNum * sizeof(float) bytes
    // 仅 T != float 时真正用到：作为 Cast 目标（将输入从 T 升精度为 float）
    // T == float 时该 buffer 未被实际读写——Cast 跳过，ReduceSum 的 sharedTmpBuffer
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpFloat;

    // SLOT_STRIDE * sizeof(float) = 16 * 4 = 64 bytes（一条 Cache Line）
    // 用于原子写回前的暂存：将 ReduceSum 结果写入此 buf，再经 SetAtomicAdd 搬至 workGm
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuffer;

    // SLOT_STRIDE * sizeof(float) = 16 * 4 = 64 bytes（32B 对齐，满足 ReduceSum dst 要求）
    // ReduceSum 的输出目标，存放当前核对 accum 向量的标量归约结果
    AscendC::TBuf<AscendC::TPosition::VECCALC> tileSumBuf;

    uint32_t blockIdx;
    uint32_t blockNum;
    uint64_t globalOffset;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t validCoreDataNum;
    uint32_t lastTileValidLen;
};

template <typename T, uint32_t schMode>
__aicore__ inline void L2Loss<T, schMode>::Init(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, const L2LossTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->blockIdx = AscendC::GetBlockIdx();
    this->blockNum = AscendC::GetBlockNum();
    uint64_t globalBufferIndex = static_cast<uint64_t>(tilingData->bigCoreDataNum) * this->blockIdx;
    this->tileDataNum = tilingData->tileDataNum;

    if (this->blockIdx < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= static_cast<uint64_t>(tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (this->blockIdx - tilingData->tailBlockNum);
    }

    // Compute the valid data count for this core to exclude padding added at tail.
    uint64_t inputNum64 = static_cast<uint64_t>(tilingData->inputNum);
    this->validCoreDataNum = inputNum64 > globalBufferIndex ?
                             static_cast<uint32_t>(inputNum64 - globalBufferIndex) : 0;
    if (this->validCoreDataNum > this->coreDataNum) {
        this->validCoreDataNum = this->coreDataNum;
    }

    // tiling 层已保证 coreDataNum/tailDataNum 是 BLOCK_SIZE/inputBytes 的整数倍，
    // 内核可直接使用，无需重算或裁剪。
    xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    zGm.SetGlobalBuffer((__gm__ T*)z, 1);
    // float 单核：workGm 直接指向 z（仅用来承接结果，实际写入走 zGm）
    // float 多核：用 workspace 做原子累加中转，最后由 block 0 写回 z
    // half/bf16：始终用 workspace 做 float 中转
    uint32_t workGmSize = SLOT_STRIDE;
    if constexpr (std::is_same_v<T, float> && schMode == ELEMENTWISE_TPL_SCH_MODE_0) {
        workGm.SetGlobalBuffer((__gm__ float*)z, workGmSize);
    } else {
        workGm.SetGlobalBuffer((__gm__ float*)workspace, workGmSize);
    }
    if constexpr (schMode != ELEMENTWISE_TPL_SCH_MODE_0) {
        if (AscendC::GetBlockIdx() == 0) {
            AscendC::InitGlobalMemory(workGm, workGmSize, (float)0.0f);
        }
        AscendC::SyncAll<true>();
    }
    pipe.InitBuffer(inQueueInput, BUFFER_NUM, this->tileDataNum * sizeof(T));
    // float 输入直接原地操作（ReinterpretCast），无需 Cast 中转 buffer
    // half/bf16 需要将输入 Cast 升精度到 float，才需要 tmpFloat
    if constexpr (!std::is_same_v<T, float>) {
        pipe.InitBuffer(tmpFloat, tileDataNum * sizeof(float));
    }
    pipe.InitBuffer(tmpBuffer, workGmSize * sizeof(float));
    pipe.InitBuffer(tileSumBuf, SLOT_STRIDE * sizeof(float)); // ReduceSum output, 32B aligned

    // 预计算最后一个 tile 的有效长度，供 CopyIn 使用 DataCopyPad
    uint64_t processedBeforeLastTile = static_cast<uint64_t>(this->tileNum - 1) * this->tileDataNum;
    this->lastTileValidLen = (this->validCoreDataNum > processedBeforeLastTile)
                              ? static_cast<uint32_t>(this->validCoreDataNum - processedBeforeLastTile)
                              : 0;

    this->globalOffset = globalBufferIndex;
}

template <typename T, uint32_t schMode>
__aicore__ inline void L2Loss<T, schMode>::CopyIn(int32_t progress)
{
    uint64_t gmOffset = static_cast<uint64_t>(progress) * this->tileDataNum;
    AscendC::LocalTensor<T> xLocal = inQueueInput.AllocTensor<T>();
    if (progress == (int32_t)(tileNum - 1)) {
        // 最后一个 tile 使用 DataCopyPad，按有效元素字节数搬运，硬件自动 pad 零
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->lastTileValidLen * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyParams, padParams);
    } else {
        AscendC::DataCopy(xLocal, xGm[gmOffset], tileDataNum);
    }
    inQueueInput.EnQue(xLocal);
}
template <typename T, uint32_t schMode>
__aicore__ inline void L2Loss<T, schMode>::Process()
{
    L2LossAxesAll();
}

template <typename T, uint32_t schMode>
__aicore__ inline void L2Loss<T, schMode>::L2LossAxesAll()
{
    const uint32_t loopCount = this->tileNum;
    const uint32_t tileLen   = this->tileDataNum;
    const uint32_t lastTileLen = this->tailDataNum;

    // localSum 作为寄存器变量，不占 UB
    float localSum = 0.0f;

    // 循环外提前获取固定 buffer 地址，避免每次循环重复计算 UB 偏移
    AscendC::LocalTensor<float> tileSumTensor = tileSumBuf.Get<float>();
    AscendC::LocalTensor<float> reduceWork;
    if constexpr (std::is_same_v<T, float>) {
        reduceWork = tmpBuffer.Get<float>();
    } else {
        reduceWork = tmpFloat.Get<float>();
    }

    // 使用 Init 阶段预计算的 lastTileValidLen
    const uint32_t curLastTileValidLen = this->lastTileValidLen;

    // --- ping-pong 双缓冲：循环前预取第 0 个 tile ---
    CopyIn(0);

    // 前 loopCount-1 个完整 tile，无需判断是否为最后一个
    for (uint32_t t = 0; t + 1 < loopCount; ++t) {
        CopyIn(t + 1);

        AscendC::LocalTensor<T> tileLocal = inQueueInput.DeQue<T>();

        AscendC::LocalTensor<float> tileFloat;
        if constexpr (std::is_same_v<T, float>) {
            tileFloat = tileLocal.template ReinterpretCast<float>();
        } else {
            tileFloat = tmpFloat.Get<float>();
            AscendC::Cast(tileFloat, tileLocal, AscendC::RoundMode::CAST_NONE, tileLen);
        }

        AscendC::Muls(tileFloat, tileFloat, INV_SQRT2, tileLen);
        AscendC::Mul(tileFloat, tileFloat, tileFloat, tileLen);
        AscendC::ReduceSum(tileSumTensor, tileFloat, reduceWork, tileLen);
        localSum += tileSumTensor.GetValue(0);

        inQueueInput.FreeTensor(tileLocal);
    }

    // 最后一个 tile 单独处理：长度为 lastTileLen，可能含 padding 需清零
    {
        AscendC::LocalTensor<T> tileLocal = inQueueInput.DeQue<T>();

        AscendC::LocalTensor<float> tileFloat;
        if constexpr (std::is_same_v<T, float>) {
            tileFloat = tileLocal.template ReinterpretCast<float>();
        } else {
            tileFloat = tmpFloat.Get<float>();
            AscendC::Cast(tileFloat, tileLocal, AscendC::RoundMode::CAST_NONE, lastTileLen);
        }

        // DataCopyPad 已在搬运时对 padding 区域补零，无需手动清零
        (void)curLastTileValidLen;

        AscendC::Muls(tileFloat, tileFloat, INV_SQRT2, lastTileLen);
        AscendC::Mul(tileFloat, tileFloat, tileFloat, lastTileLen);
        AscendC::ReduceSum(tileSumTensor, tileFloat, reduceWork, lastTileLen);
        localSum += tileSumTensor.GetValue(0);

        inQueueInput.FreeTensor(tileLocal);
    }

    AscendC::LocalTensor<float> tmpBuf = tmpBuffer.Get<float>();
    AscendC::Duplicate(tmpBuf, 0.0f, SLOT_STRIDE);
    tmpBuf.SetValue(0, localSum);

    if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_0) {
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopyExtParams outParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(zGm, tmpBuf, outParams);
        } else {
            AscendC::LocalTensor<T> dstBuf = tileSumBuf.Get<T>();
            AscendC::Cast(dstBuf, tmpBuf, AscendC::RoundMode::CAST_ROUND, 16);
            AscendC::DataCopyExtParams outParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(zGm, dstBuf, outParams);
        }
    } else {
        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopy(workGm, tmpBuf, SLOT_STRIDE);
        AscendC::SetAtomicNone();
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(workGm[0]);
        AscendC::SyncAll<true>();

        if (this->blockIdx == 0) {
            float globalSum = workGm.GetValue(0);
            AscendC::LocalTensor<float> srcBuf = tmpBuffer.Get<float>();
            srcBuf.SetValue(0, globalSum);
            if constexpr (std::is_same_v<T, float>) {
                AscendC::DataCopyExtParams outParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(zGm, srcBuf, outParams);
            } else {
                AscendC::LocalTensor<T> dstBuf = tileSumBuf.Get<T>();
                AscendC::Cast(dstBuf, srcBuf, AscendC::RoundMode::CAST_ROUND, 16);
                AscendC::DataCopyExtParams outParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPad(zGm, dstBuf, outParams);
            }
        }
    }
}
} // namespace NsL2Loss
#endif // _L2_LOSS_H
