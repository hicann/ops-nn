/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Template D: Slice-Major Stride (SM-ST) — Simplified Block-Major
// 适用条件: blockSize = 1 且 numBlocks 较小 (<= 256)
// 沿 sliceCount 分核, 每核处理若干连续 slice
// 与 Template E 的区别: 单缓冲 (无双缓冲), 适合 numBlocks 较小的场景
//   - 单缓冲: UB 占用更少, 可用更大 tileLength
//   - 无双缓冲事件管理开销, 同步更简单
//   - numBlocks 小时, 双缓冲的流水线重叠收益不足以抵消其开销

#ifndef _RENORM_SM_ST_H_
#define _RENORM_SM_ST_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "renorm_tiling_key.h"
#include "common/renorm_common.h"

namespace NsRenormSmSt {

using namespace AscendC;

constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;

// Compare/Select 对齐要求: 32 字节 = 8 个 FP32
constexpr int64_t CMP_ALIGN = 8;

template <typename D_T_X>
class RenormSmSt {
public:
    __aicore__ inline RenormSmSt() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RenormTilingData* tilingData);
    __aicore__ inline void Process();

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> dataBuf;    // 输入数据 (单缓冲)
    TBuf<QuePosition::VECCALC> workBuf;    // FP32 工作空间
    TBuf<QuePosition::VECCALC> normBuf;    // 范数累加器 (向量)
    TBuf<QuePosition::VECCALC> scaleBuf;   // 缩放因子 (向量)
    TBuf<QuePosition::VECCALC> maskBuf;    // Compare mask
    TBuf<QuePosition::VECCALC> zerosBuf;   // 零常量
    TBuf<QuePosition::VECCALC> onesBuf;    // 一常量
    TBuf<QuePosition::VECCALC> maxNormBuf; // maxNorm 广播
    TBuf<QuePosition::VECCALC> tmpBuf;     // Reciprocal 临时

    GlobalTensor<D_T_X> inputGM;
    GlobalTensor<D_T_X> outputGM;

    int64_t totalElements_ = 0;
    int64_t sliceCount_ = 0;
    int64_t numBlocks_ = 0;       // blockSize = 1
    int64_t sliceTileLength_ = 0; // sliceCount 方向的 tile 大小
    int64_t slicesPerCore_ = 0;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
    int32_t normMode_ = 0;
};

template <typename D_T_X>
__aicore__ inline void RenormSmSt<D_T_X>::Init(GM_ADDR x, GM_ADDR y, const RenormTilingData* tilingData)
{
    totalElements_ = tilingData->totalElements;
    sliceCount_ = tilingData->sliceCount;
    numBlocks_ = tilingData->numBlocks;
    sliceTileLength_ = tilingData->sliceTileLength;
    slicesPerCore_ = tilingData->slicesPerCore;
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;

    if (totalElements_ == 0 || sliceCount_ == 0 || numBlocks_ == 0 || sliceTileLength_ == 0) {
        return;
    }

    inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);

    int64_t typeSize = sizeof(D_T_X);
    int64_t alignedTile = (sliceTileLength_ + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;

    // 单缓冲: 只需 1 个 dataBuf (Template E 需要 2 个)
    // UB 开销: dataBuf(typeSize) + workBuf(4) + normBuf(4) + scaleBuf(4)
    //         + maskBuf(1) + zerosBuf(4) + onesBuf(4) + maxNormBuf(4) + tmpBuf(4)
    //         = typeSize + 25 bytes/element
    pipe.InitBuffer(dataBuf, alignedTile * typeSize);
    pipe.InitBuffer(workBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(normBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(scaleBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(maskBuf, alignedTile); // uint8_t, 1 byte per element
    pipe.InitBuffer(zerosBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(onesBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(maxNormBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(tmpBuf, alignedTile * sizeof(float));
}

template <typename D_T_X>
__aicore__ inline void RenormSmSt<D_T_X>::Process()
{
    if (totalElements_ == 0 || sliceCount_ == 0 || numBlocks_ == 0 || sliceTileLength_ == 0) {
        return;
    }

    int64_t blockIdx = GetBlockIdx();
    int64_t startSlice = blockIdx * slicesPerCore_;
    int64_t endSlice = startSlice + slicesPerCore_;
    if (endSlice > sliceCount_) {
        endSlice = sliceCount_;
    }

    LocalTensor<D_T_X> dataLocal = dataBuf.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> normLocal = normBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<float> maxNormLocal = maxNormBuf.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

    int64_t alignedTile = (sliceTileLength_ + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;

    // 预初始化常量 buffer
    Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(alignedTile));
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(alignedTile));
    Duplicate(maxNormLocal, maxNorm_, static_cast<int32_t>(alignedTile));
    PipeBarrier<PIPE_V>();

    // maxNorm=0: 直接输出全零
    if (normMode_ == NORM_MODE_MAXNORM_ZERO) {
        for (int64_t sliceTile = startSlice; sliceTile < endSlice; sliceTile += sliceTileLength_) {
            int64_t currentTile = (sliceTileLength_ < (endSlice - sliceTile)) ? sliceTileLength_ :
                                                                                (endSlice - sliceTile);
            int64_t alignedLen = (currentTile + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;

            Duplicate(dataLocal, static_cast<D_T_X>(0), static_cast<int32_t>(alignedLen));
            TEventID eventID1 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(eventID1);
            WaitFlag<HardEvent::V_MTE3>(eventID1);

            for (int64_t b = 0; b < numBlocks_; ++b) {
                int64_t offset = b * sliceCount_ + sliceTile;
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPad(outputGM[offset], dataLocal, copyParams);

                TEventID eventID2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                SetFlag<HardEvent::MTE3_MTE2>(eventID2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
            }
        }
        return;
    }

    // 按 sliceTile 迭代处理
    for (int64_t sliceTile = startSlice; sliceTile < endSlice; sliceTile += sliceTileLength_) {
        int64_t currentTile = (sliceTileLength_ < (endSlice - sliceTile)) ? sliceTileLength_ : (endSlice - sliceTile);
        int64_t alignedLen = (currentTile + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;
        int64_t padLen = alignedLen - currentTile;

        // === Pass 1: 向量累加范数 (单缓冲, block-major 遍历) ===
        Duplicate(normLocal, 0.0f, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        for (int64_t b = 0; b < numBlocks_; ++b) {
            int64_t gmOffset = b * sliceCount_ + sliceTile;

            // V→MTE2 同步 (确保上一轮 V 读 dataLocal 完成)
            TEventID eventIDV = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(eventIDV);
            WaitFlag<HardEvent::V_MTE2>(eventIDV);

            // 连续加载 currentTile 个元素
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(padLen), 0};
            DataCopyPad(dataLocal, inputGM[gmOffset], copyParams, padParams);

            // MTE2→V 同步
            TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(eventID0);
            WaitFlag<HardEvent::MTE2_V>(eventID0);

            // Cast 到 FP32
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(alignedLen));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(alignedLen));
            }
            PipeBarrier<PIPE_V>();

            // normMode 变换
            if (normMode_ == NORM_MODE_P_ZERO) {
                Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Compare(maskLocal, workLocal, zerosLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Select(workLocal, maskLocal, onesLocal, zerosLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                       static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            } else if (normMode_ == NORM_MODE_P_INF) {
                Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Max(normLocal, normLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                continue; // P_INF 不用 Add
            } else {
                Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                if (numBlocks_ == 1) {
                    // A one-element slice has ||x||_p == |x| for every p>0.
                    // Avoid high-p Log/Exp overflow and all redundant power work.
                } else if (p_ == 1.0f) {
                    // 直接累加 |x|
                } else if (p_ == 2.0f) {
                    Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                } else {
                    Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                    Log(workLocal, workLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                    Muls(workLocal, workLocal, p_, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                    Exp(workLocal, workLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                }
            }

            // 向量累加
            Add(normLocal, normLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        }

        // 后处理: Sqrt/Pow 向量化
        if (normMode_ == NORM_MODE_P_POSITIVE) {
            if (numBlocks_ == 1) {
                // The accumulation result is already the exact norm.
            } else if (p_ == 2.0f) {
                Sqrt(normLocal, normLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            } else if (p_ != 1.0f) {
                Maxs(normLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Log(normLocal, normLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Muls(normLocal, normLocal, 1.0f / p_, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Exp(normLocal, normLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            }
        }

        // === Scale: 向量化计算 ===
        // scale = (norm > maxNorm) ? maxNorm / max(norm, eps) : 1.0
        Maxs(tmpLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Reciprocal(tmpLocal, tmpLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Muls(scaleLocal, tmpLocal, maxNorm_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        Compare(maskLocal, normLocal, maxNormLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        // === Pass 2: 向量化应用 scale (单缓冲) ===
        for (int64_t b = 0; b < numBlocks_; ++b) {
            int64_t gmOffset = b * sliceCount_ + sliceTile;

            // V→MTE2 同步
            TEventID eventIDV = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(eventIDV);
            WaitFlag<HardEvent::V_MTE2>(eventIDV);

            // 加载
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(padLen), 0};
            DataCopyPad(dataLocal, inputGM[gmOffset], copyParams, padParams);

            // MTE2→V 同步
            TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(eventID0);
            WaitFlag<HardEvent::MTE2_V>(eventID0);

            // Cast → FP32
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(alignedLen));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(alignedLen));
            }
            PipeBarrier<PIPE_V>();

            // 逐元素向量乘
            Mul(workLocal, workLocal, scaleLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();

            // Cast 回原 dtype
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(dataLocal, workLocal, static_cast<int32_t>(alignedLen));
            } else {
                Cast(dataLocal, workLocal, RoundMode::CAST_RINT, static_cast<int32_t>(alignedLen));
            }

            // V→MTE3 同步
            TEventID eventID1 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(eventID1);
            WaitFlag<HardEvent::V_MTE3>(eventID1);

            // 写回 outputGM
            DataCopyExtParams storeParams;
            storeParams.blockCount = 1;
            storeParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
            storeParams.srcStride = 0;
            storeParams.dstStride = 0;
            DataCopyPad(outputGM[gmOffset], dataLocal, storeParams);

            // MTE3→MTE2 同步
            TEventID eventID2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(eventID2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
        }
    }
}

} // namespace NsRenormSmSt

#endif // _RENORM_SM_ST_H_
