/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Template E: Block-Major Vector Direct (BM-VD)
// 适用条件: blockSize = 1 且 sliceCount ≥ coreNum
// 新增模板（无 CCE 对应），为 blockSize=1 场景的性能优化
// 遍历顺序: Block-Major (先遍历 block, 再遍历 sliceTile)
// 累加方式: 向量 Add (无 ReduceSum, 无标量同步)
// 两 Pass: Pass1 计算范数向量, Pass2 应用 scale

// ───────────── 小白入门解读 ─────────────
// 这个模板处理 blockSize=1 的特殊场景。回忆 renorm 的 GM 布局:
//   [numBlocks, sliceCount, blockSize]，当 blockSize=1 时退化为 [numBlocks, sliceCount]。
// 每一行(一个 block)有 sliceCount 个元素，renorm 要沿 sliceCount 维度算范数。
//
// 多核切分策略: sliceCount 足够大(≥核数)，所以把 sliceCount 维度切给多核，
//   每核负责一段连续的 slice；每核都要遍历全部 numBlocks 个 block。
//
// 为什么用"向量累加"而不是 ReduceSum?
//   普通做法是把一个 block 的 sliceCount 个元素 ReduceSum 归约成一个标量范数，
//   但归约成标量后多核之间难以同步、且 ReduceSum 有额外开销。
//   这里换思路: normLocal 是一个长度为 alignedLen(≈sliceTileLength) 的"范数向量"，
//   每来一个 block，把它的 sliceTile 个元素 Cast 到 FP32 后直接 Add 到 normLocal 对应位置。
//   遍历完 numBlocks 个 block 后，normLocal[i] 就是第 i 个 slice 的范数(累加值)。
//   全程是向量运算，没有标量归约，也没有跨 block 的标量同步。
//
// 两个 Pass:
//   Pass1: 遍历所有 block，把每个 block 的元素按 normMode 变换后向量 Add 到 normLocal → 得到范数向量
//   Pass2: 用 normLocal 算出 scale 向量，再遍历所有 block，把 输入*scale 写回输出

#ifndef _RENORM_BM_VD_H_
#define _RENORM_BM_VD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "renorm_tiling_key.h"
#include "common/renorm_common.h"

namespace NsRenormBmVd {

using namespace AscendC;

constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;

// Compare/Select 对齐要求: 32 字节 = 8 个 FP32 (与 common/renorm_common.h FP32_ALIGN 一致)
constexpr int64_t CMP_ALIGN = 8;

template <typename D_T_X>
class RenormBmVd {
public:
    __aicore__ inline RenormBmVd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RenormTilingData* tilingData);
    __aicore__ inline void Process();

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> dataBuf0;   // 输入数据加载 (双缓冲 0)
    TBuf<QuePosition::VECCALC> dataBuf1;   // 输入数据加载 (双缓冲 1)
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

    // Tiling 参数
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
__aicore__ inline void RenormBmVd<D_T_X>::Init(GM_ADDR x, GM_ADDR y, const RenormTilingData* tilingData)
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

    // Buffer 规划 (sliceTileLength 对齐到 64)
    int64_t typeSize = sizeof(D_T_X);
    int64_t alignedTile = (sliceTileLength_ + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;

    // 双缓冲(double buffering): 分配 dataBuf0/dataBuf1 两个同样大小的输入缓冲区。
    // 流水线中 MTE2(从 GM 加载数据到 UB) 和 V(向量计算) 是异步并行的:
    //   当 V 正在处理 dataBuf0 的数据时，MTE2 可以同时把下一批数据加载到 dataBuf1；
    //   下一轮两者交换角色。这样"加载"和"计算"重叠，隐藏了 GM 访存延迟。
    // 如果只用一个 buffer，V 必须等 MTE2 加载完才能算，MTE2 必须等 V 算完才能覆写，串行执行慢一倍。
    pipe.InitBuffer(dataBuf0, alignedTile * typeSize);
    pipe.InitBuffer(dataBuf1, alignedTile * typeSize);
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
__aicore__ inline void RenormBmVd<D_T_X>::Process()
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

    LocalTensor<D_T_X> dataLocal0 = dataBuf0.Get<D_T_X>();
    LocalTensor<D_T_X> dataLocal1 = dataBuf1.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> normLocal = normBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<float> maxNormLocal = maxNormBuf.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

    // 预初始化常量 buffer
    int64_t alignedTile = (sliceTileLength_ + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;
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

            Duplicate(dataLocal0, static_cast<D_T_X>(0), static_cast<int32_t>(alignedLen));
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
                DataCopyPad(outputGM[offset], dataLocal0, copyParams);

                TEventID eventID2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                SetFlag<HardEvent::MTE3_MTE2>(eventID2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
            }
        }
        return;
    }

    // 按 sliceTile 迭代处理
    for (int64_t sliceTile = startSlice; sliceTile < endSlice; sliceTile += sliceTileLength_) {
        int64_t currentSliceTile = (sliceTileLength_ < (endSlice - sliceTile)) ? sliceTileLength_ :
                                                                                 (endSlice - sliceTile);
        int64_t alignedLen = (currentSliceTile + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;

        // === Pass 1: 向量累加范数 (双缓冲) ===
        // "向量累加"概念: normLocal 是一个 alignedLen 长度的 FP32 向量(不是标量)。
        // 每个 block 的 sliceTile 个元素 Cast 到 FP32 后，直接 Add 到 normLocal 的对应位置。
        // 注意: 这里不用 ReduceSum 把一个 block 归约成标量，而是保持向量形态，
        // 让每个 normLocal[i] 独立累加第 i 个 slice 在所有 block 上的贡献。
        // 遍历完 numBlocks 个 block 后，normLocal[i] 即为第 i 个 slice 的范数(未开根号等后处理)。
        Duplicate(normLocal, 0.0f, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        // V→MTE2 双缓冲同步: 预分配 2 个事件 ID，分别对应 dataBuf0/dataBuf1。
        // V(MTE2)流水线异步执行: 当 V 正在对 buffer 做 Cast/Add 时，MTE2 不能覆写同一 buffer。
        // 规则: 第 b 轮要覆写 dataBuf[b%2] 前，必须等待第 b-2 轮的 V(读取该 buffer)完成，
        //       即 WaitFlag(v2mIds[b%2])。这样间隔 2 轮，保证 V 已读完。
        // 预分配 2 个 V→MTE2 事件 ID (双缓冲交替复用)
        TEventID v2mIds[2];
        v2mIds[0] = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
        v2mIds[1] = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);

        for (int64_t b = 0; b < numBlocks_; ++b) {
            // blockSize=1, GM layout: [numBlocks, sliceCount]
            // 双缓冲: 偶数用 dataLocal0, 奇数用 dataLocal1
            LocalTensor<D_T_X>& dataLocal = (b % 2 == 0) ? dataLocal0 : dataLocal1;
            int64_t gmOffset = b * sliceCount_ + sliceTile;

            // V→MTE2 同步: 等待 b-2 轮 V(Cast) 完成后再覆写同一缓冲区
            if (b >= 2) {
                WaitFlag<HardEvent::V_MTE2>(v2mIds[b % 2]);
            }

            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(currentSliceTile * sizeof(D_T_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            int64_t padLen = alignedLen - currentSliceTile;
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

            // V→MTE2 信号: Cast(缓冲区读取) 已完成, b+2 可安全覆写
            SetFlag<HardEvent::V_MTE2>(v2mIds[b % 2]);

            // normMode 分支: 对 workLocal 做不同变换，为后续向量累加(Add)或取最大(Max)做准备。
            // 三个分支对应三种范数定义，全部用向量指令逐元素处理。
            if (normMode_ == NORM_MODE_P_ZERO) {
                // P_ZERO (p=0): 范数 = 非零元素个数。
                // 思路: 先 Abs，再 Compare(>0) 生成 mask，最后 Select 把"真(非零)"映射成 1.0、"假(零)"映射成 0.0。
                // 这样 workLocal 变成 0/1 向量，后面 Add 累加就是计数非零个数。
                // P_ZERO: 非零计数
                Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Compare(maskLocal, workLocal, zerosLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                // Select: 1.0 where > 0, else 0.0
                Select(workLocal, maskLocal, onesLocal, zerosLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                       static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            } else if (normMode_ == NORM_MODE_P_INF) {
                // P_INF (p=∞): 范数 = 最大绝对值。
                // 用逐元素 Max 直接更新 normLocal: normLocal[i] = max(normLocal[i], |x[i]|)。
                // 因为是取最大值而非求和，所以这里直接 continue，不走后面的 Add 累加。
                // P_INF: 取 max (用 Maxs 逐元素更新)
                Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                // 向量 Max 累加
                Max(normLocal, normLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                continue; // P_INF 不用 Add
            } else {
                // P_POSITIVE (p>0): 范数 = (Σ |x|^p)^(1/p)。
                // 先 Abs 取绝对值，再按 p 的值选最快路径:
                //   p=1: |x| 本身就是贡献，直接累加(下面 Add)
                //   p=2: |x|^2 = x*x，用 Mul 自乘(比通用 pow 快)
                //   通用 p: |x|^p = exp(p * log(|x|))，先 Maxs 防止 log(0)，再 Log→Muls(p)→Exp
                // P_POSITIVE
                Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                if (p_ == 1.0f) {
                    // 直接累加 |x|
                } else if (p_ == 2.0f) {
                    Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                } else if (p_ >= 16.0f && p_ <= 32.0f && p_ == static_cast<float>(static_cast<int32_t>(p_))) {
                    // Integer high-order powers are evaluated by vector
                    // multiplication instead of Log/Exp. This preserves the
                    // reference FP32 overflow boundary while leaving the
                    // generic path unchanged for other p values.
                    DataCopy(tmpLocal, workLocal, static_cast<int32_t>(alignedLen));
                    Duplicate(workLocal, 1.0f, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                    int32_t power = static_cast<int32_t>(p_);
                    while (power > 0) {
                        if ((power & 1) != 0) {
                            Mul(workLocal, workLocal, tmpLocal, static_cast<int32_t>(alignedLen));
                            PipeBarrier<PIPE_V>();
                        }
                        power >>= 1;
                        if (power > 0) {
                            Mul(tmpLocal, tmpLocal, tmpLocal, static_cast<int32_t>(alignedLen));
                            PipeBarrier<PIPE_V>();
                        }
                    }
                } else {
                    // Generic p: |x|^p = exp(p * log(|x|))
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

        // 消费 Pass1 剩余 V→MTE2 事件: 确保 V(Cast) 完成后再进入 Pass2 覆写缓冲区
        WaitFlag<HardEvent::V_MTE2>(v2mIds[(numBlocks_ - 1) % 2]);
        if (numBlocks_ >= 2) {
            WaitFlag<HardEvent::V_MTE2>(v2mIds[(numBlocks_ - 2) % 2]);
        }

        // 后处理: Sqrt/Pow 向量化
        if (normMode_ == NORM_MODE_P_POSITIVE) {
            if (p_ == 2.0f) {
                Sqrt(normLocal, normLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            } else if (p_ != 1.0f) {
                // norm = (sum |x|^p)^(1/p) = exp((1/p) * log(sum))
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
        // 目标: scale = (norm > maxNorm) ? maxNorm / max(norm, eps) : 1.0
        // 这是逐元素的条件表达式，用向量指令模拟"if-else"，避免标量分支:
        //   1) Maxs: tmp = max(norm, eps)            ← 防止除零
        //   2) Reciprocal: tmp = 1 / tmp
        //   3) Muls: scale = tmp * maxNorm           ← 此时 scale = maxNorm/max(norm,eps)
        //   4) Compare: mask = (norm > maxNorm)      ← 需要缩放的位置为真
        //   5) Select: scale = mask ? scale : 1.0    ← 超过 maxNorm 用 scale，否则用 1.0(不缩放)
        // 注意: 第 1-3 步对所有元素都算了 maxNorm/max(norm,eps)，第 5 步 Select 才真正"挑选"。
        // scale = (norm > maxNorm) ? maxNorm / max(norm, eps) : 1.0
        Maxs(tmpLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Reciprocal(tmpLocal, tmpLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Muls(scaleLocal, tmpLocal, maxNorm_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        // Compare: norm > maxNorm → mask
        Compare(maskLocal, normLocal, maxNormLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        // Select: mask ? scale : 1.0
        Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        // === Pass 2: 向量化应用 scale (双缓冲) ===
        // Pass2 再次遍历所有 block，把每个 block 的输入读出来，乘以 scaleLocal(向量) 后写回输出。
        // 单个 block 的处理流水: MTE2加载 → Cast(FP32) → Mul(scale) → Cast(回原dtype) → MTE3写回。
        // 同样用 dataBuf0/dataBuf1 双缓冲: 当 MTE3 正在写 dataBuf0 时，MTE2 可同时加载到 dataBuf1。
        // 同步用 MTE3→MTE2 事件: 第 b 轮覆写 buffer 前等第 b-2 轮的 MTE3(写)完成。
        // 预分配 2 个 MTE3→MTE2 事件 ID (双缓冲交替复用)
        TEventID m3m2Ids[2];
        m3m2Ids[0] = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        m3m2Ids[1] = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);

        for (int64_t b = 0; b < numBlocks_; ++b) {
            // 双缓冲: 偶数用 dataLocal0, 奇数用 dataLocal1
            LocalTensor<D_T_X>& dataLocal = (b % 2 == 0) ? dataLocal0 : dataLocal1;
            int64_t gmOffset = b * sliceCount_ + sliceTile;

            // MTE3→MTE2 同步: 等待 b-2 轮 MTE3(store) 完成后再覆写同一缓冲区
            if (b >= 2) {
                WaitFlag<HardEvent::MTE3_MTE2>(m3m2Ids[b % 2]);
            }

            // 加载
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(currentSliceTile * sizeof(D_T_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            int64_t padLen = alignedLen - currentSliceTile;
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
            storeParams.blockLen = static_cast<uint32_t>(currentSliceTile * sizeof(D_T_X));
            storeParams.srcStride = 0;
            storeParams.dstStride = 0;
            DataCopyPad(outputGM[gmOffset], dataLocal, storeParams);

            // MTE3→MTE2 信号: store 已完成, b+2 可安全覆写此缓冲区
            SetFlag<HardEvent::MTE3_MTE2>(m3m2Ids[b % 2]);
        }

        // 消费 Pass2 剩余 MTE3→MTE2 事件: 确保所有 store 完成
        WaitFlag<HardEvent::MTE3_MTE2>(m3m2Ids[(numBlocks_ - 1) % 2]);
        if (numBlocks_ >= 2) {
            WaitFlag<HardEvent::MTE3_MTE2>(m3m2Ids[(numBlocks_ - 2) % 2]);
        }
    }
}

} // namespace NsRenormBmVd

#endif // _RENORM_BM_VD_H_
