/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// Template B: Slice-Major Two-Level (SM-TL) — renorm 算子的高精度模板
// ============================================================================
//
// 【触发条件】(由 tiling 层 ChooseHighPrecision 决定):
//   1. blockSize == 1  (归一化维度 dim 是最后一维，dim 后面没有更多维度)
//   2. numBlocks >= sliceCount  (归约轴 R 足够大，沿 R 轴分核并行度更高)
//   3. sliceCount > 16  (workspace 至少 > 1 条 cache line，保证跨核可见性可靠)
//
// 【输入布局】GM layout (blockSize=1): [numBlocks, sliceCount]
//   - numBlocks = prod(shape[:dim])  (需要做归约的 "块" 数，即 R 轴)
//   - sliceCount = shape[dim]        (每块的归一化维度长度，即 A 轴)
//   - 每个块独立计算 norm，再根据 maxNorm 计算 scale 并应用
//
// 【三阶段流水线架构】
//
//   ┌─────────────────────────────────────────────────────────────┐
//   │  Pre-Pass1: core 0 清零 workspace norm slot                 │
//   └──────────────────────┬──────────────────────────────────────┘
//                          ▼ SyncAll (所有核看到清零完成)
//   ┌─────────────────────────────────────────────────────────────┐
//   │  Pass1: 每核处理自己负责的 blocks                            │
//   │  逐块: load → |x|^p → Add/Max 累加到 normLocal              │
//   │  SetAtomicAdd/Max: normLocal → workspaceGM[0..wsStride]     │
//   │  (所有核原子累加到同一个 slot，跨 cluster 可见性由硬件保证)   │
//   └──────────────────────┬──────────────────────────────────────┘
//                          ▼ SyncAll (所有核原子写完成)
//   ┌─────────────────────────────────────────────────────────────┐
//   │  Pass2 (仅 core 0): 读取聚合 norm → 计算 scale              │
//   │  → scale = (norm_root > maxNorm) ? maxNorm/norm_root : 1    │
//   │  → 写入 scale 到 scaleGM                                     │
//   └──────────────────────┬──────────────────────────────────────┘
//                          ▼ SyncAll + DCache flush
//   ┌─────────────────────────────────────────────────────────────┐
//   │  Pass3: 所有核读取 scaleGM → 逐块 load × scale → store       │
//   └─────────────────────────────────────────────────────────────┘
//
// 【Workspace 布局】(在 16MB 系统 workspace 之后)
//
//   偏移 0:  [norm slot]         ← Pass1 SetAtomicAdd 输出, Pass2 输入
//   偏移 S:  [scale factors]      ← Pass2 输出, Pass3 输入
//
//   slot 大小 = wsStride = ceil(sliceCount / 16) * 16  (64B 对齐)
//   S = wsStride  (scale 区域起始偏移)
//
// 【为什么用 SetAtomicAdd?】
//   之前的 per-core slot 方案中，每核用普通 DataCopy 写自己的 slot，
//   数据经过 L1 DCache，DataCacheCleanAndInvalid 无法保证跨 cluster L2 cache
//   可见性 (nb>16 时跨 cluster，core 0 读到旧数据)。
//   SetAtomicAdd 的原子写直接到达 GM/L2，跨 cluster 可见性由硬件保证
//   (参考 lp_norm_v3)。64B 对齐的 slot 保证多核写同一条 cache line 时
//   不会 partial write 竞态。

#ifndef _RENORM_SM_TL_STABLE_H_
#define _RENORM_SM_TL_STABLE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "renorm_tiling_key.h"
#include "common/renorm_common.h"

namespace NsRenormSmTlStable {

using namespace AscendC;

// ============================================================================
// 【常量定义】对齐与同步相关常量
// ============================================================================

// normMode: 归一化模式 (由 tiling 层根据 p 值推导)
// - P_POSITIVE: p > 0, 计算 ||x||_p = (sum |x_i|^p)^(1/p)
// - P_ZERO:     p = 0, norm = 非零元素个数
// - P_INF:      p = inf, norm = max(|x_i|)
// - MAXNORM_ZERO: 特殊模式，直接输出全零 (maxNorm=0 时)
constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;

// CMP_ALIGN: Compare/Select 向量指令的最小对齐粒度 (32B = 8 个 FP32)
// DataCopyPad 的 rightPadding 对 FP32 最大为 7，CMP_ALIGN=8 保证 padLen <= 7
constexpr int64_t CMP_ALIGN = 8;

// ATOMIC_ALIGN: workspace cache line 对齐粒度 (64B = 16 个 FP32)
// 多核写 workspace 时，每个 slot 必须按 64B 对齐，否则同一条 cache line
// 上的数据可能被不同核同时修改，导致数据不一致 (参考 lp_norm_v3 SLOT_STRIDE=16)
constexpr int64_t ATOMIC_ALIGN = 16;

template <typename D_T_X>
__aicore__ inline int64_t AlignDataElements(int64_t length)
{
    constexpr int64_t DATA_ALIGN = 32 / sizeof(D_T_X);
    return (length + DATA_ALIGN - 1) / DATA_ALIGN * DATA_ALIGN;
}

__aicore__ inline void TreeReduceBatchRows(LocalTensor<float> data, LocalTensor<float> accumulator, int64_t rowCount,
                                           int64_t rowLength)
{
    // Keep the active rows contiguous at the front of data. For an odd row
    // count, the middle row is retained for the next level.
    int64_t activeRows = rowCount;
    while (activeRows > 1) {
        int64_t pairRows = activeRows / 2;
        int64_t upperStart = activeRows - pairRows;
        int64_t addElements = pairRows * rowLength;
        Add(data, data, data[static_cast<int32_t>(upperStart * rowLength)], static_cast<int32_t>(addElements));
        PipeBarrier<PIPE_V>();
        activeRows = upperStart;
    }
    Add(accumulator, accumulator, data, static_cast<int32_t>(rowLength));
}

__aicore__ inline void TreeReduceBatchRowsMax(LocalTensor<float> data, LocalTensor<float> accumulator, int64_t rowCount,
                                              int64_t rowLength)
{
    int64_t activeRows = rowCount;
    while (activeRows > 1) {
        int64_t pairRows = activeRows / 2;
        int64_t upperStart = activeRows - pairRows;
        Max(data, data, data[static_cast<int32_t>(upperStart * rowLength)], static_cast<int32_t>(pairRows * rowLength));
        PipeBarrier<PIPE_V>();
        activeRows = upperStart;
    }
    Max(accumulator, accumulator, data, static_cast<int32_t>(rowLength));
}

// ============================================================================
// 【跨核同步常量】(参考 foreach_norm, conv3d_backprop)
// ============================================================================
// SYNC_MODE0: 向所有其他核发送 flag (barrier 模式)
constexpr uint8_t SYNC_MODE0 = 0;
// 三个同步点的 flag ID (必须唯一，范围 0-15)
// 注: 当前实现使用 SyncAll() 全局屏障，这些 flag 常量保留供未来优化使用
constexpr uint16_t FLAG_PRE_PASS1_DONE = 6; // Pre-Pass1 完成: workspace 已清零
constexpr uint16_t FLAG_PASS1_DONE = 7;     // Pass1 完成: 所有核已写入 norm slot
constexpr uint16_t FLAG_PASS2_DONE = 8;     // Pass2 完成: scale 已写入 scaleGM

template <typename D_T_X>
class RenormSmTlStable {
public:
    __aicore__ inline RenormSmTlStable() {}
    // TPipe 外置: 由调用方传入 TPipe 指针，触发 Scalar 常量折叠/传播编译优化
    // (TPipe 在类内会阻止编译器对类成员变量做常量传播，导致 Scalar 指令增多)
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    TPipe* pipe = nullptr; // 外部传入的 TPipe 指针 (不在类内构造)

    // --- UB (Unified Buffer) 局部内存 ---
    // UB 是片上高速缓存 (192KB)，所有 Vector 计算都在 UB 上进行。
    // 数据流: GM →(MTE2)→ UB →(V 计算)→ UB →(MTE3)→ GM

    // [数据搬运缓冲区] — 用于 GM ↔ UB 之间的数据搬运
    TBuf<QuePosition::VECCALC> dataBuf0; // 输入数据加载 (批量模式主缓冲区, batchSize_ * alignedTile)
    TBuf<QuePosition::VECCALC> dataBuf1; // 输入数据加载 (双缓冲后备, 1 block)
    TBuf<QuePosition::VECCALC> workBuf;  // FP32 工作缓冲区 (FP16 Cast 目标; FP32 批量模式不用)
    TBuf<QuePosition::VECCALC> workBuf1; // FP32 工作缓冲区 (双缓冲后备, 1 block)

    // [计算缓冲区] — 用于归约和 scale 计算
    TBuf<QuePosition::VECCALC> normBuf;  // norm 累加器 (Pass1: 累加 |x|^p; Pass2: 聚合多核 norm)
    TBuf<QuePosition::VECCALC> scaleBuf; // scale 因子 (Pass2 计算结果, Pass3 使用)
    TBuf<QuePosition::VECCALC> maskBuf;  // Compare 掩码 (uint8_t, 用于 Select 条件选择)
    TBuf<QuePosition::VECCALC> tmpBuf;   // 临时缓冲区 (Pass2 批量读取 + Reciprocal 等)

    // [常量缓冲区] — 预填充的常量值，用于向量化条件判断
    TBuf<QuePosition::VECCALC> zerosBuf;   // 全零向量 (用于 Compare 阈值)
    TBuf<QuePosition::VECCALC> onesBuf;    // 全一向量 (用于 Select 默认值)
    TBuf<QuePosition::VECCALC> maxNormBuf; // maxNorm 广播向量 (用于 Compare: norm > maxNorm?)

    // --- GM (Global Memory) 全局内存 ---
    GlobalTensor<D_T_X> inputGM;     // 输入张量 x
    GlobalTensor<D_T_X> outputGM;    // 输出张量 y
    GlobalTensor<float> workspaceGM; // 用户 workspace: norm slot (SetAtomicAdd) + scale factors

    // --- Tiling 参数 (由 host 侧 tiling 计算传入) ---
    int64_t totalElements_ = 0;   // 总元素数 = numBlocks * sliceCount
    int64_t sliceCount_ = 0;      // 归一化维度 (A 轴) 长度
    int64_t numBlocks_ = 0;       // 归约块数 (R 轴) = prod(shape[:dim])
    int64_t sliceTileLength_ = 0; // sliceCount 方向的 tile 大小 (UB 切分)
    int64_t blocksPerCore_ = 0;   // 每核负责的 block 数 = ceil(numBlocks / coreNum)
    float p_ = 0.0f;              // 范数阶数 (1, 2, 或任意正数)
    float maxNorm_ = 0.0f;        // 最大范数阈值
    float eps_ = 0.0f;            // 数值稳定小量 (防止除零)
    int32_t normMode_ = 0;        // 归一化模式 (见 NORM_MODE_* 常量)

    // --- 运行时派生参数 ---
    int64_t coreNum_ = 0;       // GetBlockNum() (= SetBlockDim)
    int64_t usedCoreNum_ = 0;   // 有数据的核数 = ceil(numBlocks / blocksPerCore)
    int64_t blockIdx_ = 0;      // 当前核编号
    int64_t blockStart_ = 0;    // 当前核负责的起始 block = blockIdx * blocksPerCore
    int64_t blockEnd_ = 0;      // 当前核负责的结束 block (不超过 numBlocks)
    int64_t scaleGmOffset_ = 0; // scale 区域在 workspace 中的偏移 (元素数)
    int64_t wsStride_ = 0;      // 每核 workspace slot 步长 (元素数, 64B 对齐)
    int64_t batchSize_ = 1;     // 批量处理大小 (Pass1/Pass3 一次搬运多少个 block)
};

// ============================================================================
// Init: 初始化函数 — 解析 tiling 参数、设置 GM/Workspace、规划 UB Buffer
// ============================================================================
// 调用时机: 每个核启动时调用一次，在 Process() 之前
// 主要工作:
//   1. 从 tilingData 读取形状和算法参数
//   2. 设置 GM (输入/输出/workspace) 的全局缓冲区
//   3. 计算每核负责的 block 范围和 workspace 布局
//   4. 根据 UB 容量动态计算 batchSize_ 并分配所有 UB buffer
// ============================================================================
template <typename D_T_X>
__aicore__ inline void RenormSmTlStable<D_T_X>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                     const RenormTilingData* tilingData, TPipe* pipeIn)
{
    pipe = pipeIn;
    // --- 步骤1: 读取 tiling 参数 ---
    totalElements_ = tilingData->totalElements;
    sliceCount_ = tilingData->sliceCount;
    numBlocks_ = tilingData->numBlocks;
    sliceTileLength_ = tilingData->sliceTileLength;
    blocksPerCore_ = tilingData->reduceSplitsPerCore;
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;

    if (totalElements_ == 0 || sliceCount_ == 0 || numBlocks_ == 0 || sliceTileLength_ == 0) {
        return;
    }

    // --- 步骤2: 设置 Workspace ---
    // workspace 前 16MB 是系统 workspace (供 SyncAll 等系统内建函数使用)
    // 之后是用户 workspace (norm slot + scale factors, SetAtomicAdd 方案)
    SetSysWorkspace(workspace);
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }

    // --- 步骤3: 绑定 GM 全局缓冲区 ---
    inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);
    // workspaceSize 是用户 workspace 大小 (不含 16MB 系统 workspace)
    workspaceGM.SetGlobalBuffer((__gm__ float*)userWs, tilingData->workspaceSize / sizeof(float));

    // --- 步骤4: 计算每核负责的 block 范围 ---
    // 多核沿 R 轴 (numBlocks) 分核: core i 负责 [i*blocksPerCore, (i+1)*blocksPerCore)
    blockIdx_ = GetBlockIdx();
    coreNum_ = GetBlockNum();
    blockStart_ = blockIdx_ * blocksPerCore_;
    blockEnd_ = blockStart_ + blocksPerCore_;
    if (blockEnd_ > numBlocks_) {
        blockEnd_ = numBlocks_; // 最后一核可能不满
    }
    // usedCoreNum: 实际有数据的核数 (空核不参与计算，但 SyncAll 需要全部核参与)
    usedCoreNum_ = (numBlocks_ + blocksPerCore_ - 1) / blocksPerCore_;

    // --- 步骤5: 计算 workspace 布局 ---
    // wsStride: workspace slot 步长，按 64B (16 FP32) 对齐
    // 64B 对齐保证多核 SetAtomicAdd 写同一条 cache line 时不会 partial write 竞态
    // (参考 lp_norm_v3 SLOT_STRIDE=16: "多核在64B内同时操作会导致随机覆写")
    wsStride_ = (sliceCount_ + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;

    // Workspace 布局 (SetAtomicAdd 方案，参考 lp_norm_v3):
    //   [0, wsStride):              norm slot (Pass1 SetAtomicAdd 输出, Pass2 输入)
    //   [wsStride, 2*wsStride):     scale factors (Pass2 输出, Pass3 输入)
    //
    // SetAtomicAdd 方案: 所有核用 SetAtomicAdd/Max 原子累加到同一个 norm slot。
    // SetAtomicAdd 的原子写直接到达 GM/L2，跨 cluster 可见性由硬件保证。
    // (之前 per-core slot 方案的普通 DataCopy 写经过 L1 DCache，
    //  DataCacheCleanAndInvalid 无法保证跨 cluster L2 cache 可见性)
    scaleGmOffset_ = wsStride_; // scale 区域起始偏移

    // --- 步骤6: UB Buffer 规划 ---
    // UB 总容量 192KB，需要容纳所有局部 buffer。
    // buffer 大小基于 actualTileLen = min(sliceTileLength, sliceCount)
    // (当 sc < sliceTileLength 时，每 tile 只处理 sc 个元素，不需要分配 sliceTileLength 大小)

    int64_t typeSize = sizeof(D_T_X);
    int64_t actualTileLen = (sliceTileLength_ < sliceCount_) ? sliceTileLength_ : sliceCount_;
    // GM/UB data moves need a full 32-byte line.  For FP16/BF16 this is
    // 16 elements; using the FP32 width (8 elements) leaves an unaligned
    // MTE transaction when sliceCount is not a multiple of 16.
    int64_t alignedTile = AlignDataElements<D_T_X>(actualTileLen);
    // atomicTile: workspace cache line 对齐 (ATOMIC_ALIGN=16)，用于 normBuf/scaleBuf/tmpBuf
    int64_t atomicTile = (actualTileLen + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;

    // Pass2 批量读取: 一次 DataCopy 读取多个 slot，减少 MTE2→V 同步次数
    // (core 0 在 Pass2 需要串行读取 usedCoreNum_ 个 slot)
    constexpr int64_t PASS2_MAX_BATCH = 8;

    // 计算固定 buffer 大小 (不随 batchSize_ 变化的 buffer)
    // 注意: dataBuf0/dataBuf1/workBuf(FP16)/workBuf1(FP16) 是 batchSize_ 相关的，
    //       在 perBatchBytes 中计算，不计入 fixedBytes
    int64_t fixedBytes = 0;
    if constexpr (sizeof(D_T_X) == sizeof(float)) {
        // FP32: workBuf 和 workBuf1 仅用于单 block 路径，固定大小
        fixedBytes += alignedTile * sizeof(float); // workBuf
        fixedBytes += alignedTile * sizeof(float); // workBuf1
    }
    // FP16: workBuf 和 workBuf1 随 batchSize_ 变化，在 perBatchBytes 中计算
    fixedBytes += atomicTile * sizeof(float);                   // normBuf (需 atomicLen 做 workspace 读写)
    fixedBytes += atomicTile * sizeof(float);                   // scaleBuf (需 atomicLen 做 workspace 读写)
    fixedBytes += alignedTile;                                  // maskBuf (uint8_t, 1 byte/element)
    fixedBytes += alignedTile * sizeof(float);                  // zerosBuf
    fixedBytes += alignedTile * sizeof(float);                  // onesBuf
    fixedBytes += alignedTile * sizeof(float);                  // maxNormBuf
    fixedBytes += PASS2_MAX_BATCH * atomicTile * sizeof(float); // tmpBuf (Pass2 批量读取, 需 atomicLen)

    // 动态计算 batchSize_ — Pass3 双缓冲需要 2 个等大的 data buffer
    // 批量双缓冲: 交替使用 dataBuf0/dataBuf1，让下一批 MTE2 load 与当前批 MTE3 store 重叠
    //
    // FP32: 需要 2 个 data buffer (dataBuf0 + dataBuf1)，不需要 workBuf
    // FP16: 需要 2 个 data buffer + 2 个 work buffer
    constexpr int64_t UB_SIZE_BYTES = 192 * 1024;
    constexpr int64_t RESERVED_UB_BYTES = 4 * 1024;
    int64_t availableUb = UB_SIZE_BYTES - RESERVED_UB_BYTES - fixedBytes;
    int64_t perBatchBytes;
    if constexpr (sizeof(D_T_X) == sizeof(float)) {
        perBatchBytes = 2 * alignedTile * typeSize; // FP32: dataBuf0 + dataBuf1
    } else {
        perBatchBytes = 2 * alignedTile * (typeSize + static_cast<int64_t>(sizeof(float)));
    }
    batchSize_ = availableUb / perBatchBytes;
    if (batchSize_ > 1024) {
        batchSize_ = 1024;
    } // 上限保护
    if (batchSize_ < 1) {
        batchSize_ = 1;
    }
    // DataCopyPad encodes a strided source distance in 32-byte units. A
    // non-32-byte logical row cannot use blockCount > 1 without truncating
    // the row gap; keep this stable path on its bounds-safe single-row route.
    if ((sliceCount_ * typeSize) % 32 != 0) {
        batchSize_ = 1;
    }

    // --- 步骤7: 分配所有 UB buffer ---
    // Pass3 双缓冲: dataBuf0 和 dataBuf1 等大，交替使用
    pipe->InitBuffer(dataBuf0, batchSize_ * alignedTile * typeSize);
    pipe->InitBuffer(dataBuf1, batchSize_ * alignedTile * typeSize);
    if constexpr (sizeof(D_T_X) == sizeof(float)) {
        // FP32: workBuf 仅用于 Pass1/Pass3 单 block 路径，分配最小大小
        pipe->InitBuffer(workBuf, alignedTile * sizeof(float));
    } else {
        // FP16: workBuf 和 workBuf1 需要容纳整个 batch 的 Cast 结果 (双缓冲)
        pipe->InitBuffer(workBuf, batchSize_ * alignedTile * sizeof(float));
    }
    if constexpr (sizeof(D_T_X) == sizeof(float)) {
        pipe->InitBuffer(workBuf1, alignedTile * sizeof(float));
    } else {
        pipe->InitBuffer(workBuf1, batchSize_ * alignedTile * sizeof(float));
    }
    pipe->InitBuffer(normBuf, atomicTile * sizeof(float));
    pipe->InitBuffer(scaleBuf, atomicTile * sizeof(float));
    pipe->InitBuffer(maskBuf, alignedTile); // uint8_t, 1 byte per element
    pipe->InitBuffer(zerosBuf, alignedTile * sizeof(float));
    pipe->InitBuffer(onesBuf, alignedTile * sizeof(float));
    pipe->InitBuffer(maxNormBuf, alignedTile * sizeof(float));
    pipe->InitBuffer(tmpBuf, PASS2_MAX_BATCH * atomicTile * sizeof(float));
}

// ============================================================================
// LoadAndCast: 从 GM 加载数据到 UB，并 Cast 为 FP32
// ============================================================================
// 数据流: GM →(MTE2 DataCopyPad)→ dataLocal →(V Cast/DataCopy)→ workLocal
//
// 流水线同步: MTE2 → V (SetFlag/WaitFlag)
//   MTE2 (外部存储→UB搬运) 和 V (向量计算) 是异步的，需要显式同步:
//   SetFlag<HardEvent::MTE2_V>(id) — 标记 MTE2 已完成
//   WaitFlag<HardEvent::MTE2_V>(id) — 等待 MTE2 完成后 V 才能读 dataLocal
//
// FP32 优化: sizeof(D_T_X)==sizeof(float) 时用 DataCopy (无精度损失)，
//            否则用 Cast (FP16→FP32，RoundMode::CAST_NONE 不做舍入)
// ============================================================================
template <typename D_T_X>
__aicore__ inline void LoadAndCast(LocalTensor<float>& workLocal, LocalTensor<D_T_X>& dataLocal,
                                   GlobalTensor<D_T_X>& inputGM, int64_t gmOffset, int64_t currentTile,
                                   int64_t alignedLen)
{
    // MTE2: 从 GM 搬运数据到 UB (DataCopyPad 支持非 32B 对齐的数据)
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    int64_t padLen = alignedLen - currentTile;
    DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(padLen), 0};
    DataCopyPad(dataLocal, inputGM[gmOffset], copyParams, padParams);

    // 同步: 等待 MTE2 搬运完成，V 才能读取 dataLocal
    TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(eventID0);
    WaitFlag<HardEvent::MTE2_V>(eventID0);

    // V: 将 dataLocal (可能是 FP16/FP32) 转换为 FP32 workLocal
    if constexpr (sizeof(D_T_X) == sizeof(float)) {
        DataCopy(workLocal, dataLocal, static_cast<int32_t>(alignedLen));
    } else {
        Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(alignedLen));
    }
    PipeBarrier<PIPE_V>();
}

// ============================================================================
// CastBackAndStore: 将 FP32 workLocal 转回 D_T_X 并存储到 GM
// ============================================================================
// 数据流: workLocal →(V Cast/DataCopy)→ dataLocal →(MTE3 DataCopyPad)→ GM
//
// 流水线同步:
//   1. V → MTE3: V 计算完成后，MTE3 才能搬运 dataLocal 到 GM
//   2. MTE3 → MTE2: MTE3 存储完成后，才能开始下一次 MTE2 加载
//      (确保 UB 中的 dataLocal 不会被下一次加载覆盖)
// ============================================================================
template <typename D_T_X>
__aicore__ inline void CastBackAndStore(LocalTensor<float>& workLocal, LocalTensor<D_T_X>& dataLocal,
                                        GlobalTensor<D_T_X>& outputGM, int64_t gmOffset, int64_t currentTile,
                                        int64_t alignedLen)
{
    // V: FP32 → D_T_X (FP32 直接 DataCopy，FP16 用 Cast+RINT 四舍五入)
    if constexpr (sizeof(D_T_X) == sizeof(float)) {
        DataCopy(dataLocal, workLocal, static_cast<int32_t>(alignedLen));
    } else {
        Cast(dataLocal, workLocal, RoundMode::CAST_RINT, static_cast<int32_t>(alignedLen));
    }

    // 同步: V → MTE3 (V 完成后 MTE3 才能搬运)
    TEventID eventID1 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(eventID1);
    WaitFlag<HardEvent::V_MTE3>(eventID1);

    // MTE3: 从 UB 搬运到 GM
    DataCopyExtParams storeParams;
    storeParams.blockCount = 1;
    storeParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
    storeParams.srcStride = 0;
    storeParams.dstStride = 0;
    DataCopyPad(outputGM[gmOffset], dataLocal, storeParams);

    // 同步: MTE3 → MTE2 (确保 store 完成后才能开始下一次 load)
    TEventID eventID2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
}

// ============================================================================
// Process: 主处理函数 — 三阶段流水线的核心逻辑
// ============================================================================
// 执行顺序 (每核独立执行，通过 SyncAll 在阶段间同步):
//
//   1. 初始化常量 buffer (zeros, ones, maxNorm)
//   2. [特判] MAXNORM_ZERO 模式: 直接输出全零，跳过所有 Pass
//   3. Pre-Pass1: 每核并行清零自己的 workspace slot
//      → SyncAll
//   4. Pass1: 每核遍历自己的 blocks，计算 |x|^p 并累加到 normLocal
//      → 写入自己的 workspace slot
//      → SyncAll + DCache flush
//   5. Pass2 (仅 core 0): 读取所有核的 slot 聚合 → 计算 scale → 写入 scaleGM
//      → SyncAll + DCache flush
//   6. Pass3: 所有核读取 scaleGM → 逐块 load × scale → store 到 outputGM
// ============================================================================
template <typename D_T_X>
__aicore__ inline void RenormSmTlStable<D_T_X>::Process()
{
    if (totalElements_ == 0 || sliceCount_ == 0 || numBlocks_ == 0 || sliceTileLength_ == 0) {
        return;
    }

    // --- 获取所有 UB buffer 的 LocalTensor 引用 ---
    LocalTensor<D_T_X> dataLocal0 = dataBuf0.Get<D_T_X>();
    LocalTensor<D_T_X> dataLocal1 = dataBuf1.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> workLocal1 = workBuf1.Get<float>();
    LocalTensor<float> normLocal = normBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<float> maxNormLocal = maxNormBuf.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

    // actualTileLen: 实际每 tile 处理的元素数 = min(sliceTileLength, sliceCount)
    // (当 sliceCount < sliceTileLength 时，一次就处理完整个 slice)
    int64_t actualTileLen = (sliceTileLength_ < sliceCount_) ? sliceTileLength_ : sliceCount_;
    int64_t alignedTile = AlignDataElements<D_T_X>(actualTileLen);

    // --- 预初始化常量 buffer (用于后续 Compare/Select 向量条件判断) ---
    // zerosLocal:  全零，用于 Compare(x > 0) 的阈值
    // onesLocal:   全一，用于 Select 的默认值 (scale=1 表示不缩放)
    // maxNormLocal: maxNorm 广播，用于 Compare(norm > maxNorm?) 判断是否需要缩放
    Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(alignedTile));
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(alignedTile));
    Duplicate(maxNormLocal, maxNorm_, static_cast<int32_t>(alignedTile));
    PipeBarrier<PIPE_V>();

    // --- [特判] MAXNORM_ZERO: maxNorm=0 时直接输出全零，跳过所有 Pass ---
    if (normMode_ == NORM_MODE_MAXNORM_ZERO) {
        for (int64_t b = blockStart_; b < blockEnd_; ++b) {
            for (int64_t sliceTile = 0; sliceTile < sliceCount_; sliceTile += sliceTileLength_) {
                int64_t currentTile = (sliceTileLength_ < (sliceCount_ - sliceTile)) ? sliceTileLength_ :
                                                                                       (sliceCount_ - sliceTile);
                int64_t alignedLen = AlignDataElements<D_T_X>(currentTile);

                // V: 生成全零 → MTE3: 存储到 GM
                Duplicate(dataLocal0, static_cast<D_T_X>(0), static_cast<int32_t>(alignedLen));
                TEventID eventID1 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                SetFlag<HardEvent::V_MTE3>(eventID1);
                WaitFlag<HardEvent::V_MTE3>(eventID1);

                DataCopyExtParams storeParams;
                storeParams.blockCount = 1;
                storeParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
                storeParams.srcStride = 0;
                storeParams.dstStride = 0;
                DataCopyPad(outputGM[b * sliceCount_ + sliceTile], dataLocal0, storeParams);

                TEventID eventID2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                SetFlag<HardEvent::MTE3_MTE2>(eventID2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
            }
        }
        return;
    }

    // ========================================================================
    // Pre-Pass1: core 0 清零 workspace norm slot
    // ========================================================================
    // SetAtomicAdd 方案: 所有核原子累加到同一个 slot，初始值必须为 0。
    // 只有 core 0 清零即可 (只有 1 个 slot)。
    // 参考 lp_norm_v3: SyncAll 后所有核才开始 SetAtomicAdd 写入。
    // ========================================================================
    if (blockIdx_ == 0) {
        // V: 生成全零向量
        Duplicate(normLocal, 0.0f, static_cast<int32_t>(wsStride_));
        PipeBarrier<PIPE_V>();
        // V → MTE3 同步
        TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);
        // MTE3: 写零到 norm slot (workspaceGM[0])
        DataCopy(workspaceGM[0], normLocal, static_cast<int32_t>(wsStride_));
        // MTE3 → V 同步 (确保 MTE3 完成后 normLocal 可重用)
        TEventID eventIDMte3V = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(eventIDMte3V);
        WaitFlag<HardEvent::MTE3_V>(eventIDMte3V);
    }
    // 全局屏障: 确保所有核完成清零后才进入 Pass1
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    // ========================================================================
    // Pass1: 多核并行计算 |x|^p 并累加到各自的 workspace norm slot
    // ========================================================================
    // 每核遍历自己负责的 blocks [blockStart_, blockEnd_)，
    // 对每个 block 的每个 tile:
    //   1. 从 GM 加载数据到 UB
    //   2. 计算 |x|^p (p=1: Abs; p=2: Abs+Mul; p=其他: Abs+Log+Muls+Exp)
    //   3. Add/Max 累加到 normLocal (per-tile 累加器)
    // 最后将 normLocal 写入自己的 workspace slot
    // ========================================================================
    for (int64_t sliceTile = 0; sliceTile < sliceCount_; sliceTile += sliceTileLength_) {
        // currentTile: 当前 tile 的实际元素数 (最后一个 tile 可能不满)
        // alignedLen:  按 8 元素对齐 (CMP_ALIGN)，保证 Compare/Select 正常工作
        // atomicLen:   按 16 元素对齐 (ATOMIC_ALIGN)，保证 workspace cache line 安全
        int64_t currentTile = (sliceTileLength_ < (sliceCount_ - sliceTile)) ? sliceTileLength_ :
                                                                               (sliceCount_ - sliceTile);
        int64_t alignedLen = AlignDataElements<D_T_X>(currentTile);
        int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;

        // 初始化 normLocal = 0 (使用 atomicLen 确保填充区域也为零)
        Duplicate(normLocal, 0.0f, static_cast<int32_t>(atomicLen));
        PipeBarrier<PIPE_V>();

        int64_t numBlocksThisCore = blockEnd_ - blockStart_;
        if (numBlocksThisCore <= 0) {
            numBlocksThisCore = 0;
        }

// --- 批量 Add 到 normLocal (借鉴 CCE repeatTimes 优化) ---
// 用一条带 repeatTimes 的 V Add 指令，将 src 中 batchCount 个 block 逐个累加到 normLocal
// 替代原来的 for 循环 (batchCount 次单条 Add + 每次 PipeBarrier)
// 注意: repeatTimes ≤ 255, 超过需分批
// BinaryRepeatParams 构造: (dstBlkStride=1, src0BlkStride=1, src1BlkStride=1,
//   dstRepStride=0, src0RepStride=0, src1RepStride=alignedLen/8)
//   dstRepStride=0: dst(normLocal) 不移动 (in-place 累加)
//   src0RepStride=0: src0(normLocal) 不移动 (in-place 累加)
//   src1RepStride=alignedLen/8: src1 每次移到下一个 block (8 FP32 = 32B = 1 block)

// --- RENORM_P1_REDUCE 宏: 对单个 block 执行 |x|^p 并累加到 normLocal ---
// 根据 normMode 分三种计算路径:
//   P_INF:  norm = max(|x_i|)              → Abs + Max
//   P_ZERO: norm = count(x_i != 0)         → Abs + Compare + Select + Add
//   P_POS:  norm = sum(|x_i|^p)            → Abs + (Mul/Log+Muls+Exp) + Add
//
// PipeBarrier 策略:
//   - 同一 buffer 上连续 V 指令: 不加 PipeBarrier (硬件自动处理依赖)
//   - 跨 buffer 操作 (Add/Max 到 normLocal): 保留 PipeBarrier
#define RENORM_P1_REDUCE(wBuf)                                                                   \
    do {                                                                                         \
        if (normMode_ == NORM_MODE_P_INF) {                                                      \
            Abs(wBuf, wBuf, static_cast<int32_t>(alignedLen));                                   \
            PipeBarrier<PIPE_V>();                                                               \
            Max(normLocal, normLocal, wBuf, static_cast<int32_t>(alignedLen));                   \
        } else if (normMode_ == NORM_MODE_P_ZERO) {                                              \
            Abs(wBuf, wBuf, static_cast<int32_t>(alignedLen));                                   \
            PipeBarrier<PIPE_V>();                                                               \
            Compare(maskLocal, wBuf, zerosLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen)); \
            PipeBarrier<PIPE_V>();                                                               \
            Select(wBuf, maskLocal, onesLocal, zerosLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,     \
                   static_cast<int32_t>(alignedLen));                                            \
            PipeBarrier<PIPE_V>();                                                               \
            Add(normLocal, normLocal, wBuf, static_cast<int32_t>(alignedLen));                   \
        } else {                                                                                 \
            Abs(wBuf, wBuf, static_cast<int32_t>(alignedLen));                                   \
            if (p_ == 1.0f) {                                                                    \
            } else if (p_ == 2.0f) {                                                             \
                Mul(wBuf, wBuf, wBuf, static_cast<int32_t>(alignedLen));                         \
            } else {                                                                             \
                Maxs(wBuf, wBuf, eps_, static_cast<int32_t>(alignedLen));                        \
                Log(wBuf, wBuf, static_cast<int32_t>(alignedLen));                               \
                Muls(wBuf, wBuf, p_, static_cast<int32_t>(alignedLen));                          \
                Exp(wBuf, wBuf, static_cast<int32_t>(alignedLen));                               \
            }                                                                                    \
            PipeBarrier<PIPE_V>();                                                               \
            Add(normLocal, normLocal, wBuf, static_cast<int32_t>(alignedLen));                   \
        }                                                                                        \
    } while (0)

        if (numBlocksThisCore == 1) {
            // --- 单 block 路径: 无需批量处理 ---
            // 直接 LoadAndCast + RENORM_P1_REDUCE
            int64_t gmOffset = blockStart_ * sliceCount_ + sliceTile;
            LoadAndCast<D_T_X>(workLocal, dataLocal0, inputGM, gmOffset, currentTile, alignedLen);
            RENORM_P1_REDUCE(workLocal);
        } else if (numBlocksThisCore > 1) {
            // --- 批量处理路径: 一次 DataCopyPad 搬运多个连续 block ---
            // 优化动机: profiling 显示 49.7% scalar 时间来自逐 block 的循环开销和 WaitFlag stall。
            // 批量模式用 DataCopyPad 的 blockCount 参数一次搬运多个 block，
            // 将 WaitFlag stall 从 3*nb 降到 3*ceil(nb/batchSize_)。
            //
            // GM 布局: [numBlocks, sliceCount]，连续 block 间间隔 sliceCount 元素
            // 当 currentTile == sliceCount (常见: sliceTileLength >= sliceCount) 时:
            //   block 间无 gap (srcStride=0)，可高效批量搬运
            // 当 currentTile < sliceCount 时:
            //   srcStride = (sliceCount - currentTile) * sizeof(D_T_X) / 32 (32B 单位)

            int64_t batchBytes = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
            // srcStride: GM 中相邻 block 之间的间隔 (32B 单位)
            // gap = (sliceCount - currentTile) * sizeof(D_T_X) 字节
            int64_t srcStrideBytes = (sliceCount_ - currentTile) * static_cast<int64_t>(sizeof(D_T_X));
            uint32_t srcStride = static_cast<uint32_t>(srcStrideBytes / 32);
            // dstStride: UB 中相邻 block 之间的间隔 (32B 单位)
            // block 在 UB 中按 alignedLen 间距排列，填充后无额外 gap → dstStride=0
            uint32_t dstStride = 0;
            int64_t padLen = alignedLen - currentTile;

            DataCopyExtParams batchCopyParams;
            batchCopyParams.blockLen = batchBytes;
            batchCopyParams.srcStride = srcStride;
            batchCopyParams.dstStride = dstStride;
            DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(padLen), 0};

            // 按 batchSize_ 分批处理所有 blocks
            for (int64_t batchStart = 0; batchStart < numBlocksThisCore; batchStart += batchSize_) {
                int64_t batchEnd = (batchStart + batchSize_ < numBlocksThisCore) ? (batchStart + batchSize_) :
                                                                                   numBlocksThisCore;
                int64_t batchCount = batchEnd - batchStart;

                // MTE2: 一次搬运 batchCount 个连续 block 到 dataLocal0
                int64_t gmOffset = (blockStart_ + batchStart) * sliceCount_ + sliceTile;
                batchCopyParams.blockCount = static_cast<uint32_t>(batchCount);
                DataCopyPad(dataLocal0, inputGM[gmOffset], batchCopyParams, padParams);
                // MTE2 → V 同步
                TEventID eMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(eMte2V);
                WaitFlag<HardEvent::MTE2_V>(eMte2V);

                // V: 向量化批处理 |x|^p — 对整个 batch 一次性做 Abs/Log/Muls/Exp
                // (1 条大 V 指令替代 batchCount 条小 V 指令，大幅减少 V 指令数)
                // 仅 Add (累加到 normLocal) 需要逐 block 执行 (reduce 操作)
                int64_t batchTotalElements = batchCount * alignedLen;
                if constexpr (sizeof(D_T_X) == sizeof(float)) {
                    // FP32: 直接在 dataLocal0 上计算，无需 Cast
                    if (normMode_ == NORM_MODE_P_INF) {
                        Abs(dataLocal0, dataLocal0, static_cast<int32_t>(batchTotalElements));
                        PipeBarrier<PIPE_V>();
                        TreeReduceBatchRowsMax(dataLocal0, normLocal, batchCount, alignedLen);
                        PipeBarrier<PIPE_V>();
                    } else if (normMode_ == NORM_MODE_P_POSITIVE) {
                        // |x|^p 向量化计算
                        Abs(dataLocal0, dataLocal0, static_cast<int32_t>(batchTotalElements));
                        if (p_ == 1.0f) {
                            // p=1: |x|^1 = |x|, 无需额外操作
                        } else if (p_ == 2.0f) {
                            // p=2: |x|^2 = |x| * |x|
                            Mul(dataLocal0, dataLocal0, dataLocal0, static_cast<int32_t>(batchTotalElements));
                        } else {
                            // p=其他: |x|^p = exp(p * log(max(|x|, eps)))
                            // Maxs 防止 log(0)=-inf
                            Maxs(dataLocal0, dataLocal0, eps_, static_cast<int32_t>(batchTotalElements));
                            Log(dataLocal0, dataLocal0, static_cast<int32_t>(batchTotalElements));
                            Muls(dataLocal0, dataLocal0, p_, static_cast<int32_t>(batchTotalElements));
                            Exp(dataLocal0, dataLocal0, static_cast<int32_t>(batchTotalElements));
                        }
                        PipeBarrier<PIPE_V>();
                        // UB 内按行做二叉树归约，避免 repeat Add 对同一目的地址的
                        // 跨 repeat 读写依赖。奇数行在每层保留到下一轮。
                        if (alignedLen <= 128) {
                            TreeReduceBatchRows(dataLocal0, normLocal, batchCount, alignedLen);
                        } else {
                            for (int64_t bi = 0; bi < batchCount; ++bi) {
                                Add(normLocal, normLocal, dataLocal0[static_cast<int32_t>(bi * alignedLen)],
                                    static_cast<int32_t>(alignedLen));
                            }
                        }
                        PipeBarrier<PIPE_V>();
                    } else {
                        // P_INF/P_ZERO: 使用宏路径 (涉及 Compare/Select，暂未向量化)
                        for (int64_t bi = 0; bi < batchCount; ++bi) {
                            auto blockWork = dataLocal0[bi * alignedLen];
                            RENORM_P1_REDUCE(blockWork);
                        }
                    }
                } else {
                    // FP16: 先 Cast 到 FP32 workLocal，再在 workLocal 上计算
                    Cast(workLocal, dataLocal0, RoundMode::CAST_NONE, static_cast<int32_t>(batchTotalElements));
                    PipeBarrier<PIPE_V>();
                    if (normMode_ == NORM_MODE_P_INF) {
                        Abs(workLocal, workLocal, static_cast<int32_t>(batchTotalElements));
                        PipeBarrier<PIPE_V>();
                        TreeReduceBatchRowsMax(workLocal, normLocal, batchCount, alignedLen);
                        PipeBarrier<PIPE_V>();
                    } else if (normMode_ == NORM_MODE_P_POSITIVE) {
                        Abs(workLocal, workLocal, static_cast<int32_t>(batchTotalElements));
                        if (p_ == 1.0f) {
                        } else if (p_ == 2.0f) {
                            Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(batchTotalElements));
                        } else {
                            Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(batchTotalElements));
                            Log(workLocal, workLocal, static_cast<int32_t>(batchTotalElements));
                            Muls(workLocal, workLocal, p_, static_cast<int32_t>(batchTotalElements));
                            Exp(workLocal, workLocal, static_cast<int32_t>(batchTotalElements));
                        }
                        PipeBarrier<PIPE_V>();
                        // 与 FP32 路径相同，使用无跨 repeat 写依赖的树形归约。
                        if (alignedLen <= 128) {
                            TreeReduceBatchRows(workLocal, normLocal, batchCount, alignedLen);
                        } else {
                            for (int64_t bi = 0; bi < batchCount; ++bi) {
                                Add(normLocal, normLocal, workLocal[static_cast<int32_t>(bi * alignedLen)],
                                    static_cast<int32_t>(alignedLen));
                            }
                        }
                        PipeBarrier<PIPE_V>();
                    } else {
                        for (int64_t bi = 0; bi < batchCount; ++bi) {
                            auto blockWork = workLocal[bi * alignedLen];
                            RENORM_P1_REDUCE(blockWork);
                        }
                    }
                }

                // The next batch reuses dataLocal0. Ensure all vector reads
                // and in-place reductions finish before MTE2 overwrites it.
                TEventID eVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(eVMte2);
                WaitFlag<HardEvent::V_MTE2>(eVMte2);
            }
        }
#undef RENORM_P1_REDUCE

        // --- 将本核的 partial norm 原子累加到 workspace norm slot ---
        // SetAtomicAdd/Max 方案 (参考 lp_norm_v3):
        //   - P_POSITIVE/P_ZERO: SetAtomicAdd (sum 累加)
        //   - P_INF:             SetAtomicMax (max 归约)
        // SetAtomicAdd 的原子写直接到达 GM/L2，跨 cluster 可见性由硬件保证。
        // V → MTE3 同步: 确保 V (Add) 完成后 MTE3 才能搬运 normLocal
        TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);

        // 设置原子操作模式
        if (normMode_ == NORM_MODE_P_INF) {
            SetAtomicMax<float>();
        } else {
            SetAtomicAdd<float>();
        }

        // MTE3: 原子累加到 workspaceGM[sliceTile] (所有核写同一个 slot)
        // 使用 atomicLen (64B 对齐) 保证 cache line 安全
        DataCopy(workspaceGM[sliceTile], normLocal, static_cast<int32_t>(atomicLen));

        SetAtomicNone();

        // MTE3 → V 同步: 确保 MTE3 完成后 normLocal 可在下一个 tile 重用
        TEventID eventIDMte3V = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(eventIDMte3V);
        WaitFlag<HardEvent::MTE3_V>(eventIDMte3V);
    }

    // --- Pass1 → Pass2 全局同步 ---
    // SetAtomicAdd 方案: SyncAll 即可，不需要 DataCacheCleanAndInvalid。
    // SetAtomicAdd 的原子写直接到达 GM/L2，不经过 L1 DCache，
    // 跨 cluster 可见性由硬件保证 (参考 lp_norm_v3)。
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    PipeBarrier<PIPE_ALL>();

    // ========================================================================
    // Pass2 (仅 core 0): 读取聚合 norm → 计算 scale → 写入 scaleGM
    // ========================================================================
    // SetAtomicAdd 方案: Pass1 中所有核已原子累加到同一个 slot，
    // core 0 直接读取该 slot 即可得到聚合结果，无需串行读取多个 slot。
    //
    // 步骤:
    //   1. 读取 norm slot 到 normLocal (已是所有核的聚合结果)
    //   2. 根据 normMode 计算 scale:
    //      - P_INF/P_ZERO: norm 已是最终值，scale = maxNorm / max(norm, eps) if norm > maxNorm
    //      - P_POSITIVE:   norm_root = norm^(1/p)，scale = maxNorm / max(norm_root, eps) if norm_root > maxNorm
    //   3. 写入 scale 到 scaleGM (供 Pass3 读取)
    // ========================================================================
    if (blockIdx_ == 0) {
        for (int64_t sliceTile = 0; sliceTile < sliceCount_; sliceTile += sliceTileLength_) {
            int64_t currentTile = (sliceTileLength_ < (sliceCount_ - sliceTile)) ? sliceTileLength_ :
                                                                                   (sliceCount_ - sliceTile);
            int64_t alignedLen = AlignDataElements<D_T_X>(currentTile);
            int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;

            // 步骤1: 读取聚合 norm slot 到 normLocal
            // SetAtomicAdd 已保证所有核的写入对 core 0 可见
            DataCopy(normLocal, workspaceGM[sliceTile], static_cast<int32_t>(atomicLen));
            TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(eventID0);
            WaitFlag<HardEvent::MTE2_V>(eventID0);

            // 步骤2: 计算 scale 因子
            // scale = (norm_root > maxNorm) ? maxNorm / max(norm_root, eps) : 1.0
            // 即: 如果范数超过 maxNorm，则缩放到 maxNorm；否则不缩放 (scale=1)
            if (normMode_ == NORM_MODE_P_INF || normMode_ == NORM_MODE_P_ZERO) {
                // P_INF/P_ZERO: norm 已是最终值 (max 或 count)，无需开 p 次方
                // scale = maxNorm / max(norm, eps) if norm > maxNorm, else 1.0
                Maxs(tmpLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Reciprocal(tmpLocal, tmpLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Muls(scaleLocal, tmpLocal, maxNorm_, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();

                // mask = (norm > maxNorm) → 选择 scale 或 1.0
                Compare(maskLocal, normLocal, maxNormLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                       static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            } else {
                // P_POSITIVE: 需要计算 norm_root = norm^(1/p)
                if (p_ == 2.0f) {
                    // p=2: norm_root = sqrt(norm)
                    Sqrt(normLocal, normLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                } else if (p_ != 1.0f) {
                    // p=其他: norm_root = exp((1/p) * log(max(norm, eps)))
                    // Maxs 防止 log(0)=-inf
                    Maxs(normLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                    Log(normLocal, normLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                    Muls(normLocal, normLocal, 1.0f / p_, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                    Exp(normLocal, normLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                }
                // p=1: norm_root = norm (已在 normLocal 中)

                // scale = maxNorm / max(norm_root, eps)
                Maxs(tmpLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Reciprocal(tmpLocal, tmpLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Muls(scaleLocal, tmpLocal, maxNorm_, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();

                // mask = (norm_root > maxNorm) → 选择 scale 或 1.0
                Compare(maskLocal, normLocal, maxNormLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                       static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            }

            // 步骤3: 写入 scale 到 scaleGM
            // 使用 atomicLen (64B 对齐) 保证 cache line 一致性
            // 填充区域 [alignedLen, atomicLen) 可能含垃圾数据，但 Pass3 只使用 [0, alignedLen)
            // V → MTE3 同步
            TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);

            // MTE3: 写入 scale 到 workspaceGM[scaleGmOffset_ + sliceTile]
            DataCopy(workspaceGM[scaleGmOffset_ + sliceTile], scaleLocal, static_cast<int32_t>(atomicLen));

            // MTE3 → V 同步
            TEventID eventIDMte3V = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(eventIDMte3V);
            WaitFlag<HardEvent::MTE3_V>(eventIDMte3V);
        }
    }

    // --- Pass2 → Pass3 全局同步 ---
    // core 0 用普通 DataCopy 写 scale 到 workspace，需 DataCacheCleanAndInvalid 确保对所有核可见。
    // 所有核执行 (参考 lp_norm_v3: 核0写后 flush + SyncAll)。
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(workspaceGM[0]);
    PipeBarrier<PIPE_ALL>();

    // ========================================================================
    // Pass3: 所有核并行 — 读取 scale → 逐块 load × scale → store 到 outputGM
    // ========================================================================
    // 这是最终输出阶段，所有核并行执行。
    // 每核遍历自己负责的 blocks，对每个 block 的每个 tile:
    //   1. 从 scaleGM 读取 scale 因子
    //   2. 从 inputGM 加载数据
    //   3. y = x * scale (FP32 直接 Mul; FP16 需 Cast→Mul→CastBack)
    //   4. 存储到 outputGM
    //
    // 批量优化: 一次加载多个 block，减少 MTE2→V 和 V→MTE3 同步次数
    // ========================================================================
    for (int64_t sliceTile = 0; sliceTile < sliceCount_; sliceTile += sliceTileLength_) {
        int64_t currentTile = (sliceTileLength_ < (sliceCount_ - sliceTile)) ? sliceTileLength_ :
                                                                               (sliceCount_ - sliceTile);
        int64_t alignedLen = AlignDataElements<D_T_X>(currentTile);
        int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;

        // 步骤1: 从 scaleGM 读取 scale 因子 (使用 atomicLen 保证 cache line 一致性)
        DataCopy(scaleLocal, workspaceGM[scaleGmOffset_ + sliceTile], static_cast<int32_t>(atomicLen));
        // MTE2 → V 同步
        TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(eventID0);
        WaitFlag<HardEvent::MTE2_V>(eventID0);

        // Renorm is an identity operation when every scale is exactly one.
        // For the common single-tile case, copy each core's complete,
        // contiguous block range without Cast/Mul or per-block DMA.
        TEventID scaleToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(scaleToScalar);
        WaitFlag<HardEvent::V_S>(scaleToScalar);
        bool isIdentityTile = true;
        for (int64_t s = 0; s < currentTile; ++s) {
            if (scaleLocal.GetValue(s) != 1.0f) {
                isIdentityTile = false;
                break;
            }
        }
        TEventID scalarToScale = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarToScale);
        WaitFlag<HardEvent::S_V>(scalarToScale);
        if (isIdentityTile && currentTile == sliceCount_) {
            int64_t copyStart = blockStart_ * sliceCount_;
            int64_t copyElements = (blockEnd_ - blockStart_) * sliceCount_;
            int64_t maxCopyElements = batchSize_ * alignedLen;
            DataCopyPadExtParams<D_T_X> identityPadParams = {false, 0, 0, 0};

            for (int64_t copied = 0; copied < copyElements; copied += maxCopyElements) {
                int64_t currentCopy = (maxCopyElements < copyElements - copied) ? maxCopyElements :
                                                                                  copyElements - copied;
                DataCopyExtParams identityCopyParams;
                identityCopyParams.blockCount = 1;
                identityCopyParams.blockLen = static_cast<uint32_t>(currentCopy * sizeof(D_T_X));
                identityCopyParams.srcStride = 0;
                identityCopyParams.dstStride = 0;

                DataCopyPad(dataLocal0, inputGM[copyStart + copied], identityCopyParams, identityPadParams);
                TEventID identityMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(identityMte2V);
                WaitFlag<HardEvent::MTE2_V>(identityMte2V);
                TEventID identityVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                SetFlag<HardEvent::V_MTE3>(identityVMte3);
                WaitFlag<HardEvent::V_MTE3>(identityVMte3);
                DataCopyPad(outputGM[copyStart + copied], dataLocal0, identityCopyParams);
                TEventID identityMte3Mte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                SetFlag<HardEvent::MTE3_MTE2>(identityMte3Mte2);
                WaitFlag<HardEvent::MTE3_MTE2>(identityMte3Mte2);
            }
            continue;
        }

        int64_t numBlocksThisCore = blockEnd_ - blockStart_;
        if (numBlocksThisCore <= 0) {
            continue; // 空核跳过
        }

        // 准备 DataCopyPad 参数 (单 block 路径和后备路径使用)
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        int64_t padLen = alignedLen - currentTile;
        DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(padLen), 0};

        if (numBlocksThisCore == 1) {
            // --- 单 block 路径: LoadAndCast → Mul → CastBackAndStore ---
            int64_t gmOffset = blockStart_ * sliceCount_ + sliceTile;
            LoadAndCast<D_T_X>(workLocal, dataLocal0, inputGM, gmOffset, currentTile, alignedLen);
            Mul(workLocal, workLocal, scaleLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            CastBackAndStore<D_T_X>(workLocal, dataLocal0, outputGM, gmOffset, currentTile, alignedLen);
        } else {
            // --- 批量路径: 一次加载多个 block，批量 Mul，再逐个 store ---
            // 批量 store 不可行: UB 中 block 间的填充区域 (alignedLen-currentTile) 可能
            // 不是 32B 对齐，无法用 DataCopyPad 的 srcStride 参数批量搬运。
            int64_t srcStrideBytes = (sliceCount_ - currentTile) * static_cast<int64_t>(sizeof(D_T_X));
            bool canBatchLoad = (srcStrideBytes % 32 == 0); // GM gap 必须 32B 对齐

            if (canBatchLoad) {
                // --- 批量加载路径 (GM gap 32B 对齐) — 双缓冲优化 ---
                // 交替使用 dataBuf0/dataBuf1，让下一批 MTE2 load 与当前批 MTE3 store 重叠
                // 这是 TQue 寄存器编程模式的手动实现: 两个等大 buffer 自动流水
                uint32_t srcStride = static_cast<uint32_t>(srcStrideBytes / 32);
                int64_t padLen = alignedLen - currentTile;

                DataCopyExtParams batchCopyParams;
                batchCopyParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
                batchCopyParams.srcStride = srcStride;
                batchCopyParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(padLen), 0};

                // store 参数 (逐 block 存储)
                DataCopyExtParams storeParams;
                storeParams.blockCount = 1;
                storeParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
                storeParams.srcStride = 0;
                storeParams.dstStride = 0;

                // 双缓冲事件 ID — 每个 buffer 独立的同步事件
                TEventID eMte2V[2] = {GetTPipePtr()->FetchEventID(HardEvent::MTE2_V),
                                      GetTPipePtr()->FetchEventID(HardEvent::MTE2_V)};
                TEventID eVMte3[2] = {GetTPipePtr()->FetchEventID(HardEvent::V_MTE3),
                                      GetTPipePtr()->FetchEventID(HardEvent::V_MTE3)};
                TEventID eMte3Mte2[2] = {GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2),
                                         GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2)};

                // 计算总批次数
                int64_t totalBatches = (numBlocksThisCore + batchSize_ - 1) / batchSize_;

                // 预加载第 0 批到 buf0
                int64_t batchStart0 = 0;
                int64_t batchEnd0 = (batchStart0 + batchSize_ < numBlocksThisCore) ? (batchStart0 + batchSize_) :
                                                                                     numBlocksThisCore;
                int64_t batchCount0 = batchEnd0 - batchStart0;
                int64_t gmOffset0 = (blockStart_ + batchStart0) * sliceCount_ + sliceTile;
                batchCopyParams.blockCount = static_cast<uint32_t>(batchCount0);
                DataCopyPad(dataLocal0, inputGM[gmOffset0], batchCopyParams, padParams);
                SetFlag<HardEvent::MTE2_V>(eMte2V[0]);

                // 双缓冲循环
                for (int64_t batchIdx = 0; batchIdx < totalBatches; ++batchIdx) {
                    int64_t curBuf = batchIdx % 2; // 当前批使用的 buffer 索引
                    int64_t nxtBuf = 1 - curBuf;   // 下一批使用的 buffer 索引
                    auto& curData = (curBuf == 0) ? dataLocal0 : dataLocal1;
                    auto& nxtData = (nxtBuf == 0) ? dataLocal0 : dataLocal1;
                    auto& curWork = (curBuf == 0) ? workLocal : workLocal1;
                    auto& nxtWork = (nxtBuf == 0) ? workLocal : workLocal1;

                    int64_t batchStart = batchIdx * batchSize_;
                    int64_t batchEnd = (batchStart + batchSize_ < numBlocksThisCore) ? (batchStart + batchSize_) :
                                                                                       numBlocksThisCore;
                    int64_t batchCount = batchEnd - batchStart;
                    int64_t batchTotalElements = batchCount * alignedLen;

                    // 预加载下一批 (与当前批的 V 计算和 MTE3 store 重叠)
                    if (batchIdx + 1 < totalBatches) {
                        // 等待下一批 buffer 的 MTE3 完成 (上一轮用过的同一 buffer)
                        if (batchIdx >= 1) {
                            WaitFlag<HardEvent::MTE3_MTE2>(eMte3Mte2[nxtBuf]);
                        }
                        int64_t nxtStart = (batchIdx + 1) * batchSize_;
                        int64_t nxtEnd = (nxtStart + batchSize_ < numBlocksThisCore) ? (nxtStart + batchSize_) :
                                                                                       numBlocksThisCore;
                        int64_t nxtCount = nxtEnd - nxtStart;
                        int64_t nxtGmOffset = (blockStart_ + nxtStart) * sliceCount_ + sliceTile;
                        batchCopyParams.blockCount = static_cast<uint32_t>(nxtCount);
                        DataCopyPad(nxtData, inputGM[nxtGmOffset], batchCopyParams, padParams);
                        SetFlag<HardEvent::MTE2_V>(eMte2V[nxtBuf]);
                    }

                    // 等待当前批 MTE2 完成
                    WaitFlag<HardEvent::MTE2_V>(eMte2V[curBuf]);

                    // V: 批量 y = x * scale
                    if constexpr (sizeof(D_T_X) == sizeof(float)) {
                        // FP32: 直接在 curData 上 Mul
                        if (alignedLen <= 128) {
                            uint64_t maskLo = (alignedLen >= 64) ? FULL_MASK : ((1UL << alignedLen) - 1);
                            uint64_t maskHi = (alignedLen <= 64) ?
                                                  0 :
                                                  ((alignedLen >= 128) ? FULL_MASK : ((1UL << (alignedLen - 64)) - 1));
                            SetVectorMask<float>(maskHi, maskLo);
                            BinaryRepeatParams mulParams;
                            mulParams.dstBlkStride = 1;
                            mulParams.src0BlkStride = 1;
                            mulParams.src1BlkStride = 1;
                            mulParams.dstRepStride = static_cast<int32_t>(alignedLen / 8);
                            mulParams.src0RepStride = static_cast<int32_t>(alignedLen / 8);
                            mulParams.src1RepStride = 0;
                            int64_t remaining = batchCount;
                            int64_t off = 0;
                            while (remaining > 0) {
                                uint8_t repTimes = static_cast<uint8_t>(remaining > 255 ? 255 : remaining);
                                Mul<float, false>(curData[static_cast<int32_t>(off * alignedLen)],
                                                  curData[static_cast<int32_t>(off * alignedLen)], scaleLocal,
                                                  MASK_PLACEHOLDER, repTimes, mulParams);
                                off += repTimes;
                                remaining -= repTimes;
                            }
                            SetVectorMask<float>(FULL_MASK, FULL_MASK);
                        } else {
                            for (int64_t bi = 0; bi < batchCount; ++bi) {
                                Mul(curData[bi * alignedLen], curData[bi * alignedLen], scaleLocal,
                                    static_cast<int32_t>(alignedLen));
                            }
                        }
                        PipeBarrier<PIPE_V>();
                    } else {
                        // FP16: Cast → Mul → Cast back
                        Cast(curWork, curData, RoundMode::CAST_NONE, static_cast<int32_t>(batchTotalElements));
                        PipeBarrier<PIPE_V>();
                        if (alignedLen <= 128) {
                            uint64_t maskLo = (alignedLen >= 64) ? FULL_MASK : ((1UL << alignedLen) - 1);
                            uint64_t maskHi = (alignedLen <= 64) ?
                                                  0 :
                                                  ((alignedLen >= 128) ? FULL_MASK : ((1UL << (alignedLen - 64)) - 1));
                            SetVectorMask<float>(maskHi, maskLo);
                            BinaryRepeatParams mulParams;
                            mulParams.dstBlkStride = 1;
                            mulParams.src0BlkStride = 1;
                            mulParams.src1BlkStride = 1;
                            mulParams.dstRepStride = static_cast<int32_t>(alignedLen / 8);
                            mulParams.src0RepStride = static_cast<int32_t>(alignedLen / 8);
                            mulParams.src1RepStride = 0;
                            int64_t remaining = batchCount;
                            int64_t off = 0;
                            while (remaining > 0) {
                                uint8_t repTimes = static_cast<uint8_t>(remaining > 255 ? 255 : remaining);
                                Mul<float, false>(curWork[static_cast<int32_t>(off * alignedLen)],
                                                  curWork[static_cast<int32_t>(off * alignedLen)], scaleLocal,
                                                  MASK_PLACEHOLDER, repTimes, mulParams);
                                off += repTimes;
                                remaining -= repTimes;
                            }
                            SetVectorMask<float>(FULL_MASK, FULL_MASK);
                        } else {
                            for (int64_t bi = 0; bi < batchCount; ++bi) {
                                Mul(curWork[bi * alignedLen], curWork[bi * alignedLen], scaleLocal,
                                    static_cast<int32_t>(alignedLen));
                            }
                        }
                        PipeBarrier<PIPE_V>();
                        Cast(curData, curWork, RoundMode::CAST_RINT, static_cast<int32_t>(batchTotalElements));
                        PipeBarrier<PIPE_V>();
                    }

                    // V → MTE3 同步
                    SetFlag<HardEvent::V_MTE3>(eVMte3[curBuf]);
                    WaitFlag<HardEvent::V_MTE3>(eVMte3[curBuf]);

                    // MTE3: 逐 block 存储
                    for (int64_t bi = 0; bi < batchCount; ++bi) {
                        int64_t b = blockStart_ + batchStart + bi;
                        DataCopyPad(outputGM[b * sliceCount_ + sliceTile], curData[bi * alignedLen], storeParams);
                    }

                    // MTE3 → MTE2 同步 (标记当前 buffer 的 MTE3 已完成)
                    // 下一轮使用同一 buffer 时需要等待此信号
                    SetFlag<HardEvent::MTE3_MTE2>(eMte3Mte2[curBuf]);
                }

                // 等待所有 MTE3 存储完成
                // 两个 buffer 都可能有未消费的 MTE3_MTE2 flag，都需要 Wait
                WaitFlag<HardEvent::MTE3_MTE2>(eMte3Mte2[(totalBatches - 1) % 2]);
                if (totalBatches >= 2) {
                    WaitFlag<HardEvent::MTE3_MTE2>(eMte3Mte2[totalBatches % 2]);
                }
            } else {
                // --- 后备路径: 逐 block 双缓冲 (GM gap 非 32B 对齐时) ---
                // 双缓冲: 加载 block b+1 与计算 block b 重叠，隐藏 MTE2 延迟
                // dataBuf0/dataBuf1 交替使用，workBuf/workBuf1 对应交替

                // 预加载第一个 block
                DataCopyPad(dataLocal0, inputGM[blockStart_ * sliceCount_ + sliceTile], copyParams, padParams);
                TEventID eMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(eMte2V);

                for (int64_t idx = 0; idx < numBlocksThisCore; ++idx) {
                    int64_t b = blockStart_ + idx;
                    int64_t curIdx = idx % 2; // 0 或 1，交替使用双缓冲
                    int64_t nxtIdx = 1 - curIdx;
                    auto& curData = (curIdx == 0) ? dataLocal0 : dataLocal1;
                    auto& curWork = (curIdx == 0) ? workLocal : workLocal1;
                    auto& nxtData = (nxtIdx == 0) ? dataLocal0 : dataLocal1;
                    int64_t gmOffset = b * sliceCount_ + sliceTile;

                    // 等待当前 block 的 MTE2 完成，然后 Cast 到 FP32
                    WaitFlag<HardEvent::MTE2_V>(eMte2V);
                    if constexpr (sizeof(D_T_X) == sizeof(float)) {
                        DataCopy(curWork, curData, static_cast<int32_t>(alignedLen));
                    } else {
                        Cast(curWork, curData, RoundMode::CAST_NONE, static_cast<int32_t>(alignedLen));
                    }
                    PipeBarrier<PIPE_V>();

                    // 预加载下一个 block (与下面的 V 计算重叠)
                    if (idx < numBlocksThisCore - 1) {
                        int64_t gmOffsetNext = (b + 1) * sliceCount_ + sliceTile;
                        DataCopyPad(nxtData, inputGM[gmOffsetNext], copyParams, padParams);
                        SetFlag<HardEvent::MTE2_V>(eMte2V);
                    }

                    // V: y = x * scale
                    Mul(curWork, curWork, scaleLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();

                    // Cast 回 D_T_X 并存储到 GM
                    CastBackAndStore<D_T_X>(curWork, curData, outputGM, gmOffset, currentTile, alignedLen);
                }
            }
        }
    }
}

} // namespace NsRenormSmTlStable

#endif // _RENORM_SM_TL_STABLE_H_
