/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file in_training_reduce_v2_tiling_data.h
 * \brief
 */
#ifndef _IN_TRAINING_REDUCE_V2_TILING_DATA_H_
#define _IN_TRAINING_REDUCE_V2_TILING_DATA_H_

constexpr uint64_t IN_TRAINING_REDUCE_V2_SPLIT_R_MODE = 2;

// AR full_reduce（R 全载）主路径 TilingData。
// 裁剪自 instance_norm 的 InstanceNormARFullReduceTilingData：
//   删除 epsilon / avgFactor（本算子输出原始 Σx/Σx²，不除 R、不做 rstd）。
struct INTrainingReduceV2ARFullReduceTilingData {
    int64_t numN;               // N 轴（a1）
    int64_t numC;               // C 轴（a0）
    int64_t numR;               // 规约轴长度 R = H*W（5D: D*H*W）
    uint64_t rAlign;            // R 的 32B 对齐长度；0 表示多行打包路径
    int64_t cInner;             // C tile行数；打包路径为N*C行数；split-R为每核R片段长度
    int64_t cOuter;             // C tile数；打包路径为N*C tile数；split-R为每行核数
    int64_t cTail;              // C/打包路径尾行数；split-R为每行最后一个R片段长度
    uint64_t binaryAddQuotient; // 小于 rAlign 的最大 2 的幂（pairwise 折叠点）
    int64_t perCoreCnt;         // 平均每个核处理的 tile 数
    // ---- sub-R 分块路径（R 超单次 UB 容量时启用）----
    uint64_t isSubRTiling; // 0=R 全载路径；1=单核sub-R；2=跨核split-R
    uint64_t rFactor;      // sub-R/split-R块大小；打包路径为UB行步长，0表示无行间padding
    uint64_t numChunks;    // ceil(numR / rFactor)，每行分块数
    uint64_t tailLen;      // numR - (numChunks-1)*rFactor，尾块长度
    // ---- sub-R 分组折叠（解除 R 上限）----
    // 部分和缓存不再随 R 增长：每 chunksPerGroup 个 chunk 折叠一次，结果作为 carry 参与下一组折叠。
    uint64_t chunksPerGroup;     // 每组的 chunk 数（numGroups==1 时等于 numChunks，退化为原路径）
    uint64_t numGroups;          // ceil(numChunks / chunksPerGroup)
    uint64_t tailChunks;         // 末组的 chunk 数 = numChunks - (numGroups-1)*chunksPerGroup
    int64_t totalRows;           // N*C，由 host 做溢出校验后下发
    int64_t totalElements;       // N*C*R，由 host 做溢出校验后下发
    int64_t totalTiles;          // full-load: N*cOuter；打包路径: cOuter；sub-R: totalRows；split-R:启动核数
    uint64_t inputBufferBytes;   // inQueueX 单 buffer 字节数
    uint64_t outputBufferBytes;  // sum/square_sum单buffer字节数；split-R为共用输出队列大小
    uint64_t scratchBufferBytes; // split-R最终归约的workspace输入队列字节数；其他路径为0
    uint64_t partialBufferBytes; // sub-R/split-R每个高低双分量partial buffer的字节数
};

// NHWC 物理布局 [N,R,C]，各 (N,C tile) 由唯一核沿 R 分块归约。
struct INTrainingReduceV2ARAReduceTilingData {
    uint64_t numN;
    uint64_t numR;
    uint64_t numC;
    uint64_t totalTiles;
    uint64_t tilesPerCore;
    uint64_t cOuter;
    uint64_t tileCLen;
    uint64_t tileCTail;
    uint64_t rFactor;
    uint64_t numChunks;
    uint64_t totalRows;         // N*C，host 已校验 int64 上界
    uint64_t totalElements;     // N*R*C，host 已校验 int64 上界
    uint64_t inputBufferBytes;  // 单个输入 buffer；双缓冲
    uint64_t outputBufferBytes; // 每个 fp32 输出累加器；单缓冲
};

#if defined(__CCE_AICORE__)
#define IN_TRAINING_REDUCE_V2_AICORE __aicore__
#else
#define IN_TRAINING_REDUCE_V2_AICORE
#endif
// DataCopyExtParams 的 GM stride 单位为字节；不能截断到 uint32。
IN_TRAINING_REDUCE_V2_AICORE inline constexpr int64_t CalcINTrainingReduceV2ARASrcStrideBytes(uint64_t numC,
                                                                                              uint64_t currentC,
                                                                                              uint64_t elemSize)
{
    return static_cast<int64_t>((numC - currentC) * elemSize);
}
#undef IN_TRAINING_REDUCE_V2_AICORE

#endif // _IN_TRAINING_REDUCE_V2_TILING_DATA_H_
