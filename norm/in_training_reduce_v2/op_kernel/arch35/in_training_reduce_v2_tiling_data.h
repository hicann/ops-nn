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

// AR full_reduce（R 全载）主路径 TilingData。
// 裁剪自 instance_norm 的 InstanceNormARFullReduceTilingData：
//   删除 epsilon / avgFactor（本算子输出原始 Σx/Σx²，不除 R、不做 rstd）。
struct INTrainingReduceV2ARFullReduceTilingData {
    int64_t numN;               // N 轴（a1）
    int64_t numC;               // C 轴（a0）
    int64_t numR;               // 规约轴长度 R = H*W（5D: D*H*W）
    uint64_t rAlign;            // R 按 dtype 元素宽度 32B 对齐后的长度
    int64_t cInner;             // 单 tile 内一次处理的 C 行数（sub-R 分块路径恒为 1）
    int64_t cOuter;             // C 轴切分的外层块数
    int64_t cTail;              // C 轴尾块长度
    uint64_t binaryAddQuotient; // 小于 rAlign 的最大 2 的幂（pairwise 折叠点）
    int64_t perCoreCnt;         // 平均每个核处理的 tile 数
    // ---- sub-R 分块路径（DESIGN §6.3 路 A；R 超单次 UB 容量时启用）----
    uint64_t isSubRTiling; // 0=R 全载路径；1=sub-R 分块路径
    uint64_t rFactor;      // sub-R 分块的块大小（元素数，VL_FP32 整数倍）
    uint64_t numChunks;    // ceil(numR / rFactor)，每行分块数
    uint64_t tailLen;      // numR - (numChunks-1)*rFactor，尾块长度
    // ---- sub-R 分组折叠（解除 R 上限）----
    // 部分和缓存不再随 R 增长：每 chunksPerGroup 个 chunk 折叠一次，结果作为 carry 参与下一组折叠。
    uint64_t chunksPerGroup;     // 每组的 chunk 数（numGroups==1 时等于 numChunks，退化为原路径）
    uint64_t numGroups;          // ceil(numChunks / chunksPerGroup)
    uint64_t tailChunks;         // 末组的 chunk 数 = numChunks - (numGroups-1)*chunksPerGroup
    int64_t totalRows;           // N*C，由 host 做溢出校验后下发
    int64_t totalElements;       // N*C*R，由 host 做溢出校验后下发
    int64_t totalTiles;          // full-load 路径的 N*cOuter；sub-R 路径等于 totalRows
    uint64_t inputBufferBytes;   // inQueueX 单 buffer 字节数
    uint64_t outputBufferBytes;  // sum/square_sum 各自单 buffer 字节数
    uint64_t scratchBufferBytes; // full-load 路径每个 binary-add scratch 的字节数
    uint64_t partialBufferBytes; // sub-R 路径每个 partial buffer 的字节数
};

#endif // _IN_TRAINING_REDUCE_V2_TILING_DATA_H_
