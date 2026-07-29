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
 * \file median_tiling_data.h
 * \brief median tiling data struct
 */

#ifndef MEDIAN_TILING_DATA_H
#define MEDIAN_TILING_DATA_H

// 计算路径编码（host 侧 tiling 选定，作为 tilingkey(schMode) 编译期区分；kernel 侧 if constexpr 分发。
// 见 median_tiling_key.h。dtype 由编译期 DTYPE_INPUT 决定，与本维度正交）：
//   MAIN   : redLen<=8192 多行 Sort32+MrgSort（非 int64；含 int 小 L）
//   SMALL  : redLen<=32 fp 大 batch（连续搬 + Gather 散槽 + Sort32）
//   BIGSEL : fp 8192<redLen<=24576 整行常驻向量化二分计数
//   BIG    : fp redLen>24576 或 int redLen>8192（分块二分计数）
//   HEAP   : int64 标量堆排
//   BIGSORT: fp 单行大 L 多核分段排序（nSeg>0）
enum MedianPath {
    MEDIAN_PATH_MAIN = 0,
    MEDIAN_PATH_SMALL = 1,
    MEDIAN_PATH_BIGSEL = 2,
    MEDIAN_PATH_BIG = 3,
    MEDIAN_PATH_HEAP = 4,
    MEDIAN_PATH_BIGSORT = 5,
};

struct MedianTilingData {
    int64_t batch = 0;  // 行数 = numel / redLen
    int64_t redLen = 0; // 归约长度（末轴）
    int64_t mid = 0;    // 下中位下标 (redLen-1)/2
    int64_t nSeg = 0;   // fp 单行大N(>24576) 排序路径分段数（0=不走排序路径）
    int64_t pad = 0;    // MAIN 路径的 Pad 档位（host 按 redLen 算好，仅 MAIN 用）
};
#endif // MEDIAN_TILING_DATA_H
