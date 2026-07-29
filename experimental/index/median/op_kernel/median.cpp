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
 * \file median.cpp
 * \brief Median 算子 kernel 入口：TPipe 外置；dtype 由编译期 DTYPE_INPUT 决定，计算路径由 tilingkey(schMode)
 *        编译期区分（见 median_tiling_key.h / MedianPath），kernel 侧 if constexpr 分发到对应 class，无运行期分支。
 */
#include "median.h"

// 编译期 fp 判定（half / float / bfloat16_t）；用于 if constexpr 守卫，避免为 int dtype 实例化 fp 专用 class
template <typename T>
constexpr bool IsMedianFp = std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>;

// schMode 即计算路径（tilingkey，见 MedianPath）；dtype 由 DTYPE_INPUT 编译期决定。二者均编译期，无运行期分发分支。
template <uint32_t schMode>
__global__ __aicore__ void median(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MedianTilingData);
    GET_TILING_DATA_WITH_STRUCT(MedianTilingData, td, tiling);
    using T = DTYPE_INPUT;
    uint32_t b = (uint32_t)td.batch;
    uint32_t L = (uint32_t)td.redLen;
    uint32_t m = (uint32_t)td.mid;
    uint32_t ns = (uint32_t)td.nSeg;
    uint32_t Pad = (uint32_t)td.pad;

    uint32_t bid = GetBlockIdx(), bn = GetBlockNum();
    if (bn == 0)
        return;
    TPipe pipe;

    // ----- BigSort：fp 单行大 L 多核分段（nSeg>0；仅 fp）-----
    if constexpr (schMode == MEDIAN_PATH_BIGSORT) {
        if constexpr (IsMedianFp<T>) {
            NsMedian::MedianBigSort<T> op;
            op.Init(&pipe, x, y, z, workspace, b, L, m, ns, bid, bn);
            op.Process();
        }
        return;
    } else {
        // 其余路径：先算 (s, e) per-core
        uint32_t per = (b + bn - 1) / bn, s = bid * per, e = s + per;
        if (e > b)
            e = b;
        if (s >= b)
            return;

        // ----- Heap：int64 标量堆排 -----
        if constexpr (schMode == MEDIAN_PATH_HEAP) {
            if constexpr (std::is_same_v<T, int64_t>) {
                NsMedian::MedianHeap<int64_t> op;
                op.Init(&pipe, x, y, z, s, e, L, m);
                op.Process();
            }
        }
        // ----- SMALL：fp 小 L(≤32) 大 batch（连续搬 + Gather 散槽 + Sort32）-----
        else if constexpr (schMode == MEDIAN_PATH_SMALL) {
            if constexpr (IsMedianFp<T>) {
                NsMedian::MedianSmallL<T> op;
                op.Init(&pipe, x, y, z, s, e, L, m);
                op.Process();
            }
        }
        // ----- MAIN：L≤8192 多行 Sort32+MrgSort（非 int64；含 int 小 L）；Pad 档位由 host 下发 -----
        else if constexpr (schMode == MEDIAN_PATH_MAIN) {
            if constexpr (!std::is_same_v<T, int64_t>) {
                NsMedian::MedianMain<T> op;
                op.Init(&pipe, x, y, z, s, e, L, m, Pad);
                op.Process();
            }
        }
        // ----- BigSel：fp 8192<L≤24576 整行常驻向量化二分计数 -----
        else if constexpr (schMode == MEDIAN_PATH_BIGSEL) {
            if constexpr (IsMedianFp<T>) {
                NsMedian::MedianBigSel<T> op;
                op.Init(&pipe, x, y, z, s, e, L, m);
                op.Process();
            }
        }
        // ----- Big：fp L>24576 或 int L>8192（非 int64）分块二分计数 -----
        else if constexpr (schMode == MEDIAN_PATH_BIG) {
            if constexpr (!std::is_same_v<T, int64_t>) {
                NsMedian::MedianBig<T> op;
                op.Init(&pipe, x, y, z, s, e, L, m);
                op.Process();
            }
        }
    }
}
