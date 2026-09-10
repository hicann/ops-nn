/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been technically reviewed for functional accuracy.
 */

/*!
 * \file l2_normalize_grad_tiling_data.h
 * \brief L2NormalizeGrad arch35 (Ascend950) plain tiling-data struct.
 *
 * One definition shared by host tiling (context->GetTilingData<L2NormalizeGradTilingData>())
 * and kernel (GET_TILING_DATA_WITH_STRUCT). [outer, D, inner] reduction model; cores split by outer.
 */
#ifndef _L2_NORMALIZE_GRAD_TILING_DATA_H_
#define _L2_NORMALIZE_GRAD_TILING_DATA_H_

#include <cstdint>

struct L2NormalizeGradTilingData {
    int64_t outer = 0;       // product of dims before the reduce axis
    int64_t dimLen = 0;      // D = length of the reduce axis
    int64_t inner = 0;       // product of dims after the reduce axis
    int64_t blockFactor = 0; // outer groups per (non-tail) core
    int64_t usedCoreNum = 0; // cores actually used
    int64_t colFactor = 0;   // inner columns per tile (strided 7020 / 7030)
    int64_t dFactor = 0;     // reduce-axis rows per tile (strided-split/7030 only; 0 for other keys)
    // ---- UB 缓冲字节数:一律由 host 算准下发,内核只透传,不自行推算 ----
    // (host 预算与内核分配若各算一套,少算一个通道就越界;历史 UB 越界即源于此)
    int64_t qBufBytes = 0;      // x/y/dy/dx 四个队列每块字节数(同尺寸)
    int64_t reduceBufBytes = 0; // reduceBufSq_/reduceBufS_ (7000/7010)
    int64_t accumBufBytes = 0;  // accumBufSq_/accumBufS_ (7010) / 每列累加器 (7030)
    int64_t tmpBufBytes = 0;    // tmpSumSqBuf_/tmpSumSBuf_ (7000/7010)
    int64_t ubFactorElems = 0;  // 单次处理元素数(7000 的 ubFactor / 7010 的 ubFactorD):
                                // 由 host 从平台实际 ubSize 反推,**不是写死常量**
    // ---- 对齐后的量:一律 host 算好,内核直接取用,不在内核再 AlignUp/除法 ----
    // (对齐值若两边各算一套,就又回到 host 预算与内核分配不同源的老问题)
    int64_t colsAlignBlock = 0;    // AlignUp(D, block)      —— 7000 的 GM 行宽
    int64_t colsAlignVL = 0;       // AlignUp(D, 1VL)        —— 7000 的 UB 行宽
    int64_t ubFactorN = 0;         // 每批行数 = ubFactorElems / colsAlignVL (7000)
    int64_t numChunks = 0;         // 分块数 = CeilDiv(D, ubFactorElems) (7010)
    int64_t colFactorAlign = 0;    // AlignUp(colFactor, block) —— 7020/7030 满块 UB 行距
    int64_t tailColTile = 0;       // 尾块实际列数(0=无尾块)     —— 7020/7030
    int64_t tailColAlign = 0;      // AlignUp(tailColTile, block) —— 7020/7030 尾块 UB 行距
    int64_t accElems = 0;          // 7030 每列累加器元素数(含 VL slack)
    int64_t accVLs = 0;            // 7030 累加器 VL 轮数
    int64_t chunkSlots = 0;        // 7030 跨分块结果槽位数(树形规约组大小);=1 退化为顺序累加
    int64_t slotStride = 0;        // 7030 槽位行距(fp32 block 对齐)= ReduceSum<RA> 的 srcShape[1]
    int64_t slotBufBytes = 0;      // 7030 单个槽位数组字节数 = chunkSlots * slotStride * 4
    int64_t colFactorAlignF32 = 0; // AlignUp(colFactor, fp32 block) —— fp32 中间量 tile 的行距
    int64_t tailColAlignF32 = 0;   // AlignUp(tailColTile, fp32 block)
    int64_t midBufBytes = 0;       // fp32 中间量 tile(x^2 / y*dy)每块字节数
    int64_t sumBufBytes = 0;       // 每列规约结果缓冲字节数(含 VL slack)
    float eps = 0.0f;              // denominator floor
};

#endif // _L2_NORMALIZE_GRAD_TILING_DATA_H_
