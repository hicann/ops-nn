/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdint>

struct GlobalLpPoolTilingData {
    // ─── pattern description ───
    int32_t axisNum;       // = 2 (axis count after merge)
    int64_t axisShape[9];  // axis sizes after merge: [A, R, 1, ...]
    int64_t axisStride[9]; // GM strides in elements: [C*H*W, W, 0, ...]

    // ─── multi-core split ───
    int64_t aLoopCntTotal;     // = aSplitChunkCnt (only one outer A axis)
    int64_t aSplitChunkCnt;    // = CeilDiv(axisShape[0], aUbFactor)
    int64_t aBigCoreLoopCnt;   // big core loop count (= small + 1)
    int64_t aSmallCoreLoopCnt; // small core loop count (= aLoopCntTotal / coreNum)
    int32_t aBigCoreCnt;       // big core count (= aLoopCntTotal % coreNum)
    int32_t usedCoreNum;       // actual used core count

    // ─── UB split ───
    int32_t aSplitAxisIdx;   // = 0 (outermost A)
    int32_t rSplitAxisIdx;   // = 1 (outermost R)
    int64_t aUbFactor;       // valid: A chunk element count
    int64_t aUbFactorAlign;  // padded: = aUbFactor (tail-R, A is not burst tail)
    int64_t rUbFactor;       // valid: R chunk element count
    int64_t rUbFactorAlign;  // padded: CeilAlign(rUbFactor, BS) when rSplit==LastR && isTailR
    int64_t innerAProd;      // = 1
    int64_t innerAProdAlign; // = 1
    int64_t innerRProd;      // = 1
    int64_t innerRProdAlign; // = 1

    // ─── outer R loop ───
    int64_t rLoopCntTotal; // = CeilDiv(axisShape[1], rUbFactor)

    // ─── UB buffer sizes in bytes (blockSize aligned) ───
    int64_t preReduceUbSize;  // CeilAlign(unit * sizeof(D_T), blockSize)
    int64_t postReduceUbSize; // CeilAlign(aTotal * sizeof(D_T), blockSize)
    int64_t tmpBufUbSize;     // CeilAlign(unit * sizeof(float), blockSize)
    int64_t cacheBufUbSize;   // = 16 * 1024
    int64_t cacheLevelStride; // = CeilAlign(aUbFactorAlign, 8) — cache tree per-layer
                              //   stride in fp32 lanes, 32B-aligned for VF LoadAlign
    int64_t is_fast_path;     // 1 = no need ClearInnerBurstTailPadVf

    // ─── operator-specific ───
    float p;    // Lp norm power exponent
    float invP; // = 1.0f / p, precomputed for post-elewise
};
