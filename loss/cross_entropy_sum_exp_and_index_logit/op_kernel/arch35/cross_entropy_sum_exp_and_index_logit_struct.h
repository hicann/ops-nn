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
 * \file cross_entropy_sum_exp_and_index_logit_struct.h
 * \brief A5 (ascend950) kernel-side tiling data struct
 */
#ifndef CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_STRUCT_H_
#define CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_STRUCT_H_

#include "kernel_tiling/kernel_tiling.h"

// 两级切分（核间 floor+remainder 均衡 + 每核独立内循环参数），字段顺序与源 BEGIN_TILING_DATA_DEF 一致：
//   核间：N/usedCores → tokensPerCore(头核) / tokensPerCoreTail(尾核)，headCoreNum=N%usedCores
//   核内：每核 tokens 按 rowBlockMax(UB 反推) 切，blockNum=ceil(tokens/rowBlockMax)

struct CrossEntropySumExpAndIndexLogitRegBaseTilingData {
    uint32_t N;                 // 展平后 token 总数 = prod(logits.shape[:-1])
    uint32_t vLocal;            // 最后一维长度 V_local
    uint32_t usedCores;         // 实际使用核数 = min(N, aivNum)
    uint32_t headCoreNum;       // 头核个数 = N % usedCores（前 headCoreNum 核多 1 token）
    uint32_t tokensPerCore;     // 头核 token 数 = ceil(N / usedCores)
    uint32_t tokensPerCoreTail; // 尾核 token 数 = floor(N / usedCores)
    uint32_t headBlockNum;      // 头核核内循环块数
    uint32_t tailBlockNum;      // 尾核核内循环块数
    uint32_t rowBlockMax;       // rowBlock 上限（UB 反推），kernel InitBuffer 用
    uint32_t vTile;             // 核内 V_local tile 长度
    uint32_t vLoopNum;          // ceil(vLocal / vTile)
    uint32_t lastVTile;         // vLocal - vTile*(vLoopNum-1)
    uint32_t reduceTmpBytes;    // ReduceSum(AR) sharedTmpBuffer 字节（host 官方接口算好）
    int64_t vocabStart;         // vocab_start_index
    int64_t vocabEnd;           // vocab_end_index
};

#endif // CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_STRUCT_H_
