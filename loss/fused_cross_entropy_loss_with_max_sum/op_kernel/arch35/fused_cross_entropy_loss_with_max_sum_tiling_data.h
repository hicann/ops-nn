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
 * \file fused_cross_entropy_loss_with_max_sum_tiling_data.h
 * \brief tiling data struct shared by host tiling and arch35 kernel
 */

#ifndef _FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_REGBASE_TILING_DATA_H_
#define _FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_REGBASE_TILING_DATA_H_

// 每个UB分块处理的行数（host tiling与kernel共用）
constexpr int64_t FUSED_CE_MAX_SUM_A_PER_LOOP = 8;
// v维分块列数的对齐粒度，需不小于一个向量寄存器可容纳的fp32元素数（64）
constexpr int64_t FUSED_CE_MAX_SUM_V_ALIGN = 64;

// vocab_parallel_logits dtype id（kernel运行时分发用）
constexpr int64_t FUSED_CE_MAX_SUM_DTYPE_FP32 = 0;
constexpr int64_t FUSED_CE_MAX_SUM_DTYPE_FP16 = 1;
constexpr int64_t FUSED_CE_MAX_SUM_DTYPE_BF16 = 2;

struct FusedCrossEntropyLossWithMaxSumRegBaseTilingData {
    int64_t formerCoreNum;  // 多分担一行的核数
    int64_t formerRows;     // former核每核处理的行数
    int64_t latterRows;     // latter核每核处理的行数
    int64_t vPerLoop;       // v维单次UB分块的列数（已按FUSED_CE_MAX_SUM_V_ALIGN对齐，即UB内行距）
    int64_t vLen;           // vocab维长度V
    int64_t elementsNumber; // 省显存路径下bt维单次处理的元素数（已按FUSED_CE_MAX_SUM_V_ALIGN对齐）
    int64_t vocabDtypeId;   // vocab_parallel_logits的dtype：0=fp32, 1=fp16, 2=bf16
    int64_t vCores;         // 每行在v维上切分的核数（1=不切分）；bt<coreNum且v>vPerLoop时>1
    int64_t vChunk; // v切分时每核负责的v长度（已按FUSED_CE_MAX_SUM_V_ALIGN对齐）；不切分时为vLen
};

#endif // _FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_REGBASE_TILING_DATA_H_
