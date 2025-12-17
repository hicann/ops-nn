/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/


/* !
 * \file matmul_v3_common_advanced.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_COMMON_ADVANCED_H__
#define __OP_HOST_MATMUL_V3_COMMON_ADVANCED_H__

#include <cstdint>
#include "graph/types.h"

namespace optiling {
namespace matmul_v3_advanced {
constexpr uint64_t L0A_SIZE_2 = 64 * 1024UL;
constexpr uint64_t MB_SIZE = 1024 * 1024UL;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t DB_OFF_SIZE = 1UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_64 = 64UL;
constexpr uint64_t BASIC_BLOCK_K_256_BYTE = 256UL;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t NUM_THREE = 3UL;
constexpr uint64_t NUM_FOUR = 4UL;
constexpr uint64_t CACHELINE = 512UL;
constexpr uint64_t BLOCK_BYTE_SIZE = 32UL;
constexpr uint64_t BIAS_TABLE_NUM = 256UL;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint64_t DATA_SIZE_FP16 = 2UL;
constexpr uint64_t BASE_STEP = 1UL;
constexpr uint64_t ITER_COL_FIRST = 0UL;
constexpr uint64_t ITER_ROW_FIRST = 1UL;
constexpr uint64_t BASIC_L1_BUFFER_NUM = 4UL;
constexpr uint64_t RPC_WORKSIZE = 20UL;

struct BatchMatMulV3RunInfo {
    uint64_t iterBatch = 0UL;
    uint64_t batchOutNum = 1UL;
};

struct MatMulV3BatchInfo {
    uint64_t batchA = 1UL;
    uint64_t batchA0 = 1UL;
    uint64_t batchA1 = 1UL;
    uint64_t batchA2 = 1UL;
    uint64_t batchA3 = 1UL;
    uint64_t batchB = 1UL;
    uint64_t batchB0 = 1UL;
    uint64_t batchB1 = 1UL;
    uint64_t batchB2 = 1UL;
    uint64_t batchB3 = 1UL;
    uint64_t batchC = 1UL;
    uint64_t batchC0 = 1UL;
    uint64_t batchC1 = 1UL;
    uint64_t batchC2 = 1UL;
    uint64_t batchC3 = 1UL;
    uint64_t batchBias = 1UL;
};

struct MatMulV3Args {
    const char *opName = nullptr;
    bool isATrans = false;
    bool isBTrans = false;
    bool isHf32 = false;
    bool hasBias = false;
    ge::DataType aType = ge::DT_FLOAT16;
    ge::DataType bType = ge::DT_FLOAT16;
    ge::DataType cType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_ND;
    uint64_t mValue = 0UL;
    uint64_t mOriValue = 0UL;
    uint64_t nOriValue = 0UL;
    uint64_t kValue = 0UL;
    uint64_t nValue = 0UL;
    uint64_t aDtypeSize = 1UL;
    uint64_t bDtypeSize = 1UL;
    MatMulV3BatchInfo *batchInfo = nullptr;
};

struct MatMulV3TailInfo {
    uint64_t mCnt = 1UL;
    uint64_t nCnt = 1UL;
    uint64_t kCnt = 1UL;
    uint64_t mTailMain = 0UL;
    uint64_t nTailMain = 0UL;
};

struct MatMulV3MixInfo {
    uint64_t ubDB = 1UL;
};

struct MatMulV3RunInfo {
    uint64_t usedCoreNum = 1UL;
    uint64_t singleCoreM = 1UL;
    uint64_t singleCoreN = 1UL;
    uint64_t singleCoreK = 1UL;
    uint64_t baseM = 1UL;
    uint64_t baseN = 1UL;
    uint64_t baseK = 1UL;
    uint64_t stepKa = 1UL;
    uint64_t stepKb = 1UL;
    uint64_t depthA1 = 1UL;
    uint64_t depthB1 = 1UL;
    uint64_t stepM = 1UL;
    uint64_t stepN = 1UL;
    uint64_t iterateOrder = 0UL;
    uint64_t dbL0C = 0UL;
    uint64_t iterBatchL1 = 1UL;
    uint64_t iterBatchL0 = 1UL;
    uint64_t l1BufferNum = 2UL;
    uint64_t mBaseTailSplitCnt = 1UL;
    uint64_t nBaseTailSplitCnt = 1UL;
    MatMulV3TailInfo tailInfo;
    BatchMatMulV3RunInfo bmmRunInfo;
    MatMulV3MixInfo mixInfo;
};
}
}
#endif // __OP_HOST_MATMUL_V3_COMMON_ADVANCED_H__