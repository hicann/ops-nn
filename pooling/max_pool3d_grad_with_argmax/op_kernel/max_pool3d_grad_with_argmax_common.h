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
 * \file max_pool3d_grad_with_argmax_common.h
 * \brief
 */
#ifndef MAX_POOL_GRAD3D_WITH_ARGMAX_COMMON
#define MAX_POOL_GRAD3D_WITH_ARGMAX_COMMON

#include "../pool_3d_common/arch32/max_pool3d_grad_common.h"

namespace MaxPool3DGradWithArgmaxComm {
using namespace AscendC;
using namespace MaxPool3DGradCommon;

constexpr uint64_t VL_FP32 = 64;
constexpr uint64_t VL_FP16 = 128;
constexpr uint64_t MAX_LIST_NUM = 4;
constexpr uint64_t LARGE_KERNEL = 0;
constexpr uint64_t LARGE_HO = 1;
constexpr uint64_t LARGE_WO = 2;
constexpr uint64_t UINT16_BITS = 16;
constexpr uint64_t B32_VECTOR_MASK = 64;
constexpr uint64_t B16_VECTOR_MASK = 128;
const uint64_t NUM_TWO = 2;
const uint64_t BITSIZE = 16;
constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr uint32_t MAX_REP_NUM = 255;

struct BlockParams : public BlockParamsCommon {
    uint64_t wiShapeAlign = 0;
    uint64_t offsetYD = 0;
    uint64_t offsetYH = 0;
    uint64_t offsetYW = 0;
    uint64_t padDTop = 0;
    uint64_t padHTop = 0;
    uint64_t padWTop = 0;
    uint64_t padDBottom = 0;
    uint64_t padHBottom = 0;
    uint64_t padWBottom = 0;
};

struct TilingParams : public TilingParamsCommon {
    uint64_t kd;
    uint64_t kh;
    uint64_t kw;
    uint64_t sd;
    uint64_t sh;
    uint64_t sw;
    uint64_t padDTop;
    uint64_t padDBottom;
    uint64_t padHTop;
    uint64_t padHBottom;
    uint64_t padWTop;
    uint64_t padWBottom;
    uint64_t baseDoHoWo;
    uint64_t baseDoHoWoAlign8;
    uint64_t baseDoHoWoAlign16;
    uint64_t baseDi;
    uint64_t baseHi;
    uint64_t baseWi;
    uint64_t baseWiAlign;
    uint64_t baseDiHiWi;
    uint64_t baseDiHiWiAlign8;
    uint64_t baseDiHiWiAlign16;
    uint64_t baseDiHiWiAlign;
    uint64_t padGmOffset;
    uint64_t outputDataSize;
};

// only support float/int32_t
// [row, col] -> [col, row]: row:align16,max:64, col:align8
__aicore__ inline void TransposeAddrBase16M8(
    uint64_t dstAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN], uint64_t srcAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
    uint64_t rowNum, uint64_t colNum)
{
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = colNum / BLOCK_NUM_32;
    if (transDataParams.repeatTimes == 1) {
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
    } else {
        transDataParams.srcRepStride = 1;
        transDataParams.dstRepStride = rowNum;
    }

    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        TransDataTo5HD<int32_t>(dstAddrList[r], srcAddrList[r], transDataParams);
    }
}

// only support float/int32_t
// [row, col] -> [col, row]: row:align8, col:align16,max:64
__aicore__ inline void TransposeAddrBase8M16(
    uint64_t dstAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN], uint64_t srcAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
    uint64_t rowNum, uint64_t colNum)
{
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = rowNum / BLOCK_NUM_32;
    if (transDataParams.repeatTimes == 1) {
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
    } else {
        transDataParams.srcRepStride = colNum;
        transDataParams.dstRepStride = 1;
    }
    for (uint64_t r = 0; r < colNum / TRANS_ADDR_LEN; r++) {
        TransDataTo5HD<int32_t>(dstAddrList[r], srcAddrList[r], transDataParams);
    }
}

// only support float16/bfloat16
// [row, col] -> [col, row]: row:align16, max:64, col:align16
__aicore__ inline void TransposeAddrBase16M16(
    uint64_t dstAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN], uint64_t srcAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
    uint64_t rowNum, uint64_t colNum)
{
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = colNum / BLOCK_NUM_16;
    if (transDataParams.repeatTimes == 1) {
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
    } else {
        transDataParams.srcRepStride = 1;
        transDataParams.dstRepStride = rowNum;
    }

    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        // TransData5HD does not support bfloat16, only use half
        TransDataTo5HD<half>(dstAddrList[r], srcAddrList[r], transDataParams);
    }
}

// only support float16/bfloat16
// [row, col] -> [col, row]: row:align16, col:align16, max:64
__aicore__ inline void TransposeAddrBase16M16B(
    uint64_t dstAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN], uint64_t srcAddrList[MAX_LIST_NUM][TRANS_ADDR_LEN],
    uint64_t rowNum, uint64_t colNum)
{
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = rowNum / BLOCK_NUM_16;
    if (transDataParams.repeatTimes == 1) {
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
    } else {
        transDataParams.srcRepStride = colNum;
        transDataParams.dstRepStride = 1;
    }

    for (uint64_t r = 0; r < colNum / TRANS_ADDR_LEN; r++) {
        // TransData5HD does not support bfloat16, only use half
        TransDataTo5HD<half>(dstAddrList[r], srcAddrList[r], transDataParams);
    }
}

} // namespace MaxPool3DGradWithArgmaxComm

#endif // MAX_POOL_GRAD3D_WITH_ARGMAX_COMMON