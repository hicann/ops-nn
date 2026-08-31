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
 * \file adaptive_pool_transpose_data_move.h
 * \brief Adaptive 系列池化共用的 B16/B32 分块转置接口。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_POOL_TRANSPOSE_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_POOL_TRANSPOSE_DATA_MOVE_H_

#include <cstdint>

#include "kernel_operator.h"

namespace PoolUtils {
namespace DataMove {

constexpr uint32_t ADAPTIVE_TRANS_ADDR_LEN = 16;
constexpr uint32_t ADAPTIVE_TRANS_LEN_B32 = 8;

/*
 * 功能：对 B16 数据做分块转置。
 * 说明：colNum 等于块内对齐长度时 repeat 走行方向，否则 repeat 走列方向、行方向按 16 行分组。
 */
template <typename T>
__aicore__ inline void TransposeB16(AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t rowNum,
                                    uint32_t colNum, uint32_t ubBlockSize)
{
    uint64_t dstList[ADAPTIVE_TRANS_ADDR_LEN];
    uint64_t srcList[ADAPTIVE_TRANS_ADDR_LEN];
    AscendC::TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;

    const uint32_t transPoseAlign = ubBlockSize / sizeof(T);
    if (colNum == transPoseAlign) {
        /* repeat在行方向，一次处理16*16个B16 */
        transDataParams.repeatTimes = rowNum / ADAPTIVE_TRANS_ADDR_LEN;
        /* dstStride为16*sizeof(T)，srcStride为16个dataBlock */
        transDataParams.dstRepStride = ADAPTIVE_TRANS_ADDR_LEN * sizeof(T) / ubBlockSize;
        transDataParams.srcRepStride = ADAPTIVE_TRANS_ADDR_LEN;

        for (int32_t i = 0; i < static_cast<int32_t>(ADAPTIVE_TRANS_ADDR_LEN); i++) {
            srcList[i] = static_cast<uint64_t>(src[i * transPoseAlign].GetPhyAddr());
            dstList[i] = static_cast<uint64_t>(dst[i * rowNum].GetPhyAddr());
        }

        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }

        AscendC::TransDataTo5HD<T>(dstList, srcList, transDataParams);
    } else {
        /* repeatTimes不会为1 */
        transDataParams.repeatTimes = colNum / transPoseAlign;
        transDataParams.dstRepStride = rowNum;
        transDataParams.srcRepStride = 1;

        for (int32_t rowLoopIdx = 0; rowLoopIdx < static_cast<int32_t>(rowNum / ADAPTIVE_TRANS_ADDR_LEN);
             rowLoopIdx++) {
            for (int32_t i = 0; i < static_cast<int32_t>(ADAPTIVE_TRANS_ADDR_LEN); i++) {
                srcList[i] = static_cast<uint64_t>(
                    src[rowLoopIdx * ADAPTIVE_TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
                dstList[i] = static_cast<uint64_t>(dst[rowLoopIdx * ADAPTIVE_TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
            }
            AscendC::TransDataTo5HD<T>(dstList, srcList, transDataParams);
        }
    }
}

/*
 * 功能：对 B32 数据做分块转置。
 * 说明：B32 每个 dstList 项对应两个半块，故按 8 组、每组两条地址填充。
 */
template <typename I>
__aicore__ inline void TransposeB32(AscendC::LocalTensor<I> dst, AscendC::LocalTensor<I> src, uint32_t rowNum,
                                    uint32_t colNum, uint32_t ubBlockSize)
{
    uint64_t dstList[ADAPTIVE_TRANS_ADDR_LEN];
    uint64_t srcList[ADAPTIVE_TRANS_ADDR_LEN];
    AscendC::TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;

    const uint32_t transPoseAlign = ubBlockSize / sizeof(I);
    if (colNum == transPoseAlign) {
        /* repeat在行方向，一次处理16*8个b32 */
        transDataParams.repeatTimes = rowNum / ADAPTIVE_TRANS_ADDR_LEN;
        /* dstStride为16*sizeof(I)，srcStride为16个dataBlock */
        transDataParams.dstRepStride = ADAPTIVE_TRANS_ADDR_LEN * sizeof(I) / ubBlockSize;
        transDataParams.srcRepStride = ADAPTIVE_TRANS_ADDR_LEN;

        for (int32_t i = 0; i < static_cast<int32_t>(ADAPTIVE_TRANS_ADDR_LEN); i++) {
            srcList[i] = static_cast<uint64_t>(src[i * transPoseAlign].GetPhyAddr());
        }
        for (int32_t i = 0; i < static_cast<int32_t>(ADAPTIVE_TRANS_LEN_B32); i++) {
            dstList[i * 2] = static_cast<uint64_t>(dst[i * rowNum].GetPhyAddr());
            dstList[i * 2 + 1] = static_cast<uint64_t>(dst[i * rowNum + transPoseAlign].GetPhyAddr());
        }

        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }

        AscendC::TransDataTo5HD<I>(dstList, srcList, transDataParams);
    } else {
        /* repeatTimes不会为1 */
        transDataParams.repeatTimes = colNum / transPoseAlign;
        transDataParams.dstRepStride = rowNum;
        transDataParams.srcRepStride = 1;

        for (int32_t rowLoopIdx = 0; rowLoopIdx < static_cast<int32_t>(rowNum / ADAPTIVE_TRANS_ADDR_LEN);
             rowLoopIdx++) {
            for (int32_t i = 0; i < static_cast<int32_t>(ADAPTIVE_TRANS_ADDR_LEN); i++) {
                srcList[i] = static_cast<uint64_t>(
                    src[rowLoopIdx * ADAPTIVE_TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            }
            for (int32_t i = 0; i < static_cast<int32_t>(ADAPTIVE_TRANS_LEN_B32); i++) {
                dstList[i * 2] = static_cast<uint64_t>(
                    dst[rowLoopIdx * ADAPTIVE_TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
                dstList[i * 2 + 1] = static_cast<uint64_t>(
                    dst[rowLoopIdx * ADAPTIVE_TRANS_ADDR_LEN + i * rowNum + transPoseAlign].GetPhyAddr());
            }
            AscendC::TransDataTo5HD<I>(dstList, srcList, transDataParams);
        }
    }
}

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_POOL_TRANSPOSE_DATA_MOVE_H_
