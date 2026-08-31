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
 * \file pool_calc_buffer_data_move.h
 * \brief MaxPool/MaxPoolGrad 共用的 UB 内计算缓冲搬运接口（支持 pad 偏移的 2D/3D 拷贝）。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_POOL_CALC_BUFFER_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_POOL_CALC_BUFFER_DATA_MOVE_H_

#include <cstdint>

#include "kernel_operator.h"

namespace PoolUtils {
namespace DataMove {

/*
 * 功能：把源 UB 缓冲按 batch/rows 两层循环拷贝到目的计算缓冲，目的侧按 dstRowOffset/dstColOffset 预留 pad 区。
 */
template <typename T>
__aicore__ inline void CopyToCalcBuffer2DCommon(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t batch, uint16_t rows,
                                                uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm,
                                                uint32_t srcBatchStride, uint32_t srcRowStride, uint32_t dstBatchStride,
                                                uint32_t dstRowStride, uint32_t dstRowOffset, uint32_t dstColOffset)
{
    AscendC::Reg::RegTensor<T> v0;
    AscendC::Reg::UnalignRegForStore u0;
    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t j = 0; j < rows; j++) {
            __ubuf__ T* curSrcAddr = srcAddr + i * srcBatchStride + j * srcRowStride;
            __ubuf__ T* curDstAddr = dstAddr + i * dstBatchStride + (j + dstRowOffset) * dstRowStride + dstColOffset;
            for (uint16_t k = 0; k < loopCols; k++) {
                AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                AscendC::Reg::StoreUnAlign(curDstAddr, v0, u0, repeatElm);
            }
            AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
            AscendC::Reg::StoreUnAlign(curDstAddr, v0, u0, tailCols);
            AscendC::Reg::StoreUnAlignPost(curDstAddr, u0, 0);
        }
    }
}

/*
 * 功能：CopyToCalcBuffer2DCommon 的 3D 版本，额外增加 deps 维度与对应的 dep stride/offset。
 */
template <typename T>
__aicore__ inline void CopyToCalcBuffer3DCommon(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t batch, uint16_t deps,
                                                uint16_t rows, uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm,
                                                uint32_t srcBatchStride, uint32_t srcDepStride, uint32_t srcRowStride,
                                                uint32_t dstBatchStride, uint32_t dstDepStride, uint32_t dstRowStride,
                                                uint32_t dstDepOffset, uint32_t dstRowOffset, uint32_t dstColOffset)
{
    AscendC::Reg::RegTensor<T> v0;
    AscendC::Reg::UnalignRegForStore u0;
    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t t = 0; t < deps; t++) {
            for (uint16_t j = 0; j < rows; j++) {
                __ubuf__ T* curSrcAddr = srcAddr + i * srcBatchStride + t * srcDepStride + j * srcRowStride;
                __ubuf__ T* curDstAddr = dstAddr + i * dstBatchStride + (t + dstDepOffset) * dstDepStride +
                                         (j + dstRowOffset) * dstRowStride + dstColOffset;
                for (uint16_t k = 0; k < loopCols; k++) {
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                    AscendC::Reg::StoreUnAlign(curDstAddr, v0, u0, repeatElm);
                }
                AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                AscendC::Reg::StoreUnAlign(curDstAddr, v0, u0, tailCols);
                AscendC::Reg::StoreUnAlignPost(curDstAddr, u0, 0);
            }
        }
    }
}

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_POOL_CALC_BUFFER_DATA_MOVE_H_
