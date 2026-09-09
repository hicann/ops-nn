/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pool_ub_strided_copy.h
 * \brief AvgPool/MaxPoolV3 pad kernel 共用的 UB 内按 batch/row 跨步拷贝接口。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_POOL_UB_STRIDED_COPY_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_POOL_UB_STRIDED_COPY_H_

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

namespace PoolUtils {
namespace DataMove {

constexpr int32_t POOL_COPY_B64 = 8;

/*
 * 功能：UB 内按 batch/row 跨步拷贝，主循环按 repeatElm 对齐搬入，尾段按 tailCols 搬入。
 */
template <typename T>
__aicore__ inline void CustomCopy(const __ubuf__ T* dstAddr, const __ubuf__ T* srcAddr, uint32_t srcBatchStride,
                                  uint32_t srcRowStride, uint32_t dstBatchStride, uint32_t dstRowStride,
                                  uint32_t dstRowOffset, uint32_t dstColOffset, uint16_t batch, uint16_t rows,
                                  uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm)
{
    using RegDstT = typename std::conditional<sizeof(T) == POOL_COPY_B64,
                                              AscendC::Reg::RegTensor<T, AscendC::Reg::RegTraitNumTwo>,
                                              AscendC::Reg::RegTensor<T>>::type;
    RegDstT v0;
    AscendC::Reg::UnalignRegForStore u0;

    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t j = 0; j < rows; j++) {
            __ubuf__ T* curSrcAddr = (__ubuf__ T*)srcAddr + i * srcBatchStride + j * srcRowStride;
            __ubuf__ T* curDstAddr = (__ubuf__ T*)dstAddr + i * dstBatchStride + (j + dstRowOffset) * dstRowStride +
                                     dstColOffset;
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

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_POOL_UB_STRIDED_COPY_H_
