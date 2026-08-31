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
 * \file pool_reg_element_data_move.h
 * \brief 池化系列算子共用的寄存器与 UB 之间的索引加载、单元素非对齐写出接口。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_POOL_REG_ELEMENT_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_POOL_REG_ELEMENT_DATA_MOVE_H_

#include <cstdint>

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace PoolUtils {
namespace DataMove {

template <uint16_t REG_NUM, uint16_t IDX, typename U>
__aicore__ inline void LoadIndex(__ubuf__ U* indexAddr, AscendC::Reg::RegTensor<U>& index)
{
    constexpr uint32_t repeatNum = Ops::Base::GetVRegSize() / sizeof(U);
    if constexpr (REG_NUM > IDX) {
        AscendC::Reg::LoadAlign(index, indexAddr + IDX * repeatNum);
    }
}

template <typename T>
__aicore__ inline void StoreElement(const __ubuf__ void* output, AscendC::Reg::RegTensor<T>& src, uint32_t offset,
                                    uint32_t element)
{
    AscendC::Reg::UnalignRegForStore u0;
    auto dstAddr = (__ubuf__ T*)(output) + offset;
    AscendC::Reg::StoreUnAlign(dstAddr, src, u0, element);
    AscendC::Reg::StoreUnAlignPost(dstAddr, u0, 0);
}

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_POOL_REG_ELEMENT_DATA_MOVE_H_
