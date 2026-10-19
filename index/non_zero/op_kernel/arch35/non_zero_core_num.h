/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file non_zero_core_num.h
 * \brief 跨核归约区(GM workspace 与 UB addUb)的大小计算。
 *        归约区按"每核一个槽位"组织，槽位数取决于参与计算的核数，
 *        因此不能写死核数，否则核数超过基准值的芯片(如 Ascend960)会越界。
 */

#ifndef CANN_NON_ZERO_CORE_NUM_H
#define CANN_NON_ZERO_CORE_NUM_H

#include "kernel_operator.h"

namespace NonZero {
// 跨核归约区的基准核数。实际核数不超过该值时按该值分配，与历史实现保持一致；
// 超过该值时按实际核数放大。
constexpr int32_t CORE_NUM_CAP = 72;
// small mask 路径每核占用的归约槽位数(uint32 个数)，同时也是 addUb 的每核字节数。
constexpr int32_t REDUCE_SLOT_NUM_SMALL = 32;
// big mask 路径每核占用的归约槽位数，同时也是 addUb 的每核字节数。
constexpr int32_t REDUCE_SLOT_NUM_BIG = 128;

// 按实际参与计算的核数计算归约区大小，核数不足基准值时取基准值。
__aicore__ inline int32_t GetReduceRegionSize(int32_t realCoreNum, int32_t slotNumPerCore)
{
    int32_t coreNum = realCoreNum > CORE_NUM_CAP ? realCoreNum : CORE_NUM_CAP;
    return coreNum * slotNumPerCore;
}
} // namespace NonZero

#endif // CANN_NON_ZERO_CORE_NUM_H
