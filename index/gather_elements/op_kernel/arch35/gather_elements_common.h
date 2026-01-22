/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gather_elements_common.h
 * \brief
 */
#ifndef GATHER_ELEMENTS_COMMON_H
#define GATHER_ELEMENTS_COMMON_H

#include "kernel_operator.h"

namespace GatherElements {
constexpr uint32_t THREAD_DIM_2048 = 2048;
constexpr uint32_t THREAD_DIM_1024 = 1024;
constexpr uint32_t THREAD_DIM_512 = 512;
constexpr uint32_t MAX_DIM_LEN = 8;
constexpr uint16_t DIM1 = 1;
constexpr uint16_t DIM2 = 2;
constexpr uint16_t DIM3 = 3;
constexpr uint16_t DIM4 = 4;
constexpr uint16_t DIM5 = 5;
constexpr uint16_t DIM6 = 6;
constexpr uint16_t DIM7 = 7;
constexpr uint16_t DIM8 = 8;
constexpr uint16_t DIM0_INDEX = 7;
constexpr uint16_t DIM1_INDEX = 6;
constexpr uint16_t DIM2_INDEX = 5;
constexpr uint16_t DIM3_INDEX = 4;
constexpr uint16_t DIM4_INDEX = 3;
constexpr uint16_t DIM5_INDEX = 2;
constexpr uint16_t DIM6_INDEX = 1;
constexpr uint16_t DIM7_INDEX = 0;
constexpr int64_t SUB_MINUS_SIZE = 64;
constexpr int64_t RIGHT_SHIFT_LENGTH = 32;
constexpr uint16_t MS_IDX0 = 0;
constexpr uint16_t MS_IDX1 = 1;
constexpr uint16_t MS_IDX2 = 2;
constexpr uint16_t MS_IDX3 = 3;
constexpr uint16_t MS_IDX4 = 4;
constexpr uint16_t MS_IDX5 = 5;
constexpr uint16_t MS_IDX6 = 6;
constexpr uint16_t MS_IDX7 = 7;
constexpr uint32_t M_SHIFT_OFFSET = 7;
constexpr uint16_t B32 = 4;
constexpr uint16_t TWO = 2;
constexpr uint16_t THREE = 3;
} // namespace GatherElements
#endif // GATHER_ELEMENTS_COMMON_H