/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file uniqur_consecutive_constant.h
 * \brief uniqur_consecutive_constant
 */
#ifndef UNIQUE_CONSECUTIVE_CONSTANT_H
#define UNIQUE_CONSECUTIVE_CONSTANT_H

constexpr int64_t UINT64_SHAPE_DIM_ONE = 0x80000001;
constexpr int32_t SHAPE_LEN = 27; //3 * (1 + 8), each of the 3 outputs has 1 axis number and 8 axis information

constexpr uint64_t START_POSITION = 0;
constexpr uint64_t SINGLE_OFFSET = 1;
constexpr uint64_t DOUBLE_OFFSET = 2;
constexpr uint64_t MIDPOINT_DIVIDER = 2;

constexpr int32_t SHAPE0_SIZE_IDX = 0;
constexpr int32_t SHAPE0_DIM0_IDX = 1;
constexpr int32_t SHAPE1_SIZE_IDX = 9;
constexpr int32_t SHAPE1_DIM0_IDX = 10;
constexpr int32_t SHAPE2_SIZE_IDX = 18;
constexpr int32_t SHAPE2_DIM0_IDX = 19;

#endif  // UNIQUE_CONSECUTIVE_CONSTANT_H
