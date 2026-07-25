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
 * \file hard_sigmoid_tiling_data.h
 * \brief HardSigmoid TilingData（arch35 regbase 写法）。
 */

#ifndef HARD_SIGMOID_TILING_DATA_H
#define HARD_SIGMOID_TILING_DATA_H

#include <cstdint>

// kernel 侧 in/out 队列的缓冲深度（Double Buffer）。
// Host tiling 依据它推导每元素 UB 占用（见 UbBytesPerElement），二者必须一致，
// 故置于 Host/Kernel 共享头，禁止任一侧另行定义。
constexpr int64_t HARD_SIGMOID_BUFFER_NUM = 2;

struct __attribute__((aligned(8))) HardSigmoidTilingData {
    int64_t totalElements = 0; // 元素总数
    int64_t blockFactor = 0;   // 单核处理的元素数
    int64_t ubFactor = 0;      // 单次 UB 搬运/计算的元素数
    float alpha = 1.0f / 6.0f; // y = clamp(alpha*x + beta, 0, 1)
    float beta = 0.5f;
};

#endif // HARD_SIGMOID_TILING_DATA_H
