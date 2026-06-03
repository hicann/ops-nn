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
 * \file foreach_asin_tiling_data.h
 * \brief tiling data struct for foreach_asin
 */

#ifndef FOREACH_ASIN_TILING_DATA_H
#define FOREACH_ASIN_TILING_DATA_H

#include <cstdint>

constexpr int32_t FOREACH_ASIN_MAX_TENSOR_NUM = 256;

struct ForeachAsinTilingData {
    int32_t needCoreNum;                                    // 实际使用核数
    int64_t totalElements;                                  // 所有 tensor 的总元素数
    int64_t perCoreElements;                                // 单核处理元素数
    int32_t tensorNum;                                      // tensor 数量（1~256）
    int64_t tensorCumulativeOffset[FOREACH_ASIN_MAX_TENSOR_NUM + 1]; // 累计偏移量数组
};

#endif // FOREACH_ASIN_TILING_DATA_H
