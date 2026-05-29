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
 * \file foreach_tan_tiling_data.h
 * \brief tiling data struct for foreach_tan operator
 */

#ifndef _FOREACH_TAN_TILING_DATA_H_
#define _FOREACH_TAN_TILING_DATA_H_

constexpr int32_t FOREACH_TAN_MAX_TENSOR_NUM = 256;

struct ForeachTanTilingData {
    int32_t tensorNum = 0;                                        // TensorList 中 Tensor 的数量
    int32_t needCoreNum = 1;                                      // 需要的核数
    int64_t tensorElements[FOREACH_TAN_MAX_TENSOR_NUM] = {0};    // 每个 Tensor 的元素数量
};

#endif