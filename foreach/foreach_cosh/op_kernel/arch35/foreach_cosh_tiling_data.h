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
 * \file foreach_cosh_tiling_data.h
 * \brief tiling data struct for foreach_cosh
 */

#ifndef FOREACH_COSH_TILING_DATA_H
#define FOREACH_COSH_TILING_DATA_H

#include <cstdint>

constexpr int32_t FOREACH_COSH_MAX_TENSOR_NUM = 256;

struct ForeachCoshTilingData {
    int32_t needCoreNum = 0;
    int32_t tensorCount = 0;
    int64_t totalElements = 0;
    // per-core contiguous chunk of the flattened element range; the last core uses lastCoreElements
    int64_t perCoreElements = 0;
    int64_t lastCoreElements = 0;
    // prefix starts of each tensor in the flattened range:
    // tensorStarts[0] == 0, tensorStarts[i] is the first flat index of tensor i,
    // tensorStarts[tensorCount] == totalElements (so empty tensors hold equal starts)
    int64_t tensorStarts[FOREACH_COSH_MAX_TENSOR_NUM + 1] = {0};
};

#endif // FOREACH_COSH_TILING_DATA_H
