/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_maximum_list_tiling_data.h
 * \brief tiling data struct for foreach_maximum_list
 */

#ifndef FOREACH_MAXIMUM_LIST_TILING_DATA_H
#define FOREACH_MAXIMUM_LIST_TILING_DATA_H

#include <cstdint>

struct ForeachMaximumListTilingData {
    int32_t needCoreNum = 0;
    int64_t totalElements = 0;
    int64_t perCoreElements = 0;
    int64_t lastCoreElements = 0;
    int32_t tensorNum = 0;
    int64_t tensorSizes[256] = {0};
};

#endif // FOREACH_MAXIMUM_LIST_TILING_DATA_H
