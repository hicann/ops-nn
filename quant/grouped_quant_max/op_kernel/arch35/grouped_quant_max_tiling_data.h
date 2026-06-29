/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file grouped_quant_max_tiling_data.h
 * @brief GroupedQuantMax TilingData structure definition
 */

#ifndef _GROUPED_QUANT_MAX_TILING_DATA_H_
#define _GROUPED_QUANT_MAX_TILING_DATA_H_
#include <cstdint>

class GroupedQuantMaxTilingData {
public:
    int64_t totalCoreNum;
    int64_t actualCoreNum;
    int64_t blockTailCoreNum;
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t ubFactor;
    int64_t dim1;
    int64_t roundMode;
    int64_t numGroups;
};

#endif // _GROUPED_QUANT_MAX_TILING_DATA_H_
