/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_DYNAMIC_MX_QUANT_TILINGDATA_H
#define GROUPED_DYNAMIC_MX_QUANT_TILINGDATA_H

#include <cstdint>

struct GroupedDynamicMxQuantTilingData {
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t rowSize{0};
    int64_t colSize{0};
    int64_t blockRowSize{0};
    int64_t blockColSize{0};
    int64_t blockRowTailSize{0};
    int64_t blockRowCount{0};
    int64_t groupNum{0};
    int64_t tilingKey{0};
    float invDstTypeMax{0.0f};
};
#endif // GROUPED_DYNAMIC_MX_QUANT_TILINGDATA_H
