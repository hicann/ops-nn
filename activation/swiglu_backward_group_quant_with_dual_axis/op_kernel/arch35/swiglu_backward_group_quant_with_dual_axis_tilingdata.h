/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_TILINGDATA_H
#define SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_TILINGDATA_H
#include <cstdint>

struct SwigluBackwardGroupQuantWithDualAxisMxTilingData {
    uint32_t usedCoreNum;
    uint32_t totalRows;
    uint32_t dimBatch;
    uint32_t dimM;
    uint32_t dimN;
    uint32_t numGroups;
    uint32_t quantMode;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t nTiles;
    uint32_t gradWeightTileH;
    uint32_t gradWeightTileTokens;
    float alpha;
    float clampLimit;
    float bias;
};
#endif
