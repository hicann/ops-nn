/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\
 * \file lp_loss_tiling_data.h
 * \brief tiling data struct
 */

#ifndef LP_LOSS_TILING_DATA_H
#define LP_LOSS_TILING_DATA_H

#include <cstdint>

struct LpLossTilingData {
    uint64_t totalNum = 0;
    uint64_t coreNum = 0;
    uint64_t blockSize = 0;
    uint64_t smallCoreDataNum = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t finalBigTileNum = 0;
    uint64_t finalSmallTileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t smallTailDataNum = 0;
    uint64_t bigTailDataNum = 0;
    uint64_t tailBlockNum = 0;
    uint64_t reduction = 0;
    float meanFactor = 0.0f;
    uint64_t bufferNum = 0;
};

#endif // LP_LOSS_TILING_DATA_H