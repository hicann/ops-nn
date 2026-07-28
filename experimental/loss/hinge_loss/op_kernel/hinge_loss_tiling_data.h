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
 * \file hinge_loss_tiling_data.h
 * \brief tiling data struct for multi-core hinge loss
 */

#ifndef _HINGE_LOSS_TILING_DATA_H_
#define _HINGE_LOSS_TILING_DATA_H_

#include <cstdint>

struct HingeLossTilingData {
    uint32_t smallCoreDataNum = 0;
    uint32_t bigCoreDataNum = 0;
    uint32_t finalBigTileNum = 0;
    uint32_t finalSmallTileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t smallTailDataNum = 0;
    uint32_t bigTailDataNum = 0;
    uint32_t tailBlockNum = 0;
};
#endif
