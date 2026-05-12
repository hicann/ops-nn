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
 * \file adam_apply_one_with_decay_tiling_data.h
 * \brief adam_apply_one_with_decay_tiling_data
 */

#ifndef ADAM_APPLY_ONE_WITH_DECAY_TILING_DATA_H
#define ADAM_APPLY_ONE_WITH_DECAY_TILING_DATA_H

#include <stdint.h>
struct AdamApplyOneWithDecayTilingData {
    int64_t smallCoreDataNum;
    int64_t bigCoreDataNum;
    int64_t finalBigTileNum;
    int64_t finalSmallTileNum;
    int64_t tileDataNum;
    int64_t smallTailDataNum;
    int64_t bigTailDataNum;
    int64_t tailBlockNum;
};

#endif // ADAM_APPLY_ONE_WITH_DECAY_TILING_DATA_H
