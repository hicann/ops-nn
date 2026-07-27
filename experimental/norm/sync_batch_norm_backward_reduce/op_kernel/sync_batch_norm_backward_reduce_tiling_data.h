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
 * \file sync_batch_norm_backward_reduce_tiling_data.h
 * \brief tiling data struct（host 与 device 共享）。
 */
#ifndef _SYNCBNBR_TILING_DATA_H_
#define _SYNCBNBR_TILING_DATA_H_

#include <cstdint>

// schMode (tiling key) selects the compute dtype.
#ifndef SYNCBNBR_TPL_SCH_MODE_0
#define SYNCBNBR_TPL_SCH_MODE_0 0 // half (float16)
#endif
#ifndef SYNCBNBR_TPL_SCH_MODE_1
#define SYNCBNBR_TPL_SCH_MODE_1 1 // float (float32)
#endif
#ifndef SYNCBNBR_TPL_SCH_MODE_2
#define SYNCBNBR_TPL_SCH_MODE_2 2 // bfloat16
#endif

struct SyncBatchNormBackwardReduceTilingData {
    uint64_t coreNum = 1;
    uint64_t bufferNum = 1;
    uint64_t tailElems = 0;
    uint64_t epochs = 0;
    uint64_t epochsForLastCore = 0;
    uint64_t coreLength = 0;
    uint64_t tileLength = 0;
    uint64_t tailTileLength = 0;
    uint64_t tailTileLengthForLastCore = 0;
};
#endif
