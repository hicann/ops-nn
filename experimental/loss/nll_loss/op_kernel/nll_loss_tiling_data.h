/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _NLLLOSS_TILING_DATA_H_
#define _NLLLOSS_TILING_DATA_H_

#include <cstdint>

#ifndef NLLLOSS_TPL_SCH_MODE_0
#define NLLLOSS_TPL_SCH_MODE_0 0
#endif
#ifndef NLLLOSS_TPL_SCH_MODE_1
#define NLLLOSS_TPL_SCH_MODE_1 1
#endif
#ifndef NLLLOSS_TPL_SCH_MODE_2
#define NLLLOSS_TPL_SCH_MODE_2 2
#endif

struct NllLossTilingData {
    uint64_t rowNum = 0;
    uint64_t classNum = 0;
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;
    uint64_t hasWeight = 0;
    uint64_t targetIsInt64 = 0;
    uint64_t usedCoreNum = 1;
    uint64_t rowsPerCore = 0;
    uint64_t tileRows = 0;
    uint64_t useVector = 1;
};
#endif
