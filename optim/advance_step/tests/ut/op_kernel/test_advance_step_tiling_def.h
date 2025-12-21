/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TEST_ADVANCE_STEP_H
#define TEST_ADVANCE_STEP_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define __CCE_UT_TEST__
#define __aicore__

struct AdvanceStepTilingDataTest {
    int64_t needCoreNum = 40;
    int64_t blockTablesStride = 1;
    int64_t numSeqs = 16;
    int64_t numQueries = 8;
    int64_t blockSize = 8;
    int64_t perCoreSeqs = 8;
    int64_t lastCoreSeqs = 8;
    int64_t perLoopMaxSeqs = 8;
    int64_t specNum = 8;
};

inline void IAdvanceStepTilingData(uint8_t* tiling, AdvanceStepTilingDataTest* const_data)
{
    memcpy(const_data, tiling, sizeof(AdvanceStepTilingDataTest));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    AdvanceStepTilingDataTest tilingData;              \
    IAdvanceStepTilingData(tilingPointer, &tilingData)
#endif