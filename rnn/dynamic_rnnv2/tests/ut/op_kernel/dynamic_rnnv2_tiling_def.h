/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __DYN_RNN_H
#define __DYN_RNN_H

#include "kernel_tiling/kernel_tiling.h"

#if defined(__CCE_KT_TEST__) || defined(__TIK2_REPLAY__)
#include <cstdint>
#include <cstring>
#endif

#pragma pack(1)
struct DynamicRNNTilingData {
    int64_t tilingKey = 0;
    int64_t usedCoreNum = 0;
    int64_t timeStep = 0;
    int64_t batch = 0;
    int64_t inputSize = 0;
    int64_t hiddenSize = 0;
    int64_t isBias = 0;
    int64_t isInithc = 0;
    int64_t isSeqLength = 0;
    int64_t isHF32 = 0;
    int64_t isCached = 0;
    int64_t cacheLength = 0;
    int64_t gateOrder = 0;
    int64_t direction = 0;
    int64_t isTraining = 0;
    float cellClip = 0;
    float forgetBias = 0;
    TCubeTiling inputMMParam;
    TCubeTiling hiddenMMParam;
};
#pragma pack()

#if defined(__CCE_KT_TEST__) || defined(__TIK2_REPLAY__)
inline void InitTilingData(uint8_t* tiling, DynamicRNNTilingData* const_data) {
    memcpy(const_data, tiling, sizeof(DynamicRNNTilingData));
}
#else
inline [aicore] void InitTilingData(const __gm__ uint8_t* tiling, DynamicRNNTilingData* const_data) {
    for (auto i = 0; i < sizeof(DynamicRNNTilingData) / 4; i++) {
        *(int32_t *)((int32_t *)const_data + i) = *((__gm__ int32_t *)tiling + i);
    }
}
#endif

#define GET_TILING_DATA(tiling_data, tiling_arg) \
DynamicRNNTilingData tiling_data; \
InitTilingData(tiling_arg, &tiling_data)
#endif
