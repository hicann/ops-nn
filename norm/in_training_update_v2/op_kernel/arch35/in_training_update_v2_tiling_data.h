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
 * \file in_training_update_v2_tiling_data.h
 * \brief Host/kernel ABI for INTrainingUpdateV2 on arch35.
 */

#ifndef IN_TRAINING_UPDATE_V2_TILING_DATA_H
#define IN_TRAINING_UPDATE_V2_TILING_DATA_H

#include <cstdint>

struct INTrainingUpdateV2TilingData {
    int64_t n;
    int64_t c;
    int64_t r;
    int64_t totalElements;

    // unitBlocks partitions complete 32-byte y blocks; rCores is one.  Their
    // product remains the active AI Vector core count used by the entry point.
    int64_t unitBlocks;
    int64_t rCores;
    int64_t formerBlockNum;
    int64_t formerUnits;
    int64_t latterUnits;

    int64_t tileElems;
    int64_t rTile;
    uint32_t xyBufferBytes;
    uint32_t statBufferBytes;

    int64_t hasAffine;
    int64_t hasRunning;
    int64_t gammaBatchStride;
    int64_t betaBatchStride;

    float invR;
    float bessel;
    float momentum;
    float oneMinusMomentum;
    float epsilon;
    float invRCorrection;
};

#endif // IN_TRAINING_UPDATE_V2_TILING_DATA_H
