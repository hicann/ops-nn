/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_normalization_grad_tiling_data.h
 * \brief BatchNormalizationGrad tiling data struct
 */

#ifndef BATCH_NORMALIZATION_GRAD_TILING_DATA_H_
#define BATCH_NORMALIZATION_GRAD_TILING_DATA_H_

struct BatchNormalizationGradTilingData {
    uint32_t numChannels = 0;
    uint32_t numBatches = 0;
    uint32_t numSpatial = 0;
    uint32_t m = 0;
    float mFloat = 0.0f;
    uint32_t coreNum = 0;
    uint32_t channelsPerCore = 0;
    uint32_t tailChannels = 0;
    uint32_t spatialSplitMode = 0;
    uint32_t spatialTileSize = 0;
    uint32_t spatialTileNum = 0;
    uint32_t spatialTailSize = 0;
    uint32_t batchStrideAligned = 0;
    uint32_t batchesPerTile = 0;
    uint32_t batchGroups = 0;
    uint32_t tailBatches = 0;
    uint32_t paddedSpatial = 0;
    float epsilon = 1e-5f;
    uint32_t blockSize = 0;
    uint32_t typeLength = 0;
    uint32_t elementsPerBlock = 0;
    uint32_t needPerBatchLoad = 0;
};

#endif // BATCH_NORMALIZATION_GRAD_TILING_DATA_H_
