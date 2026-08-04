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
 * \file normalize_bbox_tiling_data.h
 * \brief normalize_bbox tiling data (host/kernel shared POD)
 */

#ifndef NORMALIZE_BBOX_TILING_DATA_H
#define NORMALIZE_BBOX_TILING_DATA_H

#include <cstdint>

#pragma pack(push, 8)
struct NormalizeBBoxTilingData {
    uint64_t batch;       // boxes.shape[0]
    uint64_t num;         // frames per batch (normal: shape[1], reversed: shape[2])
    uint64_t coordNum;    // coordinate count (== 4)
    uint64_t splitMode;   // 0 = split by num (batch==1), 1 = split by batch (batch>1)
    uint64_t usedCoreNum; // diagnostic: actual cores launched (host uses for blockDim, kernel does not read)
    // split-by-batch (splitMode==1)
    uint64_t batchPerCore; // big cores each handle batchPerCore batches
    uint64_t tailBatchNum; // cores >= bigCoreNum handle (batchPerCore-1) batches
    uint64_t bigCoreNum;   // number of big cores (front cores)
    // split-by-num (splitMode==0, batch==1): front-full model — every active core takes
    // numPerCore frames; the LAST active core's count is clamped in-kernel (numStart+cnt>num),
    // so there is no distinct "small core" tier here (unlike split-by-batch above).
    uint64_t numPerCore;  // frames each active core handles (blockAlign-aligned)
    uint64_t tailNumCore; // == numPerCore (front-full model has no small-core tier; kept for kernel symmetry)
    uint64_t numBigCore;  // number of active cores (all treated as "big"; last core clamped in-kernel)
    // UB tile
    uint64_t tileLen; // element count processed per CopyIn/Compute/CopyOut tile (32B aligned)
};
#pragma pack(pop)

#endif // NORMALIZE_BBOX_TILING_DATA_H
