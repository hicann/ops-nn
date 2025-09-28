/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ctc_loss_v3_tiling.h
 * \brief
 * 
 * 
 * 
 * 
 * 
 * 
 */
#ifndef CTC_LOSS_V3_H_
#define CTC_LOSS_V3_H_
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CTCLossV3TilingData)
TILING_DATA_FIELD_DEF(int64_t, sliceLength);
TILING_DATA_FIELD_DEF(int64_t, sliceLengthTail);
TILING_DATA_FIELD_DEF(int64_t, probSliceNum);
TILING_DATA_FIELD_DEF(int64_t, maxTargetLength);
TILING_DATA_FIELD_DEF(int64_t, timeStep);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, symbleSet);
TILING_DATA_FIELD_DEF(int64_t, targetsDimNum);
TILING_DATA_FIELD_DEF(int64_t, targetsDimLength);
TILING_DATA_FIELD_DEF(int64_t, targetsNum);
TILING_DATA_FIELD_DEF(int64_t, taskPerCore);
TILING_DATA_FIELD_DEF(int64_t, taskTailCore);
TILING_DATA_FIELD_DEF(int64_t, blank);
TILING_DATA_FIELD_DEF(int64_t, zeroInfinity);
END_TILING_DATA_DEF;

struct CTCLossV3CompileInfo {
    uint32_t coreNum = 0;
    uint64_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};

REGISTER_TILING_DATA_CLASS(CTCLossV3, CTCLossV3TilingData)
} // namespace optiling

#endif // CTC_LOSS_V3_H_