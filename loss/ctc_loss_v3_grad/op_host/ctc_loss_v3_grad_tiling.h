/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ctc_loss_v3_grad.h
 * \brief
 * 
 * 
 * 
 * 
 * 
 * 
 */
#ifndef CTC_LOSS_V3_GRAD_H_
#define CTC_LOSS_V3_GRAD_H_
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CTCLossV3GradTilingData)
TILING_DATA_FIELD_DEF(int64_t, sliceLength);
TILING_DATA_FIELD_DEF(int64_t, sliceLengthTail);
TILING_DATA_FIELD_DEF(int64_t, probSliceNum);
TILING_DATA_FIELD_DEF(int64_t, alphaLength);
TILING_DATA_FIELD_DEF(int64_t, maxInputLength);
TILING_DATA_FIELD_DEF(int64_t, symbolSet);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, targetsDimNum);
TILING_DATA_FIELD_DEF(int64_t, sDimRange);
TILING_DATA_FIELD_DEF(int64_t, targetsNum);
TILING_DATA_FIELD_DEF(int64_t, taskPerCore);
TILING_DATA_FIELD_DEF(int64_t, taskTailCore);
TILING_DATA_FIELD_DEF(int64_t, BLANK);
TILING_DATA_FIELD_DEF(int64_t, zeroInfinity);
END_TILING_DATA_DEF;

struct CTCLossV3GradCompileInfo {
    uint32_t coreNum = 0;
    uint64_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};

REGISTER_TILING_DATA_CLASS(CTCLossV3Grad, CTCLossV3GradTilingData)
} // namespace optiling

#endif // CTC_LOSS_V3_GRAD_H_
