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
 * \file ctc_loss_v2_grad_tiling_arch35.h
 * \brief
 */
#ifndef CTC_LOSS_V2_GRAD_TILING__ARCH35_H_
#define CTC_LOSS_V2_GRAD_TILING__ARCH35_H_

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {
ge::graphStatus Tiling4CTCLossV2GradForAscendC(gert::TilingContext* context);

BEGIN_TILING_DATA_DEF(CTCLossV2GradTilingData4AscnedC)
TILING_DATA_FIELD_DEF(int64_t, alphaLength);
TILING_DATA_FIELD_DEF(int64_t, maxInputLength);
TILING_DATA_FIELD_DEF(int64_t, symbolSet);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, targetsDimNum);
TILING_DATA_FIELD_DEF(int64_t, sDimRange);
TILING_DATA_FIELD_DEF(int64_t, targetsNum);
TILING_DATA_FIELD_DEF(int64_t, BLANK);
TILING_DATA_FIELD_DEF(int64_t, zeroInfinity);
TILING_DATA_FIELD_DEF(int64_t, blockDimX);
TILING_DATA_FIELD_DEF(int64_t, blockDimY);
TILING_DATA_FIELD_DEF(int64_t, logAlphaT);
TILING_DATA_FIELD_DEF(int64_t, logBetaThreadNum);
TILING_DATA_FIELD_DEF(int64_t, updateLcabThreadNum);
TILING_DATA_FIELD_DEF(int64_t, calGradThreadNum);
TILING_DATA_FIELD_DEF(int64_t, initGradGmStartBlock);
TILING_DATA_FIELD_DEF(int64_t, initGradGmEndBlock);
TILING_DATA_FIELD_DEF(int64_t, initGradGmSizePerBlock);
TILING_DATA_FIELD_DEF(int64_t, initGradGmSizeLastBlock);
TILING_DATA_FIELD_DEF(int64_t, initLogBetaGmStartBlock);
TILING_DATA_FIELD_DEF(int64_t, initLogBetaGmEndBlock);
TILING_DATA_FIELD_DEF(int64_t, initLogBetaGmSizePerBlock);
TILING_DATA_FIELD_DEF(int64_t, initLogBetaGmSizeLastBlock);
TILING_DATA_FIELD_DEF(int64_t, initTempGradGmStartBlock);
TILING_DATA_FIELD_DEF(int64_t, initTempGradGmEndBlock);
TILING_DATA_FIELD_DEF(int64_t, initTempGradGmSizePerBlock);
TILING_DATA_FIELD_DEF(int64_t, initTempGradGmSizeEndBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CTCLossV2Grad, CTCLossV2GradTilingData4AscnedC)

struct CTCLossV2GradCompileInfo {
    int32_t coreNum = 1;
    int32_t ubSize = 1;
};

struct CTCLossV2GradForCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};
} // namespace optiling
#endif // CTC_LOSS_V2_GRAD_TILING__ARCH35_H_