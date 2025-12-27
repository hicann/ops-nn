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
 * \file ctc_loss_v2_tiling_arch35.h
 * \brief ctc_loss_v2_tiling_arch35
 */

#ifndef OPS_LOSS_CTC_LOSS_V2_OP_HOST_CTC_LOSS_V2_TILING_ARCH35_H_
#define OPS_LOSS_CTC_LOSS_V2_OP_HOST_CTC_LOSS_V2_TILING_ARCH35_H_
#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"

namespace optiling {
ge::graphStatus Tiling4CTCLossV2ForAscendC(gert::TilingContext* context);

BEGIN_TILING_DATA_DEF(CTCLossV2TilingData4AscendC)
TILING_DATA_FIELD_DEF(int64_t, maxInputLength);
TILING_DATA_FIELD_DEF(int64_t, maxTargetLength);
TILING_DATA_FIELD_DEF(int64_t, lpInputStride);
TILING_DATA_FIELD_DEF(int64_t, lpBatchStride);
TILING_DATA_FIELD_DEF(int64_t, lpCharStride);
TILING_DATA_FIELD_DEF(int64_t, laBatchStride);
TILING_DATA_FIELD_DEF(int64_t, laInputStride);
TILING_DATA_FIELD_DEF(int64_t, laTargetStride);
TILING_DATA_FIELD_DEF(int64_t, tgTargetStride);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, blank);
TILING_DATA_FIELD_DEF(int64_t, blockDimX);
TILING_DATA_FIELD_DEF(int64_t, blockDimY);
TILING_DATA_FIELD_DEF(int64_t, targetsDim);
TILING_DATA_FIELD_DEF(int64_t, tgBatchStride);
TILING_DATA_FIELD_DEF(int64_t, workspaceSize);
TILING_DATA_FIELD_DEF(int64_t, gridY);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CTCLossV2, CTCLossV2TilingData4AscendC)

struct CTCLossV2ForCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};

struct CTCLossV2CompileInfo {
    int32_t coreNum = 1;
    int32_t ubSize = 1;
    int32_t inputLengthsDsize = 1;
    int32_t targetsDsize = 1;
};

} // namespace optiling
#endif // OPS_LOSS_CTC_LOSS_V2_OP_HOST_CTC_LOSS_V2_TILING_ARCH35_H_