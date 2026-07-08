/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GLU_GRAD_TILING_H_
#define GLU_GRAD_TILING_H_

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GluGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, group);
TILING_DATA_FIELD_DEF(int64_t, loopNum);
TILING_DATA_FIELD_DEF(int64_t, tailLoopNum);
TILING_DATA_FIELD_DEF(int64_t, nLastTailGroup);
TILING_DATA_FIELD_DEF(int64_t, lastTailGroup);
TILING_DATA_FIELD_DEF(int64_t, splitSize);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, ny);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GLUGrad, GluGradTilingData)

struct GluGradCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

struct GluGradTilingParam {
    int64_t x = 1;
    int64_t ny = 1;
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
    int64_t blockSize = 0;
    int64_t bufSize = 1;
};
} // namespace optiling
#endif // GLU_GRAD_TILING_H_
