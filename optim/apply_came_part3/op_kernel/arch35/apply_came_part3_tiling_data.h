/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_APPLY_CAME_PART3_TILING_DATA_H
#define OPS_NN_APPLY_CAME_PART3_TILING_DATA_H

#include <cstdint>

struct ApplyCamePart3TilingData {
    int64_t usedCoreNum = 0;
    int64_t curN = 0;
    int64_t curM = 0;
    int64_t rNumCalc = 0;
    int64_t cNumCalc = 0;
    int64_t baseN = 0;
    int64_t baseM = 0;
    int64_t rCoreNum = 0;
    int64_t cCoreNum = 0;
    int64_t isGlobalShape = 0;
    int64_t useFirstMoment = 0;
};

#endif // OPS_NN_APPLY_CAME_PART3_TILING_DATA_H
