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
 * \file masked_scatter_with_position_tiling_struct.h
 * \brief tiling data struct
 */

#ifndef MASKED_SCATTER_WITH_POSITION_TILLING_DATA_H
#define MASKED_SCATTER_WITH_POSITION_TILLING_DATA_H

class MaskedScatterWithPositionTilingData {
public:
    int64_t xNum = 1;
    int64_t xInner = 1;
    int64_t updatesEleNums = 1;
    int64_t pattern = 0;  // 0:BA 1:AB
    int64_t computeType = 0;  // 0:int32 1:int64
};

#endif