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
 * \file clipped_swiglu_tiling_data.h
 * \brief TilingData struct for ClippedSwiglu Arch35
 */

#ifndef CLIPPED_SWIGLU_TILING_DATA_H
#define CLIPPED_SWIGLU_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct ClippedSwigluArch35TilingData {
    int64_t realCoreNum;
    int64_t dimBatchSize;
    int64_t dimH;
    int64_t groupNum;
    int64_t bUbFactor;
    int64_t hUbFactor;
    float gluAlpha;
    float gluLimit;
    float gluBias;
};
#endif // CLIPPED_SWIGLU_TILING_DATA_H
