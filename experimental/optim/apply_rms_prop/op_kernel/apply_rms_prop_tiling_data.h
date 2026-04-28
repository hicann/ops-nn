/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file apply_rms_prop_tiling_data.h
 * \brief ApplyRmsProp 算子 TilingData 结构体
 */

#ifndef _APPLY_RMS_PROP_TILING_DATA_H_
#define _APPLY_RMS_PROP_TILING_DATA_H_

#include <cstdint>

struct ApplyRmsPropTilingData {
    float lr    = 0.0f;
    float rho1m = 0.0f;
    float mom   = 0.0f;
    float eps   = 0.0f;

    int64_t totalLength      = 0;
    int64_t blockLength      = 0;
    int64_t lastBlockLength  = 0;
    int64_t tileLength       = 0;
    int64_t tileNum          = 0;
    int64_t lastTileLength   = 0;
    int64_t lastBlockTileNum = 0;
    int64_t lastBlockLastTileLength = 0;
    int32_t usedCoreNum      = 0;
    int32_t reserved         = 0;
};

#endif  // _APPLY_RMS_PROP_TILING_DATA_H_
