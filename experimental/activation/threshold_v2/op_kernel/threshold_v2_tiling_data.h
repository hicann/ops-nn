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
/**
 * \file threshold_v2_tiling_data.h
 * \brief ThresholdV2 TilingData
 */

#ifndef _THRESHOLD_V2_TILING_DATA_H_
#define _THRESHOLD_V2_TILING_DATA_H_

struct ThresholdV2TilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    float thresholdVal = 0.0f;
    float valueVal = 0.0f;
};

#endif
