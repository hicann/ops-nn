/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _DEQUANT_BIAS_TILING_DATA_H_
#define _DEQUANT_BIAS_TILING_DATA_H_

#include <cstdint>

#define TPL_NO_BIAS 0
#define TPL_HAS_BIAS 1
#define TPL_NO_ACTIVATE 0
#define TPL_HAS_ACTIVATE 1

#define TPL_BIAS_FLOAT 0
#define TPL_BIAS_HALF 1
#define TPL_BIAS_BFLOAT 2
#define TPL_BIAS_INT 3

struct DequantBiasArch35TilingData {
    int64_t blockFactor = 0;
    int64_t blockTailFactor = 0;
    int32_t numCore = 0;
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t baseN = 0;
    int64_t baseLen = 0;
    int64_t baseLenTail = 0;
    int64_t tileNum = 0;
    int32_t hasActivateScale = 0;
    int32_t hasBias = 0;
    int32_t outputDtype = 0;
};

#endif