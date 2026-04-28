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
 * @file quant_max_tiling_data.h
 * @brief QuantMax TilingData structure definition
 */

#ifndef _QUANT_MAX_TILING_DATA_H_
#define _QUANT_MAX_TILING_DATA_H_

class QuantMaxTilingData {
public:
    int64_t numCore;
    int64_t dim0;
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t baseLen;
    int64_t roundMode;
};

#endif // _QUANT_MAX_TILING_DATA_H_
