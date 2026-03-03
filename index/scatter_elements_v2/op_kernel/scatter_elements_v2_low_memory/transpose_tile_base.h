/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_tile_base.h
 * \brief Transpose Tile操作的宏函数定义
 */

#ifndef TRANSPOSE_TILE_BASE_H
#define TRANSPOSE_TILE_BASE_H

namespace ScatterElementsV2NS {

#define TRANSPOSE_TILE_SET_CORE_NUMS() \
    __aicore__ inline void SetCoreNums(uint32_t coreNums) { \
        this->coreNums = coreNums; \
    }

#define TRANSPOSE_TILE_SET_CORE_ID() \
    __aicore__ inline void SetCoreId(uint32_t* coreId) { \
        this->coreId = coreId; \
    }

#define TRANSPOSE_TILE_GET_NEXT_CORE() \
    __aicore__ inline void GetNextCore(uint32_t* coreId) { \
        *coreId += 1; \
        if (*coreId == this->coreNums) { \
            *coreId = 0; \
        } \
    }

#define CALC_ALIGNED_BASE_EXTRA(size, usedCores, base, extra) \
    uint32_t aliged = BYTE_ALIGNMENT / sizeof(T); \
    base = size / usedCores; \
    extra = size % usedCores; \
    if (base > 0) { \
        base = ((base + aliged - 1) / aliged) * aliged; \
        usedCores = size / base; \
        extra = size - usedCores * base; \
    }

#define INIT_DATA_COPY_PARAMS(copyParams, rowSize, colSize, rowLength) \
    uint32_t aliged = BYTE_ALIGNMENT / sizeof(T); \
    uint32_t colSizeAligned = (colSize + aliged - 1) / aliged * aliged; \
    uint32_t dstStride = (BASE_TILE_SIZE - colSizeAligned) / aliged; \
    DataCopyExtParams copyParams{ \
        static_cast<uint16_t>(rowSize), \
        static_cast<uint32_t>(colSize * sizeof(T)), \
        static_cast<uint32_t>(rowLength * sizeof(T) - colSize * sizeof(T)), \
        dstStride, \
        0 \
    };

}
#endif
