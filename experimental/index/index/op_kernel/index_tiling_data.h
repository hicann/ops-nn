/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NN_EXPERIMENTAL_INDEX_TILING_DATA_H_
#define OPS_NN_EXPERIMENTAL_INDEX_TILING_DATA_H_

#include <cstdint>

constexpr uint32_t INDEX_MAX_RANK = 8;
constexpr uint32_t INDEX_MAX_INDEX_STRIDE_SIZE = INDEX_MAX_RANK * INDEX_MAX_RANK;

#ifdef __aicore__
#define INDEX_AICORE_INLINE __aicore__ inline
#else
#define INDEX_AICORE_INLINE inline
#endif

namespace optiling {

struct IndexTilingData {
    uint64_t totalNum = 0;
    uint32_t usedCoreNum = 0;
    uint32_t xRank = 0;
    uint32_t yRank = 0;
    uint32_t indexRank = 0;
    uint32_t indexedDimNum = 0;
    uint32_t indexedDims[INDEX_MAX_RANK] = {0};
    uint64_t xShape[INDEX_MAX_RANK] = {0};
    uint64_t xStride[INDEX_MAX_RANK] = {0};
    uint64_t yShape[INDEX_MAX_RANK] = {0};
    uint64_t yStride[INDEX_MAX_RANK] = {0};
    uint64_t indexShape[INDEX_MAX_RANK] = {0};
    uint64_t indexStride[INDEX_MAX_RANK] = {0};
    uint64_t indexInputStrides[INDEX_MAX_INDEX_STRIDE_SIZE] = {0};

    INDEX_AICORE_INLINE uint64_t get_totalNum() const { return totalNum; }
    INDEX_AICORE_INLINE uint32_t get_xRank() const { return xRank; }
    INDEX_AICORE_INLINE uint32_t get_yRank() const { return yRank; }
    INDEX_AICORE_INLINE uint32_t get_indexRank() const { return indexRank; }
    INDEX_AICORE_INLINE uint32_t get_indexedDimNum() const { return indexedDimNum; }
    INDEX_AICORE_INLINE const uint32_t* get_indexedDims() const { return indexedDims; }
    INDEX_AICORE_INLINE const uint64_t* get_xShape() const { return xShape; }
    INDEX_AICORE_INLINE const uint64_t* get_xStride() const { return xStride; }
    INDEX_AICORE_INLINE const uint64_t* get_yShape() const { return yShape; }
    INDEX_AICORE_INLINE const uint64_t* get_yStride() const { return yStride; }
    INDEX_AICORE_INLINE const uint64_t* get_indexShape() const { return indexShape; }
    INDEX_AICORE_INLINE const uint64_t* get_indexStride() const { return indexStride; }
    INDEX_AICORE_INLINE const uint64_t* get_indexInputStrides() const { return indexInputStrides; }
};
}  // namespace optiling

#undef INDEX_AICORE_INLINE

#endif  // OPS_NN_EXPERIMENTAL_INDEX_TILING_DATA_H_
