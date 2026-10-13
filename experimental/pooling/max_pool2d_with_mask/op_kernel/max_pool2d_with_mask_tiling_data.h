/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL2D_WITH_MASK_TILING_DATA_H
#define MAX_POOL2D_WITH_MASK_TILING_DATA_H

#include <cstdint>

constexpr uint64_t MAX_POOL2D_WINDOW_INPUT_BYTES = 8192;
constexpr uint64_t MAX_POOL2D_WINDOW_ZERO_BYTES = 131072;
constexpr uint64_t MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS = 32;
// Two FP32 working arrays, per-repeat value/index pairs, a compare mask and the result.
constexpr uint64_t MAX_POOL2D_WINDOW_VECTOR_BYTES = 2 * (MAX_POOL2D_WINDOW_INPUT_BYTES / 2) * 4 +
                                                    (MAX_POOL2D_WINDOW_INPUT_BYTES / 2 / 64) * 8 +
                                                    MAX_POOL2D_WINDOW_INPUT_BYTES / 16 + 64;

// Larger complete-window prefetch uses one FP32 array. Small windows reuse its
// upper half as transform scratch and reuse the pair buffer as a compare mask.
constexpr uint64_t MAX_POOL2D_WINDOW_LARGE_INPUT_BYTES = 16384;
constexpr uint64_t MAX_POOL2D_WINDOW_LARGE_ZERO_BYTES = 112 * 1024;
constexpr uint64_t MAX_POOL2D_WINDOW_LARGE_VECTOR_BYTES = (MAX_POOL2D_WINDOW_LARGE_INPUT_BYTES / 2) * 4 +
                                                          (MAX_POOL2D_WINDOW_LARGE_INPUT_BYTES / 2 / 64) * 8 + 64;
static_assert(2 * MAX_POOL2D_WINDOW_LARGE_INPUT_BYTES + MAX_POOL2D_WINDOW_LARGE_ZERO_BYTES +
                      MAX_POOL2D_WINDOW_LARGE_VECTOR_BYTES <=
                  2 * MAX_POOL2D_WINDOW_INPUT_BYTES + MAX_POOL2D_WINDOW_ZERO_BYTES + MAX_POOL2D_WINDOW_VECTOR_BYTES,
              "The large-window specialization must fit the existing UB admission check");

// Trade zero-store buffer capacity for one complete medium-size input window.
constexpr uint64_t MAX_POOL2D_WINDOW_WIDE_INPUT_BYTES = 24 * 1024;
constexpr uint64_t MAX_POOL2D_WINDOW_WIDE_ZERO_BYTES = 64 * 1024;
constexpr uint64_t MAX_POOL2D_WINDOW_WIDE_VECTOR_BYTES = (MAX_POOL2D_WINDOW_WIDE_INPUT_BYTES / 2) * 4 +
                                                         (MAX_POOL2D_WINDOW_WIDE_INPUT_BYTES / 2 / 64) * 8 + 128;
static_assert(2 * MAX_POOL2D_WINDOW_WIDE_INPUT_BYTES + MAX_POOL2D_WINDOW_WIDE_ZERO_BYTES +
                      MAX_POOL2D_WINDOW_WIDE_VECTOR_BYTES <=
                  2 * MAX_POOL2D_WINDOW_INPUT_BYTES + MAX_POOL2D_WINDOW_ZERO_BYTES + MAX_POOL2D_WINDOW_VECTOR_BYTES,
              "The wider complete window must fit the existing UB admission check");

struct MaxPool2dWithMaskTilingData {
    int64_t batchSize;
    int64_t channels;
    int64_t inputHeight;
    int64_t inputWidth;
    int64_t outputHeight;
    int64_t outputWidth;

    int32_t kernelH;
    int32_t kernelW;
    int32_t strideH;
    int32_t strideW;
    int32_t padH;
    int32_t padW;
    int32_t dilationH;
    int32_t dilationW;

    int64_t totalSlices;
    int64_t slicesPerCore;
    int64_t outputRowsPerTile;
    int64_t inputWidthAligned;
    int64_t outputElementsPerTileAligned;

    int64_t validIndexBytes;
    int64_t containerBytes;
    int64_t zeroBytesPerCore;
    int64_t zeroBufferBytes;
    // Nonzero only when the 310P FLOAT32 separable reduction fits in UB.
    int64_t columnBufferElements;
    int64_t vectorBufferElements;

    // A2 vector path. The two-dimensional input is treated as NCDHW with D=1.
    uint64_t nc;
    uint64_t dx;
    uint64_t hx;
    uint64_t wx;
    uint64_t kd;
    uint64_t kh;
    uint64_t kw;
    uint64_t sd;
    uint64_t sh;
    uint64_t sw;
    uint64_t pf;
    uint64_t pb;
    uint64_t pl;
    uint64_t pr;
    uint64_t pt;
    uint64_t pd;
    uint64_t dy;
    uint64_t hy;
    uint64_t wy;
    uint64_t ncFactor;
    uint64_t ncTail;
    uint64_t ncOuter;
    uint64_t dyFactor;
    uint64_t dyTail;
    uint64_t dyOuter;
    uint64_t hyFactor;
    uint64_t hyTail;
    uint64_t hyOuter;
    uint64_t wyFactor;
    uint64_t wyTail;
    uint64_t wyOuter;
    uint64_t blockFactor;
    uint64_t blockTail;
    uint64_t totalIdx;
    uint64_t coreNums;
    uint64_t inputBufferSize;
    uint64_t outputMaxBufferSize;
    uint64_t outputIndiceBufferSize;
    uint64_t indiceTempBufferSize;
    uint64_t maskBufferSize;

    // Aligned 310P vector path, with window offsets reused across channels.
    int64_t m200MapRows;
    // Number of output rows for the aligned 16-channel 310P transpose path.
    int64_t m200NcRows;
    // Compile-time channel batch selected for the 310P transpose path.
    int64_t m200NcFactor;
    // Aligned 2x2/stride2 spatial GatherMask path, zero when not selected.
    int64_t m200SpatialRows;
    int64_t m200SpatialBatch;
    uint64_t windowOutputTile;
    uint64_t windowTailBoundaryCore;
    uint64_t windowPrefetch;
};

// Window kernels load one contiguous packet without unrelated vector/M200 fields.
struct MaxPool2dWithMaskWindowTilingData {
    int64_t inputHeight;
    int64_t inputWidth;
    int64_t outputHeight;
    int64_t outputWidth;
    int64_t validIndexBytes;
    int64_t containerBytes;
    int64_t zeroBytesPerCore;
    int64_t zeroBufferBytes;
    uint64_t nc;
    uint64_t kh;
    uint64_t kw;
    uint64_t sh;
    uint64_t sw;
    uint64_t pl;
    uint64_t pt;
    uint64_t windowOutputTile;
    uint64_t windowTailBoundaryCore;
    uint64_t windowPrefetch;
    uint64_t heavyCoreCount;
    uint64_t heavyCoreZeroBytes;
};

struct MaxPool2dWithMaskTilingPacket {
    MaxPool2dWithMaskTilingData common;
    MaxPool2dWithMaskWindowTilingData window;
};

static_assert(sizeof(MaxPool2dWithMaskWindowTilingData) == 20 * sizeof(uint64_t));

#endif // MAX_POOL2D_WITH_MASK_TILING_DATA_H
