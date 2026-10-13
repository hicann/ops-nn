/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>

#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_def_registry.h"
#include "../op_kernel/max_pool2d_with_mask_tiling_data.h"

namespace optiling {
namespace {

constexpr size_t KERNEL_SIZE_INDEX = 0;
constexpr size_t STRIDE_INDEX = 1;
constexpr size_t PADDING_INDEX = 2;
constexpr size_t DILATION_INDEX = 3;
constexpr uint64_t ALIGN_BYTES = 32;
constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t UB_RESERVE_BYTES = 4096;
constexpr uint64_t ZERO_BUFFER_BYTES = 8192;
constexpr uint64_t ZERO_BYTES_PER_CORE_TARGET = 64 * 1024;
constexpr uint64_t VECTOR_UB_RESERVE_BYTES = 8 * 1024;
constexpr uint64_t VECTOR_OR_MASK_BYTES = 256;
constexpr uint64_t VECTOR_TRANSPOSE_ALIGN = 16;
constexpr uint64_t VECTOR_CACHE_ALIGN_BYTES = 256;
constexpr uint64_t VECTOR_BLOCK_COUNT = 8;
constexpr uint64_t SCALAR_CACHE_LINE_BYTES = 64;

struct SpatialPair {
    int64_t h;
    int64_t w;
};

struct TilingInputs {
    int64_t batch;
    int64_t channels;
    int64_t inputHeight;
    int64_t inputWidth;
    int64_t outputHeight;
    int64_t outputWidth;
    SpatialPair kernel;
    SpatialPair stride;
    SpatialPair padding;
    SpatialPair dilation;
    ge::DataType dtype;
    uint64_t dtypeBytes;
    uint64_t ubBytes;
    uint64_t coreNum;
    platform_ascendc::SocVersion socVersion;
};

uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

bool SafeMultiply(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool ParsePair(const gert::TypedContinuousVector<int64_t>* values, bool allowEmpty, const SpatialPair& emptyValue,
               SpatialPair& result)
{
    if (values == nullptr) {
        return false;
    }
    const size_t size = values->GetSize();
    if (size == 0) {
        if (!allowEmpty) {
            return false;
        }
        result = emptyValue;
        return true;
    }
    if (size != 1 && size != 2) {
        return false;
    }
    const int64_t* data = values->GetData();
    result.h = data[0];
    result.w = size == 1 ? data[0] : data[1];
    return true;
}

ge::graphStatus ParseInputs(gert::TilingContext* context, TilingInputs& input)
{
    const auto* inputShapePtr = context->GetInputShape(0);
    const auto* outputShapePtr = context->GetOutputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const auto inputShape = inputShapePtr->GetStorageShape();
    const auto outputShape = outputShapePtr->GetStorageShape();
    const size_t rank = inputShape.GetDimNum();
    OP_CHECK_IF(rank != 3 && rank != 4,
                OP_LOGE(context, "MaxPool2dWithMask input rank must be 3 or 4, but got %zu", rank),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputShape.GetDimNum() != rank,
                OP_LOGE(context, "MaxPool2dWithMask output rank does not match input rank"), return ge::GRAPH_FAILED);

    if (rank == 4) {
        input.batch = inputShape.GetDim(0);
        input.channels = inputShape.GetDim(1);
    } else {
        input.batch = 1;
        input.channels = inputShape.GetDim(0);
    }
    input.inputHeight = inputShape.GetDim(rank - 2);
    input.inputWidth = inputShape.GetDim(rank - 1);
    input.outputHeight = outputShape.GetDim(rank - 2);
    input.outputWidth = outputShape.GetDim(rank - 1);
    OP_CHECK_IF(input.batch <= 0 || input.channels <= 0 || input.inputHeight <= 0 || input.inputWidth <= 0 ||
                    input.outputHeight <= 0 || input.outputWidth <= 0,
                OP_LOGE(context, "all resolved input and output dimensions must be positive"), return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_IF(!ParsePair(attrs->GetListInt(KERNEL_SIZE_INDEX), false, {}, input.kernel) || input.kernel.h <= 0 ||
                    input.kernel.w <= 0,
                OP_LOGE(context, "kernelSize must contain one or two positive values"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ParsePair(attrs->GetListInt(STRIDE_INDEX), true, input.kernel, input.stride) || input.stride.h <= 0 ||
                    input.stride.w <= 0,
                OP_LOGE(context, "stride must be empty or contain one or two positive values"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ParsePair(attrs->GetListInt(PADDING_INDEX), false, {}, input.padding) || input.padding.h < 0 ||
                    input.padding.w < 0 || input.padding.h > input.kernel.h / 2 || input.padding.w > input.kernel.w / 2,
                OP_LOGE(context, "padding is outside the supported range"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ParsePair(attrs->GetListInt(DILATION_INDEX), false, {}, input.dilation) || input.dilation.h != 1 ||
                    input.dilation.w != 1,
                OP_LOGE(context, "dilation must contain one or two values equal to 1"), return ge::GRAPH_FAILED);

    input.dtype = inputDesc->GetDataType();
    if (input.dtype == ge::DT_FLOAT) {
        input.dtypeBytes = sizeof(float);
    } else if (input.dtype == ge::DT_FLOAT16 || input.dtype == ge::DT_BF16) {
        input.dtypeBytes = sizeof(uint16_t);
    } else {
        OP_LOGE(context, "unsupported input dtype");
        return ge::GRAPH_FAILED;
    }

    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    input.coreNum = platform.GetCoreNum();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, input.ubBytes);
    input.socVersion = platform.GetSocVersion();
    OP_CHECK_IF(input.coreNum == 0 || input.ubBytes == 0,
                OP_LOGE(context, "failed to query non-zero core count and UB size"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalculateContainerSizes(gert::TilingContext* context, const TilingInputs& input, uint64_t& totalSlices,
                                        uint64_t& validBytes, uint64_t& containerBytes)
{
    uint64_t outputElements = 0;
    uint64_t kernelElements = 0;
    uint64_t maskWidth = 0;
    OP_CHECK_IF(
        !SafeMultiply(static_cast<uint64_t>(input.batch), static_cast<uint64_t>(input.channels), totalSlices) ||
            !SafeMultiply(static_cast<uint64_t>(input.outputHeight), static_cast<uint64_t>(input.outputWidth),
                          outputElements) ||
            !SafeMultiply(static_cast<uint64_t>(input.kernel.h), static_cast<uint64_t>(input.kernel.w), kernelElements),
        OP_LOGE(context, "shape product overflows uint64"), return ge::GRAPH_FAILED);

    const uint64_t maskBlocks = outputElements / 16 + (outputElements % 16 != 0) + 1;
    OP_CHECK_IF(!SafeMultiply(maskBlocks, 32, maskWidth), OP_LOGE(context, "indices mask width overflows uint64"),
                return ge::GRAPH_FAILED);
    if (kernelElements == 1) {
        uint64_t indexWidth = 0;
        OP_CHECK_IF(!SafeMultiply(outputElements / 8 + (outputElements % 8 != 0), 32, indexWidth),
                    OP_LOGE(context, "int32 indices width overflows uint64"), return ge::GRAPH_FAILED);
        maskWidth = std::max(maskWidth, indexWidth);
    }

    uint64_t validElements = 0;
    uint64_t containerElements = 0;
    OP_CHECK_IF(!SafeMultiply(totalSlices, outputElements, validElements) ||
                    !SafeMultiply(validElements, sizeof(int32_t), validBytes) ||
                    !SafeMultiply(totalSlices, kernelElements, containerElements) ||
                    !SafeMultiply(containerElements, maskWidth, containerBytes),
                OP_LOGE(context, "indices byte count overflows uint64"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(validBytes > containerBytes,
                OP_LOGE(context, "indices container has %lu bytes but int32 argmax requires %lu bytes", containerBytes,
                        validBytes),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalculateUbTiling(gert::TilingContext* context, const TilingInputs& input, uint64_t& inputWidthAligned,
                                  uint64_t& outputRowsPerTile, uint64_t& columnBufferElements,
                                  uint64_t& vectorBufferElements)
{
    const uint64_t inputRowBytes = AlignUp(static_cast<uint64_t>(input.inputWidth) * input.dtypeBytes, ALIGN_BYTES);
    inputWidthAligned = inputRowBytes / input.dtypeBytes;
    const uint64_t outputValueRowBytes = AlignUp(static_cast<uint64_t>(input.outputWidth) * input.dtypeBytes,
                                                 ALIGN_BYTES);
    const uint64_t outputIndexRowBytes = AlignUp(static_cast<uint64_t>(input.outputWidth) * sizeof(int32_t),
                                                 ALIGN_BYTES);
    const uint64_t kernelRows = static_cast<uint64_t>(input.dilation.h) * (input.kernel.h - 1) + 1;

    OP_CHECK_IF(input.ubBytes <= UB_RESERVE_BYTES + ZERO_BUFFER_BYTES,
                OP_LOGE(context, "UB is too small for fixed buffers"), return ge::GRAPH_FAILED);
    uint64_t availableBytes = input.ubBytes - UB_RESERVE_BYTES - ZERO_BUFFER_BYTES;
    const uint64_t firstRowBytes = DOUBLE_BUFFER *
                                   (kernelRows * inputRowBytes + outputValueRowBytes + outputIndexRowBytes);
    columnBufferElements = 0;
    vectorBufferElements = 0;
    if (input.socVersion == platform_ascendc::SocVersion::ASCEND310P && input.dtype == ge::DT_FLOAT) {
        const int64_t interiorStart = (input.padding.w + input.stride.w - 1) / input.stride.w;
        const int64_t lastStart = input.inputWidth + input.padding.w - input.kernel.w;
        const int64_t interiorEnd = lastStart < 0 ? 0 : std::min(input.outputWidth, lastStart / input.stride.w + 1);
        constexpr uint64_t vectorElements = 64;
        constexpr uint64_t vectorBytes = vectorElements * 6 * sizeof(uint32_t) + ALIGN_BYTES;
        if (interiorEnd - interiorStart >= 8 && firstRowBytes <= availableBytes &&
            vectorBytes <= availableBytes - firstRowBytes) {
            vectorBufferElements = vectorElements;
            availableBytes -= vectorBytes;
        }
        const uint64_t directComparisons = static_cast<uint64_t>(input.outputWidth) * input.kernel.h * input.kernel.w;
        const uint64_t separableComparisons = static_cast<uint64_t>(input.inputWidth) * input.kernel.h +
                                              static_cast<uint64_t>(input.outputWidth) * input.kernel.w;
        const uint64_t alignedColumns = AlignUp(input.inputWidth, ALIGN_BYTES / sizeof(uint32_t));
        const uint64_t columnBytes = alignedColumns * (sizeof(uint32_t) + sizeof(int32_t));
        // Equal comparison counts still reduce repeated FLOAT32 key construction.
        if (vectorBufferElements == 0 && separableComparisons <= directComparisons && firstRowBytes <= availableBytes &&
            columnBytes <= availableBytes - firstRowBytes) {
            columnBufferElements = alignedColumns;
            availableBytes -= columnBytes;
        }
    }
    if (firstRowBytes > availableBytes) {
        // The direct GM fallback only needs the fixed zero buffer.
        outputRowsPerTile = 0;
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t additionalRowBytes = DOUBLE_BUFFER * (static_cast<uint64_t>(input.stride.h) * inputRowBytes +
                                                         outputValueRowBytes + outputIndexRowBytes);
    outputRowsPerTile = 1;
    if (additionalRowBytes != 0) {
        outputRowsPerTile += (availableBytes - firstRowBytes) / additionalRowBytes;
    }
    outputRowsPerTile = std::min(outputRowsPerTile, static_cast<uint64_t>(input.outputHeight));
    return ge::GRAPH_SUCCESS;
}

struct VectorBufferSizes {
    uint64_t input = 0;
    uint64_t output = 0;
    uint64_t indices = 0;
    uint64_t indicesTemporary = 0;
    uint64_t mask = 0;
    uint64_t total = std::numeric_limits<uint64_t>::max();
};

struct VectorTiling {
    uint64_t ncFactor = 0;
    uint64_t ncTail = 0;
    uint64_t ncOuter = 0;
    uint64_t hyFactor = 0;
    uint64_t hyTail = 0;
    uint64_t hyOuter = 0;
    uint64_t wyFactor = 0;
    uint64_t wyTail = 0;
    uint64_t wyOuter = 0;
    uint64_t blockFactor = 0;
    uint64_t blockTail = 0;
    uint64_t totalIdx = 0;
    uint64_t coreNums = 0;
    uint64_t padBottom = 0;
    uint64_t padRight = 0;
    VectorBufferSizes buffers;
};

uint64_t SaturatingMultiply(uint64_t lhs, uint64_t rhs)
{
    uint64_t result = 0;
    return SafeMultiply(lhs, rhs, result) ? result : std::numeric_limits<uint64_t>::max();
}

uint64_t SaturatingAdd(uint64_t lhs, uint64_t rhs)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs + rhs;
}

uint64_t SaturatingAlignUp(uint64_t value, uint64_t alignment)
{
    if (value == std::numeric_limits<uint64_t>::max() || alignment == 0) {
        return std::numeric_limits<uint64_t>::max();
    }
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? value : SaturatingAdd(value, alignment - remainder);
}

VectorBufferSizes CalculateVectorBufferSizes(const TilingInputs& input, uint64_t ncFactor, uint64_t hyFactor,
                                             uint64_t wyFactor)
{
    VectorBufferSizes sizes;
    const uint64_t inputBytes = input.dtypeBytes;
    const uint64_t computeBytes = input.dtype == ge::DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float);
    const uint64_t blockDataCount = ALIGN_BYTES / computeBytes;
    const uint64_t ncMaximumFactor = blockDataCount * VECTOR_BLOCK_COUNT;
    const uint64_t ncAligned = SaturatingAlignUp(ncFactor, VECTOR_TRANSPOSE_ALIGN);
    const uint64_t wyAligned = SaturatingAlignUp(wyFactor, VECTOR_TRANSPOSE_ALIGN);
    const uint64_t hxFactor = SaturatingAdd(SaturatingMultiply(hyFactor - 1, static_cast<uint64_t>(input.stride.h)),
                                            static_cast<uint64_t>(input.kernel.h));
    const uint64_t wxFactor = SaturatingAdd(SaturatingMultiply(wyFactor - 1, static_cast<uint64_t>(input.stride.w)),
                                            static_cast<uint64_t>(input.kernel.w));
    const uint64_t wxAligned = SaturatingAlignUp(wxFactor, ALIGN_BYTES / inputBytes);

    sizes.output = SaturatingAlignUp(
        SaturatingMultiply(SaturatingMultiply(ncAligned, hyFactor), SaturatingMultiply(wyAligned, computeBytes)),
        VECTOR_CACHE_ALIGN_BYTES);
    sizes.indices = SaturatingAlignUp(SaturatingMultiply(SaturatingMultiply(ncMaximumFactor, hyFactor),
                                                         SaturatingMultiply(wyAligned, sizeof(int32_t))),
                                      VECTOR_CACHE_ALIGN_BYTES);
    sizes.input = SaturatingAlignUp(
        SaturatingMultiply(SaturatingMultiply(ncAligned, hxFactor), SaturatingMultiply(wxAligned, computeBytes)),
        VECTOR_CACHE_ALIGN_BYTES);
    sizes.indicesTemporary = SaturatingAlignUp(
        SaturatingMultiply(SaturatingMultiply(ncMaximumFactor, wyAligned), sizeof(int32_t)), VECTOR_CACHE_ALIGN_BYTES);
    const uint64_t maskRowBytes = AlignUp(blockDataCount * VECTOR_BLOCK_COUNT / 16 * sizeof(uint16_t), ALIGN_BYTES);
    sizes.mask = SaturatingAlignUp(SaturatingMultiply(maskRowBytes, wyAligned), VECTOR_CACHE_ALIGN_BYTES);

    uint64_t total = 0;
    for (const uint64_t part : {sizes.input, sizes.output, sizes.indices, sizes.mask}) {
        total = SaturatingAdd(total, SaturatingMultiply(part, 2));
    }
    // The kernel keeps only the column template in this size. The per-candidate index
    // tensor is gone: the window offset is now a scalar operand of the select.
    total = SaturatingAdd(total, sizes.indicesTemporary);
    sizes.total = total;
    return sizes;
}

uint64_t FindLargestVectorFactor(uint64_t upperBound, const TilingInputs& input, uint64_t ncFactor,
                                 uint64_t fixedFactor, bool searchWidth, uint64_t availableBytes)
{
    uint64_t lower = 1;
    uint64_t upper = upperBound;
    while (lower < upper) {
        const uint64_t middle = lower + (upper - lower + 1) / 2;
        const uint64_t hyFactor = searchWidth ? fixedFactor : middle;
        const uint64_t wyFactor = searchWidth ? middle : fixedFactor;
        if (CalculateVectorBufferSizes(input, ncFactor, hyFactor, wyFactor).total <= availableBytes) {
            lower = middle;
        } else {
            upper = middle - 1;
        }
    }
    return lower;
}

uint64_t SpatialTileRows(const TilingInputs& input, uint64_t totalSlices, uint64_t outputPitch)
{
    uint64_t rowAlignment = 16;
    while (rowAlignment > 1 && (input.outputWidth * (rowAlignment / 2)) % 16 == 0) {
        rowAlignment /= 2;
    }
    const uint64_t targetTiles = std::max<uint64_t>(1, input.coreNum / totalSlices);
    const uint64_t minRows = std::max<uint64_t>(1, (128 + outputPitch - 1) / outputPitch);
    return std::min<uint64_t>(
        input.outputHeight,
        AlignUp(std::max<uint64_t>(minRows, (input.outputHeight + targetTiles - 1) / targetTiles), rowAlignment));
}

bool PreferSpatialScan(const TilingInputs& input, uint64_t totalSlices, uint64_t inputPitch, uint64_t outputPitch,
                       const VectorTiling& vector, bool windowFallback)
{
    const uint64_t tileRows = SpatialTileRows(input, totalSlices, outputPitch);
    const uint64_t inputRows = (tileRows - 1) * input.stride.h + input.kernel.h;
    const uint64_t tasks = totalSlices * ((input.outputHeight + tileRows - 1) / tileRows);
    const uint64_t rounds = (tasks + input.coreNum - 1) / input.coreNum;
    uint64_t levels = 0;
    for (uint64_t width = input.kernel.w; width > 1; width /= 2) {
        ++levels;
    }
    // Source-operation estimate in 64-lane repeats, not a device-time model.
    // Geometry/encoding costs about 24 input passes; each prefix/suffix level
    // has ten passes, the horizontal query 23, and output/decode about 40.
    // Include padded lanes, halo rows, and sequential tasks on the busiest core.
    const uint64_t inputRepeats = (inputRows * inputPitch + 63) / 64;
    const uint64_t horizontalRepeats = (inputRows * outputPitch + 63) / 64;
    const uint64_t outputRepeats = (tileRows * outputPitch + 63) / 64;
    const bool clipRight = (input.outputWidth - 1) * input.stride.w + input.kernel.w > inputPitch;
    const uint64_t horizontalPasses = clipRight ? 29 : 23;
    const uint64_t scanWork = SaturatingMultiply(rounds, (24 + 20 * levels) * inputRepeats +
                                                             horizontalPasses * horizontalRepeats +
                                                             (40 + 2 * input.kernel.h) * outputRepeats);
    if (windowFallback) {
        // Compare against independent complete-window reductions, including
        // their available output parallelism. Small output counts alone do not
        // justify constructing prefix/suffix summaries for the whole input.
        const uint64_t windowRows = std::min<uint64_t>(input.inputHeight, input.kernel.h);
        const uint64_t windowPitch = AlignUp(std::min<uint64_t>(input.inputWidth, input.kernel.w), 16);
        const uint64_t windowRepeats = (windowRows * windowPitch + 63) / 64;
        const uint64_t outputs = totalSlices * input.outputHeight * input.outputWidth;
        const uint64_t windowRounds = (outputs + input.coreNum - 1) / input.coreNum;
        return scanWork < SaturatingMultiply(windowRounds, 9 * windowRepeats);
    }
    // NC pooling issues a Compare and two Select repeats per output column
    // and window position, even when most channel lanes are masked off.
    const uint64_t vectorRounds = (vector.totalIdx + input.coreNum - 1) / input.coreNum;
    const uint64_t vectorWork = SaturatingMultiply(
        vectorRounds, SaturatingMultiply(vector.hyFactor * vector.wyFactor, 3 * input.kernel.h * input.kernel.w));
    return scanWork < vectorWork;
}

bool CanCacheInputPlane(const TilingInputs& input, uint64_t totalSlices)
{
    const uint64_t outputPlane = static_cast<uint64_t>(input.outputHeight) * input.outputWidth;
    uint64_t pitch = 1;
    const uint64_t width = std::min<uint64_t>(input.inputWidth, input.kernel.w);
    while (pitch < width && pitch < 16) {
        pitch *= 2;
    }
    // Small windows benefit from batching channels in the vector path. Plane
    // caching is reserved for the large-window reduction family.
    return input.dtype == ge::DT_BF16 && static_cast<uint64_t>(input.kernel.h) * input.kernel.w >= 64 &&
           totalSlices >= input.coreNum && input.inputWidth <= 16 && outputPlane > 1 &&
           outputPlane <= 2 * MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS &&
           SaturatingMultiply(input.inputHeight, input.inputWidth) <= MAX_POOL2D_WINDOW_INPUT_BYTES / 2 &&
           SaturatingMultiply(std::min<uint64_t>(input.inputHeight, input.kernel.h), pitch) <=
               MAX_POOL2D_WINDOW_INPUT_BYTES / 2;
}

bool HasEquivalentWindows(int64_t size, int64_t kernel, int64_t stride, int64_t pad, int64_t output)
{
    const int64_t lower = std::max<int64_t>(0, size + pad - kernel);
    const int64_t first = lower / stride + (lower % stride != 0);
    const int64_t last = std::min<int64_t>(output - 1, pad / stride);
    return last > first;
}

ge::graphStatus CalculateVectorTiling(gert::TilingContext* context, const TilingInputs& input, uint64_t totalSlices,
                                      VectorTiling& result, bool& useWindowFallback)
{
    OP_CHECK_IF(input.ubBytes <= VECTOR_UB_RESERVE_BYTES + VECTOR_OR_MASK_BYTES,
                OP_LOGE(context, "UB is too small for the A2 vector path"), return ge::GRAPH_FAILED);
    // BF16 has a compute-work balancing search below. For other dtypes, retain
    // the established tile budget: larger tiles can halve active pooling cores.
    const uint64_t tileBudgetReserve = input.dtype == ge::DT_BF16 ? 0 : ZERO_BUFFER_BYTES;
    OP_CHECK_IF(input.ubBytes <= VECTOR_UB_RESERVE_BYTES + VECTOR_OR_MASK_BYTES + tileBudgetReserve,
                OP_LOGE(context, "UB is too small for the vector tile budget"), return ge::GRAPH_FAILED);
    const uint64_t availableBytes = input.ubBytes - VECTOR_UB_RESERVE_BYTES - VECTOR_OR_MASK_BYTES - tileBudgetReserve;
    const uint64_t computeBytes = input.dtype == ge::DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float);
    const uint64_t ncMaximumFactor = ALIGN_BYTES / computeBytes * VECTOR_BLOCK_COUNT;
    result.ncFactor = std::min(totalSlices, ncMaximumFactor);
    result.ncTail = totalSlices % result.ncFactor;
    result.ncTail = result.ncTail == 0 ? result.ncFactor : result.ncTail;
    result.ncOuter = (totalSlices + result.ncFactor - 1) / result.ncFactor;

    // With few output positions, large windows offer little reuse to amortize
    // three transposes. The ordered-bit reduction reads each valid window once.
    const uint64_t outputPlane = static_cast<uint64_t>(input.outputHeight) * input.outputWidth;
    // A few channels waste most lanes of the NC transpose. If every output
    // fits on a separate core, independent window reduction avoids that layout.
    const bool independentSmallOutputs = totalSlices <= 8 && totalSlices * outputPlane <= input.coreNum;
    // Tiny independent outputs cannot amortize the NC input/output transposes.
    // Clip the work estimate to the real input, as padding contributes no values.
    const uint64_t validWindowElements = SaturatingMultiply(std::min<int64_t>(input.inputHeight, input.kernel.h),
                                                            std::min<int64_t>(input.inputWidth, input.kernel.w));
    const bool tinyIndependentOutputs = independentSmallOutputs && validWindowElements <= 64;
    const bool preferWindowReduction = tinyIndependentOutputs ||
                                       (input.dtype == ge::DT_BF16 &&
                                        ((static_cast<uint64_t>(input.kernel.h) * input.kernel.w >= 64 &&
                                          outputPlane <= 16) ||
                                         independentSmallOutputs));
    if (preferWindowReduction || CanCacheInputPlane(input, totalSlices) ||
        CalculateVectorBufferSizes(input, result.ncFactor, 1, 1).total > availableBytes) {
        constexpr uint64_t windowBytes = 2 * MAX_POOL2D_WINDOW_INPUT_BYTES +
                                         2 * MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS * sizeof(uint32_t) +
                                         MAX_POOL2D_WINDOW_ZERO_BYTES;
        const uint64_t vectorBytes = input.dtypeBytes == 2 ? MAX_POOL2D_WINDOW_VECTOR_BYTES : 0;
        OP_CHECK_IF(windowBytes + vectorBytes + UB_RESERVE_BYTES > input.ubBytes,
                    OP_LOGE(context, "UB is too small for the window reduction fallback"), return ge::GRAPH_FAILED);
        useWindowFallback = true;
        result = {};
        return ge::GRAPH_SUCCESS;
    }
    result.wyFactor = FindLargestVectorFactor(input.outputWidth, input, result.ncFactor, 1, true, availableBytes);
    result.hyFactor = FindLargestVectorFactor(input.outputHeight, input, result.ncFactor, result.wyFactor, false,
                                              availableBytes);
    result.buffers = CalculateVectorBufferSizes(input, result.ncFactor, result.hyFactor, result.wyFactor);
    const uint64_t initialTasks = result.ncOuter * ((input.outputHeight + result.hyFactor - 1) / result.hyFactor) *
                                  ((input.outputWidth + result.wyFactor - 1) / result.wyFactor);
    const uint64_t outputCount = totalSlices * input.outputHeight * input.outputWidth;
    if (input.dtype == ge::DT_BF16 && initialTasks < input.coreNum && outputCount / input.coreNum >= 128) {
        // Compare complete legal tiles, including aligned data movement and the
        // maximum number of tiles assigned to a core. Filling UB alone can leave
        // most launched cores doing only index-tail stores.
        const uint64_t area = static_cast<uint64_t>(input.kernel.h) * input.kernel.w;
        auto estimate = [&](uint64_t ncFactor, uint64_t height, uint64_t width, const VectorBufferSizes& buffers) {
            const uint64_t tasks = ((totalSlices + ncFactor - 1) / ncFactor) *
                                   ((input.outputHeight + height - 1) / height) *
                                   ((input.outputWidth + width - 1) / width);
            const uint64_t rounds = (tasks + input.coreNum - 1) / input.coreNum;
            const uint64_t vectorWork = SaturatingMultiply(area + 3, SaturatingMultiply(height, AlignUp(width, 16)));
            const uint64_t movement = (buffers.input + buffers.output + buffers.indices + buffers.indicesTemporary) /
                                      32;
            const uint64_t rowSetup = SaturatingMultiply(height, SaturatingMultiply(area + 3, 64));
            return SaturatingMultiply(rounds,
                                      SaturatingAdd(256, SaturatingAdd(rowSetup, SaturatingAdd(vectorWork, movement))));
        };
        uint64_t best = estimate(result.ncFactor, result.hyFactor, result.wyFactor, result.buffers);
        const uint64_t originalVectorWork = ((initialTasks + input.coreNum - 1) / input.coreNum) * result.hyFactor *
                                            AlignUp(result.wyFactor, VECTOR_TRANSPOSE_ALIGN);
        const uint64_t originalNc = result.ncFactor;
        for (uint64_t ncCandidate :
             {originalNc, std::min(originalNc, uint64_t{32}), std::min(originalNc, uint64_t{16})}) {
            const uint64_t maxWidth = FindLargestVectorFactor(input.outputWidth, input, ncCandidate, 1, true,
                                                              availableBytes);
            for (uint64_t widthDivisor : {1U, 2U, 4U, 8U}) {
                const uint64_t width = (maxWidth + widthDivisor - 1) / widthDivisor;
                // Do not create extra width tiles smaller than one transpose
                // block. They retain a full padded output/index block while
                // duplicating halo reads and short DMA segments. Long rows
                // can still expose parallel work through complete blocks.
                if (widthDivisor > 1 && width < VECTOR_TRANSPOSE_ALIGN) {
                    continue;
                }
                const uint64_t maxHeight = FindLargestVectorFactor(input.outputHeight, input, ncCandidate, width, false,
                                                                   availableBytes);
                for (uint64_t heightDivisor : {1U, 2U, 4U, 8U}) {
                    const uint64_t height = (maxHeight + heightDivisor - 1) / heightDivisor;
                    const auto buffers = CalculateVectorBufferSizes(input, ncCandidate, height, width);
                    const uint64_t score = estimate(ncCandidate, height, width, buffers);
                    const uint64_t tasks = ((totalSlices + ncCandidate - 1) / ncCandidate) *
                                           ((input.outputHeight + height - 1) / height) *
                                           ((input.outputWidth + width - 1) / width);
                    const uint64_t vectorWork = ((tasks + input.coreNum - 1) / input.coreNum) * height *
                                                AlignUp(width, VECTOR_TRANSPOSE_ALIGN);
                    // Index arithmetic still uses a complete vector of NC lanes.
                    // Less input movement cannot justify equal or greater aligned
                    // reduction work on the critical core.
                    if (buffers.total <= availableBytes && vectorWork < originalVectorWork && score < best) {
                        best = score;
                        result.ncFactor = ncCandidate;
                        result.hyFactor = height;
                        result.wyFactor = width;
                        result.buffers = buffers;
                    }
                }
            }
        }
        result.ncTail = totalSlices % result.ncFactor;
        result.ncTail = result.ncTail == 0 ? result.ncFactor : result.ncTail;
        result.ncOuter = (totalSlices + result.ncFactor - 1) / result.ncFactor;
    }
    result.wyTail = static_cast<uint64_t>(input.outputWidth) % result.wyFactor;
    result.wyTail = result.wyTail == 0 ? result.wyFactor : result.wyTail;
    result.wyOuter = (static_cast<uint64_t>(input.outputWidth) + result.wyFactor - 1) / result.wyFactor;
    result.hyTail = static_cast<uint64_t>(input.outputHeight) % result.hyFactor;
    result.hyTail = result.hyTail == 0 ? result.hyFactor : result.hyTail;
    result.hyOuter = (static_cast<uint64_t>(input.outputHeight) + result.hyFactor - 1) / result.hyFactor;

    result.totalIdx = result.ncOuter * result.hyOuter * result.wyOuter;
    result.blockFactor = (result.totalIdx + input.coreNum - 1) / input.coreNum;
    result.blockTail = result.totalIdx % result.blockFactor;
    result.blockTail = result.blockTail == 0 ? result.blockFactor : result.blockTail;
    result.coreNums = (result.totalIdx + result.blockFactor - 1) / result.blockFactor;

    const uint64_t usedHeight = (static_cast<uint64_t>(input.outputHeight) - 1) * input.stride.h + input.kernel.h;
    const uint64_t usedWidth = (static_cast<uint64_t>(input.outputWidth) - 1) * input.stride.w + input.kernel.w;
    const uint64_t paddedInputHeight = static_cast<uint64_t>(input.inputHeight + input.padding.h);
    const uint64_t paddedInputWidth = static_cast<uint64_t>(input.inputWidth + input.padding.w);
    result.padBottom = usedHeight > paddedInputHeight ? usedHeight - paddedInputHeight : 0;
    result.padRight = usedWidth > paddedInputWidth ? usedWidth - paddedInputWidth : 0;
    return ge::GRAPH_SUCCESS;
}

bool CanReuseM200SeparableBuffers(const TilingInputs& input, uint64_t rows, bool parity = false)
{
    if (rows == 0 || input.kernel.h != input.kernel.w || (input.kernel.h != 7 && !(parity && input.kernel.h == 3)) ||
        input.stride.h != 2 || input.stride.w != 2 ||
        input.inputHeight * input.inputWidth * input.dtypeBytes % ALIGN_BYTES != 0 ||
        input.outputHeight * input.outputWidth * input.dtypeBytes % ALIGN_BYTES != 0) {
        return false;
    }
    const uint64_t inputRows = (rows - 1) * 2 + input.kernel.h;
    const uint64_t paddedWidth = (input.outputWidth - 1) * 2 + input.kernel.w;
    const bool padStatic = input.padding.h == 0 && input.padding.w == 0 &&
                           paddedWidth == static_cast<uint64_t>(input.inputWidth) &&
                           input.inputWidth * input.dtypeBytes % ALIGN_BYTES == 0 &&
                           (input.outputHeight - 1) * 2 + input.kernel.h <= input.inputHeight;
    if (padStatic) {
        return false;
    }
    // These are suballocations of existing buffers, not additional UB storage.
    // Keep the bounds in agreement with M200TransposeKernel's largest-tile check.
    const uint64_t inputSpace = AlignUp(inputRows * input.inputWidth + 64 / input.dtypeBytes, 16);
    const uint64_t rawElems = std::max(inputSpace, inputRows * paddedWidth + input.kernel.w + 16);
    const uint64_t poolCount = AlignUp(rows * input.outputWidth, 16) * 16;
    const uint64_t horizontalCount = AlignUp(inputRows * input.outputWidth * 16, 256);
    return 2 * horizontalCount * input.dtypeBytes <= inputSpace * 16 * input.dtypeBytes &&
           horizontalCount * 4 <= rawElems * 16 * input.dtypeBytes &&
           (input.kernel.w - 1) * (horizontalCount / 8) <= poolCount * 4 &&
           horizontalCount * input.dtypeBytes >= poolCount * 4 &&
           (!parity || (inputRows - rows) * input.outputWidth * 16 + poolCount <= horizontalCount);
}

bool CanSeparateM200FlatWindow(const TilingInputs& input, uint64_t rows)
{
    if (input.dtype != ge::DT_FLOAT || rows == 0 || input.kernel.h != 3 || input.kernel.w != 3 || input.stride.h != 1 ||
        input.stride.w != 1 || input.inputHeight * input.inputWidth * input.dtypeBytes % ALIGN_BYTES != 0 ||
        input.outputHeight * input.outputWidth * input.dtypeBytes % ALIGN_BYTES != 0) {
        return false;
    }
    const uint64_t pitch = (input.outputWidth + 2) * 16;
    const uint64_t count = AlignUp(rows * pitch, 64);
    const uint64_t horizontal = AlignUp(count + 2 * pitch, 64);
    const uint64_t inputSpace = AlignUp((rows + 2) * input.inputWidth + 16, 16) * 16;
    const uint64_t padded = ((rows + 2) * (input.outputWidth + 2) + 19) * 16;
    const uint64_t poolCount = AlignUp(rows * (input.outputWidth + 2), 16) * 16;
    // The ordinary row budget reserves 24 KiB, of which 8 KiB is the zero
    // source. Consume at most 8 KiB more and retain the SDK's 8 KiB scratch.
    const uint64_t extraBytes = horizontal > inputSpace ? (horizontal - inputSpace) * 4 : 0;
    return extraBytes <= 8192 && horizontal + 2 * 16 <= std::max(inputSpace, padded) &&
           2 * AlignUp(horizontal / 8, 32) <= poolCount * 4;
}

bool CanCacheM200HorizontalRow(const TilingInputs& input, uint64_t rows, uint64_t totalSlices, uint64_t usedCoreNum)
{
    if (input.kernel.h != 3 || !CanReuseM200SeparableBuffers(input, rows, true) ||
        static_cast<uint64_t>(input.outputHeight) <= rows) {
        return false;
    }
    // Tasks are contiguous tile-major ranges before the kernel visits each
    // channel group. A range no longer than the number of groups contains no
    // two adjacent tiles from the same group, so it cannot reuse a cached row.
    const uint64_t groups = totalSlices / 16;
    const uint64_t tiles = (input.outputHeight + rows - 1) / rows;
    const uint64_t tasksPerCore = (tiles * groups + usedCoreNum - 1) / usedCoreNum;
    if (groups == 0 || tasksPerCore <= groups) {
        return false;
    }
    const uint64_t inputRows = 2 * rows + 1;
    const uint64_t prefix = input.outputWidth * 16;
    const uint64_t horizontal = AlignUp(inputRows * prefix, 256);
    if (prefix + AlignUp((inputRows - 1) * prefix, 256) > horizontal) {
        return false; // The fresh suffix, including Compare/Select tails, must fit.
    }
    const uint64_t inputSpace = AlignUp(inputRows * input.inputWidth + 64 / input.dtypeBytes, 16);
    const uint64_t padded = inputRows * (2 * input.outputWidth + 1) + 19;
    const uint64_t raw = std::max(inputSpace, padded);
    const uint64_t poolCount = AlignUp(rows * input.outputWidth, 16) * 16;
    const uint64_t bytes = 24 * 1024 + (inputSpace + raw) * 16 * input.dtypeBytes +
                           poolCount * (2 * input.dtypeBytes + 12) + AlignUp(poolCount / 8, 32) +
                           prefix * (input.dtypeBytes + 4);
    return bytes <= input.ubBytes;
}

uint64_t CalculateM200TransposeRows(const TilingInputs& input, uint64_t totalSlices, uint64_t ncFactor = 16,
                                    bool reuseFlatBuffers = false)
{
    if (input.socVersion != platform_ascendc::SocVersion::ASCEND310P || totalSlices < 16 || totalSlices % 16 != 0 ||
        (input.dtype != ge::DT_FLOAT && input.dtype != ge::DT_FLOAT16) ||
        input.inputHeight * input.inputWidth > (1U << 24) || input.kernel.h > 7 || input.kernel.w > 7 ||
        input.stride.h > 4 || input.stride.w > 4) {
        return 0;
    }
    const uint64_t inputPlane = input.inputHeight * input.inputWidth;
    const uint64_t outputPlane = input.outputHeight * input.outputWidth;
    const bool packed = inputPlane * input.dtypeBytes % ALIGN_BYTES != 0 ||
                        outputPlane * input.dtypeBytes % ALIGN_BYTES != 0;
    if (packed && (inputPlane > 64 || outputPlane > 64)) {
        return 0;
    }
    const uint64_t rowGroup = packed ? input.outputHeight :
                                       ALIGN_BYTES / std::gcd(input.outputWidth * input.dtypeBytes, ALIGN_BYTES);
    const uint64_t paddedWidth = (input.outputWidth - 1) * input.stride.w + input.kernel.w;
    uint64_t selected = 0;
    for (uint64_t rows = rowGroup; rows <= static_cast<uint64_t>(input.outputHeight); rows += rowGroup) {
        const uint64_t inputRows = (rows - 1) * input.stride.h + input.kernel.h;
        const uint64_t inputSpace = AlignUp(
            packed ? inputPlane : inputRows * input.inputWidth + 2 * ALIGN_BYTES / input.dtypeBytes, 16);
        const uint64_t outputSpace = AlignUp(rows * input.outputWidth, 16);
        // Unit stride pools at the padded row pitch so the rows line up in one
        // instruction; the running buffers follow that pitch and are slightly wider.
        // Match the kernel allocation, including the final window in ceil mode.
        const bool unitStride = input.stride.h == 1 && input.stride.w == 1;
        const bool flatReuse = reuseFlatBuffers && unitStride && !packed;
        const bool compactAlias = flatReuse && inputSpace * input.dtypeBytes >= outputSpace * 4;
        const bool padStatic = !unitStride && input.padding.h == 0 && input.padding.w == 0 &&
                               paddedWidth == static_cast<uint64_t>(input.inputWidth) &&
                               (input.inputWidth * input.dtypeBytes) % 32 == 0 &&
                               (input.outputHeight - 1) * input.stride.h + input.kernel.h <= input.inputHeight;
        const uint64_t paddedElems = inputRows * paddedWidth + input.kernel.w + 16;
        const bool padAlias = !padStatic && !packed;
        const uint64_t rawElems = padAlias ? std::max(paddedElems, inputSpace) : inputSpace;
        // The raw allocation grows to hold padding, replacing its separate buffer.
        const uint64_t paddedBytes = ((rawElems - inputSpace) + (padStatic || padAlias ? 0 : paddedElems)) * ncFactor *
                                     input.dtypeBytes;
        const uint64_t poolSpace = (input.stride.h == 1 && input.stride.w == 1) ?
                                       std::max(outputSpace, AlignUp(rows * paddedWidth, 16)) :
                                       outputSpace;
        const uint64_t packedOutputs = AlignUp(outputPlane * ncFactor, 128);
        const uint64_t packedBytes = packed ? std::max(inputPlane * ncFactor * input.dtypeBytes, packedOutputs * 4) +
                                                  inputSpace * ncFactor * 4 + packedOutputs * 8 :
                                              0;
        const uint64_t bytes = 24 * 1024 + packedBytes + (packed ? 1 : 2) * inputSpace * ncFactor * input.dtypeBytes +
                               paddedBytes + poolSpace * ncFactor * ((flatReuse ? 1 : 2) * input.dtypeBytes + 8) +
                               (packed ? 0 : outputSpace * ncFactor * 4 * (unitStride && !compactAlias ? 2 : 1)) +
                               AlignUp(poolSpace * ncFactor / 8, 32);
        if (bytes > input.ubBytes) {
            break;
        }
        selected = rows;
    }
    if (reuseFlatBuffers && !packed && input.stride.h == 1 && input.stride.w == 1 && selected != 0) {
        // A larger allocation is unnecessary if a smaller aligned tile needs
        // exactly the same number of Compute calls. It also makes the final
        // input/output transposes process more unused rows. Keep the minimum
        // aligned row capacity that preserves the selected number of tiles.
        const uint64_t tiles = (input.outputHeight + selected - 1) / selected;
        selected = AlignUp((input.outputHeight + tiles - 1) / tiles, rowGroup);
    }
    return selected;
}

uint64_t CalculateM200SpatialRows(const TilingInputs& input)
{
    if (input.socVersion != platform_ascendc::SocVersion::ASCEND310P ||
        (input.dtype != ge::DT_FLOAT && input.dtype != ge::DT_FLOAT16) || input.kernel.h != 2 || input.kernel.w != 2 ||
        input.stride.h != 2 || input.stride.w != 2 || input.padding.h != 0 || input.padding.w != 0 ||
        2 * input.outputWidth != input.inputWidth || 2 * input.outputHeight > input.inputHeight ||
        input.inputWidth * input.dtypeBytes % 32 != 0 ||
        input.outputHeight * input.outputWidth * input.dtypeBytes % SCALAR_CACHE_LINE_BYTES != 0 ||
        input.inputHeight * input.inputWidth > (1U << 24) || input.inputWidth * input.dtypeBytes / 32 > 65535) {
        return 0;
    }
    const uint64_t rowGroup = SCALAR_CACHE_LINE_BYTES /
                              std::gcd(static_cast<uint64_t>(input.outputWidth) * input.dtypeBytes,
                                       SCALAR_CACHE_LINE_BYTES);
    uint64_t selected = 0;
    for (uint64_t rows = rowGroup; rows <= static_cast<uint64_t>(input.outputHeight) && rows <= 4095;
         rows += rowGroup) {
        const uint64_t capacity = AlignUp(rows * input.outputWidth, 256);
        // GatherMask uses an 8-bit repeat count; keep conservative scratch headroom.
        const uint64_t bytes = 24 * 1024 + 2 * rows * input.inputWidth * input.dtypeBytes +
                               capacity * (4 * input.dtypeBytes + 8) + AlignUp(capacity / 8, 32);
        if (2 * capacity * input.dtypeBytes / 256 > 255 || bytes > input.ubBytes) {
            break;
        }
        selected = rows;
    }
    return selected;
}

uint64_t CalculateM200SpatialBatch(const TilingInputs& input, uint64_t rows, uint64_t slicesPerCore)
{
    // A strided row copy can cross NC planes only if every input row belongs
    // to a complete window and each tile contains the entire output plane.
    if (rows == 0 || rows != static_cast<uint64_t>(input.outputHeight) || 2 * input.outputHeight != input.inputHeight) {
        return 1;
    }
    uint64_t selected = 1;
    for (uint64_t batch = 2; batch <= slicesPerCore && rows * batch <= 4095; batch *= 2) {
        const uint64_t capacity = AlignUp(rows * input.outputWidth * batch, 256);
        const uint64_t bytes = 24 * 1024 + 2 * rows * input.inputWidth * batch * input.dtypeBytes +
                               capacity * (4 * input.dtypeBytes + 8) + AlignUp(capacity / 8, 32);
        if (2 * capacity * input.dtypeBytes / 256 > 255 || bytes > input.ubBytes) {
            break;
        }
        selected = batch;
    }
    return selected;
}

bool CanPipelineM200SpatialInput(const TilingInputs& input, uint64_t rows, uint64_t batch, uint64_t slicesPerCore,
                                 bool outputPipeline = false)
{
    if (rows == 0 || batch >= slicesPerCore) {
        return false;
    }
    // Keep the existing rows and NC batch. The second input allocation must fit
    // without shortening continuous output writes or increasing Compute calls.
    const uint64_t capacity = AlignUp(rows * input.outputWidth * batch, 256);
    const uint64_t bytes = 24 * 1024 + 4 * rows * input.inputWidth * batch * input.dtypeBytes +
                           capacity * (4 * input.dtypeBytes + 8) + AlignUp(capacity / 8, 32) +
                           (outputPipeline ? capacity * (input.dtypeBytes + 4) : 0);
    return bytes <= input.ubBytes;
}

uint64_t CalculateM200IoPipelinedSpatialRows(const TilingInputs& input, uint64_t originalRows, uint64_t batch,
                                             uint64_t slicesPerCore)
{
    if (originalRows == 0 || batch != 1 || slicesPerCore <= 1 ||
        CanPipelineM200SpatialInput(input, originalRows, batch, slicesPerCore)) {
        return 0;
    }
    const uint64_t height = input.outputHeight;
    const uint64_t width = input.outputWidth;
    const uint64_t originalTiles = (height + originalRows - 1) / originalRows;
    const uint64_t originalWork = (height / originalRows) * AlignUp(originalRows * width, 256) +
                                  AlignUp((height % originalRows) * width, 256);
    const uint64_t rowGroup = SCALAR_CACHE_LINE_BYTES / std::gcd(width * input.dtypeBytes, SCALAR_CACHE_LINE_BYTES);
    for (uint64_t rows = originalRows; rows >= rowGroup; rows -= rowGroup) {
        if ((height + rows - 1) / rows > originalTiles + 1) {
            break;
        }
        const uint64_t capacity = AlignUp(rows * width, 256);
        const uint64_t work = (height / rows) * capacity + AlignUp((height % rows) * width, 256);
        // Ten explicit TBuf allocations. Each input slot contains two rounded
        // halves loaded with a GM row stride; no selected-row buffer remains.
        // Select's fixed 8KiB SDK scratch begins at 248KiB on M200.
        const uint64_t bytes = capacity * (11 * input.dtypeBytes + 12) + AlignUp(capacity / 8, 32) + 8192;
        if (bytes <= std::min<uint64_t>(input.ubBytes, 248 * 1024) && 2 * capacity * input.dtypeBytes / 256 <= 255 &&
            work <= originalWork) {
            return rows;
        }
    }
    return 0;
}

// Tiling keys split the work into classes that get their own code in the kernel binary.
// The point is not to change the algorithm but to keep the code each class runs together:
// profiling showed the remaining gap against the SDK is aiv_scalar_wait_ib_time, which the
// shared general-purpose path pays on every tile.
//
// Splitting on whether the window has padding gives each key exactly one specialisation
// instead of carrying both, and covers every shape without needing per-family thresholds.
// Measured on the narrower 3x3/stride2 FLOAT16 class this took F092's fetch wait from
// 14.20 us to 4.97 us with its vector time unchanged at 7.38 us.
//
// The class is derived from the operator parameters and the computed padding, never from
// a case name or an exact shape.
uint64_t SelectVectorTilingKey(const TilingInputs& input, const VectorTiling& vectorTiling)
{
    constexpr uint64_t VECTOR_KEY_UNPADDED = 0;
    constexpr uint64_t VECTOR_KEY_PADDED = 1;
    if (input.socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        return VECTOR_KEY_UNPADDED;
    }
    // Must match the kernel's own padding test exactly, since each key instantiates only
    // one specialisation: picking the wrong one is a correctness error, not a slow path.
    const bool hasPadding = input.padding.h != 0 || input.padding.w != 0 || vectorTiling.padBottom != 0 ||
                            vectorTiling.padRight != 0;
    // Specialize operator attributes, not individual shapes. M200 returned above.
    if (input.stride.h == 2 && input.stride.w == 2 && input.kernel.h == 3 && input.kernel.w == 3 && hasPadding) {
        return 11;
    }
    return hasPadding ? VECTOR_KEY_PADDED : VECTOR_KEY_UNPADDED;
}

uint64_t CalculateM200SpatialSeparableRows(const TilingInputs& input, uint64_t totalSlices)
{
    if (input.socVersion != platform_ascendc::SocVersion::ASCEND310P || input.dtype != ge::DT_FLOAT ||
        input.kernel.h != 3 || input.kernel.w != 3 || input.stride.h != 2 || input.stride.w != 2 ||
        input.padding.h != 1 || input.padding.w != 1 || totalSlices < 64 || input.inputWidth < 64 ||
        2 * input.outputHeight != input.inputHeight || 2 * input.outputWidth != input.inputWidth ||
        input.outputWidth * input.dtypeBytes % 32 != 0 || input.outputWidth * 4 % 32 != 0 ||
        input.inputHeight * input.inputWidth > (1U << 24)) {
        return 0;
    }
    const uint64_t available = std::min<uint64_t>(input.ubBytes, 248 * 1024);
    const uint64_t rows = input.outputHeight;
    const uint64_t countAlignment = input.inputHeight == 112 && input.inputWidth == 112 ? 64 : 256;
    const uint64_t horizontal = AlignUp((2 * rows + 1) * input.outputWidth, countAlignment);
    const uint64_t outputs = AlignUp(rows * input.outputWidth, countAlignment);
    // Complete planes keep the padded top row and aligned tail invariant.
    // Output and candidate aliases must fit strictly inside the real-input
    // interval, which DMA overwrites on the next use of this slot.
    const uint64_t aliasBytes = outputs * (input.dtypeBytes + 8);
    const uint64_t inputBytes = input.inputHeight * input.inputWidth * input.dtypeBytes;
    const uint64_t constantOrZeroBytes = std::max<uint64_t>(8192, horizontal * 4);
    const uint64_t bytes = horizontal * (6 * input.dtypeBytes + 8) + outputs * 4 + 2 * AlignUp(horizontal / 8, 32) +
                           constantOrZeroBytes;
    if (bytes > available || aliasBytes > inputBytes || 2 * horizontal * input.dtypeBytes / 256 > 255 || rows > 65535) {
        return 0;
    }
    return rows;
}

ge::graphStatus MaxPool2dWithMaskTilingFunc(gert::TilingContext* context)
{
    TilingInputs input{};
    if (ParseInputs(context, input) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t totalSlices = 0;
    uint64_t validBytes = 0;
    uint64_t containerBytes = 0;
    if (CalculateContainerSizes(context, input, totalSlices, validBytes, containerBytes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t inputWidthAligned = 0;
    uint64_t outputRowsPerTile = 0;
    uint64_t columnBufferElements = 0;
    uint64_t vectorBufferElements = 0;
    VectorTiling vectorTiling{};
    bool useWindowFallback = false;
    if (input.socVersion != platform_ascendc::SocVersion::ASCEND310P &&
        CalculateVectorTiling(context, input, totalSlices, vectorTiling, useWindowFallback) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t tailBytes = containerBytes - validBytes;
    const uint64_t zeroCoreTarget = std::max<uint64_t>(
        1, (tailBytes + ZERO_BYTES_PER_CORE_TARGET - 1) / ZERO_BYTES_PER_CORE_TARGET);
    uint64_t usedCoreNum = std::min(input.coreNum, std::max({totalSlices, zeroCoreTarget, vectorTiling.coreNums}));
    if (!useWindowFallback && input.socVersion != platform_ascendc::SocVersion::ASCEND310P &&
        input.dtype == ge::DT_BF16) {
        // The vector tiling already partitions every NC slice. Additional
        // cores are useful only when they shorten the mandatory tail stores.
        usedCoreNum = std::min(input.coreNum, std::max(zeroCoreTarget, vectorTiling.coreNums));
    }
    if (useWindowFallback) {
        const uint64_t outputs = validBytes / sizeof(int32_t);
        // Add clear-only cores when pooling occupies less than half the device.
        // Once output parallelism is high, extra clear-only cores can contend
        // with pooling stores without shortening the critical path.
        const uint64_t windowZeroCores = outputs * 2 < input.coreNum ? (tailBytes + 16383) / 16384 : zeroCoreTarget;
        // Large windows provide substantial work per output. Keep output tasks
        // independent and also provide cores for the mandatory index-tail stores.
        usedCoreNum = std::min(input.coreNum, std::max({outputs, zeroCoreTarget, windowZeroCores}));
    }
    if (input.socVersion != platform_ascendc::SocVersion::ASCEND310P && input.dtype == ge::DT_BF16) {
        usedCoreNum = useWindowFallback ? std::min(input.coreNum, validBytes / sizeof(int32_t)) : vectorTiling.coreNums;
    }
    uint64_t slicesPerCore = (totalSlices + usedCoreNum - 1) / usedCoreNum;
    if (input.socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        if (tailBytes != 0 && validBytes % SCALAR_CACHE_LINE_BYTES != 0) {
            // Scalar writes from different M200 cores must not share a cache line with the valid-index region.
            usedCoreNum = 1;
            slicesPerCore = totalSlices;
        } else {
            const uint64_t outputElementsPerSlice = validBytes / sizeof(int32_t) / totalSlices;
            const uint64_t valueBytesPerSlice = outputElementsPerSlice * input.dtypeBytes;
            const uint64_t indexBytesPerSlice = outputElementsPerSlice * sizeof(int32_t);
            const uint64_t valueSliceGroup = SCALAR_CACHE_LINE_BYTES /
                                             std::gcd(valueBytesPerSlice, SCALAR_CACHE_LINE_BYTES);
            const uint64_t indexSliceGroup = SCALAR_CACHE_LINE_BYTES /
                                             std::gcd(indexBytesPerSlice, SCALAR_CACHE_LINE_BYTES);
            const uint64_t alignedSliceGroup = std::max(valueSliceGroup, indexSliceGroup);
            slicesPerCore = std::min(totalSlices, AlignUp(slicesPerCore, alignedSliceGroup));
        }
    }
    uint64_t zeroBytesPerCore = (tailBytes + usedCoreNum - 1) / usedCoreNum;
    if (input.socVersion == platform_ascendc::SocVersion::ASCEND310P && usedCoreNum > 1) {
        zeroBytesPerCore = AlignUp(zeroBytesPerCore, SCALAR_CACHE_LINE_BYTES);
    }

    auto* packet = context->GetTilingData<MaxPool2dWithMaskTilingPacket>();
    OP_CHECK_NULL_WITH_CONTEXT(context, packet);
    auto* tiling = &packet->common;
    tiling->windowOutputTile = 1;
    tiling->windowTailBoundaryCore = 0;
    tiling->windowPrefetch = 0;
    bool largeWindowPrefetch = false;
    bool wideWindowPrefetch = false;
    bool cacheInputPlane = false;
    bool directSingleWindow = false;
    if (useWindowFallback) {
        const uint64_t target = validBytes / sizeof(int32_t) / (usedCoreNum * 2);
        while (tiling->windowOutputTile < 8 && tiling->windowOutputTile * 2 <= target) {
            tiling->windowOutputTile *= 2;
        }
        const uint64_t windowRows = std::min<uint64_t>(input.inputHeight, input.kernel.h);
        const uint64_t windowPitch = AlignUp(std::min<uint64_t>(input.inputWidth, input.kernel.w), 16);
        const uint64_t windowElements = SaturatingMultiply(windowRows, windowPitch);
        // Prefetch adds scalar planning. Reserve it for windows spanning both
        // groups of the hierarchical reduction, where there is more vector work to overlap.
        largeWindowPrefetch = input.dtype == ge::DT_BF16 && windowElements > MAX_POOL2D_WINDOW_INPUT_BYTES / 2 &&
                              windowElements <= MAX_POOL2D_WINDOW_WIDE_INPUT_BYTES / 2;
        wideWindowPrefetch = largeWindowPrefetch && windowElements > MAX_POOL2D_WINDOW_LARGE_INPUT_BYTES / 2;
        tiling->windowPrefetch = largeWindowPrefetch || (input.dtypeBytes == 2 && tiling->windowOutputTile > 1 &&
                                                         windowElements > MAX_POOL2D_WINDOW_INPUT_BYTES / 4 &&
                                                         windowElements <= MAX_POOL2D_WINDOW_INPUT_BYTES / 2);
        const uint64_t outputPlane = static_cast<uint64_t>(input.outputHeight) * input.outputWidth;
        cacheInputPlane = CanCacheInputPlane(input, totalSlices);
        // Direct buffers need one output per execution core, not one per NC.
        // Preserve reuse of identical clipped windows and cached input planes.
        const bool repeatedWindows = HasEquivalentWindows(input.inputHeight, input.kernel.h, input.stride.h,
                                                          input.padding.h, input.outputHeight) ||
                                     HasEquivalentWindows(input.inputWidth, input.kernel.w, input.stride.w,
                                                          input.padding.w, input.outputWidth);
        directSingleWindow = input.dtype == ge::DT_BF16 && validBytes / sizeof(int32_t) <= input.coreNum &&
                             windowElements <= MAX_POOL2D_WINDOW_INPUT_BYTES / 2 && !cacheInputPlane &&
                             !repeatedWindows;
        if (cacheInputPlane) {
            tiling->windowOutputTile = outputPlane;
            tiling->windowPrefetch = 0;
            largeWindowPrefetch = false;
            wideWindowPrefetch = false;
        }
        tiling->windowTailBoundaryCore = cacheInputPlane ? (totalSlices - 1) % usedCoreNum :
                                                           (validBytes / sizeof(int32_t) - 1) /
                                                               tiling->windowOutputTile % usedCoreNum;
    }
    tiling->m200MapRows = 0;
    tiling->m200SpatialRows = CalculateM200SpatialRows(input);
    tiling->m200SpatialBatch = CalculateM200SpatialBatch(input, tiling->m200SpatialRows, slicesPerCore);
    const uint64_t ioSpatialRows = CalculateM200IoPipelinedSpatialRows(input, tiling->m200SpatialRows,
                                                                       tiling->m200SpatialBatch, slicesPerCore);
    if (ioSpatialRows != 0) {
        tiling->m200SpatialRows = ioSpatialRows;
    }
    // Preserve the complete row-load admission before specializing its input
    // lifetimes. The two exact tiles need no rounded input/output tails.
    const uint64_t halfInputCapacity = 56 * 112;
    const uint64_t halfInputBytes = 40 * halfInputCapacity + AlignUp(halfInputCapacity / 8, 32);
    const bool halfInput = ioSpatialRows != 0 && input.dtype == ge::DT_FLOAT && input.inputHeight == 224 &&
                           input.inputWidth == 224 && input.outputHeight == 112 && input.outputWidth == 112 &&
                           halfInputBytes <= std::min<uint64_t>(input.ubBytes, 248 * 1024);
    if (halfInput) {
        tiling->m200SpatialRows = 56;
    }
    const uint64_t spatialSeparableRows = CalculateM200SpatialSeparableRows(input, totalSlices);
    if (spatialSeparableRows != 0) {
        tiling->m200SpatialRows = spatialSeparableRows;
        tiling->m200SpatialBatch = 1;
    }
    tiling->m200NcRows = tiling->m200SpatialRows != 0 ? 0 : CalculateM200TransposeRows(input, totalSlices, 16, true);
    const uint64_t inputPlane = SaturatingMultiply(input.inputHeight, input.inputWidth);
    const uint64_t outputPlane = SaturatingMultiply(input.outputHeight, input.outputWidth);
    const bool packedTranspose = inputPlane * input.dtypeBytes % ALIGN_BYTES != 0 ||
                                 outputPlane * input.dtypeBytes % ALIGN_BYTES != 0;
    // Spatial tiles can provide core parallelism even with fewer than 64 NC slices.
    // A larger UB-fitting tile must not disable this path when smaller aligned
    // tiles can still provide one task per core. Packed planes cannot be split.
    if (totalSlices < 64 && tiling->m200NcRows != 0) {
        const uint64_t groups = totalSlices / 16;
        const uint64_t requiredTiles = (usedCoreNum + groups - 1) / groups;
        if (!packedTranspose && requiredTiles > 1) {
            const uint64_t rowGroup = ALIGN_BYTES /
                                      std::gcd(static_cast<uint64_t>(input.outputWidth) * input.dtypeBytes,
                                               ALIGN_BYTES);
            // ceil(outputHeight / rows) >= requiredTiles iff
            // rows <= floor((outputHeight - 1) / (requiredTiles - 1)).
            const uint64_t maxRows = (input.outputHeight - 1) / (requiredTiles - 1);
            tiling->m200NcRows = std::min<uint64_t>(tiling->m200NcRows, maxRows / rowGroup * rowGroup);
        }
    }
    if (totalSlices < 64 && tiling->m200NcRows != 0) {
        const uint64_t tiles = (input.outputHeight + tiling->m200NcRows - 1) / tiling->m200NcRows;
        if (tiles * (totalSlices / 16) < usedCoreNum) {
            tiling->m200NcRows = 0;
        }
    }
    tiling->m200NcFactor = 16;
    // Small packed planes have little work per vector instruction. Batch more NC
    // slices when the complete plane fits in UB and leaves enough independent tasks.
    if (tiling->m200NcRows != 0 && packedTranspose) {
        const uint64_t batchFactors[] = {128, 64};
        for (const uint64_t ncFactor : batchFactors) {
            if (totalSlices % ncFactor != 0 || totalSlices < ncFactor * usedCoreNum) {
                continue;
            }
            const uint64_t batchRows = CalculateM200TransposeRows(input, totalSlices, ncFactor);
            if (batchRows == static_cast<uint64_t>(input.outputHeight)) {
                tiling->m200NcRows = batchRows;
                tiling->m200NcFactor = ncFactor;
                break;
            }
        }
    }
    if (tiling->m200SpatialRows == 0 && tiling->m200NcRows == 0 &&
        input.socVersion == platform_ascendc::SocVersion::ASCEND310P &&
        (input.dtype == ge::DT_FLOAT || input.dtype == ge::DT_FLOAT16) && inputPlane <= (1U << 24) &&
        outputPlane <= (1U << 24) && input.kernel.h <= 256 && input.kernel.w <= 256 && input.stride.h <= 4096 &&
        input.stride.w <= 4096 && inputPlane * input.dtypeBytes % ALIGN_BYTES == 0 &&
        outputPlane * input.dtypeBytes % SCALAR_CACHE_LINE_BYTES == 0) {
        const uint64_t rowGroup = SCALAR_CACHE_LINE_BYTES /
                                  std::gcd(static_cast<uint64_t>(input.outputWidth) * input.dtypeBytes,
                                           SCALAR_CACHE_LINE_BYTES);
        const uint64_t fixedBytes = 24 * 1024;
        // Input/output, coordinate templates, and both byte-offset and index maps.
        for (uint64_t rows = rowGroup; rows <= static_cast<uint64_t>(input.outputHeight); rows += rowGroup) {
            const uint64_t outputs = AlignUp(rows * input.outputWidth, 128);
            const uint64_t inputRows = (rows - 1) * input.stride.h + input.kernel.h;
            const uint64_t bytes = fixedBytes + AlignUp(inputRows * input.inputWidth * input.dtypeBytes, 32) +
                                   outputs * (input.dtypeBytes * 2 + 21 + input.kernel.h * input.kernel.w * 8);
            if (bytes > input.ubBytes) {
                break;
            }
            tiling->m200MapRows = rows;
        }
    }
    tiling->batchSize = input.batch;
    tiling->channels = input.channels;
    tiling->inputHeight = input.inputHeight;
    tiling->inputWidth = input.inputWidth;
    tiling->outputHeight = input.outputHeight;
    tiling->outputWidth = input.outputWidth;
    tiling->kernelH = static_cast<int32_t>(input.kernel.h);
    tiling->kernelW = static_cast<int32_t>(input.kernel.w);
    tiling->strideH = static_cast<int32_t>(input.stride.h);
    tiling->strideW = static_cast<int32_t>(input.stride.w);
    tiling->padH = static_cast<int32_t>(input.padding.h);
    tiling->padW = static_cast<int32_t>(input.padding.w);
    tiling->dilationH = static_cast<int32_t>(input.dilation.h);
    tiling->dilationW = static_cast<int32_t>(input.dilation.w);
    tiling->totalSlices = static_cast<int64_t>(totalSlices);
    tiling->slicesPerCore = static_cast<int64_t>(slicesPerCore);
    tiling->columnBufferElements = static_cast<int64_t>(columnBufferElements);
    tiling->vectorBufferElements = static_cast<int64_t>(vectorBufferElements);
    tiling->outputRowsPerTile = static_cast<int64_t>(outputRowsPerTile);
    tiling->inputWidthAligned = static_cast<int64_t>(inputWidthAligned);
    tiling->outputElementsPerTileAligned = static_cast<int64_t>(
        AlignUp(outputRowsPerTile * static_cast<uint64_t>(input.outputWidth), ALIGN_BYTES / input.dtypeBytes));
    tiling->validIndexBytes = static_cast<int64_t>(validBytes);
    tiling->containerBytes = static_cast<int64_t>(containerBytes);
    // Preserve the validated non-BF16 vector and M200 implementations. Only
    // A2 BF16 and independent window paths omit index-container tail stores.
    const bool clearIndexTail = input.socVersion == platform_ascendc::SocVersion::ASCEND310P ||
                                (input.dtype != ge::DT_BF16 && !useWindowFallback);
    tiling->zeroBytesPerCore = clearIndexTail ? static_cast<int64_t>(zeroBytesPerCore) : 0;
    tiling->zeroBufferBytes = clearIndexTail ? ZERO_BUFFER_BYTES : 0;

    tiling->nc = totalSlices;
    tiling->dx = 1;
    tiling->hx = input.inputHeight;
    tiling->wx = input.inputWidth;
    tiling->kd = 1;
    tiling->kh = input.kernel.h;
    tiling->kw = input.kernel.w;
    tiling->sd = 1;
    tiling->sh = input.stride.h;
    tiling->sw = input.stride.w;
    tiling->pf = 0;
    tiling->pb = 0;
    tiling->pl = input.padding.w;
    tiling->pr = vectorTiling.padRight;
    tiling->pt = input.padding.h;
    tiling->pd = vectorTiling.padBottom;
    tiling->dy = 1;
    tiling->hy = input.outputHeight;
    tiling->wy = input.outputWidth;
    tiling->ncFactor = vectorTiling.ncFactor;
    tiling->ncTail = vectorTiling.ncTail;
    tiling->ncOuter = vectorTiling.ncOuter;
    tiling->dyFactor = 1;
    tiling->dyTail = 1;
    tiling->dyOuter = 1;
    tiling->hyFactor = vectorTiling.hyFactor;
    tiling->hyTail = vectorTiling.hyTail;
    tiling->hyOuter = vectorTiling.hyOuter;
    tiling->wyFactor = vectorTiling.wyFactor;
    tiling->wyTail = vectorTiling.wyTail;
    tiling->wyOuter = vectorTiling.wyOuter;
    tiling->blockFactor = vectorTiling.blockFactor;
    tiling->blockTail = vectorTiling.blockTail;
    tiling->totalIdx = vectorTiling.totalIdx;
    tiling->coreNums = vectorTiling.coreNums;
    tiling->inputBufferSize = vectorTiling.buffers.input;
    tiling->outputMaxBufferSize = vectorTiling.buffers.output;
    tiling->outputIndiceBufferSize = vectorTiling.buffers.indices;
    tiling->indiceTempBufferSize = vectorTiling.buffers.indicesTemporary;
    tiling->maskBufferSize = vectorTiling.buffers.mask;

    packet->window = {tiling->inputHeight,
                      tiling->inputWidth,
                      tiling->outputHeight,
                      tiling->outputWidth,
                      tiling->validIndexBytes,
                      tiling->containerBytes,
                      tiling->zeroBytesPerCore,
                      tiling->zeroBufferBytes,
                      tiling->nc,
                      tiling->kh,
                      tiling->kw,
                      tiling->sh,
                      tiling->sw,
                      tiling->pl,
                      tiling->pt,
                      tiling->windowOutputTile,
                      tiling->windowTailBoundaryCore,
                      tiling->windowPrefetch};

    // Spatial BF16 lanes avoid NC transposes for small windows and bounded planes.
    uint64_t spatialInputPitch = 16;
    uint64_t scanInputPitch = 16;
    // Stop once capacity is already impossible; accepted pitches contain at
    // least one complete scan segment even for narrow ceilMode inputs.
    while (scanInputPitch <= static_cast<uint64_t>(input.kernel.w) / 2 && scanInputPitch <= 8192) {
        scanInputPitch *= 2;
    }
    while (scanInputPitch < static_cast<uint64_t>(input.inputWidth + input.padding.w) && scanInputPitch <= 8192) {
        scanInputPitch *= 2;
    }
    uint64_t spatialOutputPitch = 16;
    // Output dependence bounds, excluding unconsumed trailing input columns.
    const uint64_t spatialWidth = (input.outputWidth - 1) * input.stride.w + input.kernel.w;
    while (spatialInputPitch < spatialWidth && spatialInputPitch <= 8192) {
        spatialInputPitch *= 2;
    }
    const bool compactSpatial = input.kernel.h == 2 && input.kernel.w == 2 && input.stride.h == 2 &&
                                input.stride.w == 2 && input.padding.h == 0 && input.padding.w == 0 &&
                                AlignUp(spatialWidth, 16) < spatialInputPitch;
    if (compactSpatial) {
        spatialInputPitch = AlignUp(spatialWidth, 16);
    }
    while (spatialOutputPitch < static_cast<uint64_t>(input.outputWidth) && spatialOutputPitch <= 2048) {
        spatialOutputPitch *= 2;
    }
    const uint64_t spatialRows = (input.outputHeight - 1) * input.stride.h + input.kernel.h;
    const uint64_t scanRows = std::max<uint64_t>(input.inputHeight + input.padding.h, spatialRows);
    const uint64_t spatialArea = SaturatingMultiply(input.kernel.h, input.kernel.w);
    // With few channels, an NC transpose leaves most lanes unused. Spatial
    // lanes also admit rectangular windows and strides wider than the window.
    // Ordinary spatial Gather repeats geometry and conversion for every plane.
    // Once channels fill the device, retain the NC-vectorized implementation.
    // Compact 2x2 uses a separate representation and keeps its existing scope.
    const bool spatialWorkload = spatialArea >= 2 &&
                                 ((spatialArea <= 4 && (totalSlices <= input.coreNum || compactSpatial)) ||
                                  (spatialArea <= 16 && totalSlices <= 8));
    const bool spatialScanWorkload = input.socVersion != platform_ascendc::SocVersion::ASCEND310P &&
                                     input.dtype == ge::DT_BF16 && spatialArea > 16 && input.kernel.w >= 4 &&
                                     input.stride.w < input.kernel.w && input.stride.h <= input.kernel.h &&
                                     SaturatingMultiply(scanInputPitch, scanRows) <= 8192 &&
                                     SaturatingMultiply(spatialOutputPitch, input.outputHeight) <= 2048 &&
                                     PreferSpatialScan(input, totalSlices, scanInputPitch, spatialOutputPitch,
                                                       vectorTiling, useWindowFallback);
    // Few channels and several outputs per channel otherwise share one NC tile.
    // Assign a spatial output tile to each channel; retain independent-window priority.
    const bool smallSpatialOutputs = !useWindowFallback && totalSlices <= 8 && outputPlane >= 16 && outputPlane < 128 &&
                                     vectorTiling.coreNums < totalSlices;
    const bool spatialSmallWindow = input.socVersion != platform_ascendc::SocVersion::ASCEND310P &&
                                    input.dtype == ge::DT_BF16 && (spatialWorkload || spatialScanWorkload) &&
                                    (outputPlane >= 128 || spatialScanWorkload || smallSpatialOutputs) &&
                                    SaturatingMultiply(spatialScanWorkload ? scanInputPitch : spatialInputPitch,
                                                       spatialScanWorkload ? scanRows : spatialRows) <= 8192 &&
                                    SaturatingMultiply(spatialOutputPitch, input.outputHeight) <= 2048 &&
                                    input.ubBytes >=
                                        (8192 * 14 + 2048 * 26 + 2048 + 256 + 16 * 1024 + UB_RESERVE_BYTES);
    if (spatialSmallWindow) {
        // Keep interior output boundaries aligned for both BF16 and int32 DMA.
        const uint64_t tileRows = SpatialTileRows(input, totalSlices, spatialOutputPitch);
        packet->window.windowOutputTile = tileRows;
        const uint64_t rowTiles = (input.outputHeight + tileRows - 1) / tileRows;
        const uint64_t tasks = totalSlices * rowTiles;
        usedCoreNum = std::min(input.coreNum, tasks);
        packet->window.windowTailBoundaryCore = (tasks - 1) % usedCoreNum;
    }

    const bool equivalentWindows = useWindowFallback && input.dtype == ge::DT_BF16 && !largeWindowPrefetch &&
                                   !cacheInputPlane && !directSingleWindow && !tiling->windowPrefetch &&
                                   (HasEquivalentWindows(input.inputHeight, input.kernel.h, input.stride.h,
                                                         input.padding.h, input.outputHeight) ||
                                    HasEquivalentWindows(input.inputWidth, input.kernel.w, input.stride.w,
                                                         input.padding.w, input.outputWidth));

    context->SetBlockDim(usedCoreNum);
    if (input.socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        // Keep this mapping in agreement with the M200 kernel entry point.
        uint64_t key = tiling->m200MapRows != 0 ? 11 : 10;
        // The map path has cache-line-aligned row tiles. When some launched
        // cores have no channel, distribute these independent tiles as well.
        if (tiling->m200MapRows != 0 && totalSlices < usedCoreNum && tiling->m200MapRows < input.outputHeight) {
            key = 45;
            if (input.dtype == ge::DT_FLOAT16 && (input.kernel.h == 3 || input.kernel.h == 7) &&
                input.kernel.w == input.kernel.h) {
                const uint64_t outputs = AlignUp(tiling->m200MapRows * input.outputWidth, 128);
                const uint64_t rows = (tiling->m200MapRows - 1) * input.stride.h + input.kernel.h;
                const uint64_t capacity = AlignUp((AlignUp(rows * input.inputWidth * 2, 32) + 64) / 2, 128);
                const uint64_t bytes = 24 * 1024 + capacity * 18 +
                                       outputs * (26 + input.kernel.h * input.kernel.w * 4) + AlignUp(capacity / 8, 32);
                if (capacity <= 8192 && outputs <= capacity && bytes <= input.ubBytes) {
                    key = 49;
                }
            }
        }
        if (tiling->m200NcRows != 0) {
            key = tiling->m200NcFactor == 128 ? 14 : (tiling->m200NcFactor == 64 ? 13 : 12);
            if (tiling->m200NcFactor == 16 && CanReuseM200SeparableBuffers(input, tiling->m200NcRows)) {
                key = 16;
            }
            if (tiling->m200NcFactor == 16 && CanReuseM200SeparableBuffers(input, tiling->m200NcRows, true)) {
                key = 24;
            }
            if (tiling->m200NcFactor == 16 &&
                CanCacheM200HorizontalRow(input, tiling->m200NcRows, totalSlices, usedCoreNum)) {
                key = 28;
                // Only relative window offsets use half. Every integer through
                // 2048 is exact; the absolute plane index remains int32.
                if ((input.kernel.h - 1) * input.inputWidth + input.kernel.w - 1 <= 2048) {
                    key = 30;
                }
            }
            if (tiling->m200NcFactor == 16 && !packedTranspose && input.stride.h == 1 && input.stride.w == 1) {
                key = CanSeparateM200FlatWindow(input, tiling->m200NcRows) ? 48 : 27;
            }
        }
        if ((key == 13 || key == 14) && input.inputHeight == 7 && input.inputWidth == 7 && input.outputHeight == 4 &&
            input.outputWidth == 4 && input.kernel.h == 7 && input.kernel.w == 7 && input.stride.h == 2 &&
            input.stride.w == 2 && input.padding.h == 3 && input.padding.w == 3 &&
            181 * tiling->m200NcFactor * input.dtypeBytes + 172 * tiling->m200NcFactor * 4 + tiling->m200NcFactor +
                    8192 <=
                std::min<uint64_t>(input.ubBytes, 248 * 1024)) {
            key = tiling->m200NcFactor == 64 ? 46 : 47;
        }
        if (tiling->m200SpatialRows != 0) {
            key = CanPipelineM200SpatialInput(input, tiling->m200SpatialRows, tiling->m200SpatialBatch, slicesPerCore) ?
                      23 :
                      15;
            if (CanPipelineM200SpatialInput(input, tiling->m200SpatialRows, tiling->m200SpatialBatch, slicesPerCore,
                                            true)) {
                key = 25;
            }
            if (ioSpatialRows != 0) {
                key = 38;
            }
            if (halfInput) {
                key = 44;
            }
        }
        if (spatialSeparableRows != 0) {
            key = input.inputHeight == 112 && input.inputWidth == 112 ? 40 : 37;
        }
        if (key == 10) {
            if (CalculateUbTiling(context, input, inputWidthAligned, outputRowsPerTile, columnBufferElements,
                                  vectorBufferElements) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            if (outputRowsPerTile == 0) {
                key = 50;
            }
            tiling->inputWidthAligned = inputWidthAligned;
            tiling->outputRowsPerTile = outputRowsPerTile;
            tiling->columnBufferElements = columnBufferElements;
            tiling->vectorBufferElements = vectorBufferElements;
            tiling->outputElementsPerTileAligned = AlignUp(outputRowsPerTile * input.outputWidth,
                                                           ALIGN_BYTES / input.dtypeBytes);
        }
        context->SetTilingKey(key);
    } else {
        context->SetTilingKey(
            spatialSmallWindow ?
                (spatialScanWorkload ? 8 : (compactSpatial ? 10 : 6)) :
            useWindowFallback ?
                (directSingleWindow ?
                     4 :
                     (cacheInputPlane ?
                          5 :
                          (wideWindowPrefetch ? 7 : (largeWindowPrefetch ? 3 : (equivalentWindows ? 9 : 2))))) :
                SelectVectorTilingKey(input, vectorTiling));
    }

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

struct MaxPool2dWithMaskCompileInfo {};

ge::graphStatus TilingParseMaxPool2dWithMask([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_OPTILING(MaxPool2dWithMask)
    .Tiling(MaxPool2dWithMaskTilingFunc)
    .TilingParse<MaxPool2dWithMaskCompileInfo>(TilingParseMaxPool2dWithMask);

} // namespace optiling
