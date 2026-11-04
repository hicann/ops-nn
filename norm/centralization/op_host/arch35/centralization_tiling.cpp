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
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "centralization_tiling.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/centralization_tiling_data.h"

using namespace ge;
namespace optiling {
using Ops::Base::CeilDiv;

namespace {
constexpr int64_t kMaxRank = 8;
constexpr int64_t kLargeReduceThreshold = 8192;
constexpr int64_t kFastReduceLimit = 16384;
constexpr int64_t kKernelTileElements = 1024;
constexpr int64_t kGenericContiguousTileElements = 256;
constexpr int64_t kRegbaseVectorElements = 256 / sizeof(float);
constexpr int64_t kWorkspaceAlignElements = 32 / sizeof(float);
constexpr int64_t kMaxReduceSegmentOffsets = 128;
constexpr int64_t kMaxIrregularScalarKeepTile = 16;
constexpr int64_t kSmallIrregularVectorTileElements = 64;
constexpr int64_t kIrregularKeepTileTaskPressure = 4;
constexpr int64_t kSmallContiguousMaxWorkingSetElements = 1024;
constexpr int64_t kMaxShapeDim = std::numeric_limits<int64_t>::max();
constexpr int64_t kMaxShapeElements = std::numeric_limits<int64_t>::max();
constexpr size_t kSmallContiguousMaxWorkingSetBytes = 16UL * 1024UL;
constexpr size_t kSystemWorkspaceBytes = 16UL * 1024UL * 1024UL;
// UB 预留：TPipe 元数据、事件与对齐开销不参与 buffer 均分。
constexpr size_t kUbReservedBytes = 8UL * 1024UL;
constexpr size_t kUbAlignBytes = 32UL;
constexpr uint32_t kEmptyKey = 8000;
constexpr uint32_t kFastKey = 7000;
constexpr uint32_t kGenericKey = 0;
constexpr uint32_t kLargeKey = 7020;
constexpr uint32_t kLargeTrailingKey = 7030;

static ge::graphStatus ReportInvalidShape(gert::TilingContext* context, const char* field, const std::string& value,
                                          const char* reason)
{
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), field, value.c_str(), reason);
    return GRAPH_FAILED;
}

static ge::graphStatus CheckedMul(gert::TilingContext* context, const char* label, int64_t lhs, int64_t rhs,
                                  int64_t limit, int64_t& result)
{
    if (lhs < 0 || rhs < 0) {
        return ReportInvalidShape(context, label, std::to_string(lhs) + "*" + std::to_string(rhs),
                                  "shape-derived multiplication requires non-negative factors");
    }
    if (lhs != 0 && rhs > limit / lhs) {
        return ReportInvalidShape(context, label, std::to_string(lhs) + "*" + std::to_string(rhs),
                                  "shape-derived product exceeds the supported int64 range");
    }
    result = lhs * rhs;
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckedProductAppend(gert::TilingContext* context, const char* label, int64_t factor,
                                            int64_t& product)
{
    int64_t next = 0;
    if (CheckedMul(context, label, product, factor, kMaxShapeElements, next) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    product = next;
    return GRAPH_SUCCESS;
}

static ge::graphStatus ValidateRuntimeDim(gert::TilingContext* context, int64_t dim)
{
    if (dim < 0) {
        return ReportInvalidShape(context, "x", "dynamic", "runtime tiling requires concrete dimensions");
    }
    if (dim > kMaxShapeDim) {
        return ReportInvalidShape(context, "x", std::to_string(dim),
                                  "each runtime dimension must be in [0, 9223372036854775807]");
    }
    return GRAPH_SUCCESS;
}

static int64_t AlignToVectorElements(int64_t elements)
{
    return CeilDiv(elements, kRegbaseVectorElements) * kRegbaseVectorElements;
}

static size_t DataTypeBytes(ge::DataType dtype) { return dtype == DT_FLOAT ? sizeof(float) : sizeof(uint16_t); }

static ge::graphStatus SetBufferBytes(gert::TilingContext* context, const char* label, int64_t elements,
                                      size_t elementBytes, uint32_t& bytes)
{
    constexpr int64_t kByteLimit = std::numeric_limits<uint32_t>::max();
    const int64_t elementBytes64 = static_cast<int64_t>(elementBytes);
    if (elements < 0 || elementBytes64 <= 0 || elements > kByteLimit / elementBytes64) {
        const std::string product = std::to_string(elements) + "*" + std::to_string(elementBytes);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), label, product.c_str(),
                                              "buffer bytes are invalid or exceed the uint32 range");
        return GRAPH_FAILED;
    }
    bytes = static_cast<uint32_t>(elements * elementBytes64);
    return GRAPH_SUCCESS;
}

// 每条路径在 UB 上同时存活的 buffer 数量，必须与 kernel Init 里的 InitBuffer 一一对应：
//   Generic: in/out 各 depth 份 + 1 份 sum；
//   Fast:    in/out 各 bufNum 份；
//   LargeTrailing: in/out 各 2 份 + calc + mean + sum。
static int64_t GenericLiveBufferCount(const CentralizationTilingData* tiling)
{
    const bool singleDepth = tiling->smallContiguous != 0 || tiling->largeContiguousSmallInner != 0;
    return (singleDepth ? 1 : 2) * 2 + 1;
}

static size_t UbUsableBytes(uint64_t ubSize)
{
    return ubSize > kUbReservedBytes ? static_cast<size_t>(ubSize - kUbReservedBytes) : 0UL;
}

static size_t AlignDownBytes(size_t bytes) { return bytes / kUbAlignBytes * kUbAlignBytes; }

// 均值累加槽数：sum 节点可用 budgetBytes，按 sumElements * factor 的每槽占用反算。
// factor=1 仅存 sum，factor=2 同时存 sum 与 correction。
static int64_t ChooseAccSlots(int64_t count, int64_t sumElements, size_t budgetBytes, int64_t factor)
{
    if (count <= 1 || sumElements <= 0 || factor <= 0) {
        return 1;
    }
    const size_t perSlotBytes = static_cast<size_t>(sumElements) * sizeof(float) * static_cast<size_t>(factor);
    if (perSlotBytes == 0) {
        return 1;
    }
    int64_t slots = 1;
    while (slots < 16 && (slots * 2) <= count && static_cast<size_t>(slots * 2) * perSlotBytes <= budgetBytes) {
        slots *= 2;
    }
    return slots;
}

static const char* GenericPathName(const CentralizationTilingData* tiling)
{
    if (tiling->smallContiguous != 0) {
        return tiling->smallContiguousMode == 2 ? "small2" : "small1";
    }
    if (tiling->largeContiguousSmallInner != 0) {
        return "lcsi";
    }
    if (tiling->contiguousGeneric != 0) {
        return "contiguous";
    }
    if (tiling->irregularVectorizable != 0) {
        return "irregular";
    }
    if (tiling->largePath != 0) {
        return "large";
    }
    return "scalar";
}

static const char* PathName(uint32_t key, const CentralizationTilingData* tiling)
{
    switch (key) {
        case kEmptyKey:
            return "empty";
        case kFastKey:
            return "fast";
        case kLargeTrailingKey:
            return "largeTrailing";
        case kLargeKey:
            return "large";
        case kGenericKey:
            return GenericPathName(tiling);
        default:
            return "unknown";
    }
}

// 按“峰值并行存活 buffer 数 + UB 总量”给每个存活节点均分可用空间：
//   perBufferBytes = (UB - 预留 - 固定小 buffer) / 存活节点数
// 再按 perBufferBytes 反算并 clamp 各 tile，保证 sum(存活 buffer) <= UB。
static ge::graphStatus ConfigureBufferBytes(gert::TilingContext* context, CentralizationTilingData* tiling,
                                            ge::DataType dtype, uint64_t ubSize, uint32_t key)
{
    const size_t elementBytes = DataTypeBytes(dtype);
    if (key == kEmptyKey) {
        tiling->genericInputBufferBytes = 0;
        tiling->genericOutputBufferBytes = 0;
        tiling->genericSumBufferBytes = 0;
        return GRAPH_SUCCESS;
    }

    const size_t usableBytes = UbUsableBytes(ubSize);
    int64_t liveBuffers = 0;
    size_t fixedBytes = 0;
    if (key == kLargeTrailingKey) {
        liveBuffers = 6;  // in x2 + out x2 + calc + mean + sum
        fixedBytes = 128; // sumBuf_ 固定 128B
    } else if (key == kFastKey) {
        liveBuffers = 2 * (tiling->bufNum > 0 ? tiling->bufNum : 2);
    } else {
        liveBuffers = GenericLiveBufferCount(tiling);
    }
    if (liveBuffers <= 0 || usableBytes <= fixedBytes) {
        const std::string reason = "invalid UB budget for path " + std::string(PathName(key, tiling)) +
                                   ": liveBuffers=" + std::to_string(liveBuffers) +
                                   ", usableBytes=" + std::to_string(usableBytes) +
                                   ", fixedBytes=" + std::to_string(fixedBytes);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "UB", std::to_string(ubSize).c_str(),
                                              reason.c_str());
        return GRAPH_FAILED;
    }
    const size_t perBufferBytes = AlignDownBytes((usableBytes - fixedBytes) / static_cast<size_t>(liveBuffers));
    if (perBufferBytes == 0) {
        const std::string reason = "no usable UB space for path " + std::string(PathName(key, tiling)) +
                                   ": usableBytes=" + std::to_string(usableBytes) +
                                   ", fixedBytes=" + std::to_string(fixedBytes) +
                                   ", liveBuffers=" + std::to_string(liveBuffers);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "UB", std::to_string(ubSize).c_str(),
                                              reason.c_str());
        return GRAPH_FAILED;
    }

    if (key == kFastKey) {
        // Fast 的 buffer 字节由 kernel 按 ubFactor/rowsPerLoop 计算，此处只做一致性保护。
        const int64_t maxCols = static_cast<int64_t>(perBufferBytes / elementBytes);
        tiling->ubFactor = std::max<int64_t>(1, std::min<int64_t>(tiling->ubFactor, maxCols));
        return GRAPH_SUCCESS;
    }

    if (key == kLargeTrailingKey) {
        if (SetBufferBytes(context, "large trailing input buffer bytes",
                           static_cast<int64_t>(perBufferBytes / elementBytes), elementBytes,
                           tiling->largeTrailingInputBufferBytes) != GRAPH_SUCCESS ||
            SetBufferBytes(context, "large trailing output buffer bytes",
                           static_cast<int64_t>(perBufferBytes / elementBytes), elementBytes,
                           tiling->largeTrailingOutputBufferBytes) != GRAPH_SUCCESS ||
            SetBufferBytes(context, "large trailing calc buffer bytes",
                           static_cast<int64_t>(perBufferBytes / sizeof(float)), sizeof(float),
                           tiling->largeTrailingCalcBufferBytes) != GRAPH_SUCCESS ||
            SetBufferBytes(context, "large trailing mean buffer bytes",
                           static_cast<int64_t>(perBufferBytes / sizeof(float)), sizeof(float),
                           tiling->largeTrailingMeanBufferBytes) != GRAPH_SUCCESS) {
            const std::string reason = "failed to set largeTrailing buffer bytes for path " +
                                       std::string(PathName(key, tiling));
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "UB", std::to_string(perBufferBytes).c_str(),
                                                  reason.c_str());
            return GRAPH_FAILED;
        }
        // LargeTrailing 的累加槽是标量数组，不占 UB，按 reduce 链长取 2 的幂。
        tiling->numAccSlots = ChooseAccSlots(tiling->reduceCount, 1, perBufferBytes, 1);
        return GRAPH_SUCCESS;
    }

    const bool smallContiguous = tiling->smallContiguous != 0;
    const bool largeContiguousSmallInner = tiling->largeContiguousSmallInner != 0;
    int64_t inputElements = 0;
    int64_t sumElements = 0;
    int64_t sumFactor = 1;
    bool singleAccSlot = false;
    if (smallContiguous) {
        const int64_t alignedMean = AlignToVectorElements(tiling->smallContiguousMeanElements);
        if (tiling->smallContiguousMode == 2) {
            inputElements = tiling->reduceCount * alignedMean;
            sumElements = alignedMean;
            sumFactor = 1; // ProcessSmallWholeBlockTask 只维护 sum
        } else {
            inputElements = AlignToVectorElements(tiling->smallContiguousBlockElements);
            sumElements = alignedMean;
            sumFactor = 1;
            singleAccSlot = true; // ProcessSmallInnerOneTask 使用标量均值，不消费累加槽
        }
    } else if (largeContiguousSmallInner) {
        const int64_t alignedInner = AlignToVectorElements(tiling->contiguousInnerCount);
        const int64_t maxReduceTile = static_cast<int64_t>(perBufferBytes /
                                                           (static_cast<size_t>(alignedInner) * elementBytes));
        tiling->largeContiguousReduceTile = maxReduceTile > 0 ?
                                                std::max<int64_t>(
                                                    1, std::min<int64_t>(tiling->reduceCount, maxReduceTile)) :
                                                1;
        inputElements = tiling->largeContiguousReduceTile * alignedInner;
        sumElements = alignedInner;
        sumFactor = 2; // Kahan: sum + correction
    } else if (tiling->contiguousGeneric != 0) {
        const int64_t maxTile = static_cast<int64_t>(perBufferBytes / elementBytes);
        const int64_t naturalTile = std::max<int64_t>(1, tiling->contiguousInnerTileElements);
        const int64_t tile = std::max<int64_t>(
            1, std::min<int64_t>(naturalTile, std::min<int64_t>(maxTile, kGenericContiguousTileElements)));
        tiling->contiguousInnerTileElements = tile;
        tiling->contiguousInnerTileCount = CeilDiv(tiling->contiguousInnerCount, tile);
        inputElements = tile;
        sumElements = kGenericContiguousTileElements;
        sumFactor = 2; // ProcessContiguousTask 同时维护 sum 与 correction
    } else if (tiling->irregularVectorizable != 0) {
        const int64_t maxTile = static_cast<int64_t>(perBufferBytes / elementBytes);
        const int64_t naturalTile = std::max<int64_t>(1, tiling->irregularVectorTileElements);
        const int64_t tile = std::max<int64_t>(1, std::min<int64_t>(naturalTile, maxTile));
        tiling->irregularVectorTileElements = tile;
        tiling->irregularVectorTileCount = CeilDiv(tiling->irregularVectorInnerCount, tile);
        inputElements = tile;
        sumElements = kGenericContiguousTileElements;
        sumFactor = 1;
    } else {
        // large（7020）与 generic scalar：ProcessBlock 固定按 32B block 搬运
        inputElements = std::max<int64_t>(
            1, std::min<int64_t>(kGenericContiguousTileElements, static_cast<int64_t>(perBufferBytes / elementBytes)));
        sumElements = 1; // ProcessLarge/ProcessOutputBlocks 使用标量槽，不占 sumBuf_
        sumFactor = 1;
    }
    if (static_cast<size_t>(inputElements) * elementBytes > perBufferBytes) {
        const std::string reason = "input tile exceeds per-buffer budget for path " +
                                   std::string(PathName(key, tiling)) +
                                   ": inputElements=" + std::to_string(inputElements) +
                                   ", elementBytes=" + std::to_string(elementBytes) +
                                   ", perBufferBytes=" + std::to_string(perBufferBytes);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "UB", std::to_string(inputElements).c_str(),
                                              reason.c_str());
        return GRAPH_FAILED;
    }
    // dense 路径的 slot 占用 UB；large/largeTrailing 的 slot 是标量，不占 UB。
    if (smallContiguous || largeContiguousSmallInner || tiling->contiguousGeneric != 0 ||
        tiling->irregularVectorizable != 0) {
        tiling->numAccSlots = ChooseAccSlots(tiling->reduceCount, sumElements, perBufferBytes, sumFactor);
    } else {
        tiling->numAccSlots = ChooseAccSlots(tiling->reduceCount, 1, perBufferBytes, 1);
    }
    if (singleAccSlot || largeContiguousSmallInner || tiling->irregularVectorizable != 0) {
        tiling->numAccSlots = 1;
    }
    const int64_t sumBufferElements = sumElements * tiling->numAccSlots * sumFactor;
    if (static_cast<size_t>(sumBufferElements) * sizeof(float) > perBufferBytes) {
        const std::string reason = "sum buffer exceeds per-buffer budget for path " +
                                   std::string(PathName(key, tiling)) + ": sumElements=" + std::to_string(sumElements) +
                                   ", slots=" + std::to_string(tiling->numAccSlots) +
                                   ", factor=" + std::to_string(sumFactor) +
                                   ", perBufferBytes=" + std::to_string(perBufferBytes);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "UB", std::to_string(sumBufferElements).c_str(),
                                              reason.c_str());
        return GRAPH_FAILED;
    }

    if (SetBufferBytes(context, "generic input buffer bytes", static_cast<int64_t>(perBufferBytes / elementBytes),
                       elementBytes, tiling->genericInputBufferBytes) != GRAPH_SUCCESS ||
        SetBufferBytes(context, "generic output buffer bytes", static_cast<int64_t>(perBufferBytes / elementBytes),
                       elementBytes, tiling->genericOutputBufferBytes) != GRAPH_SUCCESS ||
        SetBufferBytes(context, "generic sum buffer bytes", static_cast<int64_t>(perBufferBytes / sizeof(float)),
                       sizeof(float), tiling->genericSumBufferBytes) != GRAPH_SUCCESS) {
        const std::string reason = "failed to set generic buffer bytes for path " + std::string(PathName(key, tiling));
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "UB", std::to_string(perBufferBytes).c_str(),
                                              reason.c_str());
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static int64_t CalcCoresPerRow(int64_t groupCount, int64_t coreNum, int64_t chunkCount)
{
    if (groupCount <= 0) {
        return 1;
    }
    int64_t coresPerRow = coreNum / groupCount;
    if (coresPerRow < 1) {
        coresPerRow = 1;
    }
    if (chunkCount < coresPerRow) {
        coresPerRow = chunkCount;
    }
    if (coresPerRow < 1) {
        coresPerRow = 1;
    }
    return coresPerRow;
}

static ge::graphStatus GetPlatform(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum <= 0 || ubSize == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platform", "invalid",
                                              "AI core count and UB size must be non-zero");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus ResolveAxes(gert::TilingContext* context, int64_t rank, std::vector<int64_t>& axes)
{
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* attr = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    if (attr == nullptr) {
        // 未设置：沿用 proto 默认 {-1}，归约最后一轴。
        axes.push_back(rank - 1);
        return GRAPH_SUCCESS;
    }
    if (attr->GetSize() == 0) {
        // 显式空列表：不归约（canndev DSL 下 reduce_sum(axis=[]) 为 identity）。
        return GRAPH_SUCCESS;
    }
    const auto* data = reinterpret_cast<const int64_t*>(attr->GetData());
    for (size_t i = 0; i < attr->GetSize(); ++i) {
        int64_t axis = data[i];
        if (axis < -rank) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axes", std::to_string(data[i]).c_str(),
                                                  "each axis must be within [-rank, rank)");
            return GRAPH_FAILED;
        }
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axes", std::to_string(data[i]).c_str(),
                                                  "each axis must be within [-rank, rank)");
            return GRAPH_FAILED;
        }
        axes.push_back(axis);
    }
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    return GRAPH_SUCCESS;
}

static ge::graphStatus ValidateInputs(gert::TilingContext* context, int64_t& rank, ge::DataType& dtype)
{
    const auto* xShape = context->GetInputShape(0);
    const auto* xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const auto& storageShape = xShape->GetStorageShape();
    rank = static_cast<int64_t>(storageShape.GetDimNum());
    if (storageShape.IsScalar() || rank > kMaxRank) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank).c_str(),
                                              "rank must be in [1, 8]");
        return GRAPH_FAILED;
    }
    dtype = xDesc->GetDataType();
    if (dtype != DT_FLOAT && dtype != DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(dtype).c_str(), "FLOAT or FLOAT16");
        return GRAPH_FAILED;
    }
    const auto* yDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    if (yDesc->GetDataType() != dtype) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                  "the same dtype as x");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static bool IsContinuousAxes(const std::vector<int64_t>& axes)
{
    for (size_t i = 1; i < axes.size(); ++i) {
        if (axes[i] != axes[i - 1] + 1) {
            return false;
        }
    }
    return true;
}

static int64_t CalcReduceOffset(const CentralizationTilingData* tiling, int64_t reduced, int64_t rank)
{
    int64_t offset = 0;
    int64_t value = reduced;
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t coord = value / tiling->reduceIndexStrides[i];
        value %= tiling->reduceIndexStrides[i];
        offset += coord * tiling->reduceStrides[i];
    }
    return offset;
}

static int64_t CalcReduceOuterOffset(const CentralizationTilingData* tiling, int64_t reduced)
{
    int64_t offset = 0;
    int64_t value = reduced;
    for (int64_t i = 0; i < tiling->irregularReduceOuterRank; ++i) {
        const int64_t coord = value / tiling->irregularReduceOuterIndexStrides[i];
        value %= tiling->irregularReduceOuterIndexStrides[i];
        offset += coord * tiling->reduceStrides[i];
    }
    return offset;
}

static ge::graphStatus CalcIrregularKeepOuterCount(gert::TilingContext* context, const CentralizationTilingData* tiling,
                                                   int64_t& count)
{
    count = 1;
    for (int64_t i = 0; i < tiling->irregularKeepOuterRank; ++i) {
        if (CheckedProductAppend(context, "irregular keep outer count", tiling->keepDims[i], count) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

static void ConfigureIrregularOffsetTable(CentralizationTilingData* tiling)
{
    const bool reduceSuffixMode = tiling->irregularVectorInnerCount <= 1 && tiling->irregularReduceSuffixElements > 1;
    const int64_t segmentCount = reduceSuffixMode ? tiling->irregularReduceOuterCount : tiling->reduceCount;
    tiling->reduceSegmentCount = segmentCount;
    tiling->useReduceOffsetTable = segmentCount <= kMaxReduceSegmentOffsets ? 1 : 0;
    if (tiling->useReduceOffsetTable == 0) {
        return;
    }
    for (int64_t i = 0; i < segmentCount; ++i) {
        tiling->reduceSegmentOffsets[i] = reduceSuffixMode ? CalcReduceOuterOffset(tiling, i) :
                                                             CalcReduceOffset(tiling, i, tiling->reduceRank);
    }
}

static ge::graphStatus ConfigureIrregularKeepTile(gert::TilingContext* context, CentralizationTilingData* tiling,
                                                  int64_t coreNum)
{
    int64_t irregularKeepOuterCount = 1;
    if (CalcIrregularKeepOuterCount(context, tiling, irregularKeepOuterCount) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    tiling->irregularKeepInnerCount = tiling->irregularKeepOuterRank > 0 ?
                                          tiling->keepDims[tiling->irregularKeepOuterRank - 1] :
                                          1;
    const int64_t irregularKeepPrefixCount = tiling->irregularKeepInnerCount > 0 ?
                                                 irregularKeepOuterCount / tiling->irregularKeepInnerCount :
                                                 1;

    const bool reduceSuffixMode = tiling->irregularVectorInnerCount <= 1 && tiling->irregularReduceSuffixElements > 1;
    if (reduceSuffixMode) {
        tiling->irregularKeepTileElements = std::max<int64_t>(
            1, std::min<int64_t>(tiling->irregularKeepInnerCount, kMaxIrregularScalarKeepTile));
    } else {
        const int64_t physicalTileElements = std::max<int64_t>(
            1, std::min<int64_t>(tiling->irregularVectorInnerCount, tiling->irregularVectorTileElements));
        const int64_t alignedPhysicalTileElements = CeilDiv(physicalTileElements, kRegbaseVectorElements) *
                                                    kRegbaseVectorElements;
        const int64_t ubLimitedKeepTile = std::max<int64_t>(
            1, kGenericContiguousTileElements / alignedPhysicalTileElements);
        const int64_t vectorTileCount = std::max<int64_t>(1, tiling->irregularVectorTileCount);
        int64_t baseTaskCount = 0;
        if (CheckedMul(context, "irregular base task count", irregularKeepOuterCount, vectorTileCount,
                       kMaxShapeElements, baseTaskCount) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        int64_t keepTile = 1;
        if (alignedPhysicalTileElements <= kSmallIrregularVectorTileElements &&
            baseTaskCount >= coreNum * kIrregularKeepTileTaskPressure) {
            if (tiling->irregularKeepInnerCount >= 4 && ubLimitedKeepTile >= 4) {
                keepTile = 4;
            } else if (tiling->irregularKeepInnerCount >= 2 && ubLimitedKeepTile >= 2) {
                keepTile = 2;
            }
        }
        tiling->irregularKeepTileElements = keepTile;
    }
    const int64_t vectorTileCount = std::max<int64_t>(1, tiling->irregularVectorTileCount);
    int64_t rawMaxParallelTasks = 0;
    if (CheckedMul(context, "irregular max parallel tasks", irregularKeepOuterCount, vectorTileCount, kMaxShapeElements,
                   rawMaxParallelTasks) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    const int64_t maxParallelTasks = std::max<int64_t>(1, rawMaxParallelTasks);
    const int64_t targetParallelTasks = std::max<int64_t>(1, std::min<int64_t>(coreNum, maxParallelTasks));
    while (tiling->irregularKeepTileElements > 1) {
        const int64_t keepTileCount = CeilDiv(tiling->irregularKeepInnerCount, tiling->irregularKeepTileElements);
        int64_t keepTaskCount = 0;
        int64_t taskCount = 0;
        if (CheckedMul(context, "irregular keep task count", irregularKeepPrefixCount, keepTileCount, kMaxShapeElements,
                       keepTaskCount) != GRAPH_SUCCESS ||
            CheckedMul(context, "irregular vector task count", keepTaskCount, vectorTileCount, kMaxShapeElements,
                       taskCount) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        if (taskCount >= targetParallelTasks) {
            break;
        }
        --tiling->irregularKeepTileElements;
    }
    tiling->irregularKeepTileCountPerPrefix = CeilDiv(tiling->irregularKeepInnerCount,
                                                      tiling->irregularKeepTileElements);
    if (CheckedMul(context, "irregular keep tile task count", irregularKeepPrefixCount,
                   tiling->irregularKeepTileCountPerPrefix, kMaxShapeElements,
                   tiling->irregularKeepTileTaskCount) != GRAPH_SUCCESS ||
        CheckedMul(context, "irregular vector task count", tiling->irregularKeepTileTaskCount, vectorTileCount,
                   kMaxShapeElements, tiling->irregularVectorTaskCount) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus BuildTilingData(gert::TilingContext* context, const gert::Shape& shape, int64_t rank,
                                       const std::vector<int64_t>& axes, CentralizationTilingData* tiling,
                                       int64_t& totalCount)
{
    std::array<bool, kMaxRank> isReduce{};
    for (auto axis : axes) {
        isReduce[static_cast<size_t>(axis)] = true;
    }
    *tiling = CentralizationTilingData{};
    tiling->reduceRank = static_cast<int64_t>(axes.size());
    tiling->groupCount = 1;
    tiling->reduceCount = 1;
    totalCount = 1;

    bool isEmpty = false;
    for (int64_t i = 0; i < rank; ++i) {
        tiling->dims[i] = shape.GetDim(i);
        if (ValidateRuntimeDim(context, tiling->dims[i]) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        isEmpty = isEmpty || tiling->dims[i] == 0;
    }
    if (isEmpty) {
        totalCount = 0;
        tiling->totalCount = 0;
        return GRAPH_SUCCESS;
    }

    int64_t originalStrides[kMaxRank] = {};
    int64_t stride = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
        originalStrides[i] = stride;
        if (CheckedProductAppend(context, "original stride", tiling->dims[i], stride) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    int64_t keepIndex = 0;
    int64_t reduceIndex = 0;
    for (int64_t i = 0; i < rank; ++i) {
        if (CheckedProductAppend(context, "total element count", tiling->dims[i], totalCount) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        if (isReduce[static_cast<size_t>(i)]) {
            tiling->reduceMask[i] = 1;
            tiling->reduceDims[reduceIndex] = tiling->dims[i];
            tiling->reduceStrides[reduceIndex++] = originalStrides[i];
        } else {
            tiling->keepDims[keepIndex] = tiling->dims[i];
            tiling->keepStrides[keepIndex++] = originalStrides[i];
            if (CheckedProductAppend(context, "keep element count", tiling->dims[i], tiling->groupCount) !=
                GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    }
    tiling->totalCount = totalCount;
    tiling->keepRank = keepIndex;
    for (size_t i = 0; i < axes.size(); ++i) {
        if (CheckedProductAppend(context, "reduce element count", tiling->reduceDims[i], tiling->reduceCount) !=
            GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    // 空 axes：reduceRank=0、reduceCount=1，路径选择会走 Fast，按单元素归约得到 y = x - x = 0。
    if (!axes.empty() && IsContinuousAxes(axes)) {
        const int64_t reduceStart = axes.front();
        const int64_t reduceEnd = axes.back();
        tiling->contiguousOuterCount = 1;
        for (int64_t i = 0; i < reduceStart; ++i) {
            if (CheckedProductAppend(context, "contiguous outer count", tiling->dims[i],
                                     tiling->contiguousOuterCount) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
        tiling->contiguousInnerCount = 1;
        for (int64_t i = reduceEnd + 1; i < rank; ++i) {
            if (CheckedProductAppend(context, "contiguous inner count", tiling->dims[i],
                                     tiling->contiguousInnerCount) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    }
    if (!IsContinuousAxes(axes)) {
        const int64_t lastReduceAxis = axes.back();
        tiling->irregularVectorInnerCount = 1;
        int64_t suffixRank = 0;
        for (int64_t i = lastReduceAxis + 1; i < rank; ++i) {
            if (CheckedProductAppend(context, "irregular vector inner count", tiling->dims[i],
                                     tiling->irregularVectorInnerCount) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
            ++suffixRank;
        }
        tiling->irregularKeepOuterRank = keepIndex - suffixRank;
        int64_t keepOuterStride = 1;
        for (int64_t i = tiling->irregularKeepOuterRank - 1; i >= 0; --i) {
            tiling->irregularKeepOuterIndexStrides[i] = keepOuterStride;
            if (CheckedProductAppend(context, "irregular keep outer stride", tiling->keepDims[i], keepOuterStride) !=
                GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
        int64_t reduceSuffixRank = 0;
        for (int64_t axis = rank - 1; axis >= 0 && isReduce[static_cast<size_t>(axis)]; --axis) {
            if (CheckedProductAppend(context, "irregular reduce suffix elements", tiling->dims[axis],
                                     tiling->irregularReduceSuffixElements) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
            ++reduceSuffixRank;
        }
        tiling->irregularReduceOuterRank = tiling->reduceRank - reduceSuffixRank;
        tiling->irregularReduceOuterCount = 1;
        for (int64_t i = 0; i < tiling->irregularReduceOuterRank; ++i) {
            if (CheckedProductAppend(context, "irregular reduce outer count", tiling->reduceDims[i],
                                     tiling->irregularReduceOuterCount) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
        int64_t reduceOuterStride = 1;
        for (int64_t i = tiling->irregularReduceOuterRank - 1; i >= 0; --i) {
            tiling->irregularReduceOuterIndexStrides[i] = reduceOuterStride;
            if (CheckedProductAppend(context, "irregular reduce outer stride", tiling->reduceDims[i],
                                     reduceOuterStride) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    }
    int64_t indexStride = 1;
    for (int64_t i = tiling->keepRank - 1; i >= 0; --i) {
        tiling->keepIndexStrides[i] = indexStride;
        if (CheckedProductAppend(context, "keep index stride", tiling->keepDims[i], indexStride) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    indexStride = 1;
    for (int64_t i = tiling->reduceRank - 1; i >= 0; --i) {
        tiling->reduceIndexStrides[i] = indexStride;
        if (CheckedProductAppend(context, "reduce index stride", tiling->reduceDims[i], indexStride) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    if (!IsContinuousAxes(axes)) {
        ConfigureIrregularOffsetTable(tiling);
    }
    return GRAPH_SUCCESS;
}

static bool IsTrailingAxes(const std::vector<int64_t>& axes, int64_t rank)
{
    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] != rank - static_cast<int64_t>(axes.size()) + static_cast<int64_t>(i)) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus CalcLargeWorkspaceBytes(gert::TilingContext* context, int64_t groupCount,
                                               int64_t alignedCoresPerRow, size_t& workspaceBytes)
{
    int64_t rowSlots = 0;
    int64_t workspaceElements = 0;
    int64_t userWorkspaceBytes = 0;
    if (CheckedMul(context, "large workspace row slots", groupCount, alignedCoresPerRow, kMaxShapeElements, rowSlots) !=
            GRAPH_SUCCESS ||
        CheckedMul(context, "large workspace elements", rowSlots, kWorkspaceAlignElements, kMaxShapeElements,
                   workspaceElements) != GRAPH_SUCCESS ||
        CheckedMul(context, "large workspace bytes", workspaceElements, static_cast<int64_t>(sizeof(float)),
                   std::numeric_limits<int64_t>::max() - static_cast<int64_t>(kSystemWorkspaceBytes),
                   userWorkspaceBytes) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    workspaceBytes = kSystemWorkspaceBytes + static_cast<size_t>(userWorkspaceBytes);
    return GRAPH_SUCCESS;
}

static ge::graphStatus ConfigureLargePath(gert::TilingContext* context, CentralizationTilingData* tiling,
                                          int64_t coreNum, bool trailing, int64_t largeCoresPerRow, size_t* workspace,
                                          uint32_t& key, uint32_t& blockDim)
{
    key = trailing ? kLargeTrailingKey : kLargeKey;
    const int64_t coresPerRow = largeCoresPerRow;
    tiling->coresPerRow = coresPerRow;
    tiling->alignedCoresPerRow = AlignToVectorElements(coresPerRow);
    tiling->rowParallel = (coresPerRow <= 1) ? 1 : 0;
    if (tiling->rowParallel != 0) {
        blockDim = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(coreNum, tiling->groupCount)));
        // 行并行：先把每个 group 的 mean 落 workspace，再按 32B block 并行写输出。
        if (CalcLargeWorkspaceBytes(context, tiling->groupCount, tiling->alignedCoresPerRow, workspace[0]) !=
            GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        // SyncAll requires every launched core to be co-resident.
        context->SetScheduleMode(1);
    } else {
        int64_t reduceParallelBlockDim = 0;
        if (CheckedMul(context, "large reduce block dim", tiling->groupCount, coresPerRow, kMaxShapeElements,
                       reduceParallelBlockDim) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        blockDim = static_cast<uint32_t>(std::min<int64_t>(coreNum, reduceParallelBlockDim));
        if (CalcLargeWorkspaceBytes(context, tiling->groupCount, tiling->alignedCoresPerRow, workspace[0]) !=
            GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        // SyncAll requires every launched core to be co-resident.
        context->SetScheduleMode(1);
    }
    tiling->rowsPerLoop = 1;
    tiling->rowsPerBatch = 1;
    return GRAPH_SUCCESS;
}

// DataCopyExtParams 的 rowCount/blockCount 是 uint16；计数本身用 64 位，
// 真正落到 ABI 字段的 tile 尺寸在这里 clamp，不把 shape 判非法。
// 当前 UB 下 rows 不会触发该上限，保留作为未来更大 UB 的防御性收口。
static void ClampKernelAbi(CentralizationTilingData* tiling)
{
    constexpr int64_t kAbiCountMax = std::numeric_limits<uint16_t>::max();
    tiling->rowsPerBatch = std::min(tiling->rowsPerBatch, kAbiCountMax);
    tiling->rowsPerLoop = std::min(tiling->rowsPerLoop, kAbiCountMax);
}

static ge::graphStatus FinalizePath(gert::TilingContext* context, CentralizationTilingData* tiling, ge::DataType dtype,
                                    uint32_t key, uint32_t blockDim, uint64_t ubSize)
{
    if (ConfigureBufferBytes(context, tiling, dtype, ubSize, key) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    ClampKernelAbi(tiling);
    tiling->blockFactor = tiling->groupCount > 0 ? CeilDiv(tiling->groupCount, static_cast<int64_t>(blockDim)) : 1;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(key);
    return GRAPH_SUCCESS;
}

static void ApplySmallContiguousPath(CentralizationTilingData* tiling, int64_t coreNum,
                                     int64_t smallContiguousBlockElements, size_t* workspace, uint32_t& key,
                                     uint32_t& blockDim)
{
    key = kGenericKey;
    blockDim = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(coreNum, tiling->contiguousOuterCount)));
    workspace[0] = 0;
    tiling->largePath = 0;
    tiling->contiguousGeneric = 1;
    tiling->smallContiguous = 1;
    tiling->smallContiguousMode = (tiling->contiguousInnerCount == 1) ? 1 : 2;
    tiling->smallContiguousBlockElements = smallContiguousBlockElements;
    tiling->smallContiguousMeanElements = tiling->contiguousInnerCount;
    tiling->rowsPerLoop = 1;
    tiling->rowsPerBatch = 1;
}

static void ApplyContiguousGenericPath(CentralizationTilingData* tiling, int64_t coreNum, int64_t contiguousTaskCount,
                                       bool canUseLargeContiguousSmallInner, size_t* workspace, uint32_t& key,
                                       uint32_t& blockDim)
{
    key = kGenericKey;
    blockDim = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(coreNum, contiguousTaskCount)));
    workspace[0] = 0;
    tiling->largePath = 0;
    tiling->contiguousGeneric = 1;
    tiling->smallContiguous = 0;
    tiling->largeContiguousSmallInner = canUseLargeContiguousSmallInner ? 1 : 0;
    tiling->rowsPerLoop = 1;
    tiling->rowsPerBatch = 1;
}

static void ApplyIrregularVectorPath(CentralizationTilingData* tiling, int64_t coreNum, size_t* workspace,
                                     uint32_t& key, uint32_t& blockDim)
{
    key = kGenericKey;
    blockDim = static_cast<uint32_t>(
        std::max<int64_t>(1, std::min<int64_t>(coreNum, tiling->irregularVectorTaskCount)));
    workspace[0] = 0;
    tiling->largePath = 0;
    tiling->irregularVectorizable = (tiling->irregularVectorInnerCount > 1) ? 1 : 2;
    tiling->smallContiguous = 0;
    tiling->rowsPerLoop = 1;
    tiling->rowsPerBatch = 1;
}

static bool SmallContiguousFitsUb(const CentralizationTilingData* tiling, int64_t blockElements, size_t elementBytes,
                                  uint64_t ubSize)
{
    // depth=1：in + out + sum 三个存活节点。mode 由 contiguousInnerCount 决定：
    // inner==1 走整块标量均值（mode 1），否则整块 padded 向量均值（mode 2）。
    const size_t perBuffer = AlignDownBytes(UbUsableBytes(ubSize) / 3);
    if (perBuffer == 0) {
        return false;
    }
    const int64_t innerCount = tiling->contiguousInnerCount;
    int64_t inputElements = 0;
    int64_t sumElements = 0;
    if (innerCount != 1) {
        const int64_t alignedMean = AlignToVectorElements(innerCount);
        inputElements = tiling->reduceCount * alignedMean;
        sumElements = alignedMean;
    } else {
        inputElements = AlignToVectorElements(blockElements);
        sumElements = AlignToVectorElements(innerCount);
    }
    return static_cast<size_t>(inputElements) * elementBytes <= perBuffer &&
           static_cast<size_t>(sumElements) * sizeof(float) <= perBuffer;
}

static bool LcsiUbFits(const CentralizationTilingData* tiling, size_t elementBytes, uint64_t ubSize)
{
    // depth=1：in + out + sum，lcsi 的 sum 同时存 sum/correction
    const size_t perBuffer = AlignDownBytes(UbUsableBytes(ubSize) / 3);
    const int64_t alignedInner = AlignToVectorElements(tiling->contiguousInnerCount);
    return perBuffer >= static_cast<size_t>(alignedInner) * elementBytes &&
           perBuffer >= static_cast<size_t>(alignedInner) * sizeof(float) * 2;
}

static bool ContiguousUbFits(const CentralizationTilingData* tiling, size_t elementBytes, uint64_t ubSize)
{
    // depth=2：in/out 各 2 份 + sum 1 份；sum 固定按 kGenericContiguousTileElements 分配
    const size_t perBuffer = AlignDownBytes(UbUsableBytes(ubSize) / 5);
    const int64_t tile = std::max<int64_t>(
        1, std::min<int64_t>(tiling->contiguousInnerTileElements, kGenericContiguousTileElements));
    return perBuffer >= static_cast<size_t>(tile) * elementBytes &&
           perBuffer >= static_cast<size_t>(kGenericContiguousTileElements) * sizeof(float) * 2;
}

static bool IrregularUbFits(const CentralizationTilingData* tiling, size_t elementBytes, uint64_t ubSize)
{
    // depth=2：in/out 各 2 份 + sum 1 份；不规则 vector 的 sum 固定按 256 元素分配
    const size_t perBuffer = AlignDownBytes(UbUsableBytes(ubSize) / 5);
    const int64_t tile = std::max<int64_t>(
        1, std::min<int64_t>(tiling->irregularVectorTileElements, kGenericContiguousTileElements));
    return perBuffer >= static_cast<size_t>(tile) * elementBytes &&
           perBuffer >= static_cast<size_t>(kGenericContiguousTileElements) * sizeof(float);
}

static bool FastUbFits(const CentralizationTilingData* tiling, size_t elementBytes, uint64_t ubSize,
                       size_t& perBufferBytes)
{
    const int64_t liveBuffers = 2 * (tiling->bufNum > 0 ? tiling->bufNum : 2);
    if (liveBuffers <= 0) {
        return false;
    }
    perBufferBytes = AlignDownBytes(UbUsableBytes(ubSize) / static_cast<size_t>(liveBuffers));
    if (perBufferBytes == 0) {
        return false;
    }
    return static_cast<size_t>(AlignToVectorElements(tiling->reduceCount)) * elementBytes <= perBufferBytes;
}

static bool LargeTrailingUbFits(size_t elementBytes, uint64_t ubSize)
{
    constexpr size_t kSumBytes = 128;
    const size_t usable = UbUsableBytes(ubSize);
    if (usable <= kSumBytes) {
        return false;
    }
    const size_t perBuffer = AlignDownBytes((usable - kSumBytes) / 6);
    return perBuffer >= static_cast<size_t>(kKernelTileElements) * elementBytes &&
           perBuffer >= static_cast<size_t>(kKernelTileElements) * sizeof(float);
}

static void ApplyFastPath(CentralizationTilingData* tiling, int64_t coreNum, size_t* workspace, uint32_t& key,
                          uint32_t& blockDim, size_t perBufferBytes, size_t elementBytes)
{
    key = kFastKey;
    blockDim = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(coreNum, tiling->groupCount)));
    workspace[0] = 0;
    tiling->alignedReduce = AlignToVectorElements(tiling->reduceCount);
    tiling->ubFactor = tiling->alignedReduce;
    // Fast 的 in/out 各 bufNum 份，每份 = rowsPerLoop * alignedReduce * sizeof(T)。
    const size_t rowBytes = static_cast<size_t>(tiling->alignedReduce) * elementBytes;
    const int64_t maxRows = rowBytes > 0 ? static_cast<int64_t>(perBufferBytes / rowBytes) : 1;
    const int64_t rows = std::max<int64_t>(
        1, std::min<int64_t>(maxRows, static_cast<int64_t>(std::numeric_limits<uint16_t>::max())));
    tiling->rowsPerLoop = rows;
    tiling->rowsPerBatch = rows;
}

static void ApplyGenericScalarPath(CentralizationTilingData* tiling, uint32_t& blockDim)
{
    // Generic scalar traversal must not interleave output elements across cores.
    blockDim = 1;
    tiling->rowsPerLoop = 1;
    tiling->rowsPerBatch = 1;
    tiling->smallContiguous = 0;
}

static ge::graphStatus ConfigurePath(gert::TilingContext* context, CentralizationTilingData* tiling, ge::DataType dtype,
                                     int64_t coreNum, uint64_t ubSize, bool trailing, bool continuousAxes,
                                     int64_t totalCount)
{
    const int64_t blockElems = (dtype == DT_FLOAT) ? 8LL : 16LL;
    constexpr int64_t kFastPathBufNum = 2;
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    if (totalCount == 0) {
        tiling->largePath = 0;
        tiling->rowParallel = 0;
        tiling->rowsPerLoop = 1;
        tiling->rowsPerBatch = 1;
        return FinalizePath(context, tiling, dtype, kEmptyKey, 1, ubSize);
    }

    tiling->largePath = tiling->reduceCount > kLargeReduceThreshold ? 1 : 0;
    tiling->alignedReduce = CeilDiv(tiling->reduceCount, blockElems) * blockElems;
    tiling->ubFactor = tiling->alignedReduce;
    tiling->bufNum = kFastPathBufNum;
    tiling->rowParallel = 0;
    tiling->coresPerRow = 1;
    tiling->alignedCoresPerRow = 1;
    tiling->contiguousGeneric = 0;
    tiling->contiguousInnerTileElements = kGenericContiguousTileElements;
    tiling->contiguousInnerTileCount = CeilDiv(tiling->contiguousInnerCount, kGenericContiguousTileElements);
    tiling->largeContiguousSmallInner = 0;
    tiling->largeContiguousReduceTile = 0;
    tiling->irregularVectorizable = 0;
    tiling->irregularVectorTileElements = kGenericContiguousTileElements;
    tiling->irregularVectorTileCount = CeilDiv(tiling->irregularVectorInnerCount, kGenericContiguousTileElements);
    if (ConfigureIrregularKeepTile(context, tiling, coreNum) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    uint32_t key = kGenericKey;
    uint32_t blockDim = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(coreNum, tiling->groupCount)));
    const int64_t largeChunkCount = CeilDiv(tiling->reduceCount, kKernelTileElements);
    const int64_t largeCoresPerRow = tiling->largePath != 0 ?
                                         CalcCoresPerRow(tiling->groupCount, coreNum, largeChunkCount) :
                                         1;
    const bool largeRowParallel = tiling->largePath != 0 && largeCoresPerRow <= 1;
    int64_t contiguousTaskCount = 0;
    int64_t smallContiguousBlockElements = 0;
    if (CheckedMul(context, "contiguous task count", tiling->contiguousOuterCount, tiling->contiguousInnerTileCount,
                   kMaxShapeElements, contiguousTaskCount) != GRAPH_SUCCESS ||
        CheckedMul(context, "small contiguous block elements", tiling->reduceCount, tiling->contiguousInnerCount,
                   kMaxShapeElements, smallContiguousBlockElements) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    const int64_t smallContiguousAlignedElements = AlignToVectorElements(smallContiguousBlockElements);
    int64_t smallContiguousUbElements = smallContiguousAlignedElements;
    if (tiling->contiguousInnerCount != 1) {
        smallContiguousUbElements = tiling->reduceCount * AlignToVectorElements(tiling->contiguousInnerCount);
    }
    const size_t elementBytes = dtype == DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const size_t smallContiguousWorkingSetBytes = static_cast<size_t>(smallContiguousUbElements) * elementBytes;
    const bool canUseLargeContiguousSmallInner = continuousAxes && !trailing && tiling->largePath != 0 &&
                                                 largeRowParallel && tiling->contiguousInnerCount > 1 &&
                                                 tiling->contiguousInnerCount <= kRegbaseVectorElements;
    // 小路径的 tile 不可再切（kernel 一次处理整块），必须整块放进 in/out/sum 的均分预算。
    const bool useSmallContiguous = continuousAxes && !trailing && smallContiguousBlockElements > 0 &&
                                    smallContiguousBlockElements <= kSmallContiguousMaxWorkingSetElements &&
                                    smallContiguousWorkingSetBytes <= kSmallContiguousMaxWorkingSetBytes &&
                                    SmallContiguousFitsUb(tiling, smallContiguousBlockElements, elementBytes, ubSize);
    const bool lcsiFits = canUseLargeContiguousSmallInner && LcsiUbFits(tiling, elementBytes, ubSize);
    const bool contiguousFits = ContiguousUbFits(tiling, elementBytes, ubSize);
    const bool irregularFits = IrregularUbFits(tiling, elementBytes, ubSize);
    const bool useContiguousGeneric = continuousAxes && !trailing && tiling->contiguousInnerCount > 1 &&
                                      (tiling->largePath == 0 ||
                                       contiguousTaskCount >= std::max<int64_t>(1, coreNum / 2) || largeRowParallel) &&
                                      (contiguousFits || lcsiFits);
    const bool useIrregularVector = !trailing && tiling->contiguousGeneric == 0 && tiling->largePath == 0 &&
                                    ((tiling->irregularVectorInnerCount > 1 && tiling->irregularVectorTaskCount > 0) ||
                                     tiling->irregularReduceSuffixElements > 1) &&
                                    irregularFits;
    size_t fastPerBufferBytes = 0;
    if (useSmallContiguous) {
        ApplySmallContiguousPath(tiling, coreNum, smallContiguousBlockElements, workspace, key, blockDim);
    } else if (useContiguousGeneric) {
        ApplyContiguousGenericPath(tiling, coreNum, contiguousTaskCount, lcsiFits, workspace, key, blockDim);
    } else if (useIrregularVector) {
        ApplyIrregularVectorPath(tiling, coreNum, workspace, key, blockDim);
    } else if (tiling->largePath) {
        // LargeTrailing 的 tile 固定 1024 元素；放不下时回退到 Generic large（7020）。
        const bool useLargeTrailing = trailing && LargeTrailingUbFits(elementBytes, ubSize);
        if (ConfigureLargePath(context, tiling, coreNum, useLargeTrailing, largeCoresPerRow, workspace, key,
                               blockDim) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    } else if (trailing && tiling->reduceCount <= kFastReduceLimit) {
        if (FastUbFits(tiling, elementBytes, ubSize, fastPerBufferBytes)) {
            ApplyFastPath(tiling, coreNum, workspace, key, blockDim, fastPerBufferBytes, elementBytes);
        } else {
            // Fast 的列宽不能切分，放不下时回退到按 reduce 分块的 LargeTrailing / Generic large。
            tiling->largePath = 1;
            tiling->ubFactor = 0; // 清理 Fast 残留配置，最终路径不消费这两个字段
            tiling->alignedReduce = 0;
            const bool useLargeTrailing = LargeTrailingUbFits(elementBytes, ubSize);
            if (ConfigureLargePath(context, tiling, coreNum, useLargeTrailing, 1, workspace, key, blockDim) !=
                GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    } else {
        ApplyGenericScalarPath(tiling, blockDim);
    }
    return FinalizePath(context, tiling, dtype, key, blockDim, ubSize);
}
} // namespace

static ge::graphStatus CentralizationTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    int64_t rank = 0;
    ge::DataType dtype = DT_UNDEFINED;
    if (ValidateInputs(context, rank, dtype) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    std::vector<int64_t> axes;
    if (ResolveAxes(context, rank, axes) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    CentralizationTilingData* tiling = context->GetTilingData<CentralizationTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    int64_t totalCount = 1;
    if (BuildTilingData(context, context->GetInputShape(0)->GetStorageShape(), rank, axes, tiling, totalCount) !=
        GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    if (GetPlatform(context, ubSize, coreNum) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return ConfigurePath(context, tiling, dtype, coreNum, ubSize, IsTrailingAxes(axes, rank), IsContinuousAxes(axes),
                         totalCount);
}

static ge::graphStatus CentralizationTilingParse(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<CentralizationCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo->coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Centralization)
    .Tiling(CentralizationTilingFunc)
    .TilingParse<CentralizationCompileInfo>(CentralizationTilingParse);
} // namespace optiling
