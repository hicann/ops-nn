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
 * \file wts_arq_tiling.cpp
 * \brief WtsARQ tiling for arch35 (DAV_3510 / Ascend 950).
 *
 * Flow (DESIGN.md §3.4):
 *   1. platform info (coreNum/ubSize via PlatformAscendC, never hardcoded)
 *   2. attr check: num_bits must be 8
 *   3. shape normalize (rank 0 -> [1]) + DimensionCollapse (pad / flag /
 *      merge same-flag adjacent axes / strides with broadcast axes zeroed)
 *   4. template select: shapeLen <= 4 -> RANK=4, 5~8 -> RANK=8
 *   5. UB split (OneDim: ubFormer=perBufElems, 256B align; multi-dim: 256B align)
 *   6. multi-core split with core-feeding maxElemNum shrink loop
 *   7. brcMode decision chain (NLast/dcache -> UB BRC; fp16 + 32B-aligned last
 *      axis -> UB BRC; otherwise NDDMA; no broadcast axis -> DataCopyPad)
 *   8. schMode for the NDDMA path (<=5 axes WithoutLoop, >5 WithLoop)
 *   9. workspace = 0 (pure elementwise + copy-in broadcast)
 */
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include "../../op_kernel/arch35/wts_arq_tiling_data.h"
#include "../../op_kernel/arch35/wts_arq_tiling_key.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace ge;

namespace optiling {

namespace {
constexpr size_t INPUT_W_IDX = 0;
constexpr size_t INPUT_W_MIN_IDX = 1;
constexpr size_t INPUT_W_MAX_IDX = 2;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr size_t ATTR_NUM_BITS_IDX = 0;
constexpr size_t ATTR_OFFSET_FLAG_IDX = 1;

constexpr int64_t MAX_SUPPORTED_RANK = 8;
constexpr int64_t SHAPE_SIZE_LIMIT = 1LL << 31;
constexpr float SCALE_EPS = 1.1920929e-07f; // float32 eps, spec-locked

constexpr int64_t UB_EXTRA_RESERVE = 4096;    // stack / BroadcastTiling reserve
constexpr int64_t MULTIDIM_ALIGN_BYTES = 256; // REPEAT alignment for multi-dim branch
// OneDim branch uses full-width register loads (no per-element mask), so the tile
// buffer must end on a vector-register boundary: with 128B the fp32 tile could end
// 32 elements past the last full repeat and the final LoadAlign would read out of
// bounds. 256B == one fp32 vector register (VL=64) and also covers fp16.
constexpr int64_t ONEDIM_ALIGN_BYTES = 256;
constexpr int64_t SHRINK_STEP_BYTES = 128; // core-feeding loop: shrink by CACHE_LINE each round
constexpr int64_t MIN_TILE_ELEMS = 1024;   // 4KB fp32 lower bound for the shrink loop
constexpr int64_t MAX_SHRINK_ROUNDS = 512;
constexpr int64_t NDDMA_MAX_DIMS = 5;
constexpr size_t WORKSPACE_NUM = 1;

struct CollapseResult {
    std::vector<int64_t> dims;       // collapsed output shape
    std::vector<int64_t> minStrides; // 0 on broadcast axes
    std::vector<int64_t> maxStrides;
    std::vector<bool> broadcastFlags;
    bool hasBroadcast = false;
};

struct UbSplitResult {
    int64_t axis = 0;
    int64_t ubFormer = 0;
    int64_t ubOuter = 0;
    int64_t ubTail = 0;
};

struct BlockSplitResult {
    int64_t fusedProduct = 0;
    int64_t blockFormer = 0;
    int64_t blockNum = 0;
    int64_t blockTail = 0;
};

std::string VectorToString(const std::vector<int64_t>& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

// DimensionCollapse: same-rank inputs (pad is a no-op), flag broadcast axes,
// merge adjacent same-flag axes, compute strides (broadcast axis -> 0;
// merged non-broadcast axis takes the rightmost constituent's storage stride).
CollapseResult CollapseShape(const std::vector<int64_t>& wShape, const std::vector<int64_t>& minShape,
                             const std::vector<int64_t>& maxShape)
{
    const int64_t rank = static_cast<int64_t>(wShape.size());
    std::vector<int64_t> minOrigStrides(rank, 1);
    std::vector<int64_t> maxOrigStrides(rank, 1);
    for (int64_t d = rank - 2; d >= 0; d--) {
        minOrigStrides[d] = minOrigStrides[d + 1] * minShape[d + 1];
        maxOrigStrides[d] = maxOrigStrides[d + 1] * maxShape[d + 1];
    }

    CollapseResult result;
    for (int64_t d = 0; d < rank; d++) {
        const bool isBroadcast = (minShape[d] == 1 && wShape[d] > 1);
        const int64_t minStride = isBroadcast ? 0 : minOrigStrides[d];
        const int64_t maxStride = isBroadcast ? 0 : maxOrigStrides[d];
        if (!result.dims.empty() && result.broadcastFlags.back() == isBroadcast) {
            // merge with previous axis (same broadcast flag); a merged
            // non-broadcast axis takes the rightmost constituent's stride
            result.dims.back() *= wShape[d];
            if (!isBroadcast) {
                result.minStrides.back() = minStride;
                result.maxStrides.back() = maxStride;
            }
        } else {
            result.dims.push_back(wShape[d]);
            result.minStrides.push_back(minStride);
            result.maxStrides.push_back(maxStride);
            result.broadcastFlags.push_back(isBroadcast);
        }
    }
    for (const bool flag : result.broadcastFlags) {
        result.hasBroadcast = result.hasBroadcast || flag;
    }
    return result;
}

// UB split: walk axes from inner to outer, the first axis that does not fit is
// the split axis. OneDim (shapeLen == 1): ubFormer = perBufElems.
UbSplitResult ComputeUbSplit(const std::vector<int64_t>& dims, int64_t perBufElems)
{
    UbSplitResult split;
    const int64_t shapeLen = static_cast<int64_t>(dims.size());
    if (shapeLen == 1) {
        split.axis = 0;
        split.ubFormer = std::min(perBufElems, dims[0]);
        split.ubOuter = (dims[0] + split.ubFormer - 1) / split.ubFormer;
        split.ubTail = dims[0] - (split.ubOuter - 1) * split.ubFormer;
        return split;
    }

    int64_t curProduct = 1;
    int64_t axis = 0;
    bool allFit = true;
    for (int64_t i = shapeLen - 1; i >= 0; i--) {
        if (dims[i] * curProduct > perBufElems) {
            axis = i;
            allFit = false;
            break;
        }
        curProduct *= dims[i];
    }
    if (allFit) {
        axis = 0;
        curProduct /= dims[0]; // split on the outermost axis
    }
    int64_t ubFormer = perBufElems / curProduct;
    if (ubFormer > dims[axis]) {
        ubFormer = dims[axis];
    }
    if (ubFormer < 1) {
        ubFormer = 1;
    }
    split.axis = axis;
    split.ubFormer = ubFormer;
    split.ubOuter = (dims[axis] + ubFormer - 1) / ubFormer;
    split.ubTail = dims[axis] - (split.ubOuter - 1) * ubFormer;
    return split;
}

BlockSplitResult ComputeBlockSplit(const std::vector<int64_t>& dims, const UbSplitResult& split, int64_t coreNum)
{
    BlockSplitResult block;
    int64_t outerProd = 1;
    for (int64_t d = 0; d < split.axis; d++) {
        outerProd *= dims[d];
    }
    block.fusedProduct = split.ubOuter * outerProd;
    block.blockFormer = (block.fusedProduct + coreNum - 1) / coreNum;
    block.blockNum = (block.fusedProduct + block.blockFormer - 1) / block.blockFormer;
    block.blockTail = block.fusedProduct - (block.blockNum - 1) * block.blockFormer;
    return block;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context->GetNodeName(), "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInputShapes(gert::TilingContext* context, std::vector<std::vector<int64_t>>& inputShapes,
                               std::vector<int64_t>& outputShape)
{
    for (size_t i = 0; i < 3; ++i) {
        auto shapePtr = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
        const gert::Shape s = shapePtr->GetStorageShape();
        std::vector<int64_t> dims;
        for (size_t d = 0; d < s.GetDimNum(); ++d) {
            dims.push_back(s.GetDim(d));
        }
        inputShapes.push_back(dims);
    }
    auto outShapePtr = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapePtr);
    const gert::Shape s = outShapePtr->GetStorageShape();
    for (size_t d = 0; d < s.GetDimNum(); ++d) {
        outputShape.push_back(s.GetDim(d));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapes(gert::TilingContext* context, std::vector<std::vector<int64_t>>& inputShapes,
                            std::vector<int64_t>& outputShape, int64_t& wElems)
{
    const char* nodeName = context->GetNodeName();
    // rank 0 scalar normalizes to 1-element [1]
    for (auto& shape : inputShapes) {
        if (shape.empty()) {
            shape.push_back(1);
        }
        for (const int64_t dim : shape) {
            if (dim < 0) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "w, w_min, w_max, y", "dynamic runtime shape",
                                                       "The tiling phase requires concrete runtime shapes");
                return ge::GRAPH_FAILED;
            }
        }
    }
    if (outputShape.empty()) {
        outputShape.push_back(1);
    }
    for (const int64_t dim : outputShape) {
        if (dim < 0) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "y", "dynamic runtime shape",
                                                   "The tiling phase requires concrete runtime shapes");
            return ge::GRAPH_FAILED;
        }
    }

    const int64_t wRank = static_cast<int64_t>(inputShapes[INPUT_W_IDX].size());
    if (wRank > MAX_SUPPORTED_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "w", std::to_string(wRank).c_str(),
                                                 "The rank of w must be <= 8");
        return ge::GRAPH_FAILED;
    }
    if (static_cast<int64_t>(inputShapes[INPUT_W_MIN_IDX].size()) != wRank ||
        static_cast<int64_t>(inputShapes[INPUT_W_MAX_IDX].size()) != wRank) {
        const std::string ranks = std::to_string(wRank) + ", " + std::to_string(inputShapes[INPUT_W_MIN_IDX].size()) +
                                  ", " + std::to_string(inputShapes[INPUT_W_MAX_IDX].size());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(nodeName, "w, w_min, w_max", ranks.c_str(),
                                                  "The rank of w_min and w_max must be same as w");
        return ge::GRAPH_FAILED;
    }
    if (inputShapes[INPUT_W_MIN_IDX] != inputShapes[INPUT_W_MAX_IDX]) {
        const std::string shapes = VectorToString(inputShapes[INPUT_W_MIN_IDX]) + ", " +
                                   VectorToString(inputShapes[INPUT_W_MAX_IDX]);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "w_min, w_max", shapes.c_str(),
                                               "The shape of w_min must be same as w_max");
        return ge::GRAPH_FAILED;
    }
    if (outputShape != inputShapes[INPUT_W_IDX]) {
        const std::string shapes = VectorToString(inputShapes[INPUT_W_IDX]) + ", " + VectorToString(outputShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "w, y", shapes.c_str(),
                                               "The shape of y must be the same as w");
        return ge::GRAPH_FAILED;
    }
    for (int64_t d = 0; d < wRank; ++d) {
        const int64_t minDim = inputShapes[INPUT_W_MIN_IDX][d];
        const int64_t wDim = inputShapes[INPUT_W_IDX][d];
        if (minDim != wDim && minDim != 1) {
            const std::string shapes = VectorToString(inputShapes[INPUT_W_IDX]) + ", " +
                                       VectorToString(inputShapes[INPUT_W_MIN_IDX]);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "w, w_min", shapes.c_str(),
                                                   "Each dim of w_min and w_max must be the same as w or equal to 1");
            return ge::GRAPH_FAILED;
        }
    }

    wElems = 1;
    for (const int64_t dim : inputShapes[INPUT_W_IDX]) {
        if (dim == 0) {
            wElems = 0;
            break;
        }
        if (wElems > SHAPE_SIZE_LIMIT / dim) {
            wElems = SHAPE_SIZE_LIMIT + 1;
            break;
        }
        wElems *= dim;
    }
    if (wElems > SHAPE_SIZE_LIMIT) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(nodeName, "w", std::to_string(wElems).c_str(),
                                                  "The shape size of w must be smaller than or equal to 2^31");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypeAndAttrs(gert::TilingContext* context, int64_t& typeSize, int64_t& numBits, bool& offsetFlag)
{
    const char* nodeName = context->GetNodeName();
    auto wDesc = context->GetInputDesc(INPUT_W_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    auto wMinDesc = context->GetInputDesc(INPUT_W_MIN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wMinDesc);
    auto wMaxDesc = context->GetInputDesc(INPUT_W_MAX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wMaxDesc);
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    const ge::DataType wDtype = wDesc->GetDataType();
    if (wDtype != ge::DT_FLOAT16 && wDtype != ge::DT_FLOAT) {
        const std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(wDtype);
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "w", dtypeStr.c_str(), "DT_FLOAT16, DT_FLOAT");
        return ge::GRAPH_FAILED;
    }
    // dtype mismatch: report every mismatching parameter with its actual dtype
    const ge::DataType dtypes[4] = {wDtype, wMinDesc->GetDataType(), wMaxDesc->GetDataType(), yDesc->GetDataType()};
    const char* dtypeNames[4] = {"w", "w_min", "w_max", "y"};
    std::string badDtypeNames;
    std::string badDtypeValues;
    for (size_t i = 0; i < 4; ++i) {
        if (dtypes[i] == wDtype) {
            continue;
        }
        if (!badDtypeNames.empty()) {
            badDtypeNames += ", ";
            badDtypeValues += ", ";
        }
        badDtypeNames += dtypeNames[i];
        badDtypeValues += ge::TypeUtils::DataTypeToSerialString(dtypes[i]);
    }
    if (!badDtypeNames.empty()) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nodeName, badDtypeNames.c_str(), badDtypeValues.c_str(),
                                               "The types of w_min, w_max and y must be same as w");
        return ge::GRAPH_FAILED;
    }
    // The kernel reads contiguous ND storage and never handles FRACTAL_NZ. Check the
    // *storage* format: a non-ND origin format is fine once GE normalized it with a
    // transdata, but non-ND storage arriving at Optiling would be misread as ND.
    const ge::Format storageFormats[4] = {wDesc->GetStorageFormat(), wMinDesc->GetStorageFormat(),
                                          wMaxDesc->GetStorageFormat(), yDesc->GetStorageFormat()};
    const char* formatNames[4] = {"w", "w_min", "w_max", "y"};
    std::string badFormatNames;
    std::string badFormatValues;
    for (size_t i = 0; i < 4; ++i) {
        if (storageFormats[i] == ge::FORMAT_ND) {
            continue;
        }
        if (!badFormatNames.empty()) {
            badFormatNames += ", ";
            badFormatValues += ", ";
        }
        badFormatNames += formatNames[i];
        badFormatValues += ge::TypeUtils::FormatToSerialString(storageFormats[i]);
    }
    if (!badFormatNames.empty()) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(nodeName, badFormatNames.c_str(), badFormatValues.c_str(),
                                                "The format of w, w_min, w_max and y must be ND");
        return ge::GRAPH_FAILED;
    }
    typeSize = (wDtype == ge::DT_FLOAT16) ? 2 : 4;

    numBits = 8;
    offsetFlag = false;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* numBitsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_BITS_IDX);
        numBits = (numBitsPtr != nullptr) ? *numBitsPtr : 8;
        const bool* offsetFlagPtr = attrs->GetAttrPointer<bool>(ATTR_OFFSET_FLAG_IDX);
        offsetFlag = (offsetFlagPtr != nullptr) ? *offsetFlagPtr : false;
    }
    if (numBits != 8) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "num_bits", std::to_string(numBits).c_str(),
                                              "The value of num_bits must be 8");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <int64_t R>
void FillTilingData(WtsArqTilingData<R>* tiling, const CollapseResult& collapse, const UbSplitResult& split,
                    const BlockSplitResult& block, int64_t perBufElems, uint32_t brcMode, uint32_t schMode,
                    bool offsetFlag, int64_t numBits)
{
    const int64_t shapeLen = static_cast<int64_t>(collapse.dims.size());
    const int64_t delta = R - shapeLen;
    for (int64_t d = 0; d < delta; d++) {
        tiling->dims[d] = 1;
        tiling->minStrides[d] = 0;
        tiling->maxStrides[d] = 0;
    }
    for (int64_t d = 0; d < shapeLen; d++) {
        tiling->dims[d + delta] = collapse.dims[d];
        tiling->minStrides[d + delta] = collapse.minStrides[d];
        tiling->maxStrides[d + delta] = collapse.maxStrides[d];
    }
    tiling->ubSplitAxis = split.axis + delta;
    tiling->ubFormer = split.ubFormer;
    tiling->ubOuter = split.ubOuter;
    tiling->ubTail = split.ubTail;
    tiling->fusedProduct = block.fusedProduct;
    tiling->blockFormer = block.blockFormer;
    tiling->blockNum = block.blockNum;
    tiling->blockTail = block.blockTail;
    tiling->perBufElems = perBufElems;
    tiling->shapeLen = static_cast<uint32_t>(shapeLen);
    tiling->schMode = schMode;
    tiling->brcMode = brcMode;
    tiling->offsetFlag = offsetFlag ? 1U : 0U;
    tiling->numBits = static_cast<uint32_t>(numBits);
    tiling->coreNum = static_cast<uint32_t>(block.blockNum);
    tiling->eps = SCALE_EPS;
}

template <int64_t R>
void LogTilingData(const char* nodeName, const WtsArqTilingData<R>* tiling)
{
    OP_LOGI(nodeName,
            "WtsARQ TilingData: shapeLen=%u brcMode=%u schMode=%u offsetFlag=%u numBits=%u coreNum=%u "
            "ubSplitAxis=%ld ubFormer=%ld ubOuter=%ld ubTail=%ld fusedProduct=%ld blockFormer=%ld blockNum=%ld "
            "blockTail=%ld perBufElems=%ld",
            tiling->shapeLen, tiling->brcMode, tiling->schMode, tiling->offsetFlag, tiling->numBits, tiling->coreNum,
            tiling->ubSplitAxis, tiling->ubFormer, tiling->ubOuter, tiling->ubTail, tiling->fusedProduct,
            tiling->blockFormer, tiling->blockNum, tiling->blockTail, tiling->perBufElems);
}

template <int64_t R>
ge::graphStatus DoTilingAndSet(gert::TilingContext* context, const CollapseResult& collapse, uint64_t ubSize,
                               int64_t coreNum, int64_t typeSize, bool offsetFlag, int64_t numBits, bool isEmpty)
{
    auto* tiling = context->GetTilingData<WtsArqTilingData<R>>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(WtsArqTilingData<R>), 0, sizeof(WtsArqTilingData<R>)) != EOK,
                OP_LOGE(context->GetNodeName(), "set tiling data error"), return ge::GRAPH_FAILED);

    const int64_t shapeLen = static_cast<int64_t>(collapse.dims.size());
    const bool isOneDim = (shapeLen == 1);
    const int64_t alignBytes = isOneDim ? ONEDIM_ALIGN_BYTES : MULTIDIM_ALIGN_BYTES;
    const int64_t alignElems = alignBytes / typeSize;
    // 5-buffer model: 4 fp32 buffers + 1 T-typed cast buffer per element
    const int64_t bytesPerElem = kWtsArqFloatBufs * static_cast<int64_t>(sizeof(float)) + typeSize;

    int64_t perBufElems = (static_cast<int64_t>(ubSize) - UB_EXTRA_RESERVE) / bytesPerElem;
    perBufElems = Ops::Base::FloorAlign(perBufElems, alignElems);
    OP_CHECK_IF(perBufElems <= 0, OP_LOGE(context->GetNodeName(), "UB size is too small for WtsARQ"),
                return ge::GRAPH_FAILED);

    UbSplitResult split;
    BlockSplitResult block;
    if (isEmpty) {
        // empty tensor: no real work, kernel exits on fusedProduct == 0
        split.axis = 0;
        split.ubFormer = 0;
        split.ubOuter = 0;
        split.ubTail = 0;
        block.fusedProduct = 0;
        block.blockFormer = 0;
        block.blockNum = 1;
        block.blockTail = 0;
    } else {
        split = ComputeUbSplit(collapse.dims, perBufElems);
        block = ComputeBlockSplit(collapse.dims, split, coreNum);
        // core-feeding: shrink maxElemNum by CACHE_LINE each round to feed more cores
        const int64_t shrinkStep = SHRINK_STEP_BYTES / static_cast<int64_t>(sizeof(float));
        const int64_t floorElems = std::max(MIN_TILE_ELEMS, alignElems);
        int64_t rounds = 0;
        while (block.blockNum < coreNum && perBufElems - shrinkStep >= floorElems && rounds < MAX_SHRINK_ROUNDS) {
            const int64_t candidate = Ops::Base::FloorAlign(perBufElems - shrinkStep, alignElems);
            if (candidate >= perBufElems) {
                break;
            }
            perBufElems = candidate;
            split = ComputeUbSplit(collapse.dims, perBufElems);
            block = ComputeBlockSplit(collapse.dims, split, coreNum);
            rounds++;
        }
    }

    // brcMode decision chain (DAV_3510), only for the multi-dim branch
    uint32_t brcMode = WTS_ARQ_BRC_NONE;
    if (!isOneDim && collapse.hasBroadcast) {
        const int64_t last = shapeLen - 1;
        const bool lastAxisNotBroadcast = (collapse.minStrides[last] != 0);
        bool hasOuterBroadcast = false;
        for (int64_t d = 0; d < last; d++) {
            if (collapse.minStrides[d] == 0) {
                hasOuterBroadcast = true;
                break;
            }
        }
        const int64_t dcacheHalf = static_cast<int64_t>(Ops::Base::GetNddmaDcacheSize(context)) / 2;
        if (lastAxisNotBroadcast && hasOuterBroadcast && collapse.dims[last] * typeSize >= dcacheHalf) {
            brcMode = WTS_ARQ_BRC_UB; // priority 2: NLast with large last axis
        } else if (typeSize == 2 && (collapse.dims[last] * typeSize) % 32 == 0) {
            brcMode = WTS_ARQ_BRC_UB; // priority 3: fp16 with 32B-aligned last axis
        } else {
            brcMode = WTS_ARQ_BRC_NDDMA; // priority 4: default NDDMA
        }
    }

    uint32_t schMode = 0;
    if (brcMode == WTS_ARQ_BRC_NDDMA) {
        schMode = (shapeLen - split.axis <= NDDMA_MAX_DIMS) ? WTS_ARQ_SCH_WITHOUT_LOOP : WTS_ARQ_SCH_WITH_LOOP;
    }

    FillTilingData<R>(tiling, collapse, split, block, perBufElems, brcMode, schMode, offsetFlag, numBits);

    context->SetBlockDim(static_cast<uint32_t>(block.blockNum));
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(R));
    LogTilingData<R>(context->GetNodeName(), tiling);
    return ge::GRAPH_SUCCESS;
}

} // namespace

static ge::graphStatus TilingFuncWtsArq(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for WtsARQ arch35.");

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t typeSize = 0;
    int64_t numBits = 8;
    bool offsetFlag = false;
    OP_CHECK_IF(CheckDtypeAndAttrs(context, typeSize, numBits, offsetFlag) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckDtypeAndAttrs error"), return ge::GRAPH_FAILED);

    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<int64_t> outputShape;
    OP_CHECK_IF(GetInputShapes(context, inputShapes, outputShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetInputShapes error"), return ge::GRAPH_FAILED);

    int64_t wElems = 0;
    OP_CHECK_IF(CheckShapes(context, inputShapes, outputShape, wElems) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShapes error"), return ge::GRAPH_FAILED);

    // workspace: pure elementwise + copy-in broadcast, no GM buffer needed
    size_t* workspaces = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0U;

    const bool isEmpty = (wElems == 0);
    CollapseResult collapse = CollapseShape(inputShapes[INPUT_W_IDX], inputShapes[INPUT_W_MIN_IDX],
                                            inputShapes[INPUT_W_MAX_IDX]);
    const int64_t shapeLen = static_cast<int64_t>(collapse.dims.size());

    if (shapeLen <= WTS_ARQ_RANK_4) {
        return DoTilingAndSet<WTS_ARQ_RANK_4>(context, collapse, ubSize, coreNum, typeSize, offsetFlag, numBits,
                                              isEmpty);
    }
    return DoTilingAndSet<WTS_ARQ_RANK_8>(context, collapse, ubSize, coreNum, typeSize, offsetFlag, numBits, isEmpty);
}

static ge::graphStatus TilingParseForWtsArq([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct WtsArqCompileInfo {};

IMPL_OP_OPTILING(WtsARQ).Tiling(TilingFuncWtsArq).TilingParse<WtsArqCompileInfo>(TilingParseForWtsArq);

} // namespace optiling
