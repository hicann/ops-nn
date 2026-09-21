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
 * \file cla_gate_quant_tiling_arch35.cpp
 * \brief Tiling implementation for ClaGateQuant
 */

#include "activation/cla_gate_quant/op_host/arch35/cla_gate_quant_tiling_arch35.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/op_host/util/math_util.h"
#include "activation/cla_gate_quant/op_kernel/arch35/cla_gate_quant_tiling_key.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace ClaGateQuantOp;

namespace optiling {
namespace {
// Attr order must match op_def:
// dst_type(0), round_mode(1), scale_alg(2), input_attn_layout(3), dual_axis_flag(4)
constexpr int64_t INDEX_ATTR_DST_TYPE = 0;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 2;
constexpr int64_t INDEX_ATTR_INPUT_ATTN_LAYOUT = 3;
constexpr int64_t INDEX_ATTR_DUAL_AXIS_FLAG = 4;

constexpr int64_t INPUT_GLOBAL_ATTN = 0;
constexpr int64_t INPUT_LOCAL_ATTN = 1;
constexpr int64_t INPUT_GLOBAL_GATE = 2;
constexpr int64_t INPUT_LOCAL_GATE = 3;
constexpr int64_t OUTPUT_ROW_DATA = 0;
constexpr int64_t OUTPUT_ROW_SCALE = 1;
constexpr int64_t OUTPUT_COL_DATA = 2;
constexpr int64_t OUTPUT_COL_SCALE = 3;

constexpr int64_t COL_TILE_SIZE = 256;
constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr int64_t SCALE_GROUPS_PER_PACK = 2;
constexpr int64_t SCALE_PACK_SPAN = MX_BLOCK_SIZE * SCALE_GROUPS_PER_PACK;
constexpr int64_t DUAL_AXIS_ROW_TILE_SIZE = SCALE_PACK_SPAN;
constexpr int64_t MIN_HEAD_COUNT = 1;
constexpr int64_t MAX_HEAD_COUNT = 128;
constexpr int64_t UB_BLOCK_BYTES = 32;
constexpr int64_t RESERVED_UB_BYTES = 2 * 1024;
constexpr int64_t DB_BUFFER_COUNT = 2;
constexpr int64_t BASE_INPUT_BYTES_PER_CORE = 4 * 1024;
constexpr int64_t INPUT_BYTES_PER_CORE_HINT = BASE_INPUT_BYTES_PER_CORE * DB_BUFFER_COUNT;
constexpr int64_t FP32_VF_ELEMENTS = 64;
constexpr int64_t INPUT_ELEMENT_BYTES = 2;

int64_t AlignUbBlock(int64_t bytes) { return Ops::Base::CeilDiv(bytes, UB_BLOCK_BYTES) * UB_BLOCK_BYTES; }

// Mirrors the buffers allocated by the single-axis branch of CommonBase::Init.
// colScaleReciprocalBuf_ is reused as batch-sized compact max-exponent scratch.
int64_t CalcTailAxisUbBytes(const ClaGateQuantTilingParam& params, int64_t batchSegmentCapacity)
{
    int64_t headsPerTile = COL_TILE_SIZE / params.headDim;
    int64_t halfBufferElems = batchSegmentCapacity * COL_TILE_SIZE;
    int64_t inputFrameBytes = halfBufferElems * INPUT_ELEMENT_BYTES * 2; // global + local
    // Kernel gives each global/local gate (and sigmoid) side a complete FP32
    // VF span.  The last sigmoid iteration stores one full VF without a tail
    // mask, so accounting only the logical gate count would both misalign the
    // second side and underestimate its physical scratch footprint.
    int64_t logicalGateCount = batchSegmentCapacity * headsPerTile;
    int64_t gateSideElems = Ops::Base::CeilDiv(logicalGateCount, FP32_VF_ELEMENTS) * FP32_VF_ELEMENTS;
    int64_t gateFrameBytes = gateSideElems * INPUT_ELEMENT_BYTES * 2;
    int64_t mergedBytes = halfBufferElems * INPUT_ELEMENT_BYTES;
    int64_t sigmoidBytes = gateSideElems * static_cast<int64_t>(sizeof(float)) * 2;
    int64_t rowDataFrameBytes = halfBufferElems; // Byte-addressed storage for FP8 and packed FP4.
    int64_t rowScaleFrameBytes = batchSegmentCapacity * UB_BLOCK_BYTES;
    int64_t rowReciprocalBytes = batchSegmentCapacity * UB_BLOCK_BYTES;
    int64_t fixedColReciprocalBytes = COL_TILE_SIZE * 2 * INPUT_ELEMENT_BYTES;
    constexpr int64_t B16_VF_ELEMENTS = 128;
    int64_t compactScaleCount = batchSegmentCapacity * (COL_TILE_SIZE / MX_BLOCK_SIZE);
    int64_t compactMaxExpBytes = Ops::Base::CeilDiv(compactScaleCount, B16_VF_ELEMENTS) * B16_VF_ELEMENTS *
                                 static_cast<int64_t>(sizeof(uint16_t));
    int64_t colReciprocalBytes = std::max(fixedColReciprocalBytes, compactMaxExpBytes);

    return DB_BUFFER_COUNT * AlignUbBlock(inputFrameBytes) + DB_BUFFER_COUNT * AlignUbBlock(gateFrameBytes) +
           AlignUbBlock(mergedBytes) + AlignUbBlock(sigmoidBytes) + DB_BUFFER_COUNT * AlignUbBlock(rowDataFrameBytes) +
           DB_BUFFER_COUNT * AlignUbBlock(rowScaleFrameBytes) + AlignUbBlock(rowReciprocalBytes) +
           AlignUbBlock(colReciprocalBytes);
}

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN,
                                                    ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

bool IsValidTNDShape(const gert::Shape& shape)
{
    return shape.GetDimNum() == 3 && shape.GetDim(0) > 0 && shape.GetDim(1) >= MIN_HEAD_COUNT &&
           shape.GetDim(1) <= MAX_HEAD_COUNT && (shape.GetDim(2) == 128 || shape.GetDim(2) == 256);
}

bool IsValidGateShape(const gert::Shape& shape, int64_t rowCount, int64_t headCount)
{
    return shape.GetDimNum() == 2 && shape.GetDim(0) == rowCount && shape.GetDim(1) == headCount;
}

bool ValidateInputShape(const gert::Shape& globalAttn, const gert::Shape& localAttn, const gert::Shape& globalGate,
                        const gert::Shape& localGate)
{
    return IsValidTNDShape(globalAttn) && localAttn == globalAttn &&
           IsValidGateShape(globalGate, globalAttn.GetDim(0), globalAttn.GetDim(1)) && localGate == globalGate;
}

ge::graphStatus CheckOutputShapes(const gert::Shape& globalAttn, const gert::Shape& rowData,
                                  const gert::Shape& rowScale, const gert::Shape& colData, const gert::Shape& colScale)
{
    int64_t rowCount = globalAttn.GetDim(0);
    int64_t rowLength = globalAttn.GetDim(1) * globalAttn.GetDim(2);
    if (rowData.GetDimNum() != 2 || rowData.GetDim(0) != rowCount || rowData.GetDim(1) != rowLength) {
        return ge::GRAPH_FAILED;
    }
    if (rowScale.GetDimNum() != 3 || rowScale.GetDim(0) != rowCount ||
        rowScale.GetDim(1) != Ops::Base::CeilDiv(rowLength, SCALE_PACK_SPAN) ||
        rowScale.GetDim(2) != SCALE_GROUPS_PER_PACK) {
        return ge::GRAPH_FAILED;
    }
    bool colEmpty = (colData.GetDimNum() >= 1 && colData.GetDim(0) == 0) &&
                    (colScale.GetDimNum() >= 1 && colScale.GetDim(0) == 0);
    if (!colEmpty) {
        if (colData.GetDimNum() != 2 || colData.GetDim(0) != rowCount || colData.GetDim(1) != rowLength) {
            return ge::GRAPH_FAILED;
        }
        if (colScale.GetDimNum() != 3 || colScale.GetDim(0) != Ops::Base::CeilDiv(rowCount, SCALE_PACK_SPAN) ||
            colScale.GetDim(1) != rowLength || colScale.GetDim(2) != SCALE_GROUPS_PER_PACK) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

RoundModeList ClaGateQuantTiling::GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    }
    if (roundMode == "round") {
        return RoundModeList::MODE_ROUND;
    }
    if (roundMode == "floor") {
        return RoundModeList::MODE_FLOOR;
    }
    return RoundModeList::MODE_UNDEFINED;
}

ge::graphStatus ClaGateQuantTiling::CheckInputOutput()
{
    auto globalAttnShapePtr = context_->GetInputShape(INPUT_GLOBAL_ATTN);
    auto localAttnShapePtr = context_->GetInputShape(INPUT_LOCAL_ATTN);
    auto globalGateShapePtr = context_->GetInputShape(INPUT_GLOBAL_GATE);
    auto localGateShapePtr = context_->GetInputShape(INPUT_LOCAL_GATE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, globalAttnShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, localAttnShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, globalGateShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, localGateShapePtr);
    const auto& globalAttnShape = globalAttnShapePtr->GetStorageShape();
    const auto& localAttnShape = localAttnShapePtr->GetStorageShape();
    const auto& globalGateShape = globalGateShapePtr->GetStorageShape();
    const auto& localGateShape = localGateShapePtr->GetStorageShape();
    if (!ValidateInputShape(globalAttnShape, localAttnShape, globalGateShape, localGateShape)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "inputs",
            "global_attn/local_attn must be [T,N,D], gate inputs must be [T,N], N must be in [1,128], "
            "and D must be 128 or 256");
        return ge::GRAPH_FAILED;
    }

    auto globalAttnDesc = context_->GetInputDesc(INPUT_GLOBAL_ATTN);
    auto localAttnDesc = context_->GetInputDesc(INPUT_LOCAL_ATTN);
    auto globalGateDesc = context_->GetInputDesc(INPUT_GLOBAL_GATE);
    auto localGateDesc = context_->GetInputDesc(INPUT_LOCAL_GATE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, globalAttnDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, localAttnDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, globalGateDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, localGateDesc);
    auto inputDtype = globalAttnDesc->GetDataType();
    if (INPUT_SUPPORT_DTYPE_SET.count(inputDtype) == 0 || localAttnDesc->GetDataType() != inputDtype ||
        globalGateDesc->GetDataType() != inputDtype || localGateDesc->GetDataType() != inputDtype) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "input dtypes",
                                                 "all four inputs must have the same FLOAT16 or BFLOAT16 dtype");
        return ge::GRAPH_FAILED;
    }

    auto rowDataDesc = context_->GetOutputDesc(OUTPUT_ROW_DATA);
    auto rowScaleDesc = context_->GetOutputDesc(OUTPUT_ROW_SCALE);
    auto colDataDesc = context_->GetOutputDesc(OUTPUT_COL_DATA);
    auto colScaleDesc = context_->GetOutputDesc(OUTPUT_COL_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rowDataDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rowScaleDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, colDataDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, colScaleDesc);
    if (Y_SUPPORT_DTYPE_SET.count(rowDataDesc->GetDataType()) == 0 ||
        colDataDesc->GetDataType() != rowDataDesc->GetDataType() || rowScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0 ||
        colScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "output dtypes",
            "row_data and col_data must use the requested FP4/FP8 dtype, and scale outputs must use "
            "FLOAT8_E8M0");
        return ge::GRAPH_FAILED;
    }

    auto rowDataShapePtr = context_->GetOutputShape(OUTPUT_ROW_DATA);
    auto rowScaleShapePtr = context_->GetOutputShape(OUTPUT_ROW_SCALE);
    auto colDataShapePtr = context_->GetOutputShape(OUTPUT_COL_DATA);
    auto colScaleShapePtr = context_->GetOutputShape(OUTPUT_COL_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rowDataShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rowScaleShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, colDataShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, colScaleShapePtr);
    auto rowDataShape = rowDataShapePtr->GetStorageShape();
    auto rowScaleShape = rowScaleShapePtr->GetStorageShape();
    auto colDataShape = colDataShapePtr->GetStorageShape();
    auto colScaleShape = colScaleShapePtr->GetStorageShape();
    if (CheckOutputShapes(globalAttnShape, rowDataShape, rowScaleShape, colDataShape, colScaleShape) !=
        ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "output shapes",
                                                 "output shapes do not match the input shape and axis mode");
        return ge::GRAPH_FAILED;
    }

    tilingParams_.yDtype = rowDataDesc->GetDataType();
    tilingParams_.rowCount = globalAttnShape.GetDim(0);
    tilingParams_.headCount = globalAttnShape.GetDim(1);
    tilingParams_.headDim = globalAttnShape.GetDim(2);
    tilingParams_.rowLength = tilingParams_.headCount * tilingParams_.headDim;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClaGateQuantTiling::GetPlatformInfo()
{
    auto compileInfo = context_->GetCompileInfo<ClaGateQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    tilingParams_.totalCoreNum = compileInfo->coreNum;
    tilingParams_.ubSize = compileInfo->ubSize;
    if (tilingParams_.totalCoreNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "coreNum",
                                              std::to_string(tilingParams_.totalCoreNum).c_str(),
                                              "The value of coreNum must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingParams_.ubSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize",
                                              std::to_string(tilingParams_.ubSize).c_str(),
                                              "The value of ubSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClaGateQuantTiling::GetAndCheckAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto roundModePtr = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, roundModePtr);
    string roundModeStr = roundModePtr;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    if (roundMode == RoundModeList::MODE_UNDEFINED) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "round_mode", roundModeStr.c_str(), "rint, floor or round");
        return ge::GRAPH_FAILED;
    }

    auto scaleAlgPtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleAlgPtr);
    auto dstTypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstTypePtr);
    auto dualAxisFlagPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_DUAL_AXIS_FLAG);
    auto inputAttnLayoutPtr = attrs->GetAttrPointer<char>(INDEX_ATTR_INPUT_ATTN_LAYOUT);
    tilingParams_.roundMode = static_cast<int64_t>(roundMode);
    tilingParams_.scaleAlg = *scaleAlgPtr;
    tilingParams_.dstType = *dstTypePtr;
    tilingParams_.dualAxisFlag = (dualAxisFlagPtr != nullptr && *dualAxisFlagPtr) ? 1 : 0;
    string inputAttnLayout = (inputAttnLayoutPtr != nullptr) ? string(inputAttnLayoutPtr) : string("TND");
    if (inputAttnLayout != "TND") {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "input_attn_layout", inputAttnLayout.c_str(), "TND");
        return ge::GRAPH_FAILED;
    }

    if (tilingParams_.scaleAlg != 0 && tilingParams_.scaleAlg != 1) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "scale_alg", std::to_string(tilingParams_.scaleAlg).c_str(),
                                  "0 or 1");
        return ge::GRAPH_FAILED;
    }
    if (Y_SUPPORT_DTYPE_SET.count(tilingParams_.yDtype) == 0 ||
        static_cast<int64_t>(tilingParams_.yDtype) != tilingParams_.dstType) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dst_type",
                                              std::to_string(tilingParams_.dstType).c_str(),
                                              "dst_type must match the output data dtype");
        return ge::GRAPH_FAILED;
    }
    // FP8 supports round-to-nearest-even only.
    if (Y_SUPPORT_DTYPE_FP8_SET.count(tilingParams_.yDtype) != 0 && roundMode != RoundModeList::MODE_RINT) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "round_mode", roundModeStr.c_str(), "rint");
        return ge::GRAPH_FAILED;
    }
    // FP4 supports scale algorithm 0 only.
    if (Y_SUPPORT_DTYPE_FP4_SET.count(tilingParams_.yDtype) != 0 && tilingParams_.scaleAlg != 0) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "scale_alg", std::to_string(tilingParams_.scaleAlg).c_str(),
                                  "0 for FP4 output");
        return ge::GRAPH_FAILED;
    }
    // FP4 packing requires K = N * D to be divisible by four.
    if (Y_SUPPORT_DTYPE_FP4_SET.count(tilingParams_.yDtype) != 0 && (tilingParams_.rowLength % 4) != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "K",
                                              std::to_string(tilingParams_.rowLength).c_str(),
                                              "K = N * D must be divisible by four for FP4 output");
        return ge::GRAPH_FAILED;
    }

    if (roundMode == RoundModeList::MODE_RINT) {
        tilingKeyRound_ = TPL_RINT;
    } else if (roundMode == RoundModeList::MODE_ROUND) {
        tilingKeyRound_ = TPL_ROUND;
    } else {
        tilingKeyRound_ = TPL_FLOOR;
    }
    tilingKeyDualAxisFlag_ = (tilingParams_.dualAxisFlag != 0) ? TPL_DUAL_AXIS : TPL_SINGLE_AXIS;
    return ge::GRAPH_SUCCESS;
}

void ClaGateQuantTiling::CalcTailAxisTiling()
{
    int64_t rowCount = tilingParams_.rowCount;
    int64_t rowLength = tilingParams_.rowLength;

    // Scale core use with input size, using INPUT_BYTES_PER_CORE_HINT as the
    // minimum work quantum assigned to each core.
    int64_t inputBytes = rowCount * rowLength * INPUT_ELEMENT_BYTES;
    int64_t coreHint = inputBytes / INPUT_BYTES_PER_CORE_HINT;
    if (coreHint < 1) {
        coreHint = 1;
    }
    tilingParams_.usedCoreNum = std::min(tilingParams_.totalCoreNum, coreHint);

    // A single-axis tile is a 256-element segment of the flattened [T, K]
    // stream. K is headDim-aligned, so activation and gate streams remain
    // contiguous when a segment crosses a token boundary.
    int64_t totalElements = rowCount * rowLength;
    int64_t tileWidth = COL_TILE_SIZE;
    int64_t streamTileNum = Ops::Base::CeilDiv(totalElements, tileWidth);
    tilingParams_.streamTileNum = streamTileNum;
    tilingParams_.streamTailSize = totalElements - (streamTileNum - 1) * tileWidth;
    if (tilingParams_.streamTailSize <= 0) {
        tilingParams_.streamTailSize = tileWidth;
    }

    // Derive the maximum resident 1x256 tile count from the available UB.
    int64_t totalTiles = tilingParams_.streamTileNum;
    int64_t maxTilesPerCore = Ops::Base::CeilDiv(totalTiles, tilingParams_.usedCoreNum);
    int64_t availableUb = std::max<int64_t>(0, tilingParams_.ubSize - RESERVED_UB_BYTES);
    int64_t ubBatchCapacity = 1;
    for (int64_t candidate = 1; candidate <= maxTilesPerCore; ++candidate) {
        if (CalcTailAxisUbBytes(tilingParams_, candidate) > availableUb) {
            break;
        }
        ubBatchCapacity = candidate;
    }
    int64_t selectedBatchCapacity = std::min(maxTilesPerCore, ubBatchCapacity);
    // Number of gate scalars carried by one full-width 1x256 tile.
    int64_t headsPerTile = COL_TILE_SIZE / tilingParams_.headDim;
    // Align repeated batches to a complete 64-scalar sigmoid VF. If all work
    // fits in one batch, retain the exact segment count.
    int64_t sigmoidTileQuantum = FP32_VF_ELEMENTS / headsPerTile;
    if (maxTilesPerCore > ubBatchCapacity && ubBatchCapacity >= sigmoidTileQuantum) {
        selectedBatchCapacity = (ubBatchCapacity / sigmoidTileQuantum) * sigmoidTileQuantum;
    }
    tilingParams_.batchSegmentCapacity = std::max<int64_t>(1, selectedBatchCapacity);
}

ge::graphStatus ClaGateQuantTiling::ComputeTiling()
{
    int64_t totalTaskCount;
    if (tilingParams_.dualAxisFlag == 0) {
        CalcTailAxisTiling();
        totalTaskCount = tilingParams_.streamTileNum;
    } else {
        tilingParams_.colTileNum = Ops::Base::CeilDiv(tilingParams_.rowLength, COL_TILE_SIZE);
        int64_t colTailSize = tilingParams_.rowLength % COL_TILE_SIZE;
        tilingParams_.colTailSize = colTailSize == 0 ? COL_TILE_SIZE : colTailSize;
        int64_t rowTileCount = Ops::Base::CeilDiv(tilingParams_.rowCount, DUAL_AXIS_ROW_TILE_SIZE);
        totalTaskCount = rowTileCount * tilingParams_.colTileNum;
        tilingParams_.usedCoreNum = std::min(tilingParams_.totalCoreNum, totalTaskCount);
    }
    tilingParams_.baseTaskCount = totalTaskCount / tilingParams_.usedCoreNum;
    tilingParams_.extraTaskCoreCount = totalTaskCount % tilingParams_.usedCoreNum;
    return ge::GRAPH_SUCCESS;
}

void ClaGateQuantTiling::SetTilingKeyAndCore()
{
    context_->SetBlockDim(tilingParams_.usedCoreNum);
    context_->SetTilingKey(GET_TPL_TILING_KEY(tilingKeyDualAxisFlag_, tilingKeyRound_, tilingParams_.scaleAlg));
}

ge::graphStatus ClaGateQuantTiling::SetTilingData()
{
    tilingData_ = context_->GetTilingData<ClaGateQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
    *tilingData_ = {};
    tilingData_->usedCoreNum = tilingParams_.usedCoreNum;
    tilingData_->rowCount = tilingParams_.rowCount;
    tilingData_->rowLength = tilingParams_.rowLength;
    tilingData_->colTileNum = tilingParams_.colTileNum;
    tilingData_->colTailSize = tilingParams_.colTailSize;
    tilingData_->headCount = tilingParams_.headCount;
    tilingData_->headDim = tilingParams_.headDim;
    tilingData_->streamTailSize = tilingParams_.streamTailSize;
    tilingData_->batchSegmentCapacity = tilingParams_.batchSegmentCapacity;
    tilingData_->baseTaskCount = tilingParams_.baseTaskCount;
    tilingData_->extraTaskCoreCount = tilingParams_.extraTaskCoreCount;

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams_.workspaceSize;
    return ge::GRAPH_SUCCESS;
}

void ClaGateQuantTiling::PrintTilingData() const
{
    OP_LOGD(context_->GetNodeName(),
            "ClaGateQuant tiling: usedCoreNum=%ld, dstType=%ld, rowCount=%ld, rowLength=%ld, dualAxisFlag=%ld, "
            "colTileNum=%ld, colTailSize=%ld, streamTailSize=%ld, baseTaskCount=%ld, "
            "extraTaskCoreCount=%ld, batchSegmentCapacity=%ld",
            tilingParams_.usedCoreNum, tilingParams_.dstType, tilingParams_.rowCount, tilingParams_.rowLength,
            tilingParams_.dualAxisFlag, tilingParams_.colTileNum, tilingParams_.colTailSize,
            tilingParams_.streamTailSize, tilingParams_.baseTaskCount, tilingParams_.extraTaskCoreCount,
            tilingParams_.batchSegmentCapacity);
}

ge::graphStatus ClaGateQuantTiling::DoTiling()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_);
    auto status = CheckInputOutput();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = GetPlatformInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = GetAndCheckAttrs();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = ComputeTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    SetTilingKeyAndCore();
    status = SetTilingData();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForClaGateQuant(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    ClaGateQuantTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForClaGateQuant(gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto compileInfo = context->GetCompiledInfo<ClaGateQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    if (compileInfo->coreNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum",
                                              std::to_string(compileInfo->coreNum).c_str(),
                                              "The value of coreNum must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (compileInfo->ubSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize",
                                              std::to_string(compileInfo->ubSize).c_str(),
                                              "The value of ubSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ClaGateQuant)
    .Tiling(TilingForClaGateQuant)
    .TilingParse<ClaGateQuantCompileInfo>(TilingPrepareForClaGateQuant);
} // namespace optiling
