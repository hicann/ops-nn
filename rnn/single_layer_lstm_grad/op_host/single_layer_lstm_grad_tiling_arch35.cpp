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
 * \file single_layer_lstm_grad_tiling_arch35.cpp
 * \brief regbase (Ascend950) small-shape tiling for SingleLayerLstmGrad (tiling key 20000).
 *
 * Path S = AIV-only zero-sync kernel, chosen when the whole recurrence working set fits in one
 * AIV's UB (exact same layout formula the kernel uses) and there is no seq_length. Workspace
 * is 0 for this path. Anything else falls back to the legacy pipeline untouched.
 */

#include <cstring>
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "single_layer_lstm_grad_tiling_arch35.h"
#include "../op_kernel/arch35/single_layer_lstm_grad_regbase_tiling_data.h"

namespace optiling {

namespace {
// development kill switch: set false to force every shape onto the legacy path
constexpr bool ENABLE_REGBASE_SMALL_PATH = true;

constexpr int64_t SMALL_CHUNK_COLS = 64;
constexpr int64_t SMALL_M_BLOCK = 64;
constexpr int64_t SMALL_MAX_CORES = 16;
constexpr int64_t SMALL_UB_RESERVE = 16 * 1024; // TPipe meta + safety margin
constexpr int64_t SMALL_MAX_HIDDEN = 64;        // one vector register row per gate

constexpr size_t IDX_X = 0;
constexpr size_t IDX_W = 1;
constexpr size_t IDX_BIAS = 2;
constexpr size_t IDX_INIT_H = 4;
constexpr size_t IDX_INIT_C = 5;
constexpr size_t IDX_H = 6;
constexpr size_t IDX_C = 7;
constexpr size_t IDX_DY = 8;
constexpr size_t IDX_DH = 9;
constexpr size_t IDX_DC = 10;
constexpr size_t IDX_I = 11;
constexpr size_t IDX_TANHC = 15;
constexpr size_t IDX_SEQ = 16;
constexpr size_t ATTR_DIRECTION = 0;
constexpr size_t ATTR_GATE_ORDER = 1;
constexpr size_t DIM_NUM_3 = 3;

bool InputShapeIs2D(const gert::TilingContext* context, size_t idx, int64_t d0, int64_t d1)
{
    auto s = context->GetInputShape(idx);
    if (s == nullptr) {
        return false;
    }
    const gert::Shape& shape = s->GetStorageShape();
    return shape.GetDimNum() == 2 && shape.GetDim(0) == d0 && shape.GetDim(1) == d1;
}

bool InputShapeIs3D(const gert::TilingContext* context, size_t idx, int64_t d0, int64_t d1, int64_t d2)
{
    auto s = context->GetInputShape(idx);
    if (s == nullptr) {
        return false;
    }
    const gert::Shape& shape = s->GetStorageShape();
    return shape.GetDimNum() == DIM_NUM_3 && shape.GetDim(0) == d0 && shape.GetDim(1) == d1 && shape.GetDim(2) == d2;
}

// eligible shapes bypass the legacy validation, so they must be fully re-validated here
bool ValidateSmallPathShapes(const gert::TilingContext* context, int64_t timeStep, int64_t batch, int64_t inputSize,
                             int64_t hidden, bool isBias)
{
    const int64_t gates = 4 * hidden;
    if (!InputShapeIs2D(context, IDX_W, gates, inputSize + hidden)) {
        return false;
    }
    if (!InputShapeIs3D(context, IDX_INIT_C, 1, batch, hidden) || !InputShapeIs3D(context, IDX_DH, 1, batch, hidden) ||
        !InputShapeIs3D(context, IDX_DC, 1, batch, hidden)) {
        return false;
    }
    for (size_t idx = IDX_I; idx <= IDX_TANHC; ++idx) {
        if (!InputShapeIs3D(context, idx, timeStep, batch, hidden)) {
            return false;
        }
    }
    if (!InputShapeIs3D(context, IDX_H, timeStep, batch, hidden) ||
        !InputShapeIs3D(context, IDX_C, timeStep, batch, hidden) ||
        !InputShapeIs3D(context, IDX_DY, timeStep, batch, hidden)) {
        return false;
    }
    if (isBias) {
        auto s = context->GetOptionalInputShape(IDX_BIAS);
        if (s == nullptr || s->GetStorageShape().GetDimNum() != 1 || s->GetStorageShape().GetDim(0) != gates) {
            return false;
        }
    }
    auto wDesc = context->GetInputDesc(IDX_W);
    auto xDesc = context->GetInputDesc(IDX_X);
    if (wDesc == nullptr || xDesc == nullptr || wDesc->GetDataType() != xDesc->GetDataType()) {
        return false;
    }
    return true;
}
} // namespace

ge::graphStatus TilingSingleLayerLstmGrad4RegbaseSmall(gert::TilingContext* context, bool& handled)
{
    handled = false;
    if (!ENABLE_REGBASE_SMALL_PATH) {
        return ge::GRAPH_SUCCESS;
    }
    auto xDesc = context->GetInputDesc(IDX_X);
    auto xShapePtr = context->GetInputShape(IDX_X);
    auto initHShapePtr = context->GetInputShape(IDX_INIT_H);
    if (xDesc == nullptr || xShapePtr == nullptr || initHShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS; // legacy path reports the error
    }
    ge::DataType dtype = xDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t dtypeSize = (dtype == ge::DT_FLOAT) ? 4 : 2;

    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    const gert::Shape& initHShape = initHShapePtr->GetStorageShape();
    if (xShape.GetDimNum() != DIM_NUM_3 || initHShape.GetDimNum() != DIM_NUM_3) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t timeStep = xShape.GetDim(0);
    const int64_t batch = xShape.GetDim(1);
    const int64_t inputSize = xShape.GetDim(2);
    const int64_t hidden = initHShape.GetDim(2);
    if (timeStep <= 0 || batch <= 0 || inputSize <= 0 || hidden <= 0 || hidden > SMALL_MAX_HIDDEN) {
        return ge::GRAPH_SUCCESS;
    }

    // optional inputs (mirrors legacy GetOptionalInputFlags: 0-dim placeholder == absent)
    auto seqDesc = context->GetOptionalInputDesc(IDX_SEQ);
    auto seqShape = context->GetOptionalInputShape(IDX_SEQ);
    const bool isSeqLength = (seqDesc != nullptr && seqShape != nullptr &&
                              seqShape->GetStorageShape().GetDimNum() != 0);
    if (isSeqLength) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasDesc = context->GetOptionalInputDesc(IDX_BIAS);
    auto biasShape = context->GetOptionalInputShape(IDX_BIAS);
    const bool isBias = (biasDesc != nullptr && biasShape != nullptr && biasShape->GetStorageShape().GetDimNum() != 0);

    // attrs (invalid values -> legacy path, which validates and reports)
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const char* direction = attrs->GetAttrPointer<char>(ATTR_DIRECTION);
    const char* gateOrder = attrs->GetAttrPointer<char>(ATTR_GATE_ORDER);
    if (direction == nullptr || gateOrder == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t directionVal;
    if (strcmp(direction, "UNIDIRECTIONAL") == 0) {
        directionVal = 0;
    } else if (strcmp(direction, "REDIRECTIONAL") == 0) {
        directionVal = 1;
    } else {
        return ge::GRAPH_SUCCESS;
    }
    int64_t gateOrderVal;
    if (strcmp(gateOrder, "ijfo") == 0) {
        gateOrderVal = 0;
    } else if (strcmp(gateOrder, "ifjo") == 0) {
        gateOrderVal = 1;
    } else {
        return ge::GRAPH_SUCCESS;
    }

    if (!ValidateSmallPathShapes(context, timeStep, batch, inputSize, hidden, isBias)) {
        return ge::GRAPH_SUCCESS; // legacy path validates and reports
    }

    // UB budget with the exact kernel layout formula
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const int64_t aivNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    if (ubSize == 0 || aivNum <= 0) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t mAll = timeStep * batch;
    const int64_t mBlock = (mAll < SMALL_M_BLOCK) ? mAll : SMALL_M_BLOCK;
    LstmGradRegbase::LstmGradRegbaseSmallUbLayout layout;
    layout.Fill(timeStep, batch, hidden, SMALL_CHUNK_COLS, mBlock, dtypeSize);
    if (layout.totalBytes > static_cast<int64_t>(ubSize) - SMALL_UB_RESERVE) {
        OP_LOGI(context->GetNodeName(),
                "SingleLayerLstmGrad regbase small path skipped: need %ld bytes UB, budget %ld.", layout.totalBytes,
                static_cast<int64_t>(ubSize) - SMALL_UB_RESERVE);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t numIChunks = LstmGradRegbase::CeilDivI64(inputSize, SMALL_CHUNK_COLS);
    int64_t usedCores = numIChunks + 1;
    usedCores = (usedCores > SMALL_MAX_CORES) ? SMALL_MAX_CORES : usedCores;
    usedCores = (usedCores > aivNum) ? aivNum : usedCores;
    usedCores = (usedCores < 1) ? 1 : usedCores;

    auto tilingData = context->GetTilingData<LstmGradRegbaseSmallTilingData>();
    if (tilingData == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    tilingData->timeStep = timeStep;
    tilingData->batch = batch;
    tilingData->inputSize = inputSize;
    tilingData->hiddenSize = hidden;
    tilingData->isBias = isBias ? 1 : 0;
    tilingData->direction = directionVal;
    tilingData->gateOrder = gateOrderVal;
    tilingData->usedCores = usedCores;
    tilingData->chunkCols = SMALL_CHUNK_COLS;
    tilingData->mBlock = mBlock;
    tilingData->numIChunks = numIChunks;
    tilingData->reserved0 = 0;

    context->SetTilingKey(LSTM_GRAD_TILING_KEY_REGBASE_SMALL);
    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = 0;

    OP_LOGI(context->GetNodeName(),
            "SingleLayerLstmGrad regbase small path: T=%ld B=%ld I=%ld H=%ld bias=%ld dir=%ld order=%ld cores=%ld "
            "ubBytes=%ld.",
            timeStep, batch, inputSize, hidden, tilingData->isBias, directionVal, gateOrderVal, usedCores,
            layout.totalBytes);
    handled = true;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
