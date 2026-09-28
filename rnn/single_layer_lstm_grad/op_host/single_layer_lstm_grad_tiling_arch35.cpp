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
 * includes the framework-reserved prefix and, for narrow weights, FP32 dw/db accumulators.
 * Narrow calls that this planner declines are rejected rather than entering the FP32 legacy path.
 */

#include <cstring>
#include <limits>
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
constexpr int64_t SMALL_MIN_CHUNK_COLS = 8; // narrowest column chunk the search will accept
constexpr int64_t SMALL_TB_WALK_LIMIT = 8;  // above this the time search jumps instead of stepping
constexpr int64_t SMALL_M_BLOCK = 64;
constexpr int64_t SMALL_MAX_CORES = 16;
constexpr int64_t SMALL_UB_RESERVE = 16 * 1024; // TPipe meta + safety margin
constexpr int64_t BIAS_COMPONENT_COUNT = 2;
constexpr int64_t REPLAY_PLANES = 7; // i, j, f, o, tanh(c), c, h

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
constexpr size_t IDX_J = 12;
constexpr size_t IDX_F = 13;
constexpr size_t IDX_O = 14;
constexpr size_t IDX_TANHC = 15;
constexpr size_t IDX_SEQ = 16;
constexpr size_t ATTR_DIRECTION = 0;
constexpr size_t ATTR_GATE_ORDER = 1;
constexpr size_t RANK_2D = 2; // matrix inputs: w is [4H, I+H]
constexpr size_t RANK_3D = 3; // state/sequence inputs: [T, B, H] or [1, B, H]
constexpr size_t DIM_2 = 2;   // third dim index of a 3D shape

bool InputShapeIs2D(const gert::TilingContext* context, size_t idx, int64_t d0, int64_t d1)
{
    auto s = context->GetInputShape(idx);
    if (s == nullptr) {
        return false;
    }
    const gert::Shape& shape = s->GetStorageShape();
    return shape.GetDimNum() == RANK_2D && shape.GetDim(0) == d0 && shape.GetDim(1) == d1;
}

bool InputShapeIs3D(const gert::TilingContext* context, size_t idx, int64_t d0, int64_t d1, int64_t d2)
{
    auto s = context->GetInputShape(idx);
    if (s == nullptr) {
        return false;
    }
    const gert::Shape& shape = s->GetStorageShape();
    return shape.GetDimNum() == RANK_3D && shape.GetDim(0) == d0 && shape.GetDim(1) == d1 && shape.GetDim(DIM_2) == d2;
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
        if (s == nullptr || s->GetStorageShape().GetDimNum() != 1) {
            return false;
        }
        const auto size = s->GetStorageShape().GetDim(0);
        const bool narrow = context->GetInputDesc(IDX_W)->GetDataType() != ge::DT_FLOAT;
        if (size != gates && (!narrow || size != BIAS_COMPONENT_COUNT * gates)) {
            return false;
        }
    }
    auto wDesc = context->GetInputDesc(IDX_W);
    if (wDesc == nullptr) {
        return false;
    }
    if (isBias && context->GetOptionalInputDesc(IDX_BIAS)->GetDataType() != wDesc->GetDataType()) {
        return false;
    }
    // All floating IO follow w; widening occurs only inside the kernel.
    for (size_t idx :
         {IDX_X, IDX_INIT_H, IDX_INIT_C, IDX_H, IDX_C, IDX_DY, IDX_I, IDX_J, IDX_F, IDX_O, IDX_TANHC, IDX_DH, IDX_DC}) {
        auto desc = context->GetInputDesc(idx);
        if (desc == nullptr || desc->GetDataType() != wDesc->GetDataType()) {
            return false;
        }
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
    auto wDesc = context->GetInputDesc(IDX_W);
    auto xShapePtr = context->GetInputShape(IDX_X);
    auto initHShapePtr = context->GetInputShape(IDX_INIT_H);
    if (wDesc == nullptr || xShapePtr == nullptr || initHShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS; // legacy path reports the error
    }
    /* Use w to select the common floating-point IO dtype. */
    const ge::DataType dtype = wDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t dtypeSize = (dtype == ge::DT_FLOAT) ? 4 : 2;

    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    const gert::Shape& initHShape = initHShapePtr->GetStorageShape();
    if (xShape.GetDimNum() != RANK_3D || initHShape.GetDimNum() != RANK_3D) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t timeStep = xShape.GetDim(0);
    const int64_t batch = xShape.GetDim(1);
    const int64_t inputSize = xShape.GetDim(DIM_2);
    const int64_t hidden = initHShape.GetDim(DIM_2);
    // I=0 still has recurrent/state/bias gradients. With zero input chunks,
    // the single tail core runs the recurrence without reading x or writing dx.
    if (timeStep <= 0 || batch <= 0 || inputSize < 0 || hidden <= 0) {
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
    const int64_t budget = static_cast<int64_t>(ubSize) - SMALL_UB_RESERVE;

    /* FOUR KNOBS, AND THEY RELIEVE DIFFERENT REGIONS. The plan has a row-scaled part (the seven
     * saved planes, dy and dgate, all [tBlock*bBlock, H]), a batch-scaled part (the staged initial
     * state and the ping-pong) and a column-scaled part (wChunk, dwAcc and outStage, all
     * 4H x chunkCols). tBlock and bBlock both cut rows but only bBlock cuts the second group, and
     * chunkCols and gBlock are the levers on the third. Measured over the 482 non-empty backward
     * cases: column chunking with time blocking alone covers 404, adding batch blocking all 482,
     * gate blocking instead of batch blocking 408. Gate blocking adds nothing on that set -- it is
     * what extends the feasible hidden_size beyond it: T=1 B=1 input_size=8 hidden_size=2048 at
     * float16 does not fit without it and does with it.
     *
     * chunkCols descends first because a wider column chunk is fewer DMA bursts; gBlock next,
     * because a gate chunk only re-stages the weight tile; then bBlock, because a batch block
     * re-walks the whole sequence; tBlock innermost, because a time block only re-stages planes.
     * All four are correctness-neutral: the kernel closes a short tail block on every axis.
     *
     * THE ELEMENT WIDTH REACHES THE PLAN TOO: LstmGradRegbaseSmall<T> calls Fill with sizeof(T),
     * so a host that always passed 4 would size for fp32 and the half kernel would read a layout
     * it did not lay out. */
    LstmGradRegbase::LstmGradRegbaseSmallUbLayout layout;
    int64_t tBlock = 0;
    int64_t bBlock = 0;
    int64_t gBlock = 0;
    int64_t chunkCols = 0;
    for (int64_t cc = SMALL_CHUNK_COLS; cc >= SMALL_MIN_CHUNK_COLS && tBlock == 0; cc /= 2) {
        for (int64_t gb = hidden; gb >= 1 && tBlock == 0; gb = (gb > 1) ? ((gb + 1) / 2) : 0) {
            for (int64_t bb = batch; bb >= 1 && tBlock == 0; bb = (bb > 1) ? ((bb + 1) / 2) : 0) {
                for (int64_t tb = timeStep; tb >= 1; --tb) {
                    const int64_t rows = tb * bb;
                    const int64_t mb = (rows < SMALL_M_BLOCK) ? rows : SMALL_M_BLOCK;
                    layout.Fill(tb, bb, hidden, cc, mb, dtypeSize, gb);
                    if (layout.totalBytes <= budget) {
                        tBlock = tb;
                        bBlock = bb;
                        gBlock = gb;
                        chunkCols = cc;
                        break;
                    }
                    /* The plan is linear in tb, so jump to the largest tb the measured bytes-per-step
                     * allows rather than walking one step at a time over a long sequence. */
                    if (tb > SMALL_TB_WALK_LIMIT) {
                        const int64_t per = layout.totalBytes / tb;
                        const int64_t guess = (per > 0) ? (budget / per) : 1;
                        if (guess >= 1 && guess < tb - 1) {
                            tb = guess + 2; // the decrement then lands on guess+1 and the walk continues
                        }
                    }
                }
            }
        }
    }
    if (tBlock == 0) {
        layout.Fill(1, 1, hidden, SMALL_MIN_CHUNK_COLS, 1, dtypeSize, 1);
        OP_LOGI(context->GetNodeName(),
                "SingleLayerLstmGrad regbase small path skipped: hidden_size %ld does not fit in UB even with one "
                "timestep, one batch row, one gate row and the narrowest column chunk: need %ld bytes, budget %ld.",
                hidden, layout.totalBytes, budget);
        return ge::GRAPH_SUCCESS;
    }
    const int64_t mAll = tBlock * bBlock;
    const int64_t mBlock = (mAll < SMALL_M_BLOCK) ? mAll : SMALL_M_BLOCK;
    layout.Fill(tBlock, bBlock, hidden, chunkCols, mBlock, dtypeSize, gBlock);

    const int64_t numIChunks = LstmGradRegbase::CeilDivI64(inputSize, chunkCols);
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
    tilingData->tBlock = tBlock;
    tilingData->chunkCols = chunkCols;
    tilingData->mBlock = mBlock;
    tilingData->numIChunks = numIChunks;
    tilingData->bBlock = bBlock;
    tilingData->gBlock = gBlock;
    tilingData->biasComponents = isBias ? biasShape->GetStorageShape().GetDim(0) / (4 * hidden) : 0;

    context->SetTilingKey(LSTM_GRAD_TILING_KEY_REGBASE_SMALL);
    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    /* dw and db accumulate in fp32 across time blocks, batch blocks and the cores that share a
     * gate column, and are narrowed to the output width once at the end. At fp32 the outputs ARE
     * the accumulators and no workspace is needed; otherwise the accumulator is
     * 4H x (I + H) for dw plus 4H for db, in floats. */
    const int64_t accFloats = (dtypeSize == static_cast<int64_t>(sizeof(float))) ?
                                  0 :
                                  LstmGradRegbase::LstmGradRegbaseSmallUbLayout::GATE_NUM * hidden *
                                      (inputSize + hidden + 1);
    uint64_t replayFloats = 0;
    if (dtypeSize != static_cast<int64_t>(sizeof(float))) {
        // Each AIV owns a disjoint replay cache, reused across batch blocks.
        replayFloats = REPLAY_PLANES;
        for (int64_t extent : {usedCores, timeStep, bBlock, hidden}) {
            if (replayFloats > std::numeric_limits<size_t>::max() / sizeof(float) / extent) {
                OP_LOGE(context->GetNodeName(), "Forward replay workspace size overflow.");
                return ge::GRAPH_FAILED;
            }
            replayFloats *= extent;
        }
    }
    const auto reservedBytes = ascendcPlatform.GetLibApiWorkSpaceSize();
    const uint64_t maxFloats = (std::numeric_limits<size_t>::max() - reservedBytes) / sizeof(float);
    if (static_cast<uint64_t>(accFloats) > maxFloats || replayFloats > maxFloats - accFloats) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = (static_cast<size_t>(accFloats) + replayFloats) * sizeof(float) + reservedBytes;

    OP_LOGI(context->GetNodeName(),
            "SingleLayerLstmGrad regbase small path: T=%ld B=%ld I=%ld H=%ld bias=%ld dir=%ld order=%ld cores=%ld "
            "ubBytes=%ld.",
            timeStep, batch, inputSize, hidden, tilingData->isBias, directionVal, gateOrderVal, usedCores,
            layout.totalBytes);
    handled = true;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
