/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include "exe_graph/runtime/shape.h"
#include "graph/types.h"
#include "graph/ge_error_codes.h"
#include "register/op_impl_registry.h"

namespace ops {
using namespace ge;
namespace {
// Keep Shape validation consistent with op_graph/dynamic_augru_graph_infer.cpp.
constexpr size_t kX = 0;
constexpr size_t kWeightInput = 1;
constexpr size_t kWeightHidden = 2;
constexpr size_t kWeightAttention = 3;
constexpr size_t kBiasInput = 4;
constexpr size_t kBiasHidden = 5;
constexpr size_t kSequenceLength = 6;
constexpr size_t kInitH = 7;
constexpr size_t kInputNum = 8;
constexpr size_t kOutputNum = 7;
constexpr size_t kSequenceRank = 3; // [T,B,I], [T,B,H], or initial state [1,B,H].
constexpr size_t kMatrixRank = 2;
constexpr size_t kVectorRank = 1;
constexpr size_t kTimeAxis = 0;
constexpr size_t kBatchAxis = 1;
constexpr size_t kFeatureAxis = 2;
constexpr size_t kMatrixRowAxis = 0;
constexpr size_t kMatrixColumnAxis = 1;
constexpr size_t kVectorElementAxis = 0;
constexpr size_t kInitialStateLayerAxis = 0;
constexpr int64_t kInitialStateLayers = 1;
// Unknown rank is encoded as the single-element shape [-2].
constexpr size_t kUnknownRankDimCount = 1;
constexpr size_t kUnknownRankMarkerAxis = 0;
constexpr int64_t kUnknownDim = -1;
constexpr int64_t kUnknownRank = -2;

bool IsUnknownRank(const gert::Shape* shape)
{
    return shape != nullptr && shape->GetDimNum() == kUnknownRankDimCount &&
           shape->GetDim(kUnknownRankMarkerAxis) == kUnknownRank;
}

bool IsKnownDim(int64_t value) { return value >= 0; }

bool HasValidDimensions(const gert::Shape* shape)
{
    if (shape == nullptr || IsUnknownRank(shape)) {
        return shape != nullptr;
    }
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        // -1 is the only valid unknown dimension. -2 is reserved for the
        // single-element unknown-rank representation handled above.
        if (shape->GetDim(i) < kUnknownDim) {
            return false;
        }
    }
    return true;
}

graphStatus CheckRank(const gert::Shape* shape, size_t rank)
{
    if (shape == nullptr) {
        return GRAPH_FAILED;
    }
    if (IsUnknownRank(shape)) {
        return GRAPH_SUCCESS;
    }
    return shape->GetDimNum() == rank && HasValidDimensions(shape) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus MergeKnownDimension(int64_t candidate, int64_t& resolved)
{
    if (!IsKnownDim(candidate)) {
        return GRAPH_SUCCESS;
    }
    if (!IsKnownDim(resolved)) {
        resolved = candidate;
        return GRAPH_SUCCESS;
    }
    if (candidate != resolved) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

graphStatus MergeGateWidth(int64_t gateWidth, int64_t& hiddenSize)
{
    constexpr int64_t kGateCount = 3;
    if (!IsKnownDim(gateWidth)) {
        return GRAPH_SUCCESS;
    }
    if (gateWidth % kGateCount != 0) {
        return GRAPH_FAILED;
    }
    return MergeKnownDimension(gateWidth / kGateCount, hiddenSize);
}
graphStatus ResolveDynamicAUGRUShape(const std::array<const gert::Shape*, kInputNum>& inputs, gert::Shape& output)
{
    const gert::Shape* x = inputs[kX];
    const gert::Shape* weightInput = inputs[kWeightInput];
    const gert::Shape* weightHidden = inputs[kWeightHidden];
    const gert::Shape* weightAttention = inputs[kWeightAttention];
    if (x == nullptr || weightInput == nullptr || weightHidden == nullptr || weightAttention == nullptr) {
        return GRAPH_FAILED;
    }

    if (CheckRank(x, kSequenceRank) != GRAPH_SUCCESS || CheckRank(weightInput, kMatrixRank) != GRAPH_SUCCESS ||
        CheckRank(weightHidden, kMatrixRank) != GRAPH_SUCCESS ||
        CheckRank(weightAttention, kMatrixRank) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    int64_t time = kUnknownDim;
    int64_t batch = kUnknownDim;
    int64_t inputSize = kUnknownDim;
    int64_t hiddenSize = kUnknownDim;
    if (!IsUnknownRank(x)) {
        if (MergeKnownDimension(x->GetDim(kTimeAxis), time) != GRAPH_SUCCESS ||
            MergeKnownDimension(x->GetDim(kBatchAxis), batch) != GRAPH_SUCCESS ||
            MergeKnownDimension(x->GetDim(kFeatureAxis), inputSize) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    if (!IsUnknownRank(weightHidden)) {
        if (MergeKnownDimension(weightHidden->GetDim(kMatrixRowAxis), hiddenSize) != GRAPH_SUCCESS ||
            MergeGateWidth(weightHidden->GetDim(kMatrixColumnAxis), hiddenSize) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    if (!IsUnknownRank(weightInput)) {
        if (MergeKnownDimension(weightInput->GetDim(kMatrixRowAxis), inputSize) != GRAPH_SUCCESS ||
            MergeGateWidth(weightInput->GetDim(kMatrixColumnAxis), hiddenSize) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    if (!IsUnknownRank(weightAttention)) {
        if (MergeKnownDimension(weightAttention->GetDim(kTimeAxis), time) != GRAPH_SUCCESS ||
            MergeKnownDimension(weightAttention->GetDim(kBatchAxis), batch) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }

    const gert::Shape* biasInput = inputs[kBiasInput];
    const gert::Shape* biasHidden = inputs[kBiasHidden];
    const gert::Shape* sequenceLength = inputs[kSequenceLength];
    const gert::Shape* initH = inputs[kInitH];
    for (const gert::Shape* bias : {biasInput, biasHidden}) {
        if (bias != nullptr) {
            if (CheckRank(bias, kVectorRank) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
            if (!IsUnknownRank(bias) && MergeGateWidth(bias->GetDim(kVectorElementAxis), hiddenSize) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    }
    if (sequenceLength != nullptr && !IsUnknownRank(sequenceLength)) {
        if (!HasValidDimensions(sequenceLength)) {
            return GRAPH_FAILED;
        }
        const size_t rank = sequenceLength->GetDimNum();
        if (rank != kVectorRank && rank != kSequenceRank) {
            return GRAPH_FAILED;
        }
        if (rank == kVectorRank) {
            if (MergeKnownDimension(sequenceLength->GetDim(kVectorElementAxis), batch) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        } else {
            if (MergeKnownDimension(sequenceLength->GetDim(kTimeAxis), time) != GRAPH_SUCCESS ||
                MergeKnownDimension(sequenceLength->GetDim(kBatchAxis), batch) != GRAPH_SUCCESS ||
                MergeKnownDimension(sequenceLength->GetDim(kFeatureAxis), hiddenSize) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    }
    if (initH != nullptr) {
        if (CheckRank(initH, kSequenceRank) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        if (!IsUnknownRank(initH)) {
            if (IsKnownDim(initH->GetDim(kInitialStateLayerAxis)) &&
                initH->GetDim(kInitialStateLayerAxis) != kInitialStateLayers) {
                return GRAPH_FAILED;
            }
            if (MergeKnownDimension(initH->GetDim(kBatchAxis), batch) != GRAPH_SUCCESS ||
                MergeKnownDimension(initH->GetDim(kFeatureAxis), hiddenSize) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    }

    output.SetDimNum(kSequenceRank);
    output.SetDim(kTimeAxis, time);
    output.SetDim(kBatchAxis, batch);
    output.SetDim(kFeatureAxis, hiddenSize);
    return GRAPH_SUCCESS;
}
} // namespace

static graphStatus InferShape4DynamicAUGRU(gert::InferShapeContext* context)
{
    std::array<const gert::Shape*, kInputNum> inputs{};
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = i < kBiasInput ? context->GetInputShape(i) : context->GetOptionalInputShape(i);
    }
    gert::Shape output;
    if (ResolveDynamicAUGRUShape(inputs, output) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    for (size_t i = 0; i < kOutputNum; ++i) {
        gert::Shape* out = context->GetOutputShape(i);
        if (out == nullptr) {
            return GRAPH_FAILED;
        }
        *out = output;
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicAUGRU).InferShape(InferShape4DynamicAUGRU);
} // namespace ops
