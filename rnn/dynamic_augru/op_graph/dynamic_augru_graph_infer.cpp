/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "graph/operator_reg.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include "exe_graph/runtime/shape.h"
#include "graph/types.h"
#include "graph/ge_error_codes.h"

namespace ops {
using namespace ge;
namespace {
// Keep Shape validation consistent with op_host/dynamic_augru_infershape.cpp.
constexpr size_t kX = 0;
constexpr size_t kWeightInput = 1;
constexpr size_t kWeightHidden = 2;
constexpr size_t kWeightAttention = 3;
constexpr size_t kBiasInput = 4;
constexpr size_t kBiasHidden = 5;
constexpr size_t kSequenceLength = 6;
constexpr size_t kInitH = 7;
constexpr size_t kOutputNum = 7;
constexpr int64_t kUnknownDim = -1;
constexpr int64_t kUnknownRank = -2;

bool IsUnknownRank(const gert::Shape* shape)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == kUnknownRank;
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
graphStatus ResolveDynamicAUGRUShape(const std::array<const gert::Shape*, 8>& inputs, gert::Shape& output)
{
    const gert::Shape* x = inputs[kX];
    const gert::Shape* weightInput = inputs[kWeightInput];
    const gert::Shape* weightHidden = inputs[kWeightHidden];
    const gert::Shape* weightAttention = inputs[kWeightAttention];
    if (x == nullptr || weightInput == nullptr || weightHidden == nullptr || weightAttention == nullptr) {
        return GRAPH_FAILED;
    }

    if (CheckRank(x, 3) != GRAPH_SUCCESS || CheckRank(weightInput, 2) != GRAPH_SUCCESS ||
        CheckRank(weightHidden, 2) != GRAPH_SUCCESS || CheckRank(weightAttention, 2) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    int64_t time = kUnknownDim;
    int64_t batch = kUnknownDim;
    int64_t inputSize = kUnknownDim;
    int64_t hiddenSize = kUnknownDim;
    if (!IsUnknownRank(x)) {
        if (MergeKnownDimension(x->GetDim(0), time) != GRAPH_SUCCESS ||
            MergeKnownDimension(x->GetDim(1), batch) != GRAPH_SUCCESS ||
            MergeKnownDimension(x->GetDim(2), inputSize) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    if (!IsUnknownRank(weightHidden)) {
        if (MergeKnownDimension(weightHidden->GetDim(0), hiddenSize) != GRAPH_SUCCESS ||
            MergeGateWidth(weightHidden->GetDim(1), hiddenSize) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    if (!IsUnknownRank(weightInput)) {
        if (MergeKnownDimension(weightInput->GetDim(0), inputSize) != GRAPH_SUCCESS ||
            MergeGateWidth(weightInput->GetDim(1), hiddenSize) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    if (!IsUnknownRank(weightAttention)) {
        if (MergeKnownDimension(weightAttention->GetDim(0), time) != GRAPH_SUCCESS ||
            MergeKnownDimension(weightAttention->GetDim(1), batch) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }

    const gert::Shape* biasInput = inputs[kBiasInput];
    const gert::Shape* biasHidden = inputs[kBiasHidden];
    const gert::Shape* sequenceLength = inputs[kSequenceLength];
    const gert::Shape* initH = inputs[kInitH];
    for (const gert::Shape* bias : {biasInput, biasHidden}) {
        if (bias != nullptr) {
            if (CheckRank(bias, 1) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
            if (!IsUnknownRank(bias) && MergeGateWidth(bias->GetDim(0), hiddenSize) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    }
    if (sequenceLength != nullptr && !IsUnknownRank(sequenceLength)) {
        if (!HasValidDimensions(sequenceLength)) {
            return GRAPH_FAILED;
        }
        const size_t rank = sequenceLength->GetDimNum();
        if (rank != 1 && rank != 3) {
            return GRAPH_FAILED;
        }
        if (rank == 1) {
            if (MergeKnownDimension(sequenceLength->GetDim(0), batch) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        } else {
            if (MergeKnownDimension(sequenceLength->GetDim(0), time) != GRAPH_SUCCESS ||
                MergeKnownDimension(sequenceLength->GetDim(1), batch) != GRAPH_SUCCESS ||
                MergeKnownDimension(sequenceLength->GetDim(2), hiddenSize) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    }
    if (initH != nullptr) {
        if (CheckRank(initH, 3) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        if (!IsUnknownRank(initH)) {
            if (IsKnownDim(initH->GetDim(0)) && initH->GetDim(0) != 1) {
                return GRAPH_FAILED;
            }
            if (MergeKnownDimension(initH->GetDim(1), batch) != GRAPH_SUCCESS ||
                MergeKnownDimension(initH->GetDim(2), hiddenSize) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    }

    output.SetDimNum(3);
    output.SetDim(0, time);
    output.SetDim(1, batch);
    output.SetDim(2, hiddenSize);
    return GRAPH_SUCCESS;
}
} // namespace

// GE graph preparation uses the Operator-based inference registry. Keep it in
// sync with runtime inference, including unknown ranks and absent optional inputs.
graphStatus InferShapeAndType4DynamicAUGRU(ge::Operator& op)
{
    constexpr std::array<const char*, 8> names = {"x",          "weight_input", "weight_hidden", "weight_att",
                                                  "bias_input", "bias_hidden",  "seq_length",    "init_h"};
    std::array<gert::Shape, 8> shapes;
    std::array<const gert::Shape*, 8> inputs{};
    std::array<ge::DataType, 8> types;
    types.fill(ge::DT_UNDEFINED);
    for (size_t i = 0; i < names.size(); ++i) {
        ge::TensorDesc desc;
        if (op.TryGetInputDesc(names[i], desc) != GRAPH_SUCCESS) {
            if (i < kBiasInput) {
                return GRAPH_FAILED;
            }
            continue;
        }
        const auto dims = desc.GetShape().GetDims();
        if (dims.size() > gert::Shape::kMaxDimNum) {
            return GRAPH_FAILED;
        }
        shapes[i].SetDimNum(dims.size());
        for (size_t j = 0; j < dims.size(); ++j) {
            shapes[i].SetDim(j, dims[j]);
        }
        inputs[i] = &shapes[i];
        types[i] = desc.GetDataType();
    }
    gert::Shape output;
    if (ResolveDynamicAUGRUShape(inputs, output) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    ge::DataType stateType = types[kX];
    for (size_t i : {kBiasInput, kBiasHidden, kInitH}) {
        if (types[i] != ge::DT_UNDEFINED) {
            stateType = types[i];
            break;
        }
    }
    const ge::Shape shape({output.GetDim(0), output.GetDim(1), output.GetDim(2)});
    for (uint32_t i = 0; i < kOutputNum; ++i) {
        auto desc = op.GetOutputDesc(i);
        desc.SetShape(shape);
        desc.SetOriginShape(shape);
        desc.SetDataType(stateType);
        if (op.UpdateOutputDesc(i, desc) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DynamicAUGRU, InferShapeAndType4DynamicAUGRU);

static graphStatus InferDataType4DynamicAUGRU(gert::InferDataTypeContext* context)
{
    ge::DataType stateType = context->GetOptionalInputDataType(kBiasInput);
    if (stateType == ge::DT_UNDEFINED) {
        stateType = context->GetOptionalInputDataType(kBiasHidden);
    }
    if (stateType == ge::DT_UNDEFINED) {
        stateType = context->GetOptionalInputDataType(kInitH);
    }
    if (stateType == ge::DT_UNDEFINED) {
        stateType = context->GetInputDataType(kX);
    }
    for (size_t i = 0; i < kOutputNum; ++i) {
        context->SetOutputDataType(i, stateType);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP(DynamicAUGRU).InferDataType(InferDataType4DynamicAUGRU);
} // namespace ops
