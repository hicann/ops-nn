/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_augru_tiling_arch35.cpp
 * \brief Arch35 tiling for DynamicAUGRU.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include "dynamic_augru_tiling_arch35.h"
#include "../../op_kernel/arch35/dynamic_augru_tiling_key.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#ifndef OPS_CHECK_NULL_WITH_CONTEXT
#define OPS_CHECK_NULL_WITH_CONTEXT(context, pointer)                                                 \
    do {                                                                                              \
        if ((pointer) == nullptr) {                                                                   \
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Required pointer %s is null.", #pointer); \
            return ge::GRAPH_FAILED;                                                                  \
        }                                                                                             \
    } while (0)
#endif

#ifndef VECTOR_INNER_ERR_REPORT_TILIING
#define VECTOR_INNER_ERR_REPORT_TILIING(opName, message, ...) OP_LOGE_WITHOUT_REPORT(opName, message, ##__VA_ARGS__)
#endif

#ifndef OP_TILING_CHECK
#define OP_TILING_CHECK(condition, logStatement, action) \
    do {                                                 \
        if (condition) {                                 \
            logStatement;                                \
            action;                                      \
        }                                                \
    } while (0)
#endif

namespace optiling {
namespace {
constexpr size_t kX = 0;
constexpr size_t kWeightInput = 1;
constexpr size_t kWeightHidden = 2;
constexpr size_t kWeightAttention = 3;
constexpr size_t kBiasInput = 4;
constexpr size_t kBiasHidden = 5;
constexpr size_t kSequenceLength = 6;
constexpr size_t kInitH = 7;
constexpr size_t kOutputY = 0;
constexpr size_t kOutputNum = 7;

// Attribute order follows the operator definition.
enum AttributeIndex : size_t {
    kAttrDirection = 0,
    kAttrCellDepth = 1,
    kAttrKeepProb = 2,
    kAttrCellClip = 3,
    kAttrNumProj = 4,
    kAttrTimeMajor = 5,
    kAttrActivation = 6,
    kAttrGateOrder = 7,
    kAttrResetAfter = 8,
    kAttrIsTraining = 9,
};

constexpr int64_t kGateCount = 3;   // Update, reset, candidate.
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

// Each outer task owns one Cube; Matmul chooses its own buffer allocation.
constexpr uint32_t kSingleCoreTask = 1;
constexpr int32_t kAutoMatmulBufferSize = -1;
constexpr int32_t kMatmulTilingFailure = -1;
// Serialized stateType values in the tiling ABI, not ge::DataType values.
constexpr uint32_t kStateFp16 = 0;
constexpr uint32_t kStateFp32 = 1;
constexpr size_t kWorkspaceCount = 1;
constexpr size_t kWorkspaceIndex = 0;

constexpr uint64_t kWorkspaceAlignment = 512;
constexpr uint64_t kSystemWorkspaceSize = 32UL * 1024UL * 1024UL;
constexpr uint64_t kProjectionBufferCount = 3;
constexpr uint64_t kFp32BufferCount = 10;
constexpr uint64_t kStateBufferCount = 2;
constexpr uint64_t kHalfBufferCount = 2;
constexpr uint64_t kUbBudgetNumerator = 3;
constexpr uint64_t kUbBudgetDenominator = 4;
constexpr uint32_t kVectorScratchBytes = 16U * 1024U;
constexpr uint64_t kMinParallelStateElements = 4096;
constexpr uint64_t kTargetStateElementsPerCore = 1024;
constexpr uint32_t kTargetBatchRowsPerCore = 4;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

bool IsKnownAndDifferent(int64_t lhs, int64_t rhs) { return lhs >= 0 && rhs >= 0 && lhs != rhs; }

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckShape(const gert::StorageShape* storageShape, size_t rank)
{
    return storageShape != nullptr && storageShape->GetStorageShape().GetDimNum() == rank;
}

ge::graphStatus CheckAttributes(gert::TilingContext* context, DynamicAUGRUGateOrder& gateOrder)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* direction = attrs->GetAttrPointer<char>(kAttrDirection);
    const int64_t* cellDepth = attrs->GetAttrPointer<int64_t>(kAttrCellDepth);
    const float* keepProb = attrs->GetAttrPointer<float>(kAttrKeepProb);
    const float* cellClip = attrs->GetAttrPointer<float>(kAttrCellClip);
    const int64_t* numProj = attrs->GetAttrPointer<int64_t>(kAttrNumProj);
    const bool* timeMajor = attrs->GetAttrPointer<bool>(kAttrTimeMajor);
    const char* activation = attrs->GetAttrPointer<char>(kAttrActivation);
    const char* gateOrderAttr = attrs->GetAttrPointer<char>(kAttrGateOrder);
    const bool* resetAfter = attrs->GetAttrPointer<bool>(kAttrResetAfter);
    const bool* isTraining = attrs->GetAttrPointer<bool>(kAttrIsTraining);
    OPS_CHECK_NULL_WITH_CONTEXT(context, direction);
    OPS_CHECK_NULL_WITH_CONTEXT(context, cellDepth);
    OPS_CHECK_NULL_WITH_CONTEXT(context, keepProb);
    OPS_CHECK_NULL_WITH_CONTEXT(context, cellClip);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numProj);
    OPS_CHECK_NULL_WITH_CONTEXT(context, timeMajor);
    OPS_CHECK_NULL_WITH_CONTEXT(context, activation);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gateOrderAttr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, resetAfter);
    OPS_CHECK_NULL_WITH_CONTEXT(context, isTraining);

    const DynamicAUGRUAttributes values{direction,  *cellDepth, *keepProb,     *cellClip,   *numProj,
                                        *timeMajor, activation, gateOrderAttr, *resetAfter, *isTraining};
    const char* invalidAttribute = InvalidDynamicAUGRUAttribute(values);
    OP_TILING_CHECK(
        invalidAttribute != nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(),
            "Unsupported DynamicAUGRU attribute for arch35: %s. Supported values: "
            "direction=UNIDIRECTIONAL, cell_depth=1, keep_prob=1, cell_clip=-1, num_proj=0, "
            "time_major=true, activation=tanh, gate_order=zrh/rzh, reset_after=true, is_training=true/false.",
            invalidAttribute),
        return ge::GRAPH_FAILED);
    gateOrder = values.gateOrder == "zrh" ? DynamicAUGRUGateOrder::ZRH : DynamicAUGRUGateOrder::RZH;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfigureMatmul(gert::TilingContext* context, DynamicAUGRUTilingData& tilingData, int64_t inputRows,
                                int64_t batch, int64_t inputSize, int64_t hiddenSize, bool hasBiasInput,
                                bool hasBiasHidden, ge::DataType stateType)
{
    const auto biasType = stateType == ge::DT_FLOAT ? matmul_tiling::DataType::DT_FLOAT :
                                                      matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::MultiCoreMatmulTiling inputMM;
    inputMM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    inputMM.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    inputMM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    // The outer kernel partitions batch rows. Each task owns one Cube and
    // generates a local Matmul tile; SetDim(1) is not the kernel launch count.
    inputMM.SetDim(kSingleCoreTask);
    inputMM.SetOrgShape(inputRows, kGateCount * hiddenSize, inputSize);
    inputMM.SetShape(inputRows, kGateCount * hiddenSize, inputSize);
    inputMM.SetBufferSpace(kAutoMatmulBufferSize, kAutoMatmulBufferSize, kAutoMatmulBufferSize);
    if (hasBiasInput) {
        inputMM.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, biasType);
        inputMM.SetBias(true);
    }
    OP_TILING_CHECK(
        inputMM.GetTiling(tilingData.inputMMTiling) == kMatmulTilingFailure,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to generate input projection matmul tiling."),
        return ge::GRAPH_FAILED);

    matmul_tiling::MultiCoreMatmulTiling hiddenMM;
    hiddenMM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    hiddenMM.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    hiddenMM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    hiddenMM.SetDim(kSingleCoreTask);
    hiddenMM.SetOrgShape(batch, kGateCount * hiddenSize, hiddenSize);
    hiddenMM.SetShape(batch, kGateCount * hiddenSize, hiddenSize);
    hiddenMM.SetBufferSpace(kAutoMatmulBufferSize, kAutoMatmulBufferSize, kAutoMatmulBufferSize);
    if (hasBiasHidden) {
        hiddenMM.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                             matmul_tiling::DataType::DT_FLOAT);
        hiddenMM.SetBias(true);
    }
    OP_TILING_CHECK(
        hiddenMM.GetTiling(tilingData.hiddenMMTiling) == kMatmulTilingFailure,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to generate hidden projection matmul tiling."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus Tiling4DynamicAUGRUArch35(gert::TilingContext* context, const DynamicAUGRUCompileInfo* compileInfo)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    const gert::StorageShape* xStorage = context->GetInputShape(kX);
    const gert::StorageShape* wiStorage = context->GetInputShape(kWeightInput);
    const gert::StorageShape* whStorage = context->GetInputShape(kWeightHidden);
    const gert::StorageShape* attStorage = context->GetInputShape(kWeightAttention);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    OPS_CHECK_NULL_WITH_CONTEXT(context, wiStorage);
    OPS_CHECK_NULL_WITH_CONTEXT(context, whStorage);
    OPS_CHECK_NULL_WITH_CONTEXT(context, attStorage);
    OP_TILING_CHECK(!CheckShape(xStorage, kSequenceRank) || !CheckShape(wiStorage, kMatrixRank) ||
                        !CheckShape(whStorage, kMatrixRank) || !CheckShape(attStorage, kMatrixRank),
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        context->GetNodeName(),
                        "x/weight_input/weight_hidden/weight_att ranks must be 3/2/2/2; got %zu/%zu/%zu/%zu.",
                        xStorage->GetStorageShape().GetDimNum(), wiStorage->GetStorageShape().GetDimNum(),
                        whStorage->GetStorageShape().GetDimNum(), attStorage->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);

    const gert::Shape& x = xStorage->GetStorageShape();
    const gert::Shape& wi = wiStorage->GetStorageShape();
    const gert::Shape& wh = whStorage->GetStorageShape();
    const gert::Shape& att = attStorage->GetStorageShape();
    const int64_t time = x.GetDim(kTimeAxis);
    const int64_t batch = x.GetDim(kBatchAxis);
    const int64_t inputSize = x.GetDim(kFeatureAxis);
    const int64_t hiddenSize = wh.GetDim(kMatrixRowAxis);
    OP_TILING_CHECK(time <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        context->GetNodeName(), "Runtime T/B/I/H must all be positive; got T=%ld B=%ld I=%ld H=%ld.",
                        time, batch, inputSize, hiddenSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(IsKnownAndDifferent(wi.GetDim(kMatrixRowAxis), inputSize) ||
                        wi.GetDim(kMatrixColumnAxis) != kGateCount * hiddenSize ||
                        wh.GetDim(kMatrixColumnAxis) != kGateCount * hiddenSize ||
                        IsKnownAndDifferent(att.GetDim(kTimeAxis), time) ||
                        IsKnownAndDifferent(att.GetDim(kBatchAxis), batch),
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        context->GetNodeName(),
                        "Input shape mismatch: x=[%ld,%ld,%ld], weight_input=[%ld,%ld] (expected [%ld,%ld]), "
                        "weight_hidden=[%ld,%ld] (expected [%ld,%ld]), weight_att=[%ld,%ld] (expected [%ld,%ld]).",
                        time, batch, inputSize, wi.GetDim(kMatrixRowAxis), wi.GetDim(kMatrixColumnAxis), inputSize,
                        kGateCount * hiddenSize, wh.GetDim(kMatrixRowAxis), wh.GetDim(kMatrixColumnAxis), hiddenSize,
                        kGateCount * hiddenSize, att.GetDim(kTimeAxis), att.GetDim(kBatchAxis), time, batch),
                    return ge::GRAPH_FAILED);

    const auto* xDesc = context->GetInputDesc(kX);
    const auto* wiDesc = context->GetInputDesc(kWeightInput);
    const auto* whDesc = context->GetInputDesc(kWeightHidden);
    const auto* attDesc = context->GetInputDesc(kWeightAttention);
    const auto* outputDesc = context->GetOutputDesc(kOutputY);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context, wiDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context, whDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context, attDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_TILING_CHECK(
        xDesc->GetDataType() != ge::DT_FLOAT16 || wiDesc->GetDataType() != ge::DT_FLOAT16 ||
            whDesc->GetDataType() != ge::DT_FLOAT16 || attDesc->GetDataType() != ge::DT_FLOAT16,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "x, weights and weight_att must be float16."),
        return ge::GRAPH_FAILED);
    const ge::DataType stateType = outputDesc->GetDataType();
    OP_TILING_CHECK(stateType != ge::DT_FLOAT16 && stateType != ge::DT_FLOAT,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "DynamicAUGRU state/output dtype must be float16 or float32."),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* biasInput = context->GetOptionalInputShape(kBiasInput);
    const gert::StorageShape* biasHidden = context->GetOptionalInputShape(kBiasHidden);
    const gert::StorageShape* sequence = context->GetOptionalInputShape(kSequenceLength);
    const gert::StorageShape* initH = context->GetOptionalInputShape(kInitH);
    const bool hasBiasInput = biasInput != nullptr;
    const bool hasBiasHidden = biasHidden != nullptr;
    const bool hasInitH = initH != nullptr;

    for (const auto& item : {std::make_pair(biasInput, kBiasInput), std::make_pair(biasHidden, kBiasHidden)}) {
        if (item.first == nullptr) {
            continue;
        }
        OP_TILING_CHECK(!CheckShape(item.first, kVectorRank) ||
                            item.first->GetStorageShape().GetDim(kVectorElementAxis) != kGateCount * hiddenSize,
                        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "bias must have shape [3H]."),
                        return ge::GRAPH_FAILED);
        const auto* desc = context->GetOptionalInputDesc(item.second);
        OPS_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_TILING_CHECK(
            desc->GetDataType() != stateType,
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "bias dtype must equal the output state dtype."),
            return ge::GRAPH_FAILED);
    }

    if (hasInitH) {
        OP_TILING_CHECK(!CheckShape(initH, kSequenceRank) ||
                            initH->GetStorageShape().GetDim(kInitialStateLayerAxis) != kInitialStateLayers ||
                            initH->GetStorageShape().GetDim(kBatchAxis) != batch ||
                            initH->GetStorageShape().GetDim(kFeatureAxis) != hiddenSize,
                        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "init_h must have shape [1,B,H]."),
                        return ge::GRAPH_FAILED);
        const auto* initDesc = context->GetOptionalInputDesc(kInitH);
        OPS_CHECK_NULL_WITH_CONTEXT(context, initDesc);
        OP_TILING_CHECK(
            initDesc->GetDataType() != stateType,
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "init_h dtype must equal the output state dtype."),
            return ge::GRAPH_FAILED);
    }

    DynamicAUGRUSequenceMode sequenceMode = DynamicAUGRUSequenceMode::NONE;
    if (sequence != nullptr) {
        const auto* sequenceDesc = context->GetOptionalInputDesc(kSequenceLength);
        OPS_CHECK_NULL_WITH_CONTEXT(context, sequenceDesc);
        const gert::Shape& sequenceShape = sequence->GetStorageShape();
        if (sequenceDesc->GetDataType() == ge::DT_INT32) {
            OP_TILING_CHECK(
                sequenceShape.GetDimNum() != kVectorRank || sequenceShape.GetDim(kVectorElementAxis) != batch,
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "int32 seq_length must have shape [B]."),
                return ge::GRAPH_FAILED);
            sequenceMode = DynamicAUGRUSequenceMode::LENGTH;
        } else if (sequenceDesc->GetDataType() == ge::DT_FLOAT16) {
            OP_TILING_CHECK(sequenceShape.GetDimNum() != kSequenceRank || sequenceShape.GetDim(kTimeAxis) != time ||
                                sequenceShape.GetDim(kBatchAxis) != batch ||
                                sequenceShape.GetDim(kFeatureAxis) != hiddenSize,
                            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                            "float16 compatibility mask must have shape [T,B,H]."),
                            return ge::GRAPH_FAILED);
            sequenceMode = DynamicAUGRUSequenceMode::MASK;
        } else {
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "seq_length dtype must be int32 or float16.");
            return ge::GRAPH_FAILED;
        }
    }

    DynamicAUGRUGateOrder gateOrder = DynamicAUGRUGateOrder::ZRH;
    OP_TILING_CHECK(CheckAttributes(context, gateOrder) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Attribute validation failed."),
                    return ge::GRAPH_FAILED);

    for (size_t i = 0; i < kOutputNum; ++i) {
        const auto* desc = context->GetOutputDesc(i);
        OPS_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_TILING_CHECK(
            desc->GetDataType() != stateType,
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "All seven outputs must use the same dtype."),
            return ge::GRAPH_FAILED);
    }

    uint64_t inputRows = 0;
    // All shapes use the same FP32 Cube and regbase path.
    const uint64_t computeBytes = sizeof(float);
    uint64_t gateElements = 0;
    uint64_t inputProjectionBytes = 0;
    uint64_t hiddenProjectionBytes = 0;
    uint64_t stateElements = 0;
    uint64_t hiddenWeightElements = 0;
    uint64_t hiddenWeightBytes = 0;
    OP_TILING_CHECK(
        !CheckedMul(static_cast<uint64_t>(time), static_cast<uint64_t>(batch), inputRows) ||
            !CheckedMul(inputRows, static_cast<uint64_t>(kGateCount * hiddenSize), gateElements) ||
            !CheckedMul(gateElements, computeBytes, inputProjectionBytes) ||
            !CheckedMul(static_cast<uint64_t>(batch), static_cast<uint64_t>(kGateCount * hiddenSize), gateElements) ||
            !CheckedMul(gateElements, computeBytes, hiddenProjectionBytes) ||
            !CheckedMul(static_cast<uint64_t>(batch), static_cast<uint64_t>(hiddenSize), stateElements) ||
            !CheckedMul(static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(kGateCount * hiddenSize),
                        hiddenWeightElements) ||
            !CheckedMul(hiddenWeightElements, sizeof(float), hiddenWeightBytes),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Workspace size overflow."), return ge::GRAPH_FAILED);

    const uint64_t inputProjectionOffset = 0;
    const uint64_t hiddenProjectionOffset = AlignUp(inputProjectionBytes, kWorkspaceAlignment);
    // Three aligned hidden-projection buffers: partial, sum, and Kahan residual.
    uint64_t hiddenProjectionWorkspaceBytes = 0;
    OP_TILING_CHECK(hiddenProjectionBytes > std::numeric_limits<uint64_t>::max() - (kWorkspaceAlignment - 1) ||
                        !CheckedMul(AlignUp(hiddenProjectionBytes, kWorkspaceAlignment), kProjectionBufferCount,
                                    hiddenProjectionWorkspaceBytes) ||
                        hiddenProjectionOffset > std::numeric_limits<uint64_t>::max() - hiddenProjectionWorkspaceBytes,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Hidden projection workspace overflow."),
                    return ge::GRAPH_FAILED);
    const uint64_t stateFp32Offset = hiddenProjectionOffset + hiddenProjectionWorkspaceBytes;
    const uint64_t weightHiddenFp32Offset = AlignUp(stateFp32Offset + stateElements * computeBytes,
                                                    kWorkspaceAlignment);
    const uint64_t biasHiddenBytes = hasBiasHidden ? kGateCount * hiddenSize * sizeof(float) : 0;
    const uint64_t userWorkspaceSize = AlignUp(weightHiddenFp32Offset + hiddenWeightBytes + biasHiddenBytes,
                                               kWorkspaceAlignment);

    const uint32_t availableCores = std::min(compileInfo->aicCoreNum, compileInfo->aivCoreNum);
    OP_TILING_CHECK(
        availableCores == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "No paired Cube/Vector cores: AIC=%u AIV=%u.",
                                        compileInfo->aicCoreNum, compileInfo->aivCoreNum),
        return ge::GRAPH_FAILED);
    // Keep tiny states on one task to amortize launch/synchronization overhead.
    // Otherwise scale with independent batch rows and state work, bounded by
    // the actual platform resources. Time steps cannot be split across tasks.
    const uint64_t batchTasks = (batch + kTargetBatchRowsPerCore - 1) / kTargetBatchRowsPerCore;
    const uint64_t stateTasks = std::max<uint64_t>(kSingleCoreTask, stateElements / kTargetStateElementsPerCore);
    const uint32_t usedCores = stateElements < kMinParallelStateElements ?
                                   kSingleCoreTask :
                                   static_cast<uint32_t>(
                                       std::min<uint64_t>(availableCores, std::min(batchTasks, stateTasks)));

    const uint64_t fp32VectorElements = std::max<uint64_t>(1, compileInfo->vectorLength / sizeof(float));
    const uint64_t stateBytes = stateType == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const uint64_t bytesPerElement = kFp32BufferCount * sizeof(float) + kStateBufferCount * stateBytes +
                                     kHalfBufferCount * sizeof(uint16_t);
    uint64_t tileHidden = (compileInfo->ubSize * kUbBudgetNumerator / kUbBudgetDenominator) / bytesPerElement;
    tileHidden = std::max<uint64_t>(fp32VectorElements, tileHidden / fp32VectorElements * fp32VectorElements);
    tileHidden = std::min<uint64_t>(tileHidden, AlignUp(static_cast<uint64_t>(hiddenSize), fp32VectorElements));

    DynamicAUGRUTilingData tilingData;
    tilingData.set_timeSize(time);
    tilingData.set_batchSize(batch);
    tilingData.set_inputSize(inputSize);
    tilingData.set_hiddenSize(hiddenSize);
    tilingData.set_tileHidden(static_cast<int64_t>(tileHidden));
    tilingData.set_hasBiasInput(static_cast<uint32_t>(hasBiasInput));
    tilingData.set_hasBiasHidden(static_cast<uint32_t>(hasBiasHidden));
    tilingData.set_hasInitH(static_cast<uint32_t>(hasInitH));
    tilingData.set_sequenceMode(static_cast<uint32_t>(sequenceMode));
    tilingData.set_gateOrder(static_cast<uint32_t>(gateOrder));
    tilingData.set_stateType(stateType == ge::DT_FLOAT ? kStateFp32 : kStateFp16);
    tilingData.set_usedAicCoreNum(usedCores);
    tilingData.set_usedAivCoreNum(usedCores);
    tilingData.set_blockSize(static_cast<uint32_t>(compileInfo->blockSize));
    tilingData.set_vectorLength(compileInfo->vectorLength);
    tilingData.set_inputProjectionOffset(inputProjectionOffset);
    tilingData.set_hiddenProjectionOffset(hiddenProjectionOffset);
    tilingData.set_stateFp32Offset(stateFp32Offset);
    tilingData.set_weightHiddenFp32Offset(weightHiddenFp32Offset);
    tilingData.set_userWorkspaceSize(userWorkspaceSize);

    const int64_t inputRowsPerCore = (time * batch + usedCores - 1) / usedCores;
    const int64_t batchRowsPerCore = (batch + usedCores - 1) / usedCores;
    OP_TILING_CHECK(ConfigureMatmul(context, tilingData, inputRowsPerCore, batchRowsPerCore, inputSize, hiddenSize,
                                    hasBiasInput, hasBiasHidden, stateType) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Matmul tiling failed."),
                    return ge::GRAPH_FAILED);

    gert::TilingData* raw = context->GetRawTilingData();
    OPS_CHECK_NULL_WITH_CONTEXT(context, raw);
    OP_TILING_CHECK(tilingData.GetDataSize() > raw->GetCapacity(),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Tiling data buffer is too small."),
                    return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tilingData.GetDataSize());

    auto workspaceSizes = context->GetWorkspaceSizes(kWorkspaceCount);
    OPS_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[kWorkspaceIndex] = kSystemWorkspaceSize + userWorkspaceSize;
    context->SetBlockDim(usedCores);
    // Cube's vector buffers use tileHidden * bytesPerElement bytes. Reserving
    // the entire UB would leave no shared space for the regbase pointwise VF.
    context->SetLocalMemorySize(static_cast<uint32_t>(tileHidden * bytesPerElement + kVectorScratchBytes));
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(sequenceMode));

    OP_LOGI(context->GetNodeName(),
            "DynamicAUGRU arch35 tiling: T=%ld B=%ld I=%ld H=%ld seqMode=%u tileH=%lu workspace=%lu cores=%u/%u.", time,
            batch, inputSize, hiddenSize, static_cast<uint32_t>(sequenceMode), tileHidden, userWorkspaceSize, usedCores,
            availableCores);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4DynamicAUGRU(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = context->GetCompiledInfo<DynamicAUGRUCompileInfo>();
    if (compileInfo == nullptr) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "DynamicAUGRU compile info is null during platform parsing.");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "DynamicAUGRU platform info is null.");
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->aicCoreNum = platform.GetCoreNumAic();
    compileInfo->aivCoreNum = platform.GetCoreNumAiv();
    compileInfo->isArch35 = platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    compileInfo->vectorLength = Ops::Base::GetVRegSize(context);

    if (!compileInfo->isArch35 || compileInfo->aicCoreNum == 0 || compileInfo->aivCoreNum == 0 ||
        compileInfo->ubSize == 0 || compileInfo->blockSize == 0 || compileInfo->vectorLength == 0) {
        OP_LOGE_WITHOUT_REPORT(
            context->GetNodeName(),
            "Expected Ascend950 with nonzero resources; arch35=%d AIC=%u AIV=%u UB=%lu block=%lu vector=%u.",
            compileInfo->isArch35, compileInfo->aicCoreNum, compileInfo->aivCoreNum, compileInfo->ubSize,
            compileInfo->blockSize, compileInfo->vectorLength);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4DynamicAUGRU(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = context->GetCompileInfo<DynamicAUGRUCompileInfo>();
    if (compileInfo == nullptr) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "DynamicAUGRU compile info is null during runtime tiling.");
        return ge::GRAPH_FAILED;
    }
    // Only the configured arch35 target is currently supported. Extend dispatch here when migrating 910B/910C.
    return Tiling4DynamicAUGRUArch35(context, compileInfo);
}

IMPL_OP_OPTILING(DynamicAUGRU)
    .Tiling(Tiling4DynamicAUGRU)
    .TilingParse<DynamicAUGRUCompileInfo>(TilingPrepare4DynamicAUGRU);
} // namespace optiling
