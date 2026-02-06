/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ada_layer_norm_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "tiling_base/tiling_util.h"
#include "ada_layer_norm_tiling.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

constexpr int32_t INPUT_TENSOR_NUM = 3;
constexpr int32_t INDEX_ZERO = 0;
constexpr int32_t INDEX_ONE = 1;
constexpr int32_t INDEX_TWO = 2;
constexpr int32_t INDEX_THREE = 3;
constexpr int32_t INDEX_FOUR = 4;
constexpr int32_t INDEX_FIVE = 5;
constexpr int32_t BLOCK_SIZE = 32;

constexpr int64_t MULTI_ROW_SIZE = 3072;
constexpr int64_t SINGLE_ROW_SIZE = 6144;
constexpr int64_t WELFORD_ROW_SIZE = 5120;
constexpr uint32_t FP32_BYTE = 4;
constexpr uint32_t FP16_BYTE = 2;
constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;

constexpr uint8_t BASE_OP_CODE = 1;
constexpr uint8_t BASE_V2_OP_CODE = 12;
constexpr uint8_t QUANT_OP_CODE = 2;
constexpr uint8_t TILING_KEY_ONE = 1;
constexpr uint8_t TILING_KEY_TWO = 2;
constexpr uint8_t TILING_KEY_FACTOR = 10;

class AdaLayerNormTiling
{
public:
    explicit AdaLayerNormTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus Init(uint8_t theCode);
    ge::graphStatus RunBigKernelTiling();

private:
    int32_t SplitCore(int32_t coreNumPlatform, bool isRegBase);
    void FillTilingData();
    void DoLayerNormTiling();
    uint8_t GetTilingKey(bool isRegBase);
    template <typename T1, typename T2>
    inline auto CeilA2B(T1 a, T2 b) const -> T1;

private:
    AdaLayerNormTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;

    uint8_t opCode = 0;
    int64_t batchSize = 0;
    int64_t seqLen = 0;
    int64_t hiddenDim = 0;
    int64_t hiddenDimCeil = 0;
    float epsilon = 0.0f;
    bool hasWeight = false;
    bool hasBias = false;
    int32_t hasSmooth = 0;
    bool isWeightFloat = true;
};

ge::graphStatus AdaLayerNormTiling::Init(uint8_t theCode = 0)
{
    opCode = theCode;
    // 获取输入矩阵
    auto xTensor = tilingContext->GetInputTensor(INDEX_ZERO);
    auto scaleTensor = tilingContext->GetInputTensor(INDEX_ONE);
    auto shiftTensor = tilingContext->GetInputTensor(INDEX_TWO);
    auto weightTensor = tilingContext->GetOptionalInputTensor(INDEX_THREE);
    auto biasTensor = tilingContext->GetOptionalInputTensor(INDEX_FOUR);
    if (xTensor == nullptr || scaleTensor == nullptr || shiftTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    hasWeight = (weightTensor != nullptr);
    hasBias = (biasTensor != nullptr);
    if (hasWeight && weightTensor->GetDataType() != ge::DataType::DT_FLOAT) {
        isWeightFloat = false;
    }
    if (hasBias && biasTensor->GetDataType() != ge::DataType::DT_FLOAT) {
        isWeightFloat = false;
    }
    hasSmooth = 0;
    if (opCode == QUANT_OP_CODE) {
        auto smoothTensor = tilingContext->GetOptionalInputTensor(INDEX_FIVE);
        hasSmooth = (smoothTensor != nullptr);
    }

    // 获取输入的数据类型，输入Tensor的数据类型保持一致
    for (int i = 0; i < INPUT_TENSOR_NUM; i++) {
        auto temp = tilingContext->GetInputDesc(i);
        if (dataType == ge::DT_UNDEFINED) {
            dataType = temp->GetDataType();
        } else if (dataType != temp->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaLayerNormTiling::RunBigKernelTiling()
{
    auto xShape = tilingContext->GetInputShape(INDEX_ZERO)->GetOriginShape();
    int64_t xDim = xShape.GetDimNum();
    batchSize = 1;
    for (int64_t i = 0; i < xDim - INDEX_TWO; i++) {
        batchSize *= xShape.GetDim(i);
    }
    seqLen = xShape.GetDim(xDim - INDEX_TWO);
    hiddenDim = xShape.GetDim(xDim - INDEX_ONE);
    hiddenDimCeil = CeilA2B(hiddenDim, BLOCK_SIZE) * BLOCK_SIZE;
    epsilon = *tilingContext->GetAttrs()->GetAttrPointer<float>(0);

    auto compileInfo = reinterpret_cast<const AdaLayerNormCompileInfo*>(tilingContext->GetCompileInfo());
    int32_t coreNumPlatform = compileInfo->coreNum;
    int32_t needCoreNum = SplitCore(coreNumPlatform, compileInfo->isRegBase);
    if (compileInfo->isRegBase) {
        DoLayerNormTiling();
    }

    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    if (opCode == QUANT_OP_CODE && hiddenDim > SINGLE_ROW_SIZE) {
        workspaces[0] = WORK_SPACE_SIZE + needCoreNum * hiddenDimCeil * sizeof(float);
    } else {
        workspaces[0] = WORK_SPACE_SIZE;
    }

    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(GetTilingKey(compileInfo->isRegBase));
    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

int32_t AdaLayerNormTiling::SplitCore(int32_t coreNumPlatform, bool isRegBase)
{
    int64_t sliceSize = hiddenDim;
    int64_t rowNum = 1;
    if (hiddenDim <= MULTI_ROW_SIZE) {
        rowNum = SINGLE_ROW_SIZE / hiddenDimCeil;
    } else if (hiddenDim > SINGLE_ROW_SIZE) {
        int64_t batch = CeilA2B(hiddenDim, SINGLE_ROW_SIZE);
        if (isRegBase) {
            batch = CeilA2B(hiddenDim, WELFORD_ROW_SIZE);
        }
        sliceSize = CeilA2B(hiddenDim, batch);
        sliceSize = CeilA2B(sliceSize, BLOCK_SIZE) * BLOCK_SIZE;
    }

    int64_t singleCoreNum = coreNumPlatform != 0 ? (batchSize * seqLen) / coreNumPlatform : 0;
    int64_t tailNum = coreNumPlatform != 0 ? (batchSize * seqLen) % coreNumPlatform : 0;
    tilingData.set_singleCoreNum(singleCoreNum);
    tilingData.set_tailNum(tailNum);
    tilingData.set_sliceSize(sliceSize);
    tilingData.set_rowNum(rowNum);
    return singleCoreNum > 0 ? coreNumPlatform : tailNum;
}

uint8_t AdaLayerNormTiling::GetTilingKey(bool isRegBase)
{
    uint8_t tilingKey = 0;
    if (isRegBase) {
        tilingKey = (hiddenDim > SINGLE_ROW_SIZE) ? TILING_KEY_TWO : TILING_KEY_ONE;
        tilingKey *= TILING_KEY_FACTOR;
    }
    tilingKey += (opCode == BASE_V2_OP_CODE && isWeightFloat) ? TILING_KEY_TWO : TILING_KEY_ONE;
    return tilingKey;
}

void AdaLayerNormTiling::DoLayerNormTiling()
{
    uint32_t minValue;
    uint32_t maxValue;
    if (hiddenDim > SINGLE_ROW_SIZE) {
        int64_t sliceSize = tilingData.get_sliceSize();
        int64_t tmpBufferSize = 0;
        ge::Shape inputShape({1, sliceSize});
        uint32_t welfordXByte = (dataType == ge::DataType::DT_FLOAT) ? FP32_BYTE : FP16_BYTE;
        AscendC::GetWelfordUpdateMaxMinTmpSize(inputShape, welfordXByte, FP32_BYTE, false, true, maxValue, minValue);
        tmpBufferSize += maxValue;
        AscendC::GetWelfordFinalizeMaxMinTmpSize(inputShape, FP32_BYTE, false, maxValue, minValue);
        tmpBufferSize += maxValue;
        AscendC::GetNormalizeMaxMinTmpSize(inputShape, FP32_BYTE, FP32_BYTE, false, true, false, maxValue, minValue);
        tmpBufferSize += maxValue;
        tilingData.set_tmpBufferSize(tmpBufferSize);
    } else {
        ge::Shape inputShape({tilingData.get_rowNum(), hiddenDim});
        AscendC::GetLayerNormMaxMinTmpSize(inputShape, FP32_BYTE, true, true, false, maxValue, minValue);
        AscendC::GetLayerNormNDTilingInfo(inputShape, maxValue, FP32_BYTE, true, true, tilingData.layerNormTiling);
        tilingData.set_tmpBufferSize(maxValue);
    }
}

void AdaLayerNormTiling::FillTilingData()
{
    tilingData.set_batchSize(batchSize);
    tilingData.set_seqLen(seqLen);
    tilingData.set_hiddenDim(hiddenDim);
    tilingData.set_epsilon(epsilon);
    tilingData.set_aveFactor(static_cast<float>(1.0 / hiddenDim));
    tilingData.set_hasWeight(hasWeight);
    tilingData.set_hasBias(hasBias);
    tilingData.set_hasSmooth(hasSmooth);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

template <typename T1, typename T2>
inline auto AdaLayerNormTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    return (b != 0) ? (a + b - 1) / b : a;
}

static ge::graphStatus Tiling4AdaLayerNormTiling(gert::TilingContext* context)
{
    AdaLayerNormTiling tilingObject(context);
    if (tilingObject.Init(BASE_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4AdaLayerNormV2Tiling(gert::TilingContext* context)
{
    AdaLayerNormTiling tilingObject(context);
    if (tilingObject.Init(BASE_V2_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4AdaLayerNormQuantTiling(gert::TilingContext* context)
{
    AdaLayerNormTiling tilingObject(context);
    if (tilingObject.Init(QUANT_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<AdaLayerNormCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    compileInfo->isRegBase = IsRegbaseSocVersion(context);

    OP_TILING_CHECK(
        compileInfo->coreNum <= 0,
        OP_LOGE(context->GetNodeName(), "AdaLayerNormTiling GetHardwareInfo Failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdaLayerNorm)
    .Tiling(Tiling4AdaLayerNormTiling)
    .TilingParse<AdaLayerNormCompileInfo>(TilingPrepareTiling);

IMPL_OP_OPTILING(AdaLayerNormV2)
    .Tiling(Tiling4AdaLayerNormV2Tiling)
    .TilingParse<AdaLayerNormCompileInfo>(TilingPrepareTiling);

IMPL_OP_OPTILING(AdaLayerNormQuant)
    .Tiling(Tiling4AdaLayerNormQuantTiling)
    .TilingParse<AdaLayerNormCompileInfo>(TilingPrepareTiling);
} // namespace optiling
