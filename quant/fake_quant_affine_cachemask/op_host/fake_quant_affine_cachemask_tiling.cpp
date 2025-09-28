/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fake_quant_affine_cachemask_tiling.cpp
 * \brief
 */

#include "fake_quant_affine_cachemask_tiling.h"

namespace optiling {
constexpr uint64_t FP32_MODE = 1;
constexpr uint64_t FP16_MODE = 2;
constexpr uint32_t BYTES_PER_DATA_FP32 = 68;
constexpr uint32_t BYTES_PER_DATA_FP16 = 46;
constexpr uint32_t SIZE_OF_FP16 = 2;
constexpr uint32_t SIZE_OF_FP32 = 4;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_REPEAT = 256;                 // The amount of data that can be processed by a repeat.
constexpr uint32_t SELECT_MODE_GE_ZERO_TMP_UB = 8000; // select mode 2 neead 8000B
class FakeQuantAffineCachemaskTiling
{
public:
    explicit FakeQuantAffineCachemaskTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;

private:
    void SetTilingKeyMode(ge::DataType dType);
    uint32_t GetNeedCoreNum(const uint32_t coreNumPlatform, ge::DataType dType);
    void CalTilingAligned(ge::DataType dType);
    FakeQuantAffineCachemaskTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t coreNum = 0;
    uint32_t loopNum = 0;
    uint32_t remainNum = 0;
    int64_t quantMin = 0;
    int64_t quantMax = 0;
    uint32_t dataPerRepeat = 0;
    uint32_t tileLength = 0;
    uint32_t totalLength = 1;        // the length of input
    uint32_t calcLength = 1;         // the length of input on axis != 0
    uint32_t headNum = 1;            // the length of input on axis = 0
    uint32_t alignNum = 0;           // data count per block
    uint32_t totalLengthAligned = 0; // length to align 32B
    uint64_t ubSizePlatForm = 0;
    uint32_t bytesPerData = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

void FakeQuantAffineCachemaskTiling::SetTilingKeyMode(ge::DataType dType)
{
    switch (dType) {
        case ge::DT_FLOAT16:
            tilingContext->SetTilingKey(FP16_MODE);
            bytesPerData = BYTES_PER_DATA_FP16;
            break;
        case ge::DT_FLOAT:
            tilingContext->SetTilingKey(FP32_MODE);
            bytesPerData = BYTES_PER_DATA_FP32;
            break;
        default:
            tilingContext->SetTilingKey(FP32_MODE);
            bytesPerData = BYTES_PER_DATA_FP32;
            break;
    }
}

uint32_t FakeQuantAffineCachemaskTiling::GetNeedCoreNum(const uint32_t coreNumPlatform, ge::DataType dType)
{
    if (dType == ge::DT_FLOAT16) {
        dataPerRepeat = BYTE_REPEAT / SIZE_OF_FP16;
    } else {
        dataPerRepeat = BYTE_REPEAT / SIZE_OF_FP32;
    }
    return headNum < coreNumPlatform ? headNum : coreNumPlatform;
}

void FakeQuantAffineCachemaskTiling::CalTilingAligned(ge::DataType dType)
{
    alignNum = BYTE_BLOCK / SIZE_OF_FP32;
    if (dType == ge::DT_FLOAT16) {
        alignNum = BYTE_BLOCK / SIZE_OF_FP16;
    }
    totalLengthAligned = ((calcLength + alignNum - 1) / alignNum) * alignNum;

    if (socVersion == platform_ascendc::SocVersion::ASCEND910) {
        tileLength = ubSizePlatForm / bytesPerData / dataPerRepeat * dataPerRepeat;
    } else {
        tileLength = (ubSizePlatForm - SELECT_MODE_GE_ZERO_TMP_UB) / bytesPerData / dataPerRepeat * dataPerRepeat;
    }
}

ge::graphStatus FakeQuantAffineCachemaskTiling::Init()
{
    OP_LOGD(tilingContext->GetNodeName(), "Tiling initing.");
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    auto xShape = tilingContext->GetInputShape(0)->GetStorageShape();
    totalLength = xShape.GetShapeSize();
    headNum = xShape.GetDim(0);
    if (headNum == 0) {
        return ge::GRAPH_FAILED;
    }
    calcLength = totalLength / headNum;

    OP_LOGD(
        tilingContext->GetNodeName(), "headNum %u, calcLength %u, totalLength %u.", headNum, calcLength, totalLength);

    auto compileInfo = tilingContext->GetCompileInfo<FakeQuantAffineCachemaskCompileInfo>();
    coreNum = static_cast<uint32_t>(compileInfo->totalCoreNum);
    OP_LOGD(tilingContext->GetNodeName(), "coreNum %u.", coreNum);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    ubSizePlatForm = compileInfo->ubSizePlatForm;
    OP_LOGD(tilingContext->GetNodeName(), "ub_size_platform is %lu.", ubSizePlatForm);
    auto dType = tilingContext->GetInputDesc(0)->GetDataType();
    coreNum = GetNeedCoreNum(coreNum, dType);
    loopNum = headNum / coreNum;
    remainNum = headNum % coreNum;

    auto attrs = tilingContext->GetAttrs();
    quantMin = *(attrs->GetAttrPointer<int64_t>(1));
    // 2 is quant_max in op of index
    quantMax = *(attrs->GetAttrPointer<int64_t>(2));

    SetTilingKeyMode(dType);
    CalTilingAligned(dType);
    OP_LOGD(tilingContext->GetNodeName(), "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantAffineCachemaskTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "Tiling start.");
    tilingContext->SetBlockDim(coreNum);
    tilingData.set_quantMin(quantMin);
    tilingData.set_quantMax(quantMax);
    tilingData.set_loopNum(loopNum);
    tilingData.set_remainNum(remainNum);
    tilingData.set_calcLength(calcLength);
    tilingData.set_headNum(headNum);
    tilingData.set_dataPerRepeat(dataPerRepeat);
    tilingData.set_totalLengthAligned(totalLengthAligned);
    tilingData.set_tileLength(tileLength);
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    TilingDataPrint();
    OP_LOGD(tilingContext->GetNodeName(), "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void FakeQuantAffineCachemaskTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext->GetNodeName(), "usedCoreNum: %u.", coreNum);
    OP_LOGD(tilingContext->GetNodeName(), "quantMin: %ld.", quantMin);
    OP_LOGD(tilingContext->GetNodeName(), "quantMax: %ld.", quantMax);
    OP_LOGD(tilingContext->GetNodeName(), "loopNum: %u.", loopNum);
    OP_LOGD(tilingContext->GetNodeName(), "remainNum: %u.", remainNum);
    OP_LOGD(tilingContext->GetNodeName(), "totalLength: %u.", totalLength);
    OP_LOGD(tilingContext->GetNodeName(), "calcLength: %u.", calcLength);
    OP_LOGD(tilingContext->GetNodeName(), "headNum: %u.", headNum);
    OP_LOGD(tilingContext->GetNodeName(), "alignNum: %u.", alignNum);
    OP_LOGD(tilingContext->GetNodeName(), "totalLengthAligned: %u.", totalLengthAligned);
    OP_LOGD(tilingContext->GetNodeName(), "tileLength: %u.", tileLength);
    OP_LOGD(tilingContext->GetNodeName(), "dataPerRepeat: %u.", dataPerRepeat);
}

static ge::graphStatus TilingFakeQuantAffineCachemask(gert::TilingContext* context)
{
    FakeQuantAffineCachemaskTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}

static ge::graphStatus TilingPrepareForFakeQuantAffineCachemask(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForFakeQuantAffineCachemask start.");
    auto compileInfo = context->GetCompiledInfo<FakeQuantAffineCachemaskCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context, "total_ub_size is %lu.", totalUbSize);
    OP_LOGD(context, "TilingPrepareForFakeQuantAffineCachemask end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FakeQuantAffineCachemask)
    .Tiling(TilingFakeQuantAffineCachemask)
    .TilingParse<FakeQuantAffineCachemaskCompileInfo>(TilingPrepareForFakeQuantAffineCachemask);
} // namespace optiling
