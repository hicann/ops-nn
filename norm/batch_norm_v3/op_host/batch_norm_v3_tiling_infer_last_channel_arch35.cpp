/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_v3_tiling_infer_last_channel_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include "batch_norm_v3_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_INFER_LAST_CHANNEL = 900000;

constexpr int64_t NHWC_DIM_NUM = 4;
constexpr int64_t NDHWC_DIM_NUM = 5;
constexpr int64_t WEIGHT_BIAS_NUM = 2;
constexpr int64_t MEAN_VAR_NUM = 2;
constexpr int64_t INPUT_OUTPUT_NUM = 2;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t DOUBLE_BUFFER = 2;

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

static const int32_t INDEX_EPSILON = 0;
static const int32_t INDEX_IS_TRAINING = 2;
constexpr float DEFAULT_EPSILON = 1e-5;

// 框架侧占位可以只预留32B（ttk正常），debugTool执行时需要预留16M
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
} // namespace

namespace optiling {
class BatchNormV3InferLastChannelTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BatchNormV3InferLastChannelTiling(gert::TilingContext* context)
        : Ops::NN::Optiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~BatchNormV3InferLastChannelTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        aTileBase = vlFp16;
        bytesPerElement = FLOAT16_BYTES;
        if (dataType == ge::DT_FLOAT) {
            aTileBase = vlFp32;
            bytesPerElement = FLOAT32_BYTES;
        }
        bytesPerWeightElement = weightDataType == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;

        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset();

private:
    const char* opName = "BatchNormV3InferLastChannel";

    int64_t usedCoreNums;

    uint64_t blockSize;
    uint64_t vlFp32;
    uint64_t vlFp16;
    int64_t bytesPerElement;
    int64_t bytesPerWeightElement;

    int64_t fusedALen;
    int64_t fusedBLen;
    int64_t aTileBase;
    float epsilon;

    ge::DataType dataType;
    ge::DataType weightDataType;
    BatchNormV3InferLastChannelTilingData tilingData;
};

void BatchNormV3InferLastChannelTiling::Reset()
{
    usedCoreNums = 0;
    blockSize = 0;
    vlFp32 = 0;
    vlFp16 = 0;
    bytesPerElement = 0;
    bytesPerWeightElement = 0;

    fusedALen = 0;
    fusedBLen = 0;
    aTileBase = 0;
    epsilon = 0;
}

ge::graphStatus BatchNormV3InferLastChannelTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const BatchNormV3CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    blockSize = static_cast<uint64_t>(compileInfo->blockSize);
    vlFp32 = static_cast<uint64_t>(compileInfo->vectorLength) / FLOAT32_BYTES;
    vlFp16 = static_cast<uint64_t>(compileInfo->vectorLength) / FLOAT16_BYTES;

    opName = context_->GetNodeName();
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        aicoreParams_.blockDim = compileInfo->coreNum;
        aicoreParams_.ubSize = compileInfo->ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3InferLastChannelTiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("BatchNormV3InferLastChannel", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    // 获取输入shape
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = xShape->GetStorageShape();
    auto xDesc = context_->GetInputDesc(0);
    auto weightDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    dataType = xDesc->GetDataType();
    weightDataType = weightDesc->GetDataType();
    auto format = xDesc->GetFormat().GetStorageFormat();
    // 获取attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(INDEX_EPSILON);
    epsilon = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;
    const bool* isTrainingPtr = attrs->GetBool(INDEX_IS_TRAINING);
    bool isTraining = (isTrainingPtr == nullptr) ? true : *isTrainingPtr;
    if (isTraining) {
        OP_LOGI(context_, "This node not support training.");
        return ge::GRAPH_PARAM_INVALID;
    }

    if (format == FORMAT_NHWC) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NHWC_DIM_NUM, OP_LOGE(opName, "Dims should be 4 with NHWC format."),
            return ge::GRAPH_FAILED);
        fusedALen = xStorageShape.GetDim(DIM_3);
        fusedBLen = xStorageShape.GetDim(DIM_0) * xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2);
    } else if (format == FORMAT_NDHWC) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NDHWC_DIM_NUM, OP_LOGE(opName, "Dims should be 5 with NDHWC format."),
            return ge::GRAPH_FAILED);
        fusedALen = xStorageShape.GetDim(DIM_4);
        fusedBLen = xStorageShape.GetDim(DIM_0) * xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2) *
                    xStorageShape.GetDim(DIM_3);
    } else if (format == FORMAT_NCHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NHWC_DIM_NUM, OP_LOGE(opName, "Dims should be 4 with NCHW format."),
            return ge::GRAPH_FAILED);
        bool hwIsOne = xStorageShape.GetDim(DIM_2) == 1 && xStorageShape.GetDim(DIM_3) == 1;
        if (!hwIsOne) {
            return ge::GRAPH_PARAM_INVALID;
        }
        fusedALen = xStorageShape.GetDim(DIM_1);
        fusedBLen = xStorageShape.GetDim(DIM_0);
    } else if (format == FORMAT_NCDHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NDHWC_DIM_NUM, OP_LOGE(opName, "Dims should be 5 with NCDHW format."),
            return ge::GRAPH_FAILED);
        bool dhwIsOne =
            xStorageShape.GetDim(DIM_2) == 1 && xStorageShape.GetDim(DIM_3) == 1 && xStorageShape.GetDim(DIM_4) == 1;
        if (!dhwIsOne) {
            return ge::GRAPH_PARAM_INVALID;
        }
        fusedALen = xStorageShape.GetDim(DIM_1);
        fusedBLen = xStorageShape.GetDim(DIM_0);
    } else {
        OP_LOGI(context_, "Only supported format NHWC or NDHWC.");
        return ge::GRAPH_PARAM_INVALID;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3InferLastChannelTiling::DoOpTiling()
{
    // 切分A、B基本块， （B,A） -- >(Bouter, Aouter, Binner*Ainner*ATileBase)
    int64_t aInner = 1;
    int64_t ubBufferSize = (aicoreParams_.ubSize / DOUBLE_BUFFER -
                            (MEAN_VAR_NUM * FLOAT32_BYTES + WEIGHT_BIAS_NUM * bytesPerWeightElement) * aInner * aTileBase) /
                           bytesPerElement / INPUT_OUTPUT_NUM;

    // 先按照B切分，再切A
    int64_t bFactorMax = ubBufferSize / aTileBase;
    int64_t bInner = fusedBLen <= bFactorMax ? fusedBLen : bFactorMax;
    int64_t bOuter = Ops::Base::CeilDiv(fusedBLen, bInner);
    int64_t bTail = fusedBLen % bInner;
    int64_t tileBlockBTail = bTail == 0 ? bInner : bTail;

    int64_t aFactorMax =
        aicoreParams_.ubSize / DOUBLE_BUFFER / aTileBase /
        (bInner * INPUT_OUTPUT_NUM * bytesPerElement + WEIGHT_BIAS_NUM * bytesPerWeightElement + MEAN_VAR_NUM * FLOAT32_BYTES);
    int64_t aInnerMax = fusedALen / aTileBase;
    aInner = aInnerMax <= aFactorMax ? aInnerMax : aFactorMax;

    int64_t tileBlockALen = aInner == 0 ? aTileBase : aInner * aTileBase;
    int64_t aOuter = Ops::Base::CeilDiv(fusedALen, tileBlockALen);
    int64_t aTail = fusedALen % tileBlockALen;
    int64_t tileBlockATail = aTail == 0 ? tileBlockALen : aTail;
    int64_t tileBlockAPaddingNum = tileBlockALen - tileBlockATail;

    // 切核 （Bouter, Binner, Aouter, Ainner*ATileBase） -- > (Bouter*Aouter, Binner, Ainner*ATileBase)
    int64_t totalTiles = aOuter * bOuter;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.blockDim));
    usedCoreNums = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    tilingData.set_totalTiles(totalTiles);
    tilingData.set_tilesPerCore(tilesPerCore);
    tilingData.set_usedCoreNums(usedCoreNums);
    tilingData.set_totalALen(fusedALen);
    tilingData.set_aOuter(aOuter);
    tilingData.set_bOuter(bOuter);
    tilingData.set_tileBlockALen(tileBlockALen);
    tilingData.set_tileBlockATail(tileBlockATail);
    tilingData.set_tileBlockAPaddingNum(tileBlockAPaddingNum);
    tilingData.set_tileBlockBLen(bInner);
    tilingData.set_tileBlockBTail(tileBlockBTail);
    tilingData.set_epsilon(epsilon);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3InferLastChannelTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormV3InferLastChannelTiling::GetTilingKey() const
{
    return TILINGKEY_INFER_LAST_CHANNEL;
}

ge::graphStatus BatchNormV3InferLastChannelTiling::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3InferLastChannelTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        tilingData.GetDataSize() > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
            tilingData.GetDataSize(), rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormV3", BatchNormV3InferLastChannelTiling, 90000);
} // namespace optiling