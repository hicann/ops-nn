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
 * \file batch_norm_v3_tiling_infer_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include "batch_norm_v3_tiling.h"
using namespace ge;

namespace {
constexpr int64_t TILINGKEY_INFER = 910000;

constexpr int64_t DIM_NUM_4 = 4;
constexpr int64_t DIM_NUM_5 = 5;
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
class BatchNormV3InferTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BatchNormV3InferTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~BatchNormV3InferTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        aTileBase_ = vlFp16_;
        bytesPerElement_ = FLOAT16_BYTES;
        if (dataType_ == ge::DT_FLOAT) {
            aTileBase_ = vlFp32_;
            bytesPerElement_ = FLOAT32_BYTES;
        }
        bytesPerWeightElement_ = weightDataType_ == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;

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
    const char* opName = "BatchNormV3Infer";

    int64_t usedCoreNums_;
    uint64_t blockSize_;
    uint64_t vlFp32_;
    uint64_t vlFp16_;
    int64_t bytesPerElement_;
    int64_t bytesPerWeightElement_;
    int64_t fusedB0Len_;
    int64_t fusedALen_;
    int64_t fusedB1Len_;
    int64_t aTileBase_;
    float epsilon_;

    ge::DataType dataType_;
    ge::DataType weightDataType_;
    BatchNormV3InferTilingData tilingData_;
};

void BatchNormV3InferTiling::Reset()
{
    usedCoreNums_ = 0;
    blockSize_ = 0;
    vlFp32_ = 0;
    vlFp16_ = 0;
    bytesPerElement_ = 0;
    bytesPerWeightElement_ = 0;
    fusedB0Len_ = 0;
    fusedALen_ = 0;
    fusedB1Len_ = 0;
    aTileBase_ = 0;
    epsilon_ = 0;
}

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return value;
    }

    return (value + factor - 1) / factor;
}

ge::graphStatus BatchNormV3InferTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const BatchNormV3CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    blockSize_ = static_cast<uint64_t>(compileInfo->blockSize);
    vlFp32_ = static_cast<uint64_t>(compileInfo->vectorLength) / FLOAT32_BYTES;
    vlFp16_ = static_cast<uint64_t>(compileInfo->vectorLength) / FLOAT16_BYTES;

    opName = context_->GetNodeName();
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        aicoreParams_.numBlocks = compileInfo->coreNum;
        aicoreParams_.ubSize = compileInfo->ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3InferTiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("BatchNormV3Infer", "TilingContext is nullptr.");
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
    dataType_ = xDesc->GetDataType();
    weightDataType_ = weightDesc->GetDataType();
    auto format = xDesc->GetFormat().GetStorageFormat();
    // 获取attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(INDEX_EPSILON);
    epsilon_ = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;
    const bool* isTrainingPtr = attrs->GetBool(INDEX_IS_TRAINING);
    bool isTraining = (isTrainingPtr == nullptr) ? true : *isTrainingPtr;
    if (isTraining) {
        OP_LOGI(context_->GetNodeName(), "This node not support training, continue");
        return ge::GRAPH_PARAM_INVALID;
    }

    if (format == FORMAT_NCHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != DIM_NUM_4, OP_LOGE(opName, "Dims should be 4 with NCHW format."),
            return ge::GRAPH_FAILED);
        fusedB0Len_ = xStorageShape.GetDim(DIM_0);
        fusedALen_ = xStorageShape.GetDim(DIM_1);
        fusedB1Len_ = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
    } else if (format == FORMAT_NCDHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != DIM_NUM_5, OP_LOGE(opName, "Dims should be 5 with NCDHW format."),
            return ge::GRAPH_FAILED);
        fusedB0Len_ = xStorageShape.GetDim(DIM_0);
        fusedALen_ = xStorageShape.GetDim(DIM_1);
        fusedB1Len_ = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3) * xStorageShape.GetDim(DIM_4);
    } else {
        OP_LOGI(context_->GetNodeName(), "Only supported infer NHWC & NDHWC.");
        return ge::GRAPH_PARAM_INVALID;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3InferTiling::DoOpTiling()
{
    // 切分A、B基本块， （B0,A,B1） -- >(b0Outer*aOuter*b1Outer, B0inner*Ainner*B1innerA(TileBase))
    int64_t aInner = 1;
    int64_t ubBufferSize = (aicoreParams_.ubSize / DOUBLE_BUFFER -
                            (MEAN_VAR_NUM * FLOAT32_BYTES + WEIGHT_BIAS_NUM * bytesPerWeightElement_) * aInner * aTileBase_) /
                           bytesPerElement_ / INPUT_OUTPUT_NUM;

    // 先按照B切分，再切A
    // UB可载入最大tile块数
    int64_t factorMax = ubBufferSize / aTileBase_;

    // 默认策略: 先按照B0, B1把UB切满
    int64_t b1FactorMax = CeilDiv(fusedB1Len_, aTileBase_);
    int64_t b1Inner = factorMax <= b1FactorMax ? factorMax : b1FactorMax;
    int64_t b1Outer = CeilDiv(fusedB1Len_, b1Inner * aTileBase_);

    factorMax = factorMax / b1Inner;
    int64_t b0FactorMax = fusedB0Len_;
    int64_t b0Inner = factorMax <= b0FactorMax ? factorMax : b0FactorMax;
    int64_t b0Outer = CeilDiv(fusedB0Len_, b0Inner);

    factorMax = factorMax / b0Inner;
    int64_t aFactorMax = fusedALen_;
    aInner = factorMax <= aFactorMax ? factorMax : aFactorMax;
    int64_t aOuter = CeilDiv(fusedALen_, aInner);

    int64_t totalTiles = b0Outer * aOuter * b1Outer;
    int64_t tilesPerCore = CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.numBlocks));
    usedCoreNums_ = CeilDiv(totalTiles, tilesPerCore);

    int64_t tileBlockB0Tail = fusedB0Len_ - b0Inner * (b0Outer - 1);
    int64_t tileBlockATail = fusedALen_ - aInner * (aOuter - 1);
    int64_t tileBlockB1Tail = fusedB1Len_ - b1Inner * aTileBase_ * (b1Outer - 1);

    tilingData_.set_totalTiles(totalTiles);
    tilingData_.set_tilesPerCore(tilesPerCore);
    tilingData_.set_usedCoreNums(usedCoreNums_);
    tilingData_.set_totalB0Len(fusedB0Len_);
    tilingData_.set_totalALen(fusedALen_);
    tilingData_.set_totalB1Len(fusedB1Len_);
    tilingData_.set_b0Outer(b0Outer);
    tilingData_.set_aOuter(aOuter);
    tilingData_.set_b1Outer(b1Outer);
    tilingData_.set_tileBlockB0Len(b0Inner);
    tilingData_.set_tileBlockB0Tail(tileBlockB0Tail);
    tilingData_.set_tileBlockALen(aInner);
    tilingData_.set_tileBlockATail(tileBlockATail);
    tilingData_.set_tileBlockB1Len(b1Inner * aTileBase_);
    tilingData_.set_tileBlockB1Tail(tileBlockB1Tail);
    tilingData_.set_tileBlockAPaddingNum(0);
    tilingData_.set_epsilon(epsilon_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3InferTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormV3InferTiling::GetTilingKey() const
{
    return TILINGKEY_INFER;
}

ge::graphStatus BatchNormV3InferTiling::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3InferTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        tilingData_.GetDataSize() > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
            tilingData_.GetDataSize(), rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormV3", BatchNormV3InferTiling, 91000);
} // namespace optiling
