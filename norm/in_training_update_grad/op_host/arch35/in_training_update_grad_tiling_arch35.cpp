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
 * \file in_training_update_grad_tiling_arch35.cpp
 * \brief
 */

#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "in_training_update_grad_tiling_arch35.h"

using namespace ge;
using namespace Ops::Base;

namespace {
constexpr int64_t NDC1HWC0_DIM_NUM = 6;
constexpr int64_t DIM_N = 0;
constexpr int64_t DIM_D = 1;
constexpr int64_t DIM_C1 = 2;
constexpr int64_t DIM_H = 3;
constexpr int64_t DIM_W = 4;
constexpr int64_t DIM_C0 = 5;

constexpr int64_t INPUT_DY_IDX = 0;
constexpr int64_t INPUT_X_IDX = 1;
constexpr int64_t INPUT_VARIANCE_IDX = 2;
constexpr int64_t INPUT_MEAN_IDX = 3;
constexpr int64_t OUTPUT_RES_GAMMA_IDX = 0;
constexpr int64_t OUTPUT_RES_BETA_IDX = 1;

constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t MAX_BLOCK_COUNT = 65535; // DataCopyExtParams.blockCount is uint16
constexpr int64_t SINGLE_AIV_THRESHOLD_BYTES = 32L * 1024L;

constexpr uint64_t TILINGKEY_REDUCE_EMPTY = 50000;
constexpr uint64_t TILINGKEY_FULL_LOAD = 100000;
constexpr uint64_t TILINGKEY_STREAM = 200000;

constexpr float DEFAULT_EPSILON = 1e-6f;
} // namespace

namespace optiling {

ge::graphStatus InTrainingUpdateGradTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto compileInfoPtr = reinterpret_cast<const InTrainingUpdateGradCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                return ge::GRAPH_FAILED);
    vectorLength_ = compileInfoPtr->vectorLength;
    vlfp32_ = compileInfoPtr->vectorLength / sizeof(float);
    ubBlockSize_ = compileInfoPtr->ubBlockSize;

    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        aicoreParams_.blockDim = compileInfoPtr->coreNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InTrainingUpdateGradTilingBase::CheckDtype()
{
    auto dyDesc = context_->GetInputDesc(INPUT_DY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    dyDataType_ = dyDesc->GetDataType();
    OP_CHECK_IF(
        (dyDataType_ != ge::DT_FLOAT16 && dyDataType_ != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dy", ToString(dyDataType_).c_str(), "FLOAT or FLOAT16"),
        return ge::GRAPH_FAILED);

    auto xDesc = context_->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF((xDesc->GetDataType() != dyDataType_),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "dy and x",
                                                       ToString(xDesc->GetDataType()).c_str(),
                                                       "The dtypes of input dy and input x must be the same"),
                return ge::GRAPH_FAILED);

    const std::vector<std::pair<int64_t, std::string>> fp32Params = {{INPUT_VARIANCE_IDX, "variance"},
                                                                     {INPUT_MEAN_IDX, "mean"}};
    for (const auto& [idx, name] : fp32Params) {
        auto desc = context_->GetInputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
        OP_CHECK_IF((desc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), name.c_str(),
                                              ToString(desc->GetDataType()).c_str(), "FLOAT"),
                    return ge::GRAPH_FAILED);
    }

    const std::vector<std::pair<int64_t, std::string>> fp32Outputs = {{OUTPUT_RES_GAMMA_IDX, "res_gamma"},
                                                                      {OUTPUT_RES_BETA_IDX, "res_beta"}};
    for (const auto& [idx, name] : fp32Outputs) {
        auto desc = context_->GetOutputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
        OP_CHECK_IF((desc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), name.c_str(),
                                              ToString(desc->GetDataType()).c_str(), "FLOAT"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InTrainingUpdateGradTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("INTrainingUpdateGrad", "TilingContext is nullptr."),
                return ge::GRAPH_FAILED);

    auto dyDesc = context_->GetInputDesc(INPUT_DY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    // storage format 高位可能携带 C0 编码(GE 在部分图优化路径上标注),需取 primary 再比较
    format_ = static_cast<ge::Format>(ge::GetPrimaryFormat(dyDesc->GetFormat().GetStorageFormat()));
    OP_CHECK_IF(format_ != ge::FORMAT_NDC1HWC0,
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "dy", ToString(format_).c_str(), "NDC1HWC0"),
                return ge::GRAPH_FAILED);

    auto dyShape = context_->GetInputShape(INPUT_DY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyShape);
    auto storageShape = dyShape->GetStorageShape();
    int64_t dimNum = storageShape.GetDimNum();
    OP_CHECK_IF(dimNum != NDC1HWC0_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "dy", std::to_string(dimNum).c_str(),
                                                         "The shape dim of input dy must be 6 (NDC1HWC0)"),
                return ge::GRAPH_FAILED);

    numN_ = storageShape.GetDim(DIM_N);
    numD_ = storageShape.GetDim(DIM_D);
    numC1_ = storageShape.GetDim(DIM_C1);
    numH_ = storageShape.GetDim(DIM_H);
    numW_ = storageShape.GetDim(DIM_W);
    numC0_ = storageShape.GetDim(DIM_C0);

    OP_CHECK_IF((numN_ <= 0 || numC1_ <= 0 || numC0_ <= 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy", ToString(storageShape).c_str(),
                                                      "N, C1 and C0 dims of input dy must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((numD_ < 0 || numH_ < 0 || numW_ < 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy", ToString(storageShape).c_str(),
                                                      "spatial dims (D, H, W) of input dy must not be negative"),
                return ge::GRAPH_FAILED);

    numHW_ = numH_ * numW_;
    reduceR_ = numD_ * numHW_;
    groupNum_ = numN_ * numC1_;
    blockLenElem_ = numHW_ * numC0_;
    epsilon_ = DEFAULT_EPSILON;

    return CheckDtype();
}

ge::graphStatus InTrainingUpdateGradTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus InTrainingUpdateGradTilingBase::GetWorkspaceSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------
// ReduceEmpty (TilingKey 50000): R == 0 -> write 0.0 to both outputs (N*C1*C0 elements each).
// ---------------------------------------------------------------------------------------------------
bool InTrainingUpdateGradReduceEmptyTiling::IsCapable() { return reduceR_ == 0; }

ge::graphStatus InTrainingUpdateGradReduceEmptyTiling::DoOpTiling()
{
    int64_t totalLength = groupNum_ * numC0_;
    int64_t aivNum = static_cast<int64_t>(aicoreParams_.blockDim);
    int64_t elemSize = FP32_BYTE;
    int64_t perLoopMax = static_cast<int64_t>(aicoreParams_.ubSize) / elemSize;
    int64_t alignUnit = ubBlockSize_ / elemSize; // 8 fp32 elements per 32B block

    blockNum_ = Ops::Base::CeilDiv(totalLength * elemSize, SINGLE_AIV_THRESHOLD_BYTES);
    if (blockNum_ > aivNum) {
        blockNum_ = aivNum;
    }
    if (blockNum_ < 1) {
        blockNum_ = 1;
    }

    int64_t perCoreElements = Ops::Base::CeilDiv(totalLength, blockNum_);
    int64_t lastCoreElements = totalLength - (blockNum_ - 1) * perCoreElements;
    td_.perCoreElements = static_cast<uint32_t>(perCoreElements);
    td_.lastCoreElements = static_cast<uint32_t>(lastCoreElements);

    int64_t perCorePerLoop = Ops::Base::FloorAlign(std::min(perLoopMax, perCoreElements), alignUnit);
    if (perCorePerLoop < 1) {
        perCorePerLoop = perCoreElements;
    }
    int64_t perCoreLoops = Ops::Base::CeilDiv(perCoreElements, perCorePerLoop);
    int64_t perCoreLastLoop = perCoreElements - (perCoreLoops - 1) * perCorePerLoop;
    td_.perCoreLoops = static_cast<uint32_t>(perCoreLoops);
    td_.perCorePerLoopElements = static_cast<uint32_t>(perCorePerLoop);
    td_.perCoreLastLoopElements = static_cast<uint32_t>(perCoreLastLoop);

    int64_t lastCorePerLoop = Ops::Base::FloorAlign(std::min(perLoopMax, lastCoreElements), alignUnit);
    if (lastCorePerLoop < 1) {
        lastCorePerLoop = lastCoreElements;
    }
    int64_t lastCoreLoops = Ops::Base::CeilDiv(lastCoreElements, lastCorePerLoop);
    int64_t lastCoreLastLoop = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoop;
    td_.lastCoreLoops = static_cast<uint32_t>(lastCoreLoops);
    td_.lastCorePerLoopElements = static_cast<uint32_t>(lastCorePerLoop);
    td_.lastCoreLastLoopElements = static_cast<uint32_t>(lastCoreLastLoop);
    return ge::GRAPH_SUCCESS;
}

uint64_t InTrainingUpdateGradReduceEmptyTiling::GetTilingKey() const { return TILINGKEY_REDUCE_EMPTY; }

ge::graphStatus InTrainingUpdateGradReduceEmptyTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(td_), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &td_, sizeof(td_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------
// FullLoad (TilingKey 100000): one (n,c1) group's R*C0 block fits UB (double buffered).
// ---------------------------------------------------------------------------------------------------
bool InTrainingUpdateGradFullLoadTiling::IsCapable()
{
    if (reduceR_ <= 0) {
        return false;
    }
    if (numD_ > MAX_BLOCK_COUNT) { // strided aggregate needs blockCount = D <= uint16 max
        return false;
    }
    int64_t elemSize = (dyDataType_ == ge::DT_FLOAT16) ? FP16_BYTE : FP32_BYTE;
    int64_t regPad = vectorLength_;
    int64_t spatialBytes = reduceR_ * numC0_ * elemSize;
    int64_t need = DOUBLE_BUFFER * (spatialBytes + regPad) * 2 + // dy + x
                   2 * (numC0_ * FP32_BYTE + regPad) +           // variance + mean
                   DOUBLE_BUFFER * (numC0_ * FP32_BYTE) * 2 +    // res_gamma + res_beta
                   numC0_ * FP32_BYTE;                           // rstd
    return need <= static_cast<int64_t>(aicoreParams_.ubSize);
}

ge::graphStatus InTrainingUpdateGradFullLoadTiling::DoOpTiling()
{
    int64_t perCoreGroups = Ops::Base::CeilDiv(groupNum_, static_cast<int64_t>(aicoreParams_.blockDim));
    if (perCoreGroups < 1) {
        perCoreGroups = 1;
    }
    blockNum_ = Ops::Base::CeilDiv(groupNum_, perCoreGroups);

    td_.numC1 = static_cast<uint32_t>(numC1_);
    td_.numD = static_cast<uint32_t>(numD_);
    td_.numHW = static_cast<uint32_t>(numHW_);
    td_.numC0 = static_cast<uint32_t>(numC0_);
    td_.reduceR = static_cast<uint32_t>(reduceR_);
    td_.groupNum = static_cast<uint32_t>(groupNum_);
    td_.usedCoreNum = static_cast<uint32_t>(blockNum_);
    td_.perCoreGroups = static_cast<uint32_t>(perCoreGroups);
    td_.blockLenElem = static_cast<uint32_t>(blockLenElem_);
    td_.epsilon = epsilon_;
    return ge::GRAPH_SUCCESS;
}

uint64_t InTrainingUpdateGradFullLoadTiling::GetTilingKey() const { return TILINGKEY_FULL_LOAD; }

ge::graphStatus InTrainingUpdateGradFullLoadTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(td_), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &td_, sizeof(td_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------
// Stream (TilingKey 200000): fallback for large R*C0 (or large D); stream D-slices in row chunks.
// ---------------------------------------------------------------------------------------------------
bool InTrainingUpdateGradStreamTiling::IsCapable() { return reduceR_ > 0; }

ge::graphStatus InTrainingUpdateGradStreamTiling::DoOpTiling()
{
    int64_t perCoreGroups = Ops::Base::CeilDiv(groupNum_, static_cast<int64_t>(aicoreParams_.blockDim));
    if (perCoreGroups < 1) {
        perCoreGroups = 1;
    }
    blockNum_ = Ops::Base::CeilDiv(groupNum_, perCoreGroups);

    int64_t elemSize = (dyDataType_ == ge::DT_FLOAT16) ? FP16_BYTE : FP32_BYTE;
    int64_t regPad = vectorLength_;
    int64_t overhead = 2 * (numC0_ * FP32_BYTE + regPad) +        // variance + mean
                       DOUBLE_BUFFER * (numC0_ * FP32_BYTE) * 2 + // res_gamma + res_beta
                       numC0_ * FP32_BYTE +                       // rstd
                       DOUBLE_BUFFER * regPad * 2;                // register pad reserved for dy + x
    int64_t avail = static_cast<int64_t>(aicoreParams_.ubSize) - overhead;
    int64_t perRowBytes = numC0_ * elemSize;
    int64_t streamTileRows = avail / (DOUBLE_BUFFER * perRowBytes * 2);
    if (streamTileRows < 1) {
        streamTileRows = 1;
    }
    if (streamTileRows > numHW_) {
        streamTileRows = numHW_;
    }

    td_.numC1 = static_cast<uint32_t>(numC1_);
    td_.numD = static_cast<uint32_t>(numD_);
    td_.numHW = static_cast<uint32_t>(numHW_);
    td_.numC0 = static_cast<uint32_t>(numC0_);
    td_.reduceR = static_cast<uint32_t>(reduceR_);
    td_.groupNum = static_cast<uint32_t>(groupNum_);
    td_.usedCoreNum = static_cast<uint32_t>(blockNum_);
    td_.perCoreGroups = static_cast<uint32_t>(perCoreGroups);
    td_.blockLenElem = static_cast<uint32_t>(blockLenElem_);
    td_.streamTileRows = static_cast<uint32_t>(streamTileRows);
    td_.epsilon = epsilon_;
    return ge::GRAPH_SUCCESS;
}

uint64_t InTrainingUpdateGradStreamTiling::GetTilingKey() const { return TILINGKEY_STREAM; }

ge::graphStatus InTrainingUpdateGradStreamTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(td_), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &td_, sizeof(td_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------------------------------
static ge::graphStatus Tiling4INTrainingUpdateGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("INTrainingUpdateGrad", "Tiling context is nullptr"),
                return ge::GRAPH_FAILED);
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4INTrainingUpdateGrad(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("INTrainingUpdateGrad", "TilingParse context is nullptr"),
                return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<InTrainingUpdateGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                OP_LOGE(context, "Get core num failed, core num: %lu", compileInfo->coreNum), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context, "Get ub size failed, ub size: %lu", compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF((compileInfo->ubBlockSize <= 0),
                OP_LOGE(context, "Get block size failed, block size: %lu", compileInfo->ubBlockSize),
                return ge::GRAPH_FAILED);
    compileInfo->vectorLength = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF((compileInfo->vectorLength <= 0),
                OP_LOGE(context, "Get vector length failed, vector length: %u", compileInfo->vectorLength),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(INTrainingUpdateGrad)
    .Tiling(Tiling4INTrainingUpdateGrad)
    .TilingParse<InTrainingUpdateGradCompileInfo>(TilingPrepare4INTrainingUpdateGrad);

REGISTER_OPS_TILING_TEMPLATE(INTrainingUpdateGrad, InTrainingUpdateGradReduceEmptyTiling, IN_UG_REDUCE_EMPTY_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(INTrainingUpdateGrad, InTrainingUpdateGradFullLoadTiling, IN_UG_FULL_LOAD_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(INTrainingUpdateGrad, InTrainingUpdateGradStreamTiling, IN_UG_STREAM_PRIORITY);

} // namespace optiling
