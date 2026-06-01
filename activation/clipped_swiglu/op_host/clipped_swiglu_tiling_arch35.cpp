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
 * \file clipped_swiglu_tiling_arch35.cpp
 * \brief Tiling implementation for ClippedSwiglu Arch35 (Ascend 950)
 */

#include <iostream>
#include <cstring>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "clipped_swiglu_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/arch35/clipped_swiglu_tiling_data.h"
#include "../op_kernel/arch35/clipped_swiglu_tiling_key.h"

using namespace ge;
using namespace ClippedSwigluOp;

namespace optiling {

constexpr int64_t X_INDEX = 0;
constexpr int64_t GROUP_INDEX_INDEX = 1;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t DIM_INDEX = 0;
constexpr int64_t ALPHA_INDEX = 1;
constexpr int64_t LIMIT_INDEX = 2;
constexpr int64_t BIAS_INDEX = 3;
constexpr int64_t INTERLEAVED_INDEX = 4;

constexpr int64_t CONST_2 = 2;
constexpr int64_t CONST_4 = 4;
constexpr int64_t CONST_7 = 7; // inputUb = 2x*db outputUb = x*db vectorUb = x
constexpr int64_t CONST_8 = 8; // int64 size is 8
constexpr int64_t DB_BUFFER = 2;

constexpr float CLAMP_LIMIT_DEFAULT = 7.0;
constexpr float GLU_ALPHA_DEFAULT = 1.702;
constexpr float GLU_BIAS_DEFAULT = 1.0;

static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

class ClippedSwigluArch35Tiling {
public:
    explicit ClippedSwigluArch35Tiling(gert::TilingContext* context) : context_(context) {}

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus CheckAndGetXAndAttrs();
    ge::graphStatus CheckInputX(const gert::Shape& inputShapeX, int64_t xSize);
    ge::graphStatus CheckAfterDim(const gert::Shape &xShape, const gert::Shape &inputShapeY, std::string shapeMsg);
    ge::graphStatus CheckAndGetGroupIndex();
    ge::graphStatus CheckY();
    ge::graphStatus CountUbFactor();
    void ComputeCoreSplit();
    void SetTilingKey();
    void FillTilingData();
    void PrintTilingInfo();

private:
    gert::TilingContext* context_;
    ClippedSwigluArch35TilingData* tilingData_ = nullptr;

    uint64_t coreNumAll_ = 0;
    uint64_t ubSize_ = 0;
    int64_t blockSize_ = 0;
    int64_t xDims_ = 0;
    int64_t cutDim_ = 0;
    int64_t dimBatchSize_ = 1;
    int64_t dim2H_ = 1;
    int64_t dimH_ = 1;
    int64_t xCutDimNum_ = 0;
    ge::DataType xDtype_ = ge::DT_FLOAT;
    int64_t dtypeSize_ = CONST_2;
    int64_t isGroup_ = 0;
    int64_t isInterleaved_ = 1;
    float gluLimit_ = 0.0;
    float gluAlpha_ = 0.0;
    float gluBias_ = 0.0;
    int64_t hUbFactor_ = 1;
    int64_t bUbFactor_ = 1;
    int64_t groupNum_ = 0;
    int64_t realCoreNum_ = 0;
    uint64_t tilingKey_ = 0;
    int64_t workspaceSize_ = 0;
};

ge::graphStatus ClippedSwigluArch35Tiling::Init()
{
    tilingData_ = context_->GetTilingData<ClippedSwigluArch35TilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_, "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (memset_s(tilingData_, sizeof(ClippedSwigluArch35TilingData), 0, sizeof(ClippedSwigluArch35TilingData)) != EOK),
        OP_LOGE(context_, "memset tilingdata failed"), return ge::GRAPH_FAILED);
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "GetPlatformInfo failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluArch35Tiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNumAll_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNumAll_ <= 0), OP_LOGE(context_, "core num must > 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize_ = ubSize;
    OP_CHECK_IF((ubSize_ == 0), OP_LOGE(context_, "ubSize must > 0"), return ge::GRAPH_FAILED);
    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF((blockSize_ <= 0), OP_LOGE(context_, "block size is invalid."), return ge::GRAPH_FAILED);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluArch35Tiling::DoTiling()
{
    OP_CHECK_IF(
        CheckAndGetXAndAttrs() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check x and attrs failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckAndGetGroupIndex() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check group_index failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckY() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check y failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CountUbFactor() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CountUbFactor failed."), return ge::GRAPH_FAILED);
    ComputeCoreSplit();
    SetTilingKey();
    FillTilingData();
    PrintTilingInfo();

    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(realCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluArch35Tiling::CheckInputX(const gert::Shape& inputShapeX, int64_t xSize)
{
    std::string reasonMsg = "in [" + std::to_string(0) + ", " + std::to_string(xDims_ - 1) + "]";
    OP_CHECK_IF(
        (cutDim_ > (xDims_ - 1) || cutDim_ < 0),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "dim", std::to_string(cutDim_), reasonMsg),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xSize <= 0,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context_->GetNodeName(), "x", std::to_string(xSize), "x shape size must > 0"),
        return ge::GRAPH_FAILED);
    xCutDimNum_ = inputShapeX.GetDim(cutDim_);
    dimBatchSize_ = 1;
    dim2H_ = 1;
    if (xDims_ == 1) {
        dimBatchSize_ = 1;
        dim2H_ = inputShapeX.GetDim(0);
    } else {
        for (int64_t i = 0; i < cutDim_; i++) {
            dimBatchSize_ *= inputShapeX.GetDim(i);
        }
        for (int64_t j = cutDim_; j < xDims_; j++) {
            dim2H_ *= inputShapeX.GetDim(j);
        }
    }
    dimH_ = dim2H_ / CONST_2;
    std::string reason = "xShape[ " + std::to_string(cutDim_) + "] must be divisible by 2";
    OP_CHECK_IF(
        xCutDimNum_ % 2 != 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x", Ops::Base::ToString(inputShapeX).c_str(), reason.c_str()),
        return ge::GRAPH_FAILED);
    auto descX = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, descX);
    xDtype_ = descX->GetDataType();
    OP_CHECK_IF(
        (SUPPORT_DTYPE.find(xDtype_) == SUPPORT_DTYPE.end()),
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
            "float16, bfloat16, float32"),
        return ge::GRAPH_FAILED);
    if (xDtype_ == ge::DT_FLOAT) {
        dtypeSize_ = CONST_4;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluArch35Tiling::CheckAndGetXAndAttrs()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto* attrDim = attrs->GetAttrPointer<int64_t>(DIM_INDEX);
    cutDim_ = attrDim == nullptr ? -1 : *attrDim;
    auto* attrAlpha = attrs->GetAttrPointer<float>(ALPHA_INDEX);
    gluAlpha_ = attrAlpha == nullptr ? GLU_ALPHA_DEFAULT : *attrAlpha;
    auto* attrLimit = attrs->GetAttrPointer<float>(LIMIT_INDEX);
    gluLimit_ = attrLimit == nullptr ? CLAMP_LIMIT_DEFAULT : *attrLimit;

    OP_CHECK_IF(
        gluLimit_ <= 0.0f,
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "limit", std::to_string(gluLimit_), "> 0"),
        return ge::GRAPH_FAILED);
    auto* attrBias = attrs->GetAttrPointer<float>(BIAS_INDEX);
    gluBias_ = attrBias == nullptr ? GLU_BIAS_DEFAULT : *attrBias;
    auto* attrInterleaved = attrs->GetAttrPointer<bool>(INTERLEAVED_INDEX);
    bool interleaved = attrInterleaved == nullptr ? true : *attrInterleaved;
    isInterleaved_ = interleaved ? 1 : 0;

    auto shapeX = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
    const gert::Shape& inputShapeX = shapeX->GetStorageShape();
    xDims_ = inputShapeX.GetDimNum();
    OP_CHECK_IF(
        xDims_ <= 0, OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(xDims_), "> 0"),
        return ge::GRAPH_FAILED);
    if (cutDim_ < 0) {
        cutDim_ = cutDim_ + xDims_;
    }
    int64_t xSize = inputShapeX.GetShapeSize();
    OP_CHECK_IF(
        CheckInputX(inputShapeX, xSize) != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "check input x failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluArch35Tiling::CheckAndGetGroupIndex()
{
    auto shapeGroupIndex = context_->GetOptionalInputShape(GROUP_INDEX_INDEX);
    if (shapeGroupIndex == nullptr) {
        isGroup_ = 0;
    } else {
        isGroup_ = 1;
        const gert::Shape& inputShapeGroupIndex = shapeGroupIndex->GetStorageShape();
        int64_t groupIndexDim = inputShapeGroupIndex.GetDimNum();
        auto descGroupIndex = context_->GetOptionalInputDesc(GROUP_INDEX_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, descGroupIndex);
        auto groupIndexDtype = descGroupIndex->GetDataType();
        OP_CHECK_IF(
            groupIndexDim != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "group_index", std::to_string(groupIndexDim), "1D"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            groupIndexDtype != ge::DT_INT64,
            OP_LOGE_FOR_INVALID_DTYPE(
                context_->GetNodeName(), "group_index", ge::TypeUtils::DataTypeToSerialString(groupIndexDtype).c_str(),
                "int64"),
            return ge::GRAPH_FAILED);
        groupNum_ = inputShapeGroupIndex.GetDim(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluArch35Tiling::CheckAfterDim(const gert::Shape &xShape, const gert::Shape &inputShapeY, std::string shapeMsg)
{
    if (cutDim_ < xDims_ - 1) {
        for (int64_t i = cutDim_ + 1; i < xDims_; i++) {
            int64_t xShapeValue = xShape.GetDim(i);
            int64_t yShapeValue = inputShapeY.GetDim(i);
            std::string shapeMsgValue =
                "xShape[" + std::to_string(i) + "] should be equal yShape[" + std::to_string(i) + "].";
            OP_CHECK_IF(
                xShapeValue != yShapeValue,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and y", shapeMsg, shapeMsgValue),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluArch35Tiling::CheckY()
{
    auto shapeY = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeY);
    const gert::Shape& inputShapeY = shapeY->GetStorageShape();
    int64_t yDims = inputShapeY.GetDimNum();
    auto descY = context_->GetInputDesc(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, descY);
    auto yDtype = descY->GetDataType();
    auto xShape = context_->GetInputShape(0)->GetStorageShape();
    OP_CHECK_IF(
        yDims != xDims_,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y", std::to_string(yDims), std::to_string(xDims_)),
        return ge::GRAPH_FAILED);

    std::string reasonMsg = "x shape is " + Ops::Base::ToString(xShape) + "xShape[" + std::to_string(cutDim_) +
                            "] / 2 must be equal yShape[" + std::to_string(cutDim_) + "]";
    OP_CHECK_IF(
        inputShapeY.GetDim(cutDim_) != (xCutDimNum_ / CONST_2),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "y", Ops::Base::ToString(inputShapeY).c_str(), reasonMsg.c_str()),
        return ge::GRAPH_FAILED);

    std::string dtypeMsg = "x dtype is " + ge::TypeUtils::DataTypeToSerialString(xDtype_) + " , y dtype is " +
                           ge::TypeUtils::DataTypeToSerialString(yDtype);
    OP_CHECK_IF(
        yDtype != xDtype_,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x and y", dtypeMsg.c_str(), "y dtype must be same as x dtype"),
        return ge::GRAPH_FAILED);
    std::string shapeMsg =
        "x shape is " + Ops::Base::ToString(xShape) + " y shape is " + Ops::Base::ToString(inputShapeY);
    for (int64_t i = 0; i < cutDim_; i++) {
        int64_t xShapeValue = xShape.GetDim(i);
        int64_t yShapeValue = inputShapeY.GetDim(i);
        std::string shapeMsgValue =
            "xShape[" + std::to_string(i) + "] should be equal yShape[" + std::to_string(i) + "].";
        OP_CHECK_IF(
            xShapeValue != yShapeValue,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and y", shapeMsg, shapeMsgValue),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(CheckAfterDim(xShape, inputShapeY, shapeMsg) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CheckAfterDim failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluArch35Tiling::CountUbFactor()
{
    hUbFactor_ = 1;
    int64_t groupIndexBuf = 0;
    if (isGroup_ != 0) {
        groupIndexBuf = blockSize_;
    }
    int64_t oneBlockNum = blockSize_ / dtypeSize_;
    int64_t oneBlockNumG = blockSize_ / CONST_8;
    int64_t allUbNum = (static_cast<int64_t>(ubSize_) - groupIndexBuf);
    int64_t ubFactor = allUbNum / (CONST_7 * dtypeSize_);
    ubFactor = Ops::Base::FloorDiv(ubFactor, oneBlockNum) * oneBlockNum;
    hUbFactor_ = ubFactor;
    int64_t groupNumAlign = Ops::Base::CeilDiv(groupNum_, oneBlockNumG) * oneBlockNumG;
    int64_t groupUb = groupNumAlign * CONST_8;
    OP_CHECK_IF(
        (groupUb > (ubFactor * CONST_2) || ubFactor <= 0),
        OP_LOGE(
            context_, "ubFactor must > 0 and groupUb <= (ubFactor*CONST_2), but ubFactor is %ld, groupUb is %ld",
            ubFactor, groupUb),
        return ge::GRAPH_FAILED);
    if ((dimH_ < hUbFactor_) && (isInterleaved_ == 0)) {
        hUbFactor_ = Ops::Base::FloorDiv(dimH_, oneBlockNum) * oneBlockNum;
        if (hUbFactor_ <= 0) {
            hUbFactor_ = oneBlockNum;
        }
        bUbFactor_ = ubFactor / hUbFactor_;
    } else {
        bUbFactor_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

void ClippedSwigluArch35Tiling::ComputeCoreSplit()
{
    if (isInterleaved_ != 0) {
        int64_t pairTotal = dim2H_ * dimBatchSize_ / CONST_2;
        int64_t blockFactor = (pairTotal + coreNumAll_ - 1) / coreNumAll_;
        realCoreNum_ = (pairTotal + blockFactor - 1) / blockFactor;
    } else {
        int64_t hCore = 1;
        int64_t bBlockFactor = (dimBatchSize_ + coreNumAll_ - 1) / coreNumAll_;
        int64_t bCore = (dimBatchSize_ + bBlockFactor - 1) / bBlockFactor;
        int64_t core = coreNumAll_ / bCore;
        if (core > 1) {
            hCore = core;
            int64_t hBlockFactor = (dimH_ + hCore - 1) / hCore;
            hCore = (dimH_ + hBlockFactor - 1) / hBlockFactor;
        }
        realCoreNum_ = bCore * hCore;
        if (isGroup_ == 1) {
            realCoreNum_ = coreNumAll_;
        }
    }
    return;
}

void ClippedSwigluArch35Tiling::SetTilingKey()
{
    uint64_t isInterleavedKey = (isInterleaved_ != 0) ? TPL_INTERLEAVED_TRUE : TPL_INTERLEAVED_FALSE;
    uint64_t isGroupKey = (isGroup_ != 0) ? TPL_GROUP_INDEX : TPL_NO_GROUP_INDEX;
    OP_LOGI(context_->GetNodeName(), "isInterleavedKey = %lu, isGroupKey = %lu", isInterleavedKey, isGroupKey);
    tilingKey_ = GET_TPL_TILING_KEY(isInterleavedKey, isGroupKey);
}

void ClippedSwigluArch35Tiling::FillTilingData()
{
    tilingData_->dimBatchSize = dimBatchSize_;
    tilingData_->dimH = dimH_;
    tilingData_->gluAlpha = gluAlpha_;
    tilingData_->gluLimit = gluLimit_;
    tilingData_->gluBias = gluBias_;
    tilingData_->hUbFactor = hUbFactor_;
    tilingData_->bUbFactor = bUbFactor_;
    tilingData_->groupNum = groupNum_;
    tilingData_->realCoreNum = realCoreNum_;
}

void ClippedSwigluArch35Tiling::PrintTilingInfo()
{
    std::ostringstream info;
    info << "Print tilingData: tilingKey_: " << tilingKey_;
    info << ", coreNumAll: " << coreNumAll_;
    info << ", ubSize_: " << ubSize_;
    info << ", dimBatchSize: " << dimBatchSize_;
    info << ", dim2H: " << dim2H_;
    info << ", dimH: " << dimH_;
    info << ", isGroup: " << isGroup_;
    info << ", isInterleaved: " << isInterleaved_;
    info << ", gluLimit: " << gluLimit_;
    info << ", gluAlpha: " << gluAlpha_;
    info << ", gluBias: " << gluBias_;
    info << ", hUbFactor: " << hUbFactor_;
    info << ", bUbFactor: " << bUbFactor_;
    info << ", groupNum: " << groupNum_;
    info << ", realCoreNum: " << realCoreNum_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus Tiling4ClippedSwigluArch35(gert::TilingContext* context)
{
    OP_LOGI("ClippedSwigluArch35Tiling", "Enter Tiling4ClippedSwigluArch35");
    ClippedSwigluArch35Tiling tilingImpl = ClippedSwigluArch35Tiling(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4ClippedSwigluArch35 init failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4ClippedSwigluArch35 do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI("ClippedSwigluArch35Tiling", "Tiling4ClippedSwigluArch35 done.");
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
