/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_fill_d_tiling.cc
 * \brief
 */
#include "index_fill_d_tiling.h"
#include "log/log.h"
#include "op_host/util/math_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
using namespace Ops::Base;
const int64_t BUFFER_NUM            = 2;
const int64_t ALL_BUFFERS           = 4;
const int64_t INPUT_X_IDX           = 0;
const int64_t INPUT_ASSIST1_IDX     = 1;
const int64_t INPUT_ASSIST2_IDX     = 2;
const int64_t OUTPUT_Y_IDX          = 0;
const int64_t MIN_DATA_SIZE         = 1024;
const int64_t ASCEND_WORKSPACE      = 16 * 1024 * 1024;
const uint64_t TILING_KEY_COMMON    = 200;
const uint64_t ALIGN_SIZE           = 512;

static const std::set<ge::DataType> SUPPORT_DTYPE = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64,ge::DT_BOOL, ge::DT_BF16
};

ge::graphStatus IndexFillDTiling::CheckShape()
{
    auto xShapePtr = context_->GetRequiredInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    auto assist1ShapePtr = context_->GetRequiredInputShape(INPUT_ASSIST1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, assist1ShapePtr);
    auto assist1Shape = assist1ShapePtr->GetStorageShape();

    auto assist2ShapePtr = context_->GetRequiredInputShape(INPUT_ASSIST2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, assist2ShapePtr);
    auto assist2Shape = assist2ShapePtr->GetStorageShape();

    auto yShapePtr = context_->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape != assist1Shape || xShape != assist2Shape || xShape != yShape,
               OP_LOGE(context_->GetNodeName(),
               "input x, assist1, assist2 and y shape must be same, please check"),
               return ge::GRAPH_FAILED);
    inputXShape_ = xShape;
    return ge::GRAPH_SUCCESS;
}

inline static bool IsSupportDtype(const std::set<ge::DataType> &supportDtype, const ge::DataType dtype)
{
    return (supportDtype.count(dtype) != 0);
}

ge::graphStatus IndexFillDTiling::CheckDataType()
{
    auto inputXPtr = context_->GetRequiredInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    dType_ = inputXPtr->GetDataType();
    auto assist1Ptr = context_->GetRequiredInputDesc(INPUT_ASSIST1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, assist1Ptr);
    auto assist1DType = assist1Ptr->GetDataType();
    auto assist2Ptr = context_->GetRequiredInputDesc(INPUT_ASSIST2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, assist2Ptr);
    auto assist2DType = assist2Ptr->GetDataType();
    auto outputPtr = context_->GetOutputDesc(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputPtr);
    auto outputDtype = outputPtr->GetDataType();
    OP_CHECK_IF(!IsSupportDtype(SUPPORT_DTYPE, dType_), OP_LOGE(context_->GetNodeName(),
        "The dtype only support float32, float16, int32, int64, bool, bfloat16 \
        currently, please check."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dType_ != assist1DType || dType_ != assist2DType || dType_ != outputDtype,
               OP_LOGE(context_->GetNodeName(),
               "input x, assist1, assist2 and y dtype must be same, please check"),
               return ge::GRAPH_FAILED);
    dataTypeSize_ = ge::GetSizeByDataType(dType_);
    OP_CHECK_IF(dataTypeSize_ == -1, OP_LOGE(context_->GetNodeName(),
        "Get the size of dtype failed, please check."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool IndexFillDTiling::IsCapable() {
    return true;
}

ge::graphStatus IndexFillDTiling::GetShapeAttrsInfo() {
    if (CheckDataType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return CheckShape();
}

ge::graphStatus IndexFillDTiling::GetPlatformInfo() {
    auto compileInfo = reinterpret_cast<const IndexFillDCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillDTiling::DoOpTiling() {
    inputShapeSize_ = inputXShape_.GetShapeSize();
    int64_t maxUbAvailable = ubSize_ / (BUFFER_NUM * ALL_BUFFERS * dataTypeSize_);
    normalCoreData_ = std::max(CeilDiv(inputShapeSize_, totalCoreNum_), MIN_DATA_SIZE);
    usedCoreNum_ = CeilDiv(inputShapeSize_, normalCoreData_);
    tailCoreData_ = inputShapeSize_ - (usedCoreNum_ - 1) * normalCoreData_;
    ubFactor_ = (maxUbAvailable / ALIGN_SIZE) * ALIGN_SIZE;
    normalCoreLoop_ = (normalCoreData_ + ubFactor_ - 1) / ubFactor_;
    tailUbFactor_ = normalCoreData_ - (normalCoreLoop_ - 1) * ubFactor_;
    tailCoreLoop_ = (tailCoreData_ + ubFactor_ - 1) / ubFactor_;
    tailCoreTailUbFactor_ = tailCoreData_ - (tailCoreLoop_ - 1) * ubFactor_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillDTiling::DoLibApiTiling() {
    return ge::GRAPH_SUCCESS;
}

uint64_t IndexFillDTiling::GetTilingKey() const {
    return TILING_KEY_COMMON;
}

ge::graphStatus IndexFillDTiling::GetWorkspaceSize() {
    workspaceSize_ = ASCEND_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillDTiling::PostTiling() {
    tilingData_.set_normalCoreData(normalCoreData_);
    tilingData_.set_tailCoreData(tailCoreData_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_tailUbFactor(tailUbFactor_);
    tilingData_.set_tailCoreTailUbFactor(tailCoreTailUbFactor_);
    tilingData_.set_normalCoreLoop(normalCoreLoop_);
    tilingData_.set_tailCoreLoop(tailCoreLoop_);
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetBlockDim(usedCoreNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void IndexFillDTiling::DumpTilingInfo() {
    std::ostringstream info;
    info << "usedCoreNum: " << usedCoreNum_;
    info << ", normalCoreData: " << normalCoreData_;
    info << ", tailCoreData: " << tailCoreData_;
    info << ", ubFactor: " << ubFactor_;
    info << ", tailUbFactor: " << tailUbFactor_;
    info << ", tailCoreTailUbFactor: " << tailCoreTailUbFactor_;
    info << ", normalCoreLoop: " << normalCoreLoop_;
    info << ", tailCoreLoop: " << tailCoreLoop_;
    info << ", tilingKey: " << GetTilingKey();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

static ge::graphStatus Tiling4IndexFillD(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "TilingIndexFillD rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const IndexFillDCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    IndexFillDTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.DoTiling();
}

ge::graphStatus TilingPrepareForIndexFillD(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForIndexFillD is running.");
    auto compileInfo = context->GetCompiledInfo<IndexFillDCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IndexFillD).Tiling(Tiling4IndexFillD).TilingParse<IndexFillDCompileInfo>(TilingPrepareForIndexFillD);
}  // namespace optiling