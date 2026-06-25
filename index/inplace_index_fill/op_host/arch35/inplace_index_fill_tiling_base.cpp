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
 * \file inplace_index_fill_tiling_base.cpp
 * \brief
 */

#include "inplace_index_fill_tiling_base.h"
#include "log/log.h"
#include "../../op_kernel/arch35/inplace_index_fill_tiling_key.h"
#include "op_host/tiling_templates_registry.h"

using namespace InplaceIndexFill;

namespace optiling {
// x value数据类型相同，在aclnn处理value数据类型，与x对齐
static const std::set<ge::DataType> X_SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_FLOAT16, ge::DT_BF16,
                                                       ge::DT_INT8,  ge::DT_UINT8,  ge::DT_INT16,   ge::DT_INT32,
                                                       ge::DT_INT64, ge::DT_BOOL};
static const std::set<ge::DataType> INDICES_SUPPORT_DTYPE = {ge::DT_INT32, ge::DT_INT64};
constexpr uint64_t DACHE_SIZE = 32 * 1024;

// 校验数据类型
inline static bool IsSupportDtype(const std::set<ge::DataType>& supportDtype, const ge::DataType dtype)
{
    return (supportDtype.count(dtype) != 0);
}
ge::graphStatus InplaceIndexFillTilingBase::CheckDataType()
{
    const char* opName_ = "InplaceIndexFill";
    auto inputXShape_ = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXShape_);
    int64_t xShapeSize = inputXShape_->GetStorageShape().GetShapeSize();
    if (xShapeSize <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "x", std::to_string(xShapeSize).c_str(),
            "shape size must be greater than zero");
        return ge::GRAPH_FAILED;
    }
    auto indicesShape_ = context_->GetInputShape(INPUT_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShape_);
    int64_t indicesShapeSize = indicesShape_->GetStorageShape().GetShapeSize();
    if (indicesShapeSize <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "indices", std::to_string(indicesShapeSize).c_str(),
            "shape size must be greater than zero");
        return ge::GRAPH_FAILED;
    }
    // 校验x的dtype是否满足
    auto inputXDesc = context_->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    auto xDType = inputXDesc->GetDataType();
    if (!IsSupportDtype(X_SUPPORT_DTYPE, xDType)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "x",
            std::to_string(static_cast<int32_t>(xDType)).c_str(),
            "dtype must be in [DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_BOOL]");
        return ge::GRAPH_FAILED;
    }
    // 校验x和y的dtype是否相同
    auto outputDesc = context_->GetOutputDesc(OUTPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto yDType = outputDesc->GetDataType();
    if (xDType != yDType) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "x, y",
            (std::to_string(static_cast<int32_t>(xDType)) + ", " + std::to_string(static_cast<int32_t>(yDType))).c_str(),
            "x and y must have the same dtype");
        return ge::GRAPH_FAILED;
    }
    // 校验x和value的dtype是否相同
    auto valueDesc = context_->GetInputDesc(INPUT_VALUE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueDesc);
    auto valueDType = valueDesc->GetDataType();
    if (xDType != valueDType) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "x, value",
            (std::to_string(static_cast<int32_t>(xDType)) + ", " + std::to_string(static_cast<int32_t>(valueDType))).c_str(),
            "x and value must have the same dtype");
        return ge::GRAPH_FAILED;
    }
    // 校验indices数据类型
    auto indicesDesc = context_->GetInputDesc(INPUT_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    auto indicesDType = indicesDesc->GetDataType();
    if (!IsSupportDtype(INDICES_SUPPORT_DTYPE, indicesDType)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "indices",
            std::to_string(static_cast<int32_t>(indicesDType)).c_str(),
            "dtype must be DT_INT32 or DT_INT64");
        return ge::GRAPH_FAILED;
    }
    inputData.xDtypeSize = ge::GetSizeByDataType(xDType);
    inputData.indicesDtypeSize = ge::GetSizeByDataType(indicesDType);
    inputData.indicesDtype = indicesDType;
    return ge::GRAPH_SUCCESS;
}

// 校验shape与合轴处理
void InplaceIndexFillTilingBase::CalculatePQ(
    const gert::Shape& xShape, int64_t dim, int64_t xDim, InplaceIndexFIllInputInfo& inputDataParam)
{
    for (int64_t i = 0; i < dim; i++) {
        inputDataParam.preDimProduct *= xShape.GetDim(i);
    }
    for (int64_t i = dim + 1; i < xDim; i++) {
        inputDataParam.postDimProduct *= xShape.GetDim(i);
    }
}
ge::graphStatus InplaceIndexFillTilingBase::GetShapeAttrsInfo()
{
    const char* opName_ = "InplaceIndexFill";
    if (CheckDataType() != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "x, y, value, indices", "unknown",
            "please check the data types of input and output");
        return ge::GRAPH_FAILED;
    }

    // 校验x和y的shape是否相同，value、indices无需校验shape
    auto xShapePtr = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    int64_t xDim = xShape.GetDimNum();
    auto yShapePtr = context_->GetInputShape(OUTPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    if (xShape != yShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "x, y", "xShape, yShape",
            "x and y must have the same shape");
        return ge::GRAPH_FAILED;
    }
    // 校验dim满足xShape
    auto dimAttr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, dimAttr);
    auto dimIdx = 0;
    auto dimP = dimAttr->GetAttrPointer<int64_t>(dimIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dimP);
    auto dim_ = *dimP;
    if (!((-xDim <= dim_) && (dim_ < xDim))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "dim", std::to_string(dim_).c_str(),
            "dim index must be in range [-xDim, xDim)");
        return ge::GRAPH_FAILED;
    }
    // dim处理
    auto curDim = dim_ >= 0 ? dim_ : dim_ + xDim;
    inputData.dim = curDim;
    inputData.dimSize = xShape.GetDim(curDim);
    CalculatePQ(xShape, curDim, xDim, inputData);
    // totalDataSize处理
    auto indicesShapePtr = context_->GetInputShape(INPUT_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShapePtr);
    auto indicesShape = indicesShapePtr->GetStorageShape();
    inputData.indicesNum = indicesShape.GetDim(0);
    inputData.totalDataSize = inputData.preDimProduct * inputData.indicesNum * inputData.postDimProduct;
    inputData.numel = inputData.preDimProduct * inputData.dimSize * inputData.postDimProduct;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplaceIndexFillTilingBase::GetPlatformInfo()
{
    const char* opName_ = "InplaceIndexFill";
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const InplaceIndexFillCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<uint64_t>(ubSizePlatform);
    }
    if (coreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "coreNum", "0", "coreNum must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForInplaceIndexFill(gert::TilingParseContext* context_)
{
    OP_LOGD(context_->GetNodeName(), "TilingPrepareForInplaceIndexFill is running.");

    auto compileInfoPtr = context_->GetCompiledInfo<InplaceIndexFillCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

bool InplaceIndexFillTilingBase::IsCapable()
{
    return true;
}

ge::graphStatus InplaceIndexFillTilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplaceIndexFillTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplaceIndexFillTilingBase::GetWorkspaceSize()
{
    size_t sysWorkspaceSize = SYS_WORKSPACE_SIZE;
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    }
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplaceIndexFillTilingBase::PostTiling()
{
    // 设置blockDim，即参与计算的Vector核数
    context_->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

uint64_t InplaceIndexFillTilingBase::GetTilingKey() const
{
    uint64_t dtypeMode = (inputData.xDtypeSize <= 4)
        ? static_cast<uint64_t>(TPL_MODE_DTYPE_B32)
        : static_cast<uint64_t>(TPL_MODE_DTYPE_B64);
    return GET_TPL_TILING_KEY(TPL_MODE_TEMPLATE_SIMT, dtypeMode);
}

static ge::graphStatus InplaceIndexFillTilingArch35(gert::TilingContext* context_)
{
    OP_LOGD(context_->GetNodeName(), "Tiling for InplaceIndexFill start.");
    ge::graphStatus result = optiling::Tiling4InplaceIndexFillArch35(context_);
    OP_LOGD(context_->GetNodeName(), "Tiling for InplaceIndexFill end.");
    return result;
}

ge::graphStatus Tiling4InplaceIndexFillArch35(gert::TilingContext* context_)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context_);
}

IMPL_OP_OPTILING(InplaceIndexFill)
    .Tiling(Tiling4InplaceIndexFillArch35)
    .TilingParse<InplaceIndexFillCompileInfo>(TilingPrepareForInplaceIndexFill);
} // namespace optiling