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
    auto inputXShape_ = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXShape_);
    int64_t xShapeSize = inputXShape_->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(xShapeSize <= 0, OP_LOGE(context_->GetNodeName(), "input x shape size %ld is less than or equal to zero", xShapeSize), return ge::GRAPH_FAILED);
    auto indicesShape_ = context_->GetInputShape(INPUT_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShape_);
    int64_t indicesShapeSize = indicesShape_->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(indicesShapeSize <= 0, OP_LOGE(context_->GetNodeName(), "indices shape size %ld is less than or equal to zero", indicesShapeSize), return ge::GRAPH_FAILED);
    // 校验x的dtype是否满足
    auto inputXDesc = context_->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    auto xDType = inputXDesc->GetDataType();
    OP_CHECK_IF(
        !IsSupportDtype(X_SUPPORT_DTYPE, xDType),
        OP_LOGE(
            context_->GetNodeName(),
            "The dtype only support float32, float16, bfloat16, \
                int32, int64, bool, int8, uint8, int16, double, but got [%s], please check.",
                Ops::Base::ToString(xDType).c_str()),
        return ge::GRAPH_FAILED);
    // 校验x和y的dtype是否相同
    auto outputDesc = context_->GetOutputDesc(OUTPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto yDType = outputDesc->GetDataType();
    OP_CHECK_IF(
        xDType != yDType, OP_LOGE(context_->GetNodeName(), "x and y should have same dtype, please check."),
        return ge::GRAPH_FAILED);
    // 校验x和value的dtype是否相同
    auto valueDesc = context_->GetInputDesc(INPUT_VALUE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueDesc);
    auto valueDType = valueDesc->GetDataType();
    OP_CHECK_IF(
        xDType != valueDType, OP_LOGE(context_->GetNodeName(), "x and value should have same dtype, please check."),
        return ge::GRAPH_FAILED);
    // 校验indices数据类型
    auto indicesDesc = context_->GetInputDesc(INPUT_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    auto indicesDType = indicesDesc->GetDataType();
    OP_CHECK_IF(
        !IsSupportDtype(INDICES_SUPPORT_DTYPE, indicesDType),
        OP_LOGE(context_->GetNodeName(), "indices should be of type int32 or int64."), return ge::GRAPH_FAILED);
    inputData.xDtypeSize = ge::GetSizeByDataType(xDType);
    inputData.indicesDtypeSize = ge::GetSizeByDataType(indicesDType);

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
    OP_CHECK_IF(CheckDataType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "please check the data types of input and output!"),
                return ge::GRAPH_FAILED);

    // 校验x和y的shape是否相同，value、indices无需校验shape
    auto xShapePtr = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    int64_t xDim = xShape.GetDimNum();
    auto yShapePtr = context_->GetInputShape(OUTPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != yShape, OP_LOGE(context_->GetNodeName(), "input x and output y shape must be same, please check."),
        return ge::GRAPH_FAILED);
    // 校验dim满足xShape
    auto dimAttr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, dimAttr);
    auto dimIdx = 0;
    auto dimP = dimAttr->GetAttrPointer<int64_t>(dimIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dimP);
    auto dim_ = *dimP;
    OP_CHECK_IF(
        !((-xDim <= dim_) && (dim_ < xDim)),
        OP_LOGE(context_->GetNodeName(), "dim index out of range, please check."), return ge::GRAPH_FAILED);
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

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplaceIndexFillTilingBase::GetPlatformInfo()
{
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
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
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
    return GET_TPL_TILING_KEY(TPL_MODE_SIMT, TPL_MODE_ADDR_INT32);
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