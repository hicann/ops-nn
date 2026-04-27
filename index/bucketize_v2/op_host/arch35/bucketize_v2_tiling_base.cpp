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
 * \file bucketize_v2_tiling_base.cpp
 * \brief
 */

#include "exe_graph/runtime/shape.h"
#include "op_host/tiling_util.h"
#include "bucketize_v2_tiling.h"

using namespace ge;

namespace optiling {
const int32_t INPUT_IDX_X = 0;
const int32_t INPUT_BOUND_IDX = 1;
const int32_t OUT_INT32 = 0;
const int32_t RIGHT_POS = 1;

const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

ge::graphStatus BucketizeV2BaseTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const BucketizeV2CompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum_ = ascendcPlatform.GetCoreNum();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<uint64_t>(ubSizePlatform);
    }
    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

int64_t BucketizeV2BaseTiling::GetUpLog2(int64_t n)
{
    int64_t res = 1;
    int64_t pow2Value = 1;
    while (pow2Value < n) {
        pow2Value = pow2Value << 1;
        res += 1;
    }
    return res;
}

bool BucketizeV2BaseTiling::IsInvalidType(const DataType dataType)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT64,
                                                   ge::DT_INT32, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8};
    bool dtypeInValid = (supportedDtype.count(dataType) == 0);
    return dtypeInValid;
}

ge::graphStatus BucketizeV2BaseTiling::GetShapeAndDtype()
{
    auto xShapePtr = context_->GetRequiredInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    xSize_ = xShape.GetShapeSize();
    auto inputXPtr = context_->GetRequiredInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    xDtype_ = inputXPtr->GetDataType();    
    xDtypeSize_ = ge::GetSizeByDataType(xDtype_);
    OP_CHECK_IF(
        xDtypeSize_ <= 0, OP_LOGE(context_, "xDtypeSize must be greater than 0, xDtypeSize: %ld", xDtypeSize_),
        return ge::GRAPH_FAILED);

    auto boundShapePtr = context_->GetRequiredInputShape(INPUT_BOUND_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, boundShapePtr);
    auto boundShape = boundShapePtr->GetStorageShape();
    boundSize_ = boundShape.GetShapeSize();
    size_t boundDim = boundShape.GetDimNum();
    OP_CHECK_IF(
        boundDim > 1, OP_LOGE(context_, "boundaries must be 1 dimension, but got : dim(%zu)", boundDim),
        return ge::GRAPH_FAILED);

    auto inputBoundPtr = context_->GetRequiredInputDesc(INPUT_BOUND_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputBoundPtr);
    boundDtype_ = inputBoundPtr->GetDataType();    
    boundDtypeSize_ = ge::GetSizeByDataType(boundDtype_);
    OP_CHECK_IF(
        boundDtypeSize_ <= 0, OP_LOGE(context_, "boundDtypeSize must be greater than 0, boundDtypeSize: %ld", boundDtypeSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        boundDtype_ != xDtype_, OP_LOGE(context_, "boundaries dtype must be same with x dtype"),
        return ge::GRAPH_FAILED);

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    ySize_ = yShape.GetShapeSize();   
    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    yDtype_ = outputYPtr->GetDataType();
    yDtypeSize_ = ge::GetSizeByDataType(yDtype_);
    OP_CHECK_IF(
        yDtypeSize_ <= 0, OP_LOGE(context_, "yDtypeSize must be greater than 0, yDtypeSize: %ld", yDtypeSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        IsInvalidType(xDtype_), 
        OP_LOGE(context_, "input X dtype only support fp16, fp32, bf16 int8, int16, int32, int64, uint8 currently, please check."), 
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BucketizeV2BaseTiling::GetAttrsInfo()
{
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    const bool* outInt32Ptr = runtimeAttrs->GetAttrPointer<bool>(OUT_INT32);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outInt32Ptr);
    outInt32_ = *outInt32Ptr;
    const bool* rightPtr = runtimeAttrs->GetAttrPointer<bool>(RIGHT_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rightPtr);
    right_ = *rightPtr;
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BucketizeV2BaseTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(GetAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetAttrsInfo fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetShapeAndDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetShapeAndDtype fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool BucketizeV2BaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus BucketizeV2BaseTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BucketizeV2BaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t BucketizeV2BaseTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus BucketizeV2BaseTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BucketizeV2BaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling