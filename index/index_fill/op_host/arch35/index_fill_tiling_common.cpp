/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file index_fill_tiling_common.cpp
 * \brief
 */

#include "op_api/runtime2_util.h"
#include "op_api/op_util.h"
#include "log/log.h"
#include "op_common/log/log.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "index_fill_tiling_arch35.h"
#include "index_fill_tiling_common.h"

namespace optiling
{
void CalculatePNQ(gert::Shape& xShape, int64_t dim, IndexFillInputInfo& inputData)
{
    // 非尾轴场景，统一合轴为shape: (P, N, Q); 尾轴场景，统一合轴为shape: (P, N),该shape可以看成是(P, N, 1)
    inputData.N = xShape.GetDim(dim);
    inputData.P = 1;
    inputData.Q = 1;
    for (int64_t i = 0; i < dim; i++) {
        inputData.P *= xShape.GetDim(i);
    }
    for (int64_t i = dim + 1; i < xShape.GetDimNum(); i++) {
        inputData.Q *= xShape.GetDim(i);
    }
}

ge::graphStatus GetIndexFillShapeAttrsInfo(gert::TilingContext* context, IndexFillInputInfo& inputData)
{
    // 校验x和y的shape是否一致；并且dtype是否相同
    auto inputDesc = context->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType xDtype = inputDesc->GetDataType();

    auto outputDesc = context->GetOutputDesc(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    ge::DataType yDtype = outputDesc->GetDataType();
    OP_CHECK_IF(xDtype != yDtype,
                    OP_LOGE(context->GetNodeName(), "x and y should have same dtype, please check."),
                    return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    auto yShapePtr = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    OP_CHECK_IF(xShape != yShape,
                    OP_LOGE(context->GetNodeName(), "x and y should have same shape, please check."),
                    return ge::GRAPH_FAILED);
    inputData.dtype = xDtype;
    inputData.dtypeSize = ge::GetSizeByDataType(xDtype);

    // 校验x和value是否是相同的类型，aclnn侧已经将value转为了x的同类型，这里需要进一步添加校验。
    auto valueDesc = context->GetInputDesc(INDEX_INPUT_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueDesc);
    ge::DataType valueDtype = valueDesc->GetDataType();
    OP_CHECK_IF(xDtype != valueDtype,
                    OP_LOGE(context->GetNodeName(), "x and value should have same dtype, please check."),
                    return ge::GRAPH_FAILED);

    int64_t xDims = xShape.GetDimNum();
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t dim = *(attrs->GetAttrPointer<int64_t>(0)); // 获取算子输入的dim属性值.
    dim = dim >= 0 ? dim : dim + xDims;
    // dim如果不在[0, xDims)这个范围内，需要校验报错
    OP_CHECK_IF(dim < 0 || dim >= xDims,
                    OP_LOGE(context->GetNodeName(), "Dim value error, abs(input dim) is greater than the dim of x, please check."),
                    return ge::GRAPH_FAILED);

    inputData.dim = dim;
    inputData.xDims = xDims;

    CalculatePNQ(xShape, dim, inputData);

    // 获取总处理数据
    auto indicesShape = context->GetInputShape(INDEX_INPUT_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    uint64_t indicesNum = indicesShape->GetStorageShape().GetShapeSize();
    inputData.indicesNum = indicesNum;
    inputData.numel = xShape.GetShapeSize();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillCommonTiling::GetShapeAttrsInfo()
{
    return optiling::GetIndexFillShapeAttrsInfo(context_, inputData);
}

ge::graphStatus IndexFillCommonTiling::GetPlatformInfo()
{
    auto compileInfo = context_->GetCompileInfo<IndexFillCompileInfoArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from compile info.");
        coreNum_ = compileInfo->totalCoreNum;
        ubSize_ = static_cast<int64_t>(compileInfo->ubSizePlatForm);
        sysWorkspaceSize_ = compileInfo->sysWorkspaceSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<uint64_t>(ubSizePlatform);

        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        sysWorkspaceSize_ = (sysWorkspaceSize <= 0) ? WS_SYS_SIZE : sysWorkspaceSize;
    }

    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillCommonTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillCommonTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillCommonTiling::PostTiling()
{
    // 设置blockDim，即参与计算的Vector核数
    context_->SetBlockDim(usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4IndexFillArch35(gert::TilingContext* context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

}  // namespace optiling
