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
 * \file group_quant_tiling.cc
 * \brief
 */
#include "group_quant_tiling.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_base.h"
#include "util/math_util.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace gert;

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
static const uint64_t TILING_OFFSET = 1000000000000UL;
static const size_t INPUT_INDEX_OF_X = 0;
static const size_t INPUT_INDEX_OF_SCALE = 1;
static const size_t INPUT_INDEX_OF_GROUP_INDEX = 2;
static const size_t INPUT_INDEX_OF_OFFSET = 3;
static const size_t OUTPUT_INDEX_OF_Y = 0;
static const size_t ATTR_INDEX_OF_DST_TYPE = 0;
static const size_t DIM_NUM_OF_X = 2;
static const size_t DIM_NUM_OF_SCALE = 2;
static const size_t DIM_NUM_OF_GROUP_INDEX = 1;
static const size_t DIM_NUM_OF_OFFSET = 1;
static const size_t DIM_NUM_OF_Y = 2;
static const size_t DIM_INDEX_0 = 0;
static const size_t DIM_INDEX_1 = 1;
static const int32_t DTYPE_INT8 = 2;
static const int32_t DTYPE_INT4 = 29;
static const size_t WORKSPACES_DEFAULT_SIZE_32B = 32;
static const int64_t EVEN_FACTOR = 2;

bool GroupQuantTiling::IsCapable()
{
    return true;
}
const gert::Shape g_vec_1_shape = {1};
const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus GroupQuantTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "GetPlatformInfo begin.");

    auto compileInfo = context_->GetCompileInfo<GroupQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        coreNumVar = compileInfo->coreNum;
        OP_LOGD(context_->GetNodeName(), "Get core num from compile info. Core num is %ld.", coreNumVar);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNumVar = ascendcPlatform.GetCoreNumAiv();
        OP_LOGD(context_->GetNodeName(), "Get core num from platform. Core num is %ld.", coreNumVar);
    }

    OP_LOGD(context_->GetNodeName(), "GetPlatformInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "GetShapeAttrsInfo begin.");

    // check for input x
    auto inputX = context_->GetInputShape(INPUT_INDEX_OF_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputXDesc = context_->GetInputDesc(INPUT_INDEX_OF_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    auto xDtype = inputXDesc->GetDataType();
    OP_CHECK_IF(
        (xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16),
        OP_LOGE(
            context_->GetNodeName(), "x datatype only support FLOAT32, FLOAT16 or BFLOAT16, but is %s",
            Ops::Base::ToString(xDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto xShape = EnsureNotScalar(inputX->GetStorageShape());
    OP_CHECK_IF(
        xShape.GetDimNum() != DIM_NUM_OF_X,
        OP_LOGE(
            context_->GetNodeName(), "x shape length should be 2, but shape is %s",
            Ops::Base::ToString(xShape).c_str()),
        return ge::GRAPH_FAILED);
    dimS = xShape.GetDim(DIM_INDEX_0);
    dimH = xShape.GetDim(DIM_INDEX_1);

    // check for input scale
    auto inputScale = context_->GetInputShape(INPUT_INDEX_OF_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputScale);
    auto inputScaleDesc = context_->GetInputDesc(INPUT_INDEX_OF_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputScaleDesc);
    auto scaleDtype = inputScaleDesc->GetDataType();
    OP_CHECK_IF(
        (scaleDtype != ge::DT_FLOAT && scaleDtype != ge::DT_FLOAT16 && scaleDtype != ge::DT_BF16),
        OP_LOGE(
            context_->GetNodeName(), "scale datatype only support FLOAT32, FLOAT16 or BFLOAT16, but is %s",
            Ops::Base::ToString(scaleDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto scaleShape = EnsureNotScalar(inputScale->GetStorageShape());
    OP_CHECK_IF(
        scaleShape.GetDimNum() != DIM_NUM_OF_SCALE,
        OP_LOGE(
            context_->GetNodeName(), "scale shape length should be 2, but shape is %s",
            Ops::Base::ToString(scaleShape).c_str()),
        return ge::GRAPH_FAILED);
    dimE = scaleShape.GetDim(DIM_INDEX_0);
    OP_CHECK_IF(
        dimE < 1,
        OP_LOGE(
            context_->GetNodeName(), "no support for the first dim of scale shape(%s) is less than 1",
            Ops::Base::ToString(scaleShape).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        scaleShape.GetDim(DIM_INDEX_1) != dimH,
        OP_LOGE(
            context_->GetNodeName(),
            "the second dim of scale shape(%s) should be same as the second dim of x shape(%s)",
            Ops::Base::ToString(scaleShape).c_str(), Ops::Base::ToString(xShape).c_str()),
        return ge::GRAPH_FAILED);

    // check for input group_index
    auto inputGrpIdx = context_->GetInputShape(INPUT_INDEX_OF_GROUP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputGrpIdx);
    auto inputGrpIdxDesc = context_->GetInputDesc(INPUT_INDEX_OF_GROUP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputGrpIdxDesc);
    auto grpIdxDtype = inputGrpIdxDesc->GetDataType();
    OP_CHECK_IF(
        (grpIdxDtype != ge::DT_INT32 && grpIdxDtype != ge::DT_INT64),
        OP_LOGE(
            context_->GetNodeName(), "group_index datatype only support INT32 or INT64, but is %s",
            Ops::Base::ToString(grpIdxDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto grpIdxShape = EnsureNotScalar(inputGrpIdx->GetStorageShape());
    OP_CHECK_IF(
        grpIdxShape.GetDimNum() != DIM_NUM_OF_GROUP_INDEX,
        OP_LOGE(
            context_->GetNodeName(), "group_index shape length should be 1, but shape is %s",
            Ops::Base::ToString(grpIdxShape).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        grpIdxShape.GetDim(DIM_INDEX_0) != dimE,
        OP_LOGE(
            context_->GetNodeName(),
            "The first dim of group_index shape(%s) should be same as the first dim of scale shape(%s)",
            Ops::Base::ToString(grpIdxShape).c_str(), Ops::Base::ToString(scaleShape).c_str()),
        return ge::GRAPH_FAILED);

    // check for input offset which is an optional input
    auto inputOffset = context_->GetInputShape(INPUT_INDEX_OF_OFFSET);
    if (inputOffset == nullptr) {
        hasOffset = 0;
        OP_LOGD(context_->GetNodeName(), "Input offset is not exist.");
    } else {
        hasOffset = 1;
        auto inputOffsetDesc = context_->GetInputDesc(INPUT_INDEX_OF_OFFSET);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputOffsetDesc);
        auto offsetDtype = inputOffsetDesc->GetDataType();
        OP_CHECK_IF(
            offsetDtype != scaleDtype,
            OP_LOGE(
                context_->GetNodeName(), "offset datatype(%s) should be same as scale datatype(%s)",
                Ops::Base::ToString(offsetDtype).c_str(), Ops::Base::ToString(scaleDtype).c_str()),
            return ge::GRAPH_FAILED);
        auto offsetShape = EnsureNotScalar(inputOffset->GetStorageShape());
        OP_CHECK_IF(
            offsetShape.GetDimNum() != DIM_NUM_OF_OFFSET || offsetShape.GetDim(DIM_INDEX_0) != 1,
            OP_LOGE(
                context_->GetNodeName(), "offset should be scalar or vector with shape is [1, ], but shape is %s",
                Ops::Base::ToString(offsetShape).c_str()),
            return ge::GRAPH_FAILED);
    }

    // check attr dst_type
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int32_t* pDstType = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
    if (pDstType != nullptr) {
        int32_t dstType = *pDstType;
        OP_CHECK_IF(
            dstType != DTYPE_INT8 && dstType != DTYPE_INT4,
            OP_LOGE(
                context_->GetNodeName(), "dst_type should be DT_INT4 or DT_INT8, but is %s",
                Ops::Base::ToString(static_cast<ge::DataType>(dstType)).c_str()),
            return ge::GRAPH_FAILED);
    }

    // check for output y
    auto outputY = context_->GetOutputShape(OUTPUT_INDEX_OF_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputY);
    auto outputYDesc = context_->GetOutputDesc(OUTPUT_INDEX_OF_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    auto yDtype = outputYDesc->GetDataType();
    OP_CHECK_IF(
        yDtype != ge::DT_INT4 && yDtype != ge::DT_INT8,
        OP_LOGE(
            context_->GetNodeName(), "y datatype only support INT4 or INT8, but is %s",
            Ops::Base::ToString(yDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto yShape = EnsureNotScalar(outputY->GetStorageShape());
    OP_CHECK_IF(
        yShape.GetDimNum() != DIM_NUM_OF_Y || yShape.GetDim(DIM_INDEX_0) != dimS || yShape.GetDim(DIM_INDEX_1) != dimH,
        OP_LOGE(
            context_->GetNodeName(), "y shape(%s) should be same as x shape(%s)", Ops::Base::ToString(yShape).c_str(),
            Ops::Base::ToString(xShape).c_str()),
        return ge::GRAPH_FAILED);
    if (yDtype == ge::DT_INT4) {
        OP_CHECK_IF(
            dimH % EVEN_FACTOR != 0,
            OP_LOGE(
                context_->GetNodeName(), "y datatype is int4, the second dim of shape(%s) should be an even number",
                Ops::Base::ToString(yShape).c_str()),
            return ge::GRAPH_FAILED);
    }

    OP_LOGD(context_->GetNodeName(), "GetShapeAttrsInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "DoOpTiling begin.");

    if (dimS == 0) {
        // kernel support empty tensor
        needCoreNum = 1;
    } else if (dimS < coreNumVar) {
        needCoreNum = dimS;
    } else {
        needCoreNum = coreNumVar;
    }
    OP_CHECK_IF(
        needCoreNum < 1, OP_LOGE(context_->GetNodeName(), "need core num should be greater than 0"),
        return ge::GRAPH_FAILED);
    if (dimS % needCoreNum == 0) {
        preCoreNum = needCoreNum;
    } else {
        preCoreNum = dimS % needCoreNum;
    }
    xRowNumPreCore = Ops::Base::CeilDiv(dimS, needCoreNum);
    xRowNumPostCore = dimS / needCoreNum;

    tilingData.set_dimS(dimS);
    tilingData.set_dimE(dimE);
    tilingData.set_dimH(dimH);
    tilingData.set_hasOffset(hasOffset);
    tilingData.set_needCoreNum(needCoreNum);
    tilingData.set_preCoreNum(preCoreNum);
    tilingData.set_xRowNumPreCore(xRowNumPreCore);
    tilingData.set_xRowNumPostCore(xRowNumPostCore);

    OP_LOGD(context_->GetNodeName(), "DoOpTiling end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupQuantTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    return tilingKey % TILING_OFFSET;
}

ge::graphStatus GroupQuantTiling::GetWorkspaceSize()
{
    OP_LOGD(context_->GetNodeName(), "GetWorkspaceSize begin.");

    workspaceSize_ = WORKSPACES_DEFAULT_SIZE_32B;

    OP_LOGD(context_->GetNodeName(), "GetWorkspaceSize end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "PostTiling begin.");

    context_->SetBlockDim(tilingData.get_needCoreNum());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;

    gert::TilingData* rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        rawTilingData == nullptr, OP_LOGE(context_->GetNodeType(), "GetRawTilingData failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tilingData.GetDataSize() > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_, "actual tiling data size %zu > context tiling data size %zu", tilingData.GetDataSize(),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    OP_LOGD(context_->GetNodeName(), "PostTiling end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GroupQuant(gert::TilingContext* context)
{
    // 初始化算子Tiling类
    GroupQuantTiling tiling(context);
    // 执行算子tiling框架
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4GroupQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4GroupQuant begin.");

    auto compileInfo = context->GetCompiledInfo<GroupQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        OP_LOGE(
            context->GetNodeName(), "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0),
        OP_LOGE(
            context->GetNodeName(), "Get ub size failed, ub size: %u",
            static_cast<uint32_t>(compileInfo->ubSizePlatForm)),
        return ge::GRAPH_FAILED);

    OP_LOGD(context, "TilingPrepare4GroupQuant end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupQuant).Tiling(Tiling4GroupQuant).TilingParse<GroupQuantCompileInfo>(TilingPrepare4GroupQuant);
} // namespace optiling
