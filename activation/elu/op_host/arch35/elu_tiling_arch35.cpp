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
 * \file elu_tiling_arch35.cpp
 * \brief
 */
#include "elu_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "../../op_kernel/arch35/elu_dag.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include <iostream>

using namespace Ops::Base;
using namespace ge;
using namespace EluNs;

namespace optiling {
constexpr uint64_t ELU_TILING_KEY_ELEMENTWISE = 101;
constexpr uint64_t ELU_WORKSPACE_RESERVE_BYTE = 16777216; // 16 * 1024 * 1024
const int ATTR_ELU_ALPHA_POS = 0;
const int ATTR_ELU_SCALE_POS = 1;
const int ATTR_ELU_INPUT_SCALE_POS = 2;
const gert::Shape g_vec_1_shape = {1};

inline static const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus TilingPrepareForElu(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EluTiling::SetTilingData()
{
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = ELU_WORKSPACE_RESERVE_BYTE;
    tilingContext->SetTilingKey(ELU_TILING_KEY_ELEMENTWISE);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EluTiling::CalcOutputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(
        inputDtype != this->outputDtype, OP_LOGE(tilingContext, "input and output dtype is diff, check failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EluTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "EluTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputShape = EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputShape != outputShape, OP_LOGE(tilingContext->GetNodeName(), "input x and output y shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EluTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    tiling = tilingContext->GetTilingData<EluTilingData>();
    OP_CHECK_IF(
        (tiling == nullptr), OP_LOGE(tilingContext, "Get EleBaseTilingData from context failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus res = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<EluDag<half, float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<EluDag<float, float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<EluDag<bfloat16_t, float>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(tilingContext, "data type check failed. dtype：%d", this->outputDtype);
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "DoTiling failed"), return ge::GRAPH_FAILED);

    auto runtimeAttrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, runtimeAttrs);
    tiling->alpha = 1;
    tiling->scale = 1;
    tiling->inputScale = 1;
    const float* alphaPtr = runtimeAttrs->GetAttrPointer<float>(ATTR_ELU_ALPHA_POS);
    if (alphaPtr != nullptr) {
        tiling->alpha = (*alphaPtr);
    }
    const float* scalePtr = runtimeAttrs->GetAttrPointer<float>(ATTR_ELU_SCALE_POS);
    if (scalePtr != nullptr) {
        tiling->scale = (*scalePtr);
    }
    const float* inputScalePtr = runtimeAttrs->GetAttrPointer<float>(ATTR_ELU_INPUT_SCALE_POS);
    if (inputScalePtr != nullptr) {
        tiling->inputScale = (*inputScalePtr);
    }

    ge::graphStatus result = SetTilingData();
    return result;
}

static ge::graphStatus TilingForElu(gert::TilingContext* context)
{
    OP_LOGD("EluTiling", "Enter TilingForElu");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "Tiling context is null"), return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const EluCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("EluTiling", "Enter new EluTiling");
    EluTiling eluTiling(context);
    return eluTiling.RunTiling();
}

IMPL_OP_OPTILING(Elu).Tiling(TilingForElu).TilingParse<EluCompileInfo>(TilingPrepareForElu);
} // namespace optiling