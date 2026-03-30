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
 * \file erfinv_tiling_arch35.cpp
 * \brief erfinv tiling arch35
 */

#include <iostream>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "activation/erfinv/op_kernel/arch35/erfinv_dag.h"
#include "activation/erfinv/op_kernel/arch35/erfinv_struct.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "erfinv_tiling_arch35.h"
#include "error_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {
const int64_t ASCEND_WORKSPACE = 16777216; // 16M
const int64_t ASCEND_API_BUFFER = 122880;  // 120K
const int64_t DCACHE_SIZE = 32768;

ge::graphStatus ErfinvTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "ErfinvTiling SetTilingData enter.");
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = ASCEND_WORKSPACE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(tiling->scheMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->blockNum);

    uint64_t ubSize = 0;
    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const ElewiseCompileInfo*>(tilingContext->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(tilingContext, "compile info is null"), return ge::GRAPH_FAILED);
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize = ubSizePlatForm;
    }
    tilingContext->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ErfinvTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ErfinvTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE(
            tilingContext->GetNodeName(), "input x dtype[%s] not support",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ErfinvTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "ErfinvTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputXShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputXShape != outputYShape, OP_LOGE(tilingContext->GetNodeName(), "input x and output y shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ErfinvTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ErfinvTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype,
        OP_LOGE(tilingContext->GetNodeName(), "output y dtype not same as input x"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ErfinvTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "ErfinvTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF(
        (tiling == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get ErfinvTiling from GE context failed"),
        return ge::GRAPH_FAILED);
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult =
            elewiseBaseTiling.DoTiling<ErfinvOp::ErfinvDAG<half>::OpDag>(*tiling, ASCEND_API_BUFFER + DCACHE_SIZE);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<ErfinvOp::ErfinvDAG<bfloat16_t>::OpDag>(
            *tiling, ASCEND_API_BUFFER + DCACHE_SIZE);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult =
            elewiseBaseTiling.DoTiling<ErfinvOp::ErfinvDAG<float>::OpDag>(*tiling, ASCEND_API_BUFFER + DCACHE_SIZE);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus Tiling4Erfinv(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Erfinv rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    ErfinvTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepare4Erfinv([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Erfinv).Tiling(Tiling4Erfinv).TilingParse<ElewiseCompileInfo>(TilingPrepare4Erfinv);
} // namespace optiling