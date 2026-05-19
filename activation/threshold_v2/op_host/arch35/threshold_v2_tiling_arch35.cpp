/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file threshold_v2_tiling_arch35.cpp
 * \brief
 */

#include "threshold_v2_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "activation/threshold_v2/op_kernel/arch35/threshold_dag.h"
#include "activation/threshold_v2/op_kernel/arch35/threshold_tiling_struct.h"
#include "register/op_impl_registry.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "util/math_util.h"
#include <unordered_map>

using namespace AscendC;
using namespace ge;
using namespace ThresholdV2Ns;
using namespace ThresholdOp;
using namespace platform_ascendc;
using namespace Ops::Base;

namespace optiling {
const int64_t ASCEND_WORKSPACE = 16777216; // 16 * 1024 * 1024

ge::graphStatus ThresholdTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ThresholdV2Tiling CalcInputDtype enter.");

    auto IsSupportedDtype = [](ge::DataType dtype) -> bool {
        return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_INT8 || dtype == ge::DT_UINT8 ||
               dtype == ge::DT_INT32 || dtype == ge::DT_FLOAT || dtype == ge::DT_INT64;
    };

    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        !IsSupportedDtype(this->inputDtype), OP_LOGE(tilingContext->GetNodeName(), "input x dtype not support"),
        return ge::GRAPH_FAILED);

    auto thresholdDesc = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, thresholdDesc);
    auto thresholdDtype = thresholdDesc->GetDataType();
    OP_CHECK_IF(
        !IsSupportedDtype(thresholdDtype), OP_LOGE(tilingContext->GetNodeName(), "input threshold dtype not support"),
        return ge::GRAPH_FAILED);

    auto valueDesc = tilingContext->GetOptionalInputDesc(2);
    if (valueDesc != nullptr) {
        auto valueDtype = valueDesc->GetDataType();
        OP_CHECK_IF(
            !IsSupportedDtype(valueDtype), OP_LOGE(tilingContext->GetNodeName(), "input value dtype not support"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ThresholdTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ThresholdV2Tiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype, OP_LOGE(tilingContext->GetNodeName(), "Output dtype not same as input"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ThresholdTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "ThresholdV2Tiling CheckShape enter.");
    auto xStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xStorageShape);
    const gert::Shape& inputXShape = EnsureNotScalar(xStorageShape->GetStorageShape());

    auto thresholdStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, thresholdStorageShape);
    const gert::Shape& inputThresholdShape = EnsureNotScalar(thresholdStorageShape->GetStorageShape());

    auto valueStorageShape = tilingContext->GetOptionalInputShape(2);
    if (valueStorageShape != nullptr) {
        const gert::Shape& inputValueShape = EnsureNotScalar(valueStorageShape->GetStorageShape());
    } else {
        hasValue = false;
    }

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputXShape != outputYShape,
        OP_LOGE(tilingContext->GetNodeName(), "input x and output y shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ThresholdTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);

    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "invalid in dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "invalid out dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "invalid shape"), return ge::GRAPH_FAILED);

    auto tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF(
        (tiling == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get ThresholdV2Tiling from GE context failed"),
        return ge::GRAPH_FAILED);

    static const std::unordered_map<ge::DataType, uint64_t> dtypeToTplMap = {
        {ge::DT_FLOAT16, TPL_FP16}, {ge::DT_BF16, TPL_BF16}, {ge::DT_FLOAT, TPL_FP32},
        {ge::DT_INT8, TPL_INT8}, {ge::DT_UINT8, TPL_UINT8}, {ge::DT_INT32, TPL_INT32}, {ge::DT_INT64, TPL_INT64}
    };
    auto it = dtypeToTplMap.find(this->inputDtype);
    OP_CHECK_IF(it == dtypeToTplMap.end(), OP_LOGE(tilingContext->GetNodeName(), "Input dtype not support."),
        return ge::GRAPH_FAILED);
    dType = it->second;

    ge::graphStatus res = ge::GRAPH_FAILED;
    if (dType == TPL_BF16) {
        res = hasValue ? elewiseBaseTiling.DoTiling<ThresholdCastDag<bfloat16_t, float>::OpDag>(*tiling)
            : elewiseBaseTiling.DoTiling<ThresholdCastDagNoValue<bfloat16_t, float>::OpDag>(*tiling);
    } else if (dType == TPL_FP16) {
        res = hasValue ? elewiseBaseTiling.DoTiling<ThresholdDag<half>::OpDag>(*tiling)
            : elewiseBaseTiling.DoTiling<ThresholdDagNoValue<half>::OpDag>(*tiling);
    } else if (dType == TPL_FP32) {
        res = hasValue ? elewiseBaseTiling.DoTiling<ThresholdDag<float>::OpDag>(*tiling)
            : elewiseBaseTiling.DoTiling<ThresholdDagNoValue<float>::OpDag>(*tiling);
    } else if (dType == TPL_INT8) {
        res = hasValue ? elewiseBaseTiling.DoTiling<ThresholdDag<int8_t>::OpDag>(*tiling)
            : elewiseBaseTiling.DoTiling<ThresholdDagNoValue<int8_t>::OpDag>(*tiling);
    } else if (dType == TPL_UINT8) {
        res = hasValue ? elewiseBaseTiling.DoTiling<ThresholdDag<uint8_t>::OpDag>(*tiling)
            : elewiseBaseTiling.DoTiling<ThresholdDagNoValue<uint8_t>::OpDag>(*tiling);
    } else if (dType == TPL_INT32) {
        res = hasValue ? elewiseBaseTiling.DoTiling<ThresholdDag<int32_t>::OpDag>(*tiling)
            : elewiseBaseTiling.DoTiling<ThresholdDagNoValue<int32_t>::OpDag>(*tiling);
    } else if (dType == TPL_INT64) {
        res = hasValue ? elewiseBaseTiling.DoTiling<ThresholdDag<int64_t>::OpDag>(*tiling)
            : elewiseBaseTiling.DoTiling<ThresholdDagNoValue<int64_t>::OpDag>(*tiling);
    }

    OP_CHECK_IF(res == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);
    tilingContext->GetWorkspaceSizes(1)[0] = ASCEND_WORKSPACE;

    uint8_t valueMode = hasValue ? TPL_HAS_VALUE : TPL_NO_VALUE;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tiling->scheMode, valueMode, dType);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->blockNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForThreshold(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext->GetNodeName(), "TilingForThreshold arch35 is running");
    auto compileInfo = tilingContext->GetCompileInfo<ThresholdCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    ThresholdTiling baseOpTiling(tilingContext);
    OP_LOGD(tilingContext->GetNodeName(), "ThresholdV2Tiling RunTiling start.");
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForThreshold(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ThresholdCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ThresholdV2).Tiling(TilingForThreshold).TilingParse<ThresholdCompileInfo>(TilingPrepareForThreshold);

} // namespace optiling