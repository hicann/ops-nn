/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "activation/fast_gelu/op_kernel/arch35/fast_gelu_dag.h"
#include "activation/fast_gelu/op_kernel/arch35/fast_gelu_struct.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "fast_gelu_tiling_arch35.h"

using namespace FastGeluOp;

namespace optiling {
const int64_t SYSWORKSPACE = 16777216; // 16 * 1024 * 1024

ge::graphStatus FastGeluTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            tilingContext->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(this->inputDtype)),
            "The dtype of x must be DT_FLOAT16, DT_BF16, or DT_FLOAT"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FastGeluTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            tilingContext->GetNodeName(), "y",
            ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(this->outputDtype)),
            "The dtype of y must be DT_FLOAT16, DT_BF16, or DT_FLOAT"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    tilingContext->GetNodeName(), "x, y",
                    ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(this->inputDtype)) + ", " +
                        ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(this->outputDtype)),
                    "The dtypes of x and y must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FastGeluTiling::CheckShape()
{
    auto selfStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, selfStorageShape);
    const gert::Shape& inputShape = Ops::Base::EnsureNotScalar(selfStorageShape->GetStorageShape());

    auto outStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outStorageShape);
    const gert::Shape& outputShape = Ops::Base::EnsureNotScalar(outStorageShape->GetStorageShape());

    OP_CHECK_IF(inputShape != outputShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    tilingContext->GetNodeName(), "x, y",
                    Ops::Base::ToString(inputShape) + ", " + Ops::Base::ToString(outputShape),
                    "The shapes of x and y must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

std::string FastGeluTiling::DataTypeToSerialString(const ge::DataType type) const
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "x",
                                              ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(type)),
                                              "The dtype of x must be DT_FLOAT16, DT_BF16, or DT_FLOAT");
        return "UNDEFINED";
    }
}

ge::graphStatus FastGeluTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    // 获取tiling计算所需的参数
    ge::graphStatus status = ge::GRAPH_FAILED;
    status = CalcInputDtype();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get input dtype failed"), return ge::GRAPH_FAILED);
    status = CalcOutputDtype();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    status = CheckShape();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    auto tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF((tiling == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get FastGeluTiling from GE context failed"),
                return ge::GRAPH_FAILED);
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        status = elewiseBaseTiling.DoTiling<FastGeluDag::FastGeluNeedCast<half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        status = elewiseBaseTiling.DoTiling<FastGeluDag::FastGeluNeedCast<bfloat16_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        status = elewiseBaseTiling.DoTiling<FastGeluDag::FastGeluNoCast<float>::OpDag>(*tiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            tilingContext->GetNodeName(), "y",
            ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(this->outputDtype)),
            "The dtype of y must be DT_FLOAT16, DT_BF16, or DT_FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"),
                return ge::GRAPH_FAILED);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tiling->scheMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%ld.", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->blockNum);
    size_t usrWorkspaceSize = 0;
    size_t sysWorkspaceSize = SYSWORKSPACE;
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncFastGelu(gert::TilingContext* tilingContext)
{
    auto compileInfo = tilingContext->GetCompileInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);

    OP_LOGD(tilingContext->GetNodeName(), "START FastGelu AscendC Tiling \n");
    FastGeluTiling FastGeluTiling(tilingContext);
    return FastGeluTiling.RunTiling();
}

ge::graphStatus TilingPrepareForFastGelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FastGelu).Tiling(TilingFuncFastGelu).TilingParse<ElewiseCompileInfo>(TilingPrepareForFastGelu);
} // namespace optiling