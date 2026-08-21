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
 * \file relu_v2_tiling_arch35.cpp
 * \brief
 */
#include "relu_v2_tiling_arch35.h"
#include <iostream>
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/relu_v2_dag.h"
#include "atvoss/elewise/elewise_base_struct.h"

namespace optiling {
using namespace ge;
using namespace Ops::Base;

constexpr uint64_t SYS_WORKSPACE = 16777216; // 16M
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_FP16 = 101;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_BF16 = 102;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_FP32 = 103;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_INT8 = 104;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_INT32 = 105;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_INT64 = 106;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_INT16 = 107;

ge::graphStatus ReluV2Tiling::CalcOutputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(inputDtype != this->outputDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x, y",
                                                       ge::TypeUtils::DataTypeToSerialString(inputDtype) + ", " +
                                                           ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
                                                       "The dtypes of x and y must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReluV2Tiling::RunTiling()
{
    auto tiling = tilingContext->GetTilingData<Ops::Base::EleBaseTilingData16B>();
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
                return ge::GRAPH_FAILED);
    ge::graphStatus res = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<ReluV2Op::GraphReluV2<half, half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<ReluV2Op::GraphReluV2<float, float>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<ReluV2Op::GraphReluV2<half, float>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT8) {
        res = elewiseBaseTiling.DoTiling<ReluV2Op::GraphReluV2<int8_t, half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT16) {
        res = elewiseBaseTiling.DoTiling<ReluV2Op::GraphReluV2<int16_t, float>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        res = elewiseBaseTiling.DoTiling<ReluV2Op::GraphReluV2<int32_t, int32_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT64) {
        res = elewiseBaseTiling.DoTiling<ReluV2Op::GraphReluV2Max<int64_t>::OpDag>(*tiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y",
                                  ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
                                  "DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "DoTiling failed"), return ge::GRAPH_FAILED);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = SYS_WORKSPACE;
    if (this->outputDtype == ge::DT_FLOAT16) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_FP16);
    } else if (this->outputDtype == ge::DT_BF16) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_BF16);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_FP32);
    } else if (this->outputDtype == ge::DT_INT8) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_INT8);
    } else if (this->outputDtype == ge::DT_INT16) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_INT16);
    } else if (this->outputDtype == ge::DT_INT32) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_INT32);
    } else if (this->outputDtype == ge::DT_INT64) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_INT64);
    }

    tilingContext->SetBlockDim(elewiseBaseTiling.GetBlockDim());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ReluV2(gert::TilingContext* context)
{
    OP_LOGD("ReluV2Tiling", "Enter Tiling4ReluV2");
    auto compileInfo = reinterpret_cast<const ReluV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // 走新的模板tiling
    OP_LOGD("ReluV2Tiling", "Enter new ReluV2Tiling");
    ReluV2Tiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForReluV2(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ReluV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReluV2).Tiling(Tiling4ReluV2).TilingParse<ReluV2CompileInfo>(TilingPrepareForReluV2);
} // namespace optiling
