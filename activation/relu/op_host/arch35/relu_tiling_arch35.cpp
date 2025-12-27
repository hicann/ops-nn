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
 * \file relu_tiling_arch35.cpp
 * \brief
 */
#include "relu_tiling_arch35.h"
#include <iostream>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "activation/relu/op_kernel/arch35/relu_dag.h"
#include "atvoss/elewise/elewise_base_struct.h"

using namespace ge;

namespace optiling {
constexpr uint64_t SYS_WORKSPACE = 16777216; //16M
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_FP16 = 101;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_BF16 = 102;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_FP32 = 103;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_INT8 = 104;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_INT32 = 105;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_INT64 = 106;

ge::graphStatus ReluTiling::CalcOutputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(inputDtype != this->outputDtype,
        OP_LOGE(tilingContext, "input and output dtype is diff, check failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReluTiling::RunTiling()
{
    auto tiling = tilingContext->GetTilingData<Ops::Base::EleBaseTilingData16B>();
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get output dtype failed"),
               return ge::GRAPH_FAILED);
    ge::graphStatus res = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<GraphRelu<half, half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<GraphRelu<float, float>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<GraphRelu<half, float>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT8) {
        res = elewiseBaseTiling.DoTiling<GraphRelu<int8_t, half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        res = elewiseBaseTiling.DoTiling<GraphRelu<int32_t, int32_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT64) {
        res = elewiseBaseTiling.DoTiling<GraphReluMax<int64_t>::OpDag>(*tiling);
    } else {
        OP_LOGE(tilingContext, "data type check failed. getype：%d", this->outputDtype);
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext, "DoTiling failed"),
        return ge::GRAPH_FAILED);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYS_WORKSPACE;
    if (this->outputDtype == ge::DT_FLOAT16) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_FP16);
    } else if (this->outputDtype == ge::DT_BF16) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_BF16);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_FP32);
    } else if (this->outputDtype == ge::DT_INT8) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_INT8);
    } else if (this->outputDtype == ge::DT_INT32) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_INT32);
    } else if (this->outputDtype == ge::DT_INT64) {
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_INT64);
    }

    tilingContext->SetBlockDim(elewiseBaseTiling.GetBlockDim());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Relu(gert::TilingContext *context)
{
    OP_LOGD("ReluTiling", "Enter Tiling4Relu");
    if (context == nullptr) {
        OP_LOGE("ReluTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const ReluCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // 走新的模板tiling
    OP_LOGD("ReluTiling", "Enter new ReluTiling");
    ReluTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForRelu(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ReluCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Relu).Tiling(Tiling4Relu)
                            .TilingParse<ReluCompileInfo>(TilingPrepareForRelu);
}  // namespace optiling