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
 * \file leaky_relu_tiling_arch35.cpp
 * \brief
 */
#include "leaky_relu_tiling_arch35.h"

#include <graph/utils/type_utils.h>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

#include "register/op_def_registry.h"
#include "atvoss/elewise/elewise_tiling.h"

#include "log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/leaky_relu_dag.h"
#include "../../op_kernel/arch35/leaky_relu_struct.h"

#include <iostream>

using namespace ge;
using namespace LeakyReluOp;

namespace optiling
{
const uint64_t SYS_WORKSPACE = 16777216; // 16M
  
class LeakyReluTiling
{
public:
    explicit LeakyReluTiling(gert::TilingContext *context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    LeakyReluTilingData *tiling = nullptr;

protected:
    ge::graphStatus SetTilingData();

private:
    uint64_t schMode = 0;
    uint64_t dType = 0;
    ge::DataType outputDtype;
    gert::TilingContext *tilingContext;
};

ge::graphStatus LeakyReluTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "Enter SetTilingData");

    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);

    size_t *currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYS_WORKSPACE;

    schMode = tiling->baseTiling.scheMode;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tiling_key = %ld.", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeakyReluTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "LeakyReluTiling RunTiling enter.");
    ElewiseBaseTiling eleBaseTiling(tilingContext);

    tiling = tilingContext->GetTilingData<LeakyReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);

    this->outputDtype = outputDesc->GetDataType();

    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);

    const float* scaleValueAttr = attrs->GetAttrPointer<float>(0);
    float negativeSlope = scaleValueAttr != nullptr ? *scaleValueAttr : 0.0;

    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = static_cast<uint64_t>(TPL_FP16);
        // fp16需要cast成fp32处理
        OP_CHECK_IF(eleBaseTiling.DoTiling<LeakyReluCastDag<half>::OpDag>(tiling->baseTiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for fp16"),
                        return ge::GRAPH_FAILED);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = static_cast<uint64_t>(TPL_BF16);
        // bf16需要cast成fp32处理
        OP_CHECK_IF(eleBaseTiling.DoTiling<LeakyReluCastDag<half>::OpDag>(tiling->baseTiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for bf16"),
                        return ge::GRAPH_FAILED);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        // fp32直接使用原生DAG
        dType = static_cast<uint64_t>(TPL_FP32);
        OP_CHECK_IF(eleBaseTiling.DoTiling<LeakyReluDag<float>::OpDag>(tiling->baseTiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for fp32"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "current dtype not supported");
        return ge::GRAPH_FAILED;
    }
    tiling->negativeSlope = negativeSlope;
    
    return SetTilingData();
}

static ge::graphStatus TilingForLeakyRelu(gert::TilingContext *context)
{
    OP_LOGD("LeakyReluTiling", "Enter TilingForLeakyRelu");
    if (context == nullptr) {
        OP_LOGE("LeakyReluTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const LeakrReluCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("LeakyReluTiling", "Enter new LeakyReluTiling");
    LeakyReluTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForLeakyRelu(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context -> GetCompiledInfo<LeakrReluCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LeakyRelu).Tiling(TilingForLeakyRelu).TilingParse<LeakrReluCompileInfo>(TilingPrepareForLeakyRelu);
}  