/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mish_tiling_arch35.h"
#include "tiling/platform/platform_ascendc.h"
#include <graph/utils/type_utils.h>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/arch35/mish_dag.h"
#include "../op_kernel/arch35/mish_struct.h"
#include "common/inc/tiling_base/tiling_util.h"
#include <iostream>

using namespace ge;

namespace optiling
{
const int64_t ASCEND_WORKSPACE = 0;
constexpr int64_t MAX_DIM_NUM = 8;
const gert::Shape g_vec_1_shape = {1};

inline static const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}
ge::graphStatus MishTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "MishTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT && this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16,
        OP_LOGE(tilingContext->GetNodeName(), "input x dtype not support [%s],only support [DT_FLOAT, DT_FLOAT16, DT_BF16]", ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MishTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "MishTiling Checkshape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputYShape = EnsureNotScalar(inputStorageShape->GetStorageShape());
    
    OP_CHECK_IF(
        inputYShape.GetDimNum() > MAX_DIM_NUM,
        OP_LOGE(tilingContext->GetNodeName(), "input x dim num should be no more than 8"),
        return ge::GRAPH_FAILED);
     
    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputZShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputYShape != outputZShape,
        OP_LOGE(tilingContext->GetNodeName(), "output y shape not same as input x"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus MishTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "MishTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype,
        OP_LOGE(tilingContext->GetNodeName(), "output y dtype not same as input x"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MishTiling::RunTiling()
{
    auto tiling = tilingContext->GetTilingData<EleBaseTilingData16B>();
    OP_LOGD(tilingContext->GetNodeName(), "MishTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext, "get input dtype failed"), 
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext, "get output dtype failed"), 
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, 
        OP_LOGE(tilingContext, "check shape failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<MishDAG<half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<MishDAG<bfloat16_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<MishDAG<float>::OpDag>(*tiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext, "elewiseBaseTiling failed"), 
        return ge::GRAPH_FAILED);
        
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(1, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(elewiseBaseTiling.GetBlockDim());
    return ge::GRAPH_SUCCESS;
}
 
static ge::graphStatus Tiling4Mish(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Mish rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const MishCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);

    MishTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}
 
ge::graphStatus TilingPrepareForMish(gert::TilingParseContext* context)
 {
    auto compileInfoPtr = context->GetCompiledInfo<MishCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(Mish).Tiling(Tiling4Mish).TilingParse<MishCompileInfo>(TilingPrepareForMish);
}  // namespace optiling