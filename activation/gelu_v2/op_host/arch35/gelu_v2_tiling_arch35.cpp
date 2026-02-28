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
* \file gelu_v2_tiling_arch35.cpp
* \brief
*/

#include "gelu_v2_tiling_arch35.h"
#include "tiling/platform/platform_ascendc.h"
#include <graph/utils/type_utils.h>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/arch35/gelu_v2_dag.h"
#include "../op_kernel/arch35/gelu_v2_struct.h"
#include "op_host/tiling_util.h"

#include <iostream>

using namespace GeluV2Op;

namespace optiling
{
const size_t ASCEND_WORKSPACE = 16777216; // 16M
const int64_t ASCEND_API_BUFFER = 122880; //120K
const int ATTR_APPROXIMATE_POS = 0;
static const gert::Shape g_vec_1_shape = {1};

const gert::Shape& GeluV2Tiling::EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

ge::graphStatus GeluV2Tiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "GeluV2Tiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext, "input x dtype [%s] not supported, only support [DT_FLOAT16, DT_BF16, DT_FLOAT]",
                ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluV2Tiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "GeluV2Tiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
                OP_LOGE(tilingContext, "output y dtype not same as input x"),
                return ge::GRAPH_FAILED);
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
   } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
   } else {
        OP_LOGE(tilingContext, "output y dtype [%s] not supported, only support [DT_FLOAT16, DT_BF16, DT_FLOAT]",
                ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluV2Tiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "GeluV2Tiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputXShape = EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(inputXShape != outputYShape,
                OP_LOGE(tilingContext, "input x and output y shape not same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluV2Tiling::CheckValid() 
{
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
                OP_LOGE(tilingContext, "get input dtype failed"), 
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
                OP_LOGE(tilingContext, "get output dtype failed"), 
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, 
                OP_LOGE(tilingContext, "check shape failed"), 
                return ge::GRAPH_FAILED);

    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const auto *approximatePtr = attrs->GetAttrPointer<char>(ATTR_APPROXIMATE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, approximatePtr);
    approximateStr = approximatePtr;
    if (approximateStr == "none") {
        approximate = TPL_NONE;
    } else if (approximateStr == "tanh") {
        approximate = TPL_TANH;
    } else {
        OP_LOGE(tilingContext, "approximate [%s] not supported, only support [none, tanh]", approximateStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluV2Tiling::RunTiling()
{
    auto tiling = tilingContext->GetTilingData<Ops::Base::EleBaseTilingData16B>();
    OP_LOGD(tilingContext->GetNodeName(), "GeluV2Tiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CheckValid() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "validity check failed"), 
                return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (approximate == TPL_NONE) {
        if (dType == TPL_FP16) {
            baseTilingResult = elewiseBaseTiling.DoTiling<GeluV2Op::GeluV2Erf16BDag<half>::OpDag>(*tiling, ASCEND_API_BUFFER);
        } else if ( dType == TPL_BF16) {
            baseTilingResult = elewiseBaseTiling.DoTiling<GeluV2Op::GeluV2Erf16BDag<bfloat16_t>::OpDag>(*tiling, ASCEND_API_BUFFER);
        } else if (dType == TPL_FP32) {
            baseTilingResult = elewiseBaseTiling.DoTiling<GeluV2Op::GeluV2Erf32BDag<float>::OpDag>(*tiling, ASCEND_API_BUFFER);
        } else {
            OP_LOGE(tilingContext, "output y dtype [%s] not supported, only support [DT_FLOAT16, DT_BF16, DT_FLOAT]",
                    ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (approximate == TPL_TANH) {  
        if (dType == TPL_FP16) {
            baseTilingResult = elewiseBaseTiling.DoTiling<GeluV2Op::GeluV2TanhDag<half>::OpDag>(*tiling, ASCEND_API_BUFFER);
        } else if (dType == TPL_BF16) {
            baseTilingResult = elewiseBaseTiling.DoTiling<GeluV2Op::GeluV2TanhDag<bfloat16_t>::OpDag>(*tiling, ASCEND_API_BUFFER);
        } else if (dType == TPL_FP32) {
            baseTilingResult = elewiseBaseTiling.DoTiling<GeluV2Op::GeluV2TanhDag<float>::OpDag>(*tiling, ASCEND_API_BUFFER);
        } else {
            OP_LOGE(tilingContext, "output y dtype [%s] not supported, only support [DT_FLOAT16, DT_BF16, DT_FLOAT]",
                    ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(tilingContext, "approximate [%s] not supported, only support [none, tanh]", approximateStr.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"), 
                return ge::GRAPH_FAILED);
            
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = ASCEND_WORKSPACE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(1, approximate, dType);
    OP_LOGD(tilingContext, "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(elewiseBaseTiling.GetBlockDim());
                
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GeluV2Arch35(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<GeluV2CompileInfoArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GeluV2Arch35(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4GeluV2 rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const GeluV2CompileInfoArch35*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);

    GeluV2Tiling GeluV2OpTiling(tilingContextGen);
    return GeluV2OpTiling.RunTiling();
}

IMPL_OP_OPTILING(GeluV2).Tiling(Tiling4GeluV2Arch35).TilingParse<GeluV2CompileInfoArch35>(TilingPrepare4GeluV2Arch35);
}  // namespace optiling