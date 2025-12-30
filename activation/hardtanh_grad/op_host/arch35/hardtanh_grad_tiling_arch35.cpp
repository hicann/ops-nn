/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include "hardtanh_grad_tiling_arch35.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "register/tilingdata_base.h"
#include "activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_dag.h"
#include "activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_struct.h"

#include <iostream>

using namespace HardtanhGradOp;
namespace optiling
{
const uint64_t ASCEND_WORKSPACE = 16777216;
const std::int32_t ATTR_HARDTANH_GRAD_MINVAL_POS = 0;
const std::int32_t ATTR_HARDTANH_GRAD_MAXVAL_POS = 1;

class HardtanhGradTiling
{
public:
    explicit HardtanhGradTiling(gert::TilingContext* context) : tilingContext(context){
        // get tilingdata address in context
        tiling = tilingContext->GetTilingData<HardtanhGradTilingData>();
    };
    ge::graphStatus RunTiling();
    HardtanhGradTilingData *tiling = nullptr;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetTilingData();

private:
    uint64_t dType = 0;
    uint64_t schMode = 0;
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    ge::DataType inputDtype;
    ge::DataType inputDtype1;
};

ge::graphStatus HardtanhGradTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "HardtanhGradTiling SetTilingData enter.");

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(ASCEND_WORKSPACE);

    schMode = tiling->baseTiling.scheMode;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HardtanhGradTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE("HardtanhGrad", "input result dtype not support"),
        return ge::GRAPH_FAILED);
    auto inputDesc1 = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc1);
    this->inputDtype1 = inputDesc1->GetDataType();
    OP_CHECK_IF(
        this->inputDtype1 != ge::DT_FLOAT16 && this->inputDtype1 != ge::DT_BF16 && this->inputDtype1 != ge::DT_FLOAT,
        OP_LOGE("HardtanhGrad", "input grad dtype not support"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(this->inputDtype1 != this->inputDtype,
               OP_LOGE("HardtanhGrad", "input grad dtype not same as input result"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HardtanhGradTiling::CheckShape()
{
    auto resultStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, resultStorageShape);
    const gert::Shape& inputResultShape = Ops::Base::EnsureNotScalar(resultStorageShape->GetStorageShape());
    auto gradStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradStorageShape);
    const gert::Shape& inputGradShape = Ops::Base::EnsureNotScalar(gradStorageShape->GetStorageShape());

    auto yStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yStorageShape);
    const gert::Shape& outputYShape = Ops::Base::EnsureNotScalar(yStorageShape->GetStorageShape());

    OP_CHECK_IF(inputResultShape != inputGradShape,
               OP_LOGE("HardtanhGrad", "input result and grad shape not same"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputResultShape != outputYShape,
               OP_LOGE("HardtanhGrad", "input result and output y shape not same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HardtanhGradTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext, "output dtype not support"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
               OP_LOGE("HardtanhGrad", "output y dtype not same as input result"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HardtanhGradTiling::RunTiling()
{
    OP_LOGD("HardtanhGrad", "HardtanhGradTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get input dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
               return ge::GRAPH_FAILED);
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = static_cast<uint64_t>(HardtanhGrad_TPL_FP16);
        baseTilingResult = elewiseBaseTiling.DoTiling<HardtanhGradOp::HardtanhGradDag<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = static_cast<uint64_t>(HardtanhGrad_TPL_BF16);
        baseTilingResult = elewiseBaseTiling.DoTiling<HardtanhGradOp::HardtanhGradDag<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = static_cast<uint64_t>(HardtanhGrad_TPL_FP32);
        baseTilingResult = elewiseBaseTiling.DoTiling<HardtanhGradOp::HardtanhGradDag<float>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE("HardtanhGrad", "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);

    auto runtimeAttrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, runtimeAttrs);
    const float *minValPtr = runtimeAttrs->GetAttrPointer<float>(ATTR_HARDTANH_GRAD_MINVAL_POS);
    if (minValPtr != nullptr) {
        tiling->minVal = (*minValPtr);
    }
    const float *maxValPtr = runtimeAttrs->GetAttrPointer<float>(ATTR_HARDTANH_GRAD_MAXVAL_POS);
    if (maxValPtr != nullptr) {
        tiling->maxVal = (*maxValPtr);
    }

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4HardtanhGrad(gert::TilingContext *context)
{
    OP_LOGD("HardtanhGrad", "Tiling4HardtanhGrad rt2.0 is running.");
    OP_CHECK_IF(context == nullptr,
        OP_LOGE(context, "Tiling context is null."),
        return ge::GRAPH_FAILED);

    auto compileInfo = static_cast<const ElewiseCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("HardtanhGrad", "Enter new HardtanhGrad.");
    HardtanhGradTiling baseOpTiling(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, baseOpTiling.tiling);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForHardtanhGrad([[maybe_unused]] gert::TilingParseContext *context){
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HardtanhGrad).Tiling(Tiling4HardtanhGrad).TilingParse<ElewiseCompileInfo>(TilingPrepareForHardtanhGrad);
}  // namespace optiling
