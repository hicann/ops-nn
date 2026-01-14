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
 * \file softplus_tiling_arch35.cpp
 * \brief
 */
#include <iostream>
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "softplus_tiling_arch35.h"
#include "activation/softplus/op_kernel/arch35/softplus_dag.h"


using namespace ge;
using namespace SoftplusOp;

namespace optiling {
constexpr uint64_t SOFTPLUS_TILING_KEY_ELEMENTWISE = 101;
constexpr uint64_t SOFTPLUS_WORKSPACE_RESERVE_BYTE = 16777216; // 16 * 1024 * 1024

ge::graphStatus SoftplusTiling::SetTilingData()
{
    size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkspace[0] = SOFTPLUS_WORKSPACE_RESERVE_BYTE;
    tilingContext_->SetTilingKey(SOFTPLUS_TILING_KEY_ELEMENTWISE);
    tilingContext_->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftplusTiling::CalcOutputDtype()
{
    auto inputDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(
        inputDtype != this->outputDtype,
        OP_LOGE(
            tilingContext_, "input dtype %s and output dtype %s are different, which should be same",
            Ops::Base::ToString(inputDtype).c_str(), Ops::Base::ToString(this->outputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftplusTiling::CheckShape()
{
    OP_LOGD(tilingContext_->GetNodeName(), "SoftplusTiling CheckShape enter.");
    auto inputStorageShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputStorageShape);
    const gert::Shape& inputShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputStorageShape);
    const gert::Shape& outputShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputShape != outputShape,
        OP_LOGE(
            tilingContext_->GetNodeName(), "input shape %s and output shape %s are different, which should be same",
            Ops::Base::ToString(inputShape).c_str(), Ops::Base::ToString(outputShape).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftplusTiling::RunTiling()
{
    // get tilingdata address in context
    tiling = tilingContext_->GetTilingData<SoftplusTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, tiling);

    ElewiseBaseTiling elewiseBaseTiling(tilingContext_);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext_, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, 
        OP_LOGE(tilingContext_, "check shape failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus res = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<SoftplusDag<half, float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<SoftplusDag<float, float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<SoftplusDag<bfloat16_t, float>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(tilingContext_, "data type check failed. getype: %s", Ops::Base::ToString(this->outputDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext_, "DoTiling failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus result = SetTilingData();
    return result;
}

static ge::graphStatus TilingForSoftplus(gert::TilingContext *context)
{
    OP_LOGD("SoftplusTiling", "Enter TilingForSoftplus");
    OP_CHECK_IF(context == nullptr,
        OP_LOGE(context, "Tiling context is null"),
        return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo *>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD("SoftplusTiling", "Enter new SoftplusTiling");
    SoftplusTiling softplusTiling(context);
    return softplusTiling.RunTiling();
}

ge::graphStatus TilingPrepareForSoftplus([[maybe_unused]] gert::TilingParseContext *context){
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Softplus).Tiling(TilingForSoftplus)
    .TilingParse<ElewiseCompileInfo>(TilingPrepareForSoftplus);
}  // namespace optiling
