/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swish_tiling_arch35.cpp
 * \brief
 */

#include "swish_tiling_arch35.h"
#include "log/log.h"
#include "error_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "activation/swish/op_kernel/arch35/swish_dag.h"
#include "activation/swish/op_kernel/arch35/swish_struct.h"

using namespace AscendC;
using namespace ge;
using namespace SwishOp;

namespace optiling {
static constexpr uint64_t OP_KEY_INVALID = 0;
static constexpr uint64_t OP_KEY_1 = 1;
static constexpr uint64_t OP_KEY_2 = 2;
static constexpr uint64_t OP_KEY_3 = 3;
static constexpr uint64_t INDEX_0 = 0;
static constexpr uint64_t WORKSPACE_SIZE = 32;
const int64_t ASCEND_WORKSPACE = 16777216; // 16 * 1024 * 1024
static constexpr float NEG_ONE = -1.0f;
static constexpr float ZERO = 0.0;
const gert::Shape g_vec_1_shape = {1};

inline static const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus SwishTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "SwishTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_TILING_CHECK(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "input x dtype not support %d", this->inputDtype),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwishTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "SwishTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_TILING_CHECK(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "output dtype not support"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(this->outputDtype != this->inputDtype,
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "output y dtype not same as input x"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwishTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "SwishTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputYShape = EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputZShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_TILING_CHECK(inputYShape != outputZShape,
               VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "input x and output y shape not same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwishTiling::SetAttr()
{
    OP_LOGD(tilingContext->GetNodeName(), "SwishTiling GetAttrs enter.");
    auto attrs = tilingContext->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const float* scaleValueAttr = attrs->GetAttrPointer<float>(SwishDag::PLACEHOLDER_INDEX_0);
    float scale = scaleValueAttr == nullptr ? 1.0f : *scaleValueAttr;

    attrScale = scale;

    if (scale == NEG_ONE) {
        attrWork = static_cast<uint64_t>(TPL_SCALE_NEG_ONE);
    } else if (scale == ZERO) {
        attrWork = static_cast<uint64_t>(TPL_SCALE_ZERO);
    } else {
        attrWork = static_cast<uint64_t>(TPL_SCALE_OTHER);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwishTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "SwishTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);

    OP_TILING_CHECK(CalcInputDtype() == ge::GRAPH_FAILED,
               OPS_REPORT_VECTOR_INNER_ERR(tilingContext, "get input dtype failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CalcOutputDtype() == ge::GRAPH_FAILED,
               OPS_REPORT_VECTOR_INNER_ERR(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckShape() == ge::GRAPH_FAILED, OPS_REPORT_VECTOR_INNER_ERR(tilingContext, "check shape failed"),
               return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetAttr() == ge::GRAPH_FAILED, OPS_REPORT_VECTOR_INNER_ERR(tilingContext, "set Attr failed"),
               return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        if (attrWork == static_cast<uint64_t>(TPL_SCALE_NEG_ONE)) {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<SwishDag::SwishNegOne<half>::OpDag>();
        } else if (attrWork == static_cast<uint64_t>(TPL_SCALE_ZERO)) {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<SwishDag::SwishZero<half>::OpDag>();
        } else {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<SwishDag::SwishOther<half>::OpDag>();
        }
    } else if (this->outputDtype == ge::DT_BF16) {
        if (attrWork == static_cast<uint64_t>(TPL_SCALE_NEG_ONE)) {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<SwishDag::SwishNegOne<bfloat16_t>::OpDag>();
        } else if (attrWork == static_cast<uint64_t>(TPL_SCALE_ZERO)) {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<SwishDag::SwishZero<bfloat16_t>::OpDag>();
        } else {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<SwishDag::SwishOther<bfloat16_t>::OpDag>();
        }
    } else if (this->outputDtype == ge::DT_FLOAT) {
        if (attrWork == static_cast<uint64_t>(TPL_SCALE_NEG_ONE)) {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<SwishDag::SwishNegOne<float>::OpDag>();
        } else if (attrWork == static_cast<uint64_t>(TPL_SCALE_ZERO)) {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<SwishDag::SwishZero<float>::OpDag>();
        } else {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<SwishDag::SwishOther<float>::OpDag>();
        }
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(baseTilingResult == ge::GRAPH_FAILED,
               OPS_REPORT_VECTOR_INNER_ERR(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);
    elewiseBaseTiling.SetScalar<float>(attrScale);
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, attrWork);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(elewiseBaseTiling.GetBlockDim());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSwish(gert::TilingContext *tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "TilingForSwish rt2.0 is running");
    SwishTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepare4Swish([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Swish).Tiling(TilingForSwish).TilingParse<SwishCompileInfo>(TilingPrepare4Swish);
} // namespace optiling
