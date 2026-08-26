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

#include "tiling/platform/platform_ascendc.h"

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "atvoss/elewise/elewise_tiling.h"

#include "log/log.h"
#include "../../op_kernel/arch35/leaky_relu_dag.h"
#include "../../op_kernel/arch35/leaky_relu_struct.h"

#include <iostream>

using namespace ge;
using namespace LeakyReluOp;
using namespace Ops::Base;

namespace optiling {
const size_t ASCEND_WORKSPACE = 16777216; // 16M
const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus LeakyReluTiling::CalcInputDtype()
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

ge::graphStatus LeakyReluTiling::CheckShape()
{
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputYShape = EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputZShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(inputYShape != outputZShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    tilingContext->GetNodeName(), "x, y",
                    Ops::Base::ToString(inputYShape) + ", " + Ops::Base::ToString(outputZShape),
                    "The shapes of x and y must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeakyReluTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    tilingContext->GetNodeName(), "x, y",
                    ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(this->inputDtype)) + ", " +
                        ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(this->outputDtype)),
                    "The dtypes of x and y must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeakyReluTiling::RunTiling()
{
    ElewiseBaseTiling eleBaseTiling(tilingContext);

    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get input dtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
                return ge::GRAPH_FAILED);

    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const float* scaleValueAttr = attrs->GetAttrPointer<float>(0);
    float negativeSlope = scaleValueAttr != nullptr ? *scaleValueAttr : 0.0f;

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = static_cast<uint64_t>(TPL_FP16);
        baseTilingResult = eleBaseTiling.DoTiling24B<LeakyReluCastDag<half, float>::OpDag>();
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = static_cast<uint64_t>(TPL_BF16);
        baseTilingResult = eleBaseTiling.DoTiling24B<LeakyReluCastDag<bfloat16_t, float>::OpDag>();
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = static_cast<uint64_t>(TPL_FP32);
        baseTilingResult = eleBaseTiling.DoTiling24B<LeakyReluDag<float, float>::OpDag>();
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y",
                                  ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
                                  "DT_FLOAT16, DT_BF16, DT_FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
                OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed, output dtype: %s.",
                        ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str()),
                return ge::GRAPH_FAILED);

    eleBaseTiling.SetScalar<float>(negativeSlope);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(eleBaseTiling.GetBlockDim());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForLeakyRelu(gert::TilingContext* context)
{
    OP_LOGD("LeakyReluTiling", "Enter TilingForLeakyRelu");
    if (context == nullptr) {
        OP_LOGE("LeakyReluTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = context->GetCompileInfo<LeakrReluCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    LeakyReluTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForLeakyRelu(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<LeakrReluCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LeakyRelu).Tiling(TilingForLeakyRelu).TilingParse<LeakrReluCompileInfo>(TilingPrepareForLeakyRelu);
} // namespace optiling
