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
 * \file rms_norm_quant_v2_tiling.cpp
 * \brief
 */
#include "rms_norm_quant_v2_tiling.h"
#include "op_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "kernel_tiling/kernel_tiling.h"
#include "log/log.h"
#include <iostream>
#include "tiling_base/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_impl_registry.h"

using namespace Ops::Base;
using namespace AscendC;
namespace optiling {
constexpr size_t DEST_MAX = 100;
constexpr size_t MAX_LEN_SIMPLIFIED_KEY = 256;

ge::graphStatus Tiling4RmsNormQuantV2(gert::TilingContext* context)
{
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
        return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }
    OP_LOGE(context, "RmsNormQuantV2 is not supported on the current chip!");
    return ge::GRAPH_FAILED;
}

ge::graphStatus TilingPrepare4RmsNormQuantV2(gert::TilingParseContext* context)
{
    OP_CHECK_IF(nullptr == context, OP_LOGE("RmsNormQuantV2", "Context is null"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter TilingPrepare4RmsNormQuantV2.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfoPtr == nullptr, OP_LOGE("RmsNormQuantV2", "PlatformInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<RmsNormQuantV2CompileInfo>();
    OP_CHECK_IF(
        compileInfoPtr == nullptr, OP_LOGE("RmsNormQuantV2", "CompileInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetCurNpuArch();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GenSimplifiedKey4RmsNormQuantV2(gert::TilingContext* context, ge::char_t* simplifiedKey)
{
    OP_CHECK_IF(nullptr == context, OP_LOGE("RmsNormQuantV2", "Context is null"), return ge::GRAPH_FAILED);
    OP_LOGW(context, "Enter RmsNormQuantV2 genSimplifiedKey.");

    OP_CHECK_IF(
        nullptr == simplifiedKey, OP_LOGE(context, "SimplifiedKey is null"), return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(GAMMA_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(SCALES1_INDEX));

    int32_t xDtype = static_cast<int32_t>(context->GetInputDesc(X_INDEX)->GetDataType());
    int32_t gammaDtype = static_cast<int32_t>(context->GetInputDesc(GAMMA_INDEX)->GetDataType());
    int32_t scales1Dtype = static_cast<int32_t>(context->GetInputDesc(SCALES1_INDEX)->GetDataType());

    int32_t scales2Dtype = -1;
    int32_t zeroPoints1Dtype = -1;
    int32_t zeroPoints2Dtype = -1;
    int32_t betaDtype = -1;

    int32_t y1Dtype = static_cast<int32_t>(context->GetOutputDesc(Y1_INDEX)->GetDataType());

    OP_CHECK_IF(
        context->GetOptionalInputDesc(SCALES2_INDEX) != nullptr,
        OP_LOGW(context, "Optional input scale2 exist"),
        scales2Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(SCALES2_INDEX)->GetDataType()));
    OP_CHECK_IF(
        context->GetOptionalInputDesc(ZERO_POINTS1_INDEX) != nullptr,
        OP_LOGW(context, "Optional input zeroPoints1 exist"),
        zeroPoints1Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(ZERO_POINTS1_INDEX)->GetDataType()));
    OP_CHECK_IF(
        context->GetOptionalInputDesc(ZERO_POINTS2_INDEX) != nullptr,
        OP_LOGW(context, "Optional input zeroPoints2 exist"),
        zeroPoints2Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(ZERO_POINTS2_INDEX)->GetDataType()));
    OP_CHECK_IF(
        context->GetOptionalInputDesc(BETA_INDEX) != nullptr,
        OP_LOGW(context, "Optional input beta exist"),
        betaDtype = static_cast<int32_t>(context->GetOptionalInputDesc(BETA_INDEX)->GetDataType()));

    std::string simpleKeyTemp = "";
    strcat_s(simplifiedKey, DEST_MAX, "diy,");
    simpleKeyTemp
        .append(std::to_string(xDtype)) // x
        .append("/")
        .append(std::to_string(gammaDtype)) // gammaDtype
        .append("/")
        .append(std::to_string(scales1Dtype)) // scales1
        .append("/")
        .append(std::to_string(scales2Dtype)) // scales2
        .append("/")
        .append(std::to_string(zeroPoints1Dtype)) // zeroPoints1
        .append("/")
        .append(std::to_string(zeroPoints2Dtype)) // zeroPoints2
        .append("/")
        .append(std::to_string(betaDtype)) // beta
        .append("/")
        .append(std::to_string(y1Dtype)); // y1Dtype
    OP_LOGW(context, "SimpleKeyTemp: %s", simpleKeyTemp.c_str());
    errno_t err = strcat_s(simplifiedKey, DEST_MAX, simpleKeyTemp.c_str());
    OP_CHECK_IF(
        (err != 0), OP_LOGE(context, "Error: strcat_s failed with error code %d.", err), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        strlen(simplifiedKey) > MAX_LEN_SIMPLIFIED_KEY, OP_LOGE(context, "Len of simplifiedKey exceeds max length."),
        return ge::GRAPH_FAILED);
    OP_LOGW(context, "Finish RmsNormQuantV2 genSimplifiedKey.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RmsNormQuantV2)
    .Tiling(Tiling4RmsNormQuantV2)
    .TilingParse<RmsNormQuantV2CompileInfo>(TilingPrepare4RmsNormQuantV2)
    .GenSimplifiedKey(GenSimplifiedKey4RmsNormQuantV2);

} // namespace optiling