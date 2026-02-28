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
 * \file layer_norm_grad_v3_tiling.cc
 * \brief
 */
#include<iostream>
#include<string>
#include "op_host/tiling_util.h"
#include "layer_norm_grad_v3_tiling.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

static ge::graphStatus Tiling4LayerNormGradV3(gert::TilingContext *context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4LayerNormGradV3(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepare4LayerNormGradV3 enter.");

    auto compileInfo = context->GetCompiledInfo<LayerNormGradV3CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    compileInfo->vlFp32 = Ops::Base::GetVRegSize(context) / FLOAT_SIZE;

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Get core num failed, core num: %u",
        static_cast<uint32_t>(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0),
        OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %u",
        static_cast<uint32_t>(compileInfo->ubSizePlatForm)),
        return ge::GRAPH_FAILED);
    compileInfo->isRegBase = IsRegbaseSocVersion(context) ? true : false;

    OP_LOGD(context->GetNodeName(),
            "TilingPrepare4LayerNormGradV3 exit, coreNum: %u, blockSize: %ld, ubSize: %ld, vlFp32: %ld.",
            compileInfo->coreNum, compileInfo->blockSize, compileInfo->ubSizePlatForm, compileInfo->vlFp32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LayerNormGradV3)
    .Tiling(Tiling4LayerNormGradV3)
    .TilingParse<LayerNormGradV3CompileInfo>(TilingPrepare4LayerNormGradV3);
} // namespace optiling
