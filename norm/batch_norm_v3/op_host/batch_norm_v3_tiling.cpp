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
 * \file batch_norm_v3_tiling.cc
 * \brief
 */

#include "batch_norm_v3_tiling.h"

namespace optiling {

static ge::graphStatus Tiling4BatchNormV3(gert::TilingContext* context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4BatchNormV3(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4BatchNormV3 enter.");

    auto compileInfo = context->GetCompiledInfo<BatchNormV3CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        OP_LOGE(
            context, "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = ubSizePlatForm;
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0),
        OP_LOGE(context, "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ubSize)),
        return ge::GRAPH_FAILED);
    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(
        (compileInfo->blockSize <= 0),
        OP_LOGE(
            context, "Get block Size failed, block size: %u",
            static_cast<uint32_t>(compileInfo->blockSize)),
        return ge::GRAPH_FAILED);
    compileInfo->vectorLength = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF(
        (compileInfo->vectorLength <= 0),
        OP_LOGE(
            context, "Get vector Length failed, vector Length: %u",
            static_cast<uint32_t>(compileInfo->vectorLength)),
        return ge::GRAPH_FAILED);

    OP_LOGD(context, "TilingPrepare4BatchNormV3 exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BatchNormV3).Tiling(Tiling4BatchNormV3).TilingParse<BatchNormV3CompileInfo>(TilingPrepare4BatchNormV3);

} // namespace optiling
