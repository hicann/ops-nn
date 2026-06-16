/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /* !
 * \file sync_batch_norm_gather_stats_fused_tiling.cpp
 * \brief
 */

#include<iostream>
#include<string>
#include "op_host/tiling_util.h"
#include "sync_batch_norm_gather_stats_fused_tiling.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

static ge::graphStatus Tiling4SyncBatchNormGatherStatsFused(gert::TilingContext *context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4SyncBatchNormGatherStatsFused(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepare4SyncBatchNormGatherStatsFused enter.");

    auto compileInfo = context->GetCompiledInfo<SyncBatchNormGatherStatsFusedCompileInfo>();
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
            "TilingPrepare4SyncBatchNormGatherStatsFused exit, coreNum: %u, blockSize: %ld, ubSize: %ld, vlFp32: %ld.",
            compileInfo->coreNum, compileInfo->blockSize, compileInfo->ubSizePlatForm, compileInfo->vlFp32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SyncBatchNormGatherStatsFused)
    .Tiling(Tiling4SyncBatchNormGatherStatsFused)
    .TilingParse<SyncBatchNormGatherStatsFusedCompileInfo>(TilingPrepare4SyncBatchNormGatherStatsFused);
} // namespace optiling