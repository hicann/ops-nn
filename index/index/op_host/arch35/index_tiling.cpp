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
 * \file index_tiling.cpp
 * \brief Index operator tiling entry and registration
 */

#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "index_tiling.h"

using namespace ge;
using Ops::NN::Optiling::TilingRegistry;

namespace optiling {

static ge::graphStatus Tiling4Index(gert::TilingContext* context)
{
    OP_LOGD("Index", "Tiling4Index rt2.0 is running.");
    auto compile_info = static_cast<const IndexCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "Tiling4Index dsl compile_info is Null, running Simt tiling.");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareIndexForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init IndexSimtTiling.");
    auto ci = context->GetCompiledInfo<IndexCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (ci->core_num <= 0),
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "core_num", std::to_string(ci->core_num).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    uint64_t indexUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, indexUbSize);
    ci->ubSize = static_cast<int64_t>(indexUbSize);
    OP_CHECK_IF((ci->ubSize <= 0),
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "ubSize", std::to_string(ci->ubSize).c_str(), "> 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Index(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<IndexCompileInfo>();
    OP_CHECK_IF(compile_info == nullptr,
                OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(context->GetNodeName(), "tiling", "compile_info", "null",
                                                       "compile info cannot be null"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4Index Simt Mode success!");
    auto ret = TilingPrepareIndexForAscendC(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Index)
    .Tiling(Tiling4Index)
    .TilingInputsDataDependency({1})
    .TilingParse<IndexCompileInfo>(TilingPrepare4Index);
} // namespace optiling
