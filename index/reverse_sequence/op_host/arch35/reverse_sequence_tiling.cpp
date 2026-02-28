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
* \file reverse_sequence_tiling.cc
* \brief
*/
#include "reverse_sequence_tiling.h"
#include "log/log.h"
#include "op_host/util/math_util.h"
#include "op_host/tiling_templates_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling
{
using namespace Ops::Base;

static ge::graphStatus Tiling4ReverseSequence(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ReverseSequence is running.");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForReverseSequence(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForReverseSequence is running.");
    auto compileInfo = context->GetCompiledInfo<ReverseSequenceCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReverseSequence).Tiling(Tiling4ReverseSequence).TilingParse<ReverseSequenceCompileInfo>(TilingPrepareForReverseSequence);
}  // namespace optiling