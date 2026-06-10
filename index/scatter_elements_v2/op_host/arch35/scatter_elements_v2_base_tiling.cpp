/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_v2_base_tiling.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "op_host/util/math_util.h"
#include "log/log.h"
#include "op_common/log/log.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "scatter_elements_v2_base_tiling.h"
#include "scatter_elements_v2_asc_tiling.h"

namespace optiling {
static ge::graphStatus ScatterElementsV2TilingForArch35(gert::TilingContext* context)
{
  return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingArch35PrepareForScatterElementsV2(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ScatterElementsV2CompileInfoArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<uint64_t>(ubSizePlatForm);
    if (compileInfo->ubSizePlatForm <= 0) {
      OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "compileInfo->ubSizePlatForm",
                                                std::to_string(compileInfo->ubSizePlatForm).c_str(),
                                                "ubSize must be greater than 0");
    }

    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterElementsV2)
    .Tiling(ScatterElementsV2TilingForArch35)
    .TilingParse<ScatterElementsV2CompileInfoArch35>(TilingArch35PrepareForScatterElementsV2);
}
