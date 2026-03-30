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
 * \file index_fill_tiling_arch35.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "op_host/util/math_util.h"
#include "log/log.h"
#include "op_common/log/log.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "index_fill_tiling_arch35.h"
#include "index_fill_tiling_common.h"

namespace optiling {
constexpr uint64_t SIZE_OF_HALF = 2;
constexpr uint64_t ALIGNED_NUM = 32;
constexpr size_t SYS_WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint64_t HALF_ALIGNED = 32;
constexpr uint64_t P_LIMIT = 1024;
constexpr uint64_t N_LIMIT = 64;
constexpr uint64_t P_GM_NUMS = 8192;
constexpr uint64_t FLOAT_ALIGNED = 8;
constexpr uint32_t FRONT_P_KEY = 2;
constexpr uint32_t TAIL_N_KEY = 3;
constexpr uint32_t TAIL_P_KEY = 4;
constexpr uint32_t TAIL_N_LIMIT = 256;
constexpr uint32_t COMPARE_ALIGNED = 128;

class IndexFillTilingArch35 {
public:
  explicit IndexFillTilingArch35(gert::TilingContext* context_) : context(context_){};
  void Init();
private:
  gert::TilingContext* context = nullptr;
  TilingDataStructIndexFillArch35 tilingData;
};

static ge::graphStatus IndexFillTilingForArch35(gert::TilingContext* context)
{
  OP_LOGD(context->GetNodeName(), "Tiling for IndexFill start.");
  ge::graphStatus result = optiling::Tiling4IndexFillArch35(context);
  OP_LOGD(context->GetNodeName(), "Tiling for IndexFill end.");
  return result;
}

ge::graphStatus TilingArch35PrepareForIndexFill(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For IndexFill start.");
    auto compileInfo = context->GetCompiledInfo<IndexFillCompileInfoArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<uint64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0),
                    OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                    return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfo->sysWorkspaceSize = (sysWorkspaceSize <= 0) ? WS_SYS_SIZE : sysWorkspaceSize;
    OP_LOGD(context->GetNodeName(), "sysWorkspaceSize is %lu.", compileInfo->sysWorkspaceSize);

    OP_LOGD(context->GetNodeName(), "Tiling Prepare For IndexFill end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IndexFill)
    .Tiling(IndexFillTilingForArch35)
    .TilingParse<IndexFillCompileInfoArch35>(TilingArch35PrepareForIndexFill);
}
