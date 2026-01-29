/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dual_level_quant_batch_matmul_tiling_registry.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "dual_level_quant_batch_matmul_adaptive_sliding_window_tiling.h"
#include "dual_level_quant_batch_matmul_tiling_base.h"
#include "op_cache_tiling.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "common/inc/error_util.h"

using Ops::NN::TilingPrepareForOpCache;
using Ops::NN::Optiling::TilingRegistry;

namespace optiling {
using dual_level_quant_batch_matmul::DualLevelQuantBatchMatmulTilingASW;

// tiling模板查找的key
constexpr int32_t ASW_CUBE_BOUND_TEMPLATE = 0;
REGISTER_TILING_TEMPLATE("DualLevelQuantBatchMatmul", DualLevelQuantBatchMatmulTilingASW, ASW_CUBE_BOUND_TEMPLATE);

static ge::graphStatus DualLevelQuantBatchMatmulTilingFunc(gert::TilingContext* context)
{
    OP_LOGD("DualLevelQuantBatchMatmul TilingFunc called");
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "DualLevelQuantBatchMatmul", "tilingContext is null");

    NpuArch npuArch = NpuArch::DAV_RESV;

    auto* compileInfoPtr = reinterpret_cast<const DualLevelQuantBatchMatmulCompileInfo*>(context->GetCompileInfo());
    if (compileInfoPtr == nullptr) {
        auto platformInfoPtr = context->GetPlatformInfo();
        OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        npuArch = ascendcPlatform.GetCurNpuArch();
    } else {
        npuArch = compileInfoPtr->npuArch;
    }
    OP_LOGE_IF(npuArch != NpuArch::DAV_3510, ge::GRAPH_FAILED, context->GetNodeName(), "Platform not supported");
    std::vector<int32_t> registerList = {ASW_CUBE_BOUND_TEMPLATE};
    return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
}

static ge::graphStatus TilingParseForDualLevelQuantBatchMatmul(gert::TilingParseContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "DualLevelQuantBatchMatmul", "tilingParseContext is null");
    auto platformInfoPtr = context->GetPlatformInfo();
    // 在tilingParse获取不到GetPlatformInfo时，返回成功。会在之后的InitCompileInfo阶段设置compileInfo信息。
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_SUCCESS, context->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<DualLevelQuantBatchMatmulCompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");

    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    OP_LOGE_IF(compileInfoPtr->aicNum == 0, ge::GRAPH_FAILED, context->GetNodeName(), "aicNum is 0");
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    compileInfoPtr->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();

    TilingPrepareForOpCache(context);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DualLevelQuantBatchMatmul)
    .Tiling(DualLevelQuantBatchMatmulTilingFunc)
    .TilingParse<optiling::DualLevelQuantBatchMatmulCompileInfo>(
        TilingParseForDualLevelQuantBatchMatmul); // 向框架注册入口函数
} // namespace optiling
