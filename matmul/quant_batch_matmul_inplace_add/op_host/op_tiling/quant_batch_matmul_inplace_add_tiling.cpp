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
 * \file quant_batch_matmul_inplace_add_tiling.cpp
 * \brief QuantBatchMatmulInplaceAdd tiling entry.
 */
#include "quant_batch_matmul_inplace_add_tiling.h"

#include <string>
#include <vector>
#include "error_util.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/platform_util.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_compile_info.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.h"

using Ops::NN::Optiling::TilingRegistryArch;

namespace optiling {
namespace {
constexpr int32_t MX_BASIC_API_TILING_PRIORITY = 0;
constexpr int32_t CUBE_BASIC_API_TILING_PRIORITY = 1;
constexpr const char *OP_NAME = "QuantBatchMatmulInplaceAdd";
const std::vector<int32_t> INPLACE_ADD_TILING_PRIORITIES = {
    MX_BASIC_API_TILING_PRIORITY, CUBE_BASIC_API_TILING_PRIORITY};

template <typename Context>
const char *GetValidOpName(Context *context)
{
    if (context == nullptr) {
        return OP_NAME;
    }
    const char *nodeName = context->GetNodeName();
    if (nodeName != nullptr && nodeName[0] != '\0') {
        return nodeName;
    }
    return OP_NAME;
}
} // namespace

static ge::graphStatus QuantBatchMatmulInplaceAddTilingFunc(gert::TilingContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "QuantBatchMatmulInplaceAdd", "TilingContext is null!");
    const char *opName = GetValidOpName(context);

    auto compileInfoPtr = context->GetCompileInfo<QuantBatchMatmulV3CompileInfo>();
    OP_LOGE_IF(
        compileInfoPtr == nullptr, ge::GRAPH_FAILED, opName, "The compileInfoPtr is null!");
    if (!compileInfoPtr->supportL12BtBf16) {
        OP_LOGD(
            "QuantBatchMatmulInplaceAddTilingFunc",
            "Do op tiling failed, only supports on Ascend 950PR/Ascend 950DT for now.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD("QuantBatchMatmulInplaceAddTilingFunc", "Using the basic api tiling strategy.");
    ResetQuantBatchMatmulV3InputParams();
    return TilingRegistryArch::GetInstance().DoTilingImpl(
        context, INPLACE_ADD_TILING_PRIORITIES, static_cast<int32_t>(compileInfoPtr->npuArch));
}

static ge::graphStatus TilingPrepareForQuantBatchMatmulInplaceAdd(gert::TilingParseContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "QuantBatchMatmulInplaceAdd", "TilingParseContext is null!");
    const char *opName = GetValidOpName(context);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, opName, "The platformInfoPtr is null!");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<QuantBatchMatmulV3CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, opName, "The compileInfoPtr is null!");

    PlatformUtil::ParseRuntimePlatformInfo(*compileInfoPtr, opName, *platformInfoPtr);

    compileInfoPtr->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();

    std::string platformRes;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", platformRes);
    compileInfoPtr->supportL0c2Out = !platformRes.empty();
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", platformRes);
    compileInfoPtr->supportL12BtBf16 = (platformRes.find("bf16") != std::string::npos);
    compileInfoPtr->supportMmadS8S4 = false;
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    platformInfoPtr->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantBatchMatmulInplaceAdd)
    .Tiling(QuantBatchMatmulInplaceAddTilingFunc)
    .TilingParse<QuantBatchMatmulV3CompileInfo>(TilingPrepareForQuantBatchMatmulInplaceAdd);
} // namespace optiling
