/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_quant_matmul_tiling.cpp
 * \brief
 */

#include <map>
#include <numeric>

#include "common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "op_cache_tiling.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"

#include "./arch35/fused_quant_matmul_asw_tiling.h"
#include "./arch35/fused_quant_matmul_swiglu_tiling.h"
#include "fused_quant_matmul_unary_tiling.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/platform_util.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

using AscendC::BLOCK_CUBE;
using Ops::NN::Optiling::TilingRegistry;
using Ops::NN::TilingPrepareForOpCache;

namespace optiling {

REGISTER_TILING_TEMPLATE("FusedQuantMatMul", FusedQuantMatMulASWTiling, 0);
REGISTER_TILING_TEMPLATE("FusedQuantMatMul", FusedQuantMatMulSwigluTiling, 1);
REGISTER_TILING_TEMPLATE("FusedQuantMatMul", FusedQuantMatMulUnaryTiling, 2);

ge::graphStatus FusedQuantMatMulTilingFunc(gert::TilingContext *context) {
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "FusedQuantMatMul", "TilingContext is null!");
    auto compileInfoPtr = context->GetCompileInfo<QuantBatchMatmulV3CompileInfo>();
    if (compileInfoPtr->supportMmadS8S4) {
        std::vector<int32_t> registerList = {1, 0};
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    } else if (compileInfoPtr->supportL0c2Out) {
        std::vector<int32_t> registerList = {2};
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    }
}

ge::graphStatus TilingParseForFusedQuantMatMul(gert::TilingParseContext *context) {
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "FusedQuantMatMul", "TilingParseContext is null!");
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "The platformInfoPtr is null!");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<QuantBatchMatmulV3CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "The compileInfoPtr is null!");

    PlatformUtil::ParseRuntimePlatformInfo(*compileInfoPtr, context->GetNodeName(), *platformInfoPtr);

    compileInfoPtr->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();

    std::string platformRes;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", platformRes);
    compileInfoPtr->supportL0c2Out = !platformRes.empty();
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", platformRes);
    compileInfoPtr->supportL12BtBf16 = (platformRes.find("bf16") != std::string::npos);
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", platformRes);
    compileInfoPtr->supportMmadS8S4 = (platformRes.find("s8s4") != std::string::npos);
    platformInfoPtr->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    if (!TilingPrepareForOpCache(context)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedQuantMatMul)
    .Tiling(FusedQuantMatMulTilingFunc)
    .TilingParse<FusedQuantMatMulCompileInfo>(TilingParseForFusedQuantMatMul); // 向框架注册入口函数
} // namespace optiling
