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
 * \file transpose_quant_batch_mat_mul_tiling.cc
 * \brief
 */

#include "transpose_quant_batch_mat_mul_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "./arch35/transpose_quant_batch_mat_mul_tiling_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h"
#include "op_cache_tiling.h"
#include "error_util.h"

using Ops::NN::TilingPrepareForOpCache;

namespace optiling {

static ge::graphStatus TransposeQuantBatchMatMulTilingFunc(gert::TilingContext* context)
{
    return transpose_quant_batch_mat_mul_advanced::TransposeQuantBatchMatMulTiling(context).DoTiling();
}

static ge::graphStatus TilingPrepareForTransposeQuantBatchMatMul(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("TransposeQuantBatchMatMul", "context is null"),
                    return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
                    return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MatmulV3CompileInfo>();
    OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    std::string val;
    std::string dataMoveL12Bt;
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
    compileInfoPtr->supportL0c2out = !val.empty();
    compileInfoPtr->supportL12BtBf16 = (dataMoveL12Bt.find("bf16") != std::string::npos);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfoPtr->btSize = compileInfoPtr->supportL0c2out ? 1024UL : 0UL;                      // 1024 is btSize
    compileInfoPtr->btSize = compileInfoPtr->supportL12BtBf16 ? 4096UL : compileInfoPtr->btSize; // 4096 is btSize
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    if (!TilingPrepareForOpCache(context)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TransposeQuantBatchMatMul)
    .Tiling(TransposeQuantBatchMatMulTilingFunc)
    .TilingParse<MatmulV3CompileInfo>(TilingPrepareForTransposeQuantBatchMatMul);
} // namespace optiling