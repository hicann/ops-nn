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
 * \file quant_batch_matmul_v4_tiling_registry.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "quant_batch_matmul_v4_msd_tiling.h"
#include "quant_batch_matmul_v4_perblock_tiling.h"
#include "quant_batch_matmul_v4_pergroup_tiling.h"
#include "quant_batch_matmul_v4_compile_info.h"
#include "error_util.h"
#include "platform/platform_infos_def.h"

namespace optiling {
using Ops::NN::Optiling::TilingRegistry;

constexpr int32_t BASIS_PRIORITY = 0;
constexpr int32_t MSD_PRIORITY = 1;
constexpr int32_t PERBLOCK_PRIORITY = 2;
constexpr int32_t PERGROUP_PRIORITY = 3;

REGISTER_TILING_TEMPLATE("QuantBatchMatmulV4", QuantBatchMatmulV4MsdTiling, MSD_PRIORITY);
REGISTER_TILING_TEMPLATE("QuantBatchMatmulV4", QuantBatchMatmulV4PerblockTiling, PERBLOCK_PRIORITY);
REGISTER_TILING_TEMPLATE("QuantBatchMatmulV4", QuantBatchMatmulV4PergroupTiling, PERGROUP_PRIORITY);

ge::graphStatus QuantBatchMatmulV4TilingFunc(gert::TilingContext *context)
{
    platform_ascendc::SocVersion socVersion;
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "QuantBatchMatmulV4", "tilingContext is null");
    auto compileInfoPtr = reinterpret_cast<const QuantBatchMatmulV4CompileInfo *>(context->GetCompileInfo());
    if (compileInfoPtr == nullptr) {
        auto platformInfoPtr = context->GetPlatformInfo();
        OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        socVersion = ascendcPlatform.GetSocVersion();
    } else {
        socVersion = compileInfoPtr->socVersion;
    }
    if (socVersion == platform_ascendc::SocVersion::ASCEND910B ||
            socVersion == platform_ascendc::SocVersion::ASCEND910_93) {
        std::vector<int32_t> regitserList = {MSD_PRIORITY, PERBLOCK_PRIORITY, PERGROUP_PRIORITY};
        return TilingRegistry::GetInstance().DoTilingImpl(context, regitserList);
    }
    std::vector<int32_t> registerList = {BASIS_PRIORITY};
    return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
}

ge::graphStatus TilingParseForQuantBatchMatmulV4(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    // 在tilingParse获取不到GetPlatformInfo时，返回成功。会在之后的InitCompileInfo阶段设置compileInfo信息。
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_SUCCESS, context->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<QuantBatchMatmulV4CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");

    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    compileInfoPtr->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();

    std::string val;
    (void)platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    compileInfoPtr->supportL0c2Out = !val.empty();

    std::string dataMoveL12Bt;
    (void)platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);

    compileInfoPtr->supportL12BtBf16 = dataMoveL12Bt.find("bf16") != string::npos;

    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantBatchMatmulV4)
    .Tiling(QuantBatchMatmulV4TilingFunc)
    .TilingParse<QuantBatchMatmulV4CompileInfo>(TilingParseForQuantBatchMatmulV4);  // 向框架注册入口函数
} // namespace optiling
