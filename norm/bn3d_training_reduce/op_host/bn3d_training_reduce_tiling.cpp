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
 * \file bn3d_training_reduce_tiling.cpp
 * \brief
 */

#include "bn3d_training_reduce_tiling.h"

namespace optiling {
static ge::graphStatus Tiling4BN3DTrainingReduce(gert::TilingContext* context)
{
    // 本算子仅在 Ascend950（regbase / DAV_3510 / arch35）交付，其余芯片直接失败，
    // 不改变 ascend910b / ascend910_93 的任何既有行为。
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
        return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }
    OP_LOGE(context, "BN3DTrainingReduce is not supported on the current chip!");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TilingPrepare4BN3DTrainingReduce(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4BN3DTrainingReduce enter.");

    auto compileInfo = context->GetCompiledInfo<BN3DTrainingReduceCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    // 下面三项在 CompileInfo 里都是 uint64_t：判 <= 0 等价于 == 0（编译器还会告警），
    // 且日志用 %u 打印时把 uint64 截成 uint32 会把异常值报成另一个数，故一律 == 0 + %lu。
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum == 0),
                OP_LOGE(context, "Get core num failed, core num: %lu", compileInfo->coreNum), return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = ubSizePlatForm;
    OP_CHECK_IF((compileInfo->ubSize == 0), OP_LOGE(context, "Get ub size failed, ub size: %lu", compileInfo->ubSize),
                return ge::GRAPH_FAILED);

    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF((compileInfo->ubBlockSize == 0),
                OP_LOGE(context, "Get block size failed, block size: %lu", compileInfo->ubBlockSize),
                return ge::GRAPH_FAILED);

    compileInfo->vectorLength = Ops::Base::GetVRegSize(context);
    // 下界取 sizeof(float) 而不是 0：vectorLength 是字节宽度，tiling 侧一律换算成
    // vlfp32_ = vectorLength / sizeof(float) 后用作除数（r0Factor 向下对齐那一步）。
    // 只挡 <= 0 时，1~3 字节仍会让 vlfp32_ 归零 → 整型除零。
    OP_CHECK_IF((compileInfo->vectorLength < static_cast<uint32_t>(sizeof(float))),
                OP_LOGE(context, "Get vector length failed, vector length: %u",
                        static_cast<uint32_t>(compileInfo->vectorLength)),
                return ge::GRAPH_FAILED);

    OP_LOGD(context, "TilingPrepare4BN3DTrainingReduce exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BN3DTrainingReduce)
    .Tiling(Tiling4BN3DTrainingReduce)
    .TilingParse<BN3DTrainingReduceCompileInfo>(TilingPrepare4BN3DTrainingReduce);

} // namespace optiling
