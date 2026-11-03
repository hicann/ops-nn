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
 * \file modulate_grad_tiling_arch35.cpp
 * \brief modulate_grad regbase tiling for arch35 (ascend950)
 */

#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "../op_kernel/arch35/modulate_grad_struct.h"
#include "../op_kernel/arch35/modulate_grad_regbase_tiling_key.h"

namespace optiling {
ge::graphStatus ModulateGradTilingArch35(gert::TilingContext* context, bool isScale, bool isShift)
{
    OP_LOGD(context->GetNodeName(), "Begin the regbase tiling process for arch35.");
    constexpr uint32_t INPUT_INDEX = 1;
    const gert::StorageShape* input_shape = context->GetInputShape(INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    const gert::Shape inputStorageShape = input_shape->GetStorageShape();
    if (inputStorageShape.GetDimNum() != 3) {
        OP_LOGE(context, "modulate_grad input rank must be 3, got %zu.", inputStorageShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    const uint64_t B = inputStorageShape.GetDim(0);
    const uint64_t L = inputStorageShape.GetDim(1);
    const uint64_t D = inputStorageShape.GetDim(2);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t block_dim = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    ModulateGradRegbaseTilingData tiling = {};
    tiling.B = B;
    tiling.L = L;
    tiling.D = D;
    tiling.has_scale = isScale ? 1 : 0;
    tiling.has_shift = isShift ? 1 : 0;
    uint8_t tilingStrategy = MODULATE_GRAD_TILING_B;
    if (B == 0 || D == 0 || B * D == 0) {
        // Degenerate batch/feature dims: single idle core, no division, no data tiles.
        tiling.coresPerB = 1;
        tiling.usedCores = 1;
        tiling.formerNum = 0;
        tiling.formerLength = 0;
        tiling.tailNum = 1;
        tiling.tailLength = 0;
    } else {
        uint64_t numBlocks = static_cast<uint64_t>(block_dim);
        if (numBlocks == 0) {
            OP_LOGE(context, "modulate_grad got zero AIV core count.");
            return ge::GRAPH_FAILED;
        }
        numBlocks = numBlocks < B * D ? numBlocks : B * D;
        const uint64_t splitBlock = numBlocks / 2;
        const bool splitByD = (B < splitBlock);
        if (splitByD) {
            tilingStrategy = MODULATE_GRAD_TILING_D;
            const uint64_t coresPerB = numBlocks / B;
            tiling.coresPerB = static_cast<uint32_t>(coresPerB);
            tiling.usedCores = static_cast<uint32_t>(coresPerB * B);
            const uint64_t dPerCore = D / coresPerB;
            const uint64_t remainderD = D % coresPerB;
            tiling.formerNum = static_cast<uint32_t>(remainderD);
            tiling.formerLength = static_cast<uint32_t>(dPerCore + 1);
            tiling.tailNum = static_cast<uint32_t>(coresPerB - remainderD);
            tiling.tailLength = static_cast<uint32_t>(dPerCore);
        } else {
            numBlocks = numBlocks < B ? numBlocks : B;
            tiling.coresPerB = 1;
            const uint64_t bPerCore = B / numBlocks;
            tiling.usedCores = static_cast<uint32_t>(numBlocks);
            const uint64_t remainderB = B % numBlocks;
            tiling.formerNum = static_cast<uint32_t>(remainderB);
            tiling.formerLength = static_cast<uint32_t>(remainderB == 0 ? 0 : bPerCore + 1);
            tiling.tailNum = static_cast<uint32_t>(remainderB == 0 ? numBlocks : numBlocks - remainderB);
            tiling.tailLength = static_cast<uint32_t>(bPerCore);
        }
    }

    OP_LOGD(context, "regbase tiling: strategy=%u isScale=%u isShift=%u B=%lu L=%lu D=%lu usedCores=%u.",
            static_cast<uint32_t>(tilingStrategy), static_cast<uint32_t>(isScale), static_cast<uint32_t>(isShift), B, L,
            D, tiling.usedCores);

    tiling.split_by_d = (tilingStrategy == MODULATE_GRAD_TILING_D) ? 1 : 0;
    context->SetTilingKey(GET_TPL_TILING_KEY(tilingStrategy, isScale, isShift));
    auto res = context->SetBlockDim(tiling.usedCores);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context, "SetBlockDim failed."), return ge::GRAPH_FAILED);
    res = context->SetLocalMemorySize(ubSize);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context, "SetLocalMemorySize failed."), return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(), &tiling,
                           sizeof(ModulateGradRegbaseTilingData));
    if (ret != EOK) {
        OP_LOGE(context, "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(sizeof(ModulateGradRegbaseTilingData));

    auto workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
