/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "chunk_loss_grad_tiling_arch22.h"
#include "../chunk_loss_grad_tiling_common.h"
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* context)
{
    // Complete Cube tiles use the one-window route. Keep tails on key0 until
    // the MDL tail-path L1 initialization diagnostics are resolved.
    if (context && context->GetInputShape(0) && context->GetInputShape(1)) {
        const auto& hiddenShape = context->GetInputShape(0)->GetStorageShape();
        const auto& weightShape = context->GetInputShape(1)->GetStorageShape();
        if (hiddenShape.GetDimNum() == 2 && weightShape.GetDimNum() == 2 && hiddenShape[0] >= 128 &&
            hiddenShape[0] <= 512 && hiddenShape[1] >= 128 && hiddenShape[1] <= 8192 && hiddenShape[0] % 128 == 0 &&
            hiddenShape[1] % 128 == 0 && weightShape[0] >= 1024 && weightShape[0] % 128 == 0) {
            const auto result = BuildChunkLossGradTiling<ChunkLossGradArch22TilingData, 512, 128>(context);
            if (result == ge::GRAPH_SUCCESS) {
                context->SetTilingKey(1);
                return result;
            }
            // Retain the existing resource-compatible route if a platform
            // cannot construct the larger Cube tile.
        }
    }
    return BuildChunkLossGradTiling<ChunkLossGradArch22TilingData, 32, 128>(context);
}
IMPL_OP_OPTILING(ChunkLossGrad).Tiling(Tiling).TilingParse<ChunkLossGradCompileInfo>(ParseChunkLossGrad);
} // namespace optiling
