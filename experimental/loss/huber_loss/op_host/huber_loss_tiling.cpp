/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "graph/utils/type_utils.h"
#include "../op_kernel/huber_loss_tiling_data.h"
#include <limits>

namespace optiling {
using Ops::Base::CeilDiv;
constexpr uint32_t kVectorElements = 64;
constexpr uint32_t kQueueCount = 6;         // two inputs and one output, all double buffered
constexpr uint32_t kFloatBuffers = 4;       // diff, abs(diff), quadratic, linear
constexpr uint32_t kUpcastFloatBuffers = 2; // converted predictions and targets
constexpr uint32_t kMaskBytes = 1;
static bool FitsU32(uint64_t value) { return value <= std::numeric_limits<uint32_t>::max(); }
static ge::graphStatus HuberLossTiling(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("HuberLoss", "tiling context is null"), return ge::GRAPH_FAILED);
    const auto* inputShape = context->GetInputShape(0);
    const auto* targetShape = context->GetInputShape(1);
    const auto* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    const gert::Shape& predictions = inputShape->GetStorageShape();
    const gert::Shape& targets = targetShape->GetStorageShape();
    const gert::Shape& loss = outputShape->GetStorageShape();
    OP_CHECK_IF(predictions != targets || predictions != loss, OP_LOGE(context, "HuberLoss requires equal shapes"),
                return ge::GRAPH_FAILED);
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* targetDesc = context->GetInputDesc(1);
    const auto* outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16,
                OP_LOGE(context, "HuberLoss unsupported dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(targetDesc->GetDataType() != dtype || outputDesc->GetDataType() != dtype,
                OP_LOGE(context, "HuberLoss requires equal dtypes"), return ge::GRAPH_FAILED);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* deltaAttr = attrs->GetAttrPointer<float>(0);
    const float delta = deltaAttr == nullptr ? 1.0f : *deltaAttr;
    OP_CHECK_IF(!(delta > 0.0f), OP_LOGE(context, "HuberLoss requires delta > 0"), return ge::GRAPH_FAILED);
    uint32_t elemBytes = 0;
    OP_CHECK_IF(!ge::TypeUtils::GetDataTypeLength(dtype, elemBytes) || elemBytes == 0,
                OP_LOGE(context, "HuberLoss failed to get dtype length"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos* info = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, info);
    platform_ascendc::PlatformAscendC platform(info);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t cores = platform.GetCoreNumAiv();
    OP_CHECK_IF(ubBytes == 0 || cores == 0, OP_LOGE(context, "HuberLoss invalid platform information"),
                return ge::GRAPH_FAILED);
    const int64_t signedTotal = predictions.GetShapeSize();
    OP_CHECK_IF(signedTotal < 0, OP_LOGE(context, "HuberLoss does not support unknown shape size"),
                return ge::GRAPH_FAILED);
    const uint64_t total = static_cast<uint64_t>(signedTotal);
    const uint64_t floatBuffers = kFloatBuffers + (dtype == ge::DT_FLOAT ? 0 : kUpcastFloatBuffers);
    const uint64_t bytesPerElement = kQueueCount * elemBytes + floatBuffers * sizeof(float) + kMaskBytes;
    const uint64_t rawTile = ubBytes / bytesPerElement;
    const uint64_t tile = (rawTile / kVectorElements) * kVectorElements;
    OP_CHECK_IF(tile == 0 || !FitsU32(tile), OP_LOGE(context, "HuberLoss UB cannot hold one aligned tile"),
                return ge::GRAPH_FAILED);
    const uint64_t usedCores = total == 0 ? 1 : (total < cores ? total : cores);
    const uint64_t small = total / usedCores;
    const uint64_t tailCores = total % usedCores;
    const uint64_t big = small + (tailCores == 0 ? 0 : 1);
    const uint64_t smallTiles = small == 0 ? 0 : CeilDiv(small, tile);
    const uint64_t bigTiles = big == 0 ? 0 : CeilDiv(big, tile);
    const uint64_t smallTail = small == 0 ? 0 : small - (smallTiles - 1) * tile;
    const uint64_t bigTail = big == 0 ? 0 : big - (bigTiles - 1) * tile;
    OP_CHECK_IF(tailCores * big + (usedCores - tailCores) * small != total,
                OP_LOGE(context, "HuberLoss core split does not conserve the logical element count"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!FitsU32(small) || !FitsU32(big) || !FitsU32(smallTiles) || !FitsU32(bigTiles) || !FitsU32(smallTail) ||
                    !FitsU32(bigTail) || !FitsU32(tailCores),
                OP_LOGE(context, "HuberLoss tiling field exceeds uint32 range"), return ge::GRAPH_FAILED);
    auto* tiling = context->GetTilingData<HuberLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->smallCoreDataNum = static_cast<uint32_t>(small);
    tiling->bigCoreDataNum = static_cast<uint32_t>(big);
    tiling->finalBigTileNum = static_cast<uint32_t>(bigTiles);
    tiling->finalSmallTileNum = static_cast<uint32_t>(smallTiles);
    tiling->tileDataNum = static_cast<uint32_t>(tile);
    tiling->smallTailDataNum = static_cast<uint32_t>(smallTail);
    tiling->bigTailDataNum = static_cast<uint32_t>(bigTail);
    tiling->tailBlockNum = static_cast<uint32_t>(tailCores);
    tiling->delta = delta;
    context->SetTilingKey(0);
    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HuberLoss).Tiling(HuberLossTiling);
} // namespace optiling
