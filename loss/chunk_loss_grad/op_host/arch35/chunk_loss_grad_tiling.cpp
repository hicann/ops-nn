/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "chunk_loss_grad_tiling_arch35.h"
#include "../chunk_loss_grad_tiling_common.h"
#include "../../op_kernel/arch35/chunk_loss_grad_direct_options.h"

namespace optiling {
namespace {
constexpr uint64_t RESERVE = 16 * 1024;
uint64_t Align(uint64_t value, uint64_t numTokens) { return (value + numTokens - 1) / numTokens * numTokens; }
} // namespace
static ge::graphStatus Tiling(gert::TilingContext* context)
{
    // Existing validation and the old Cube tilings are retained for the tiny
    // and large-stride fallback. arch22 instantiations are unchanged.
    const auto result = BuildChunkLossGradTiling<ChunkLossGradArch35TilingData, 64, 128>(context);
    if (result != ge::GRAPH_SUCCESS) {
        return result;
    }
    const auto& hiddenShape = context->GetInputShape(0)->GetStorageShape();
    const auto& weightShape = context->GetInputShape(1)->GetStorageShape();
    const uint64_t numTokens = hiddenShape[0], hiddenDim = hiddenShape[1], vocabSize = weightShape[0];
    if (numTokens <= 4 && hiddenDim <= 32 && vocabSize <= 32) {
        // Compile the tiny route independently of the larger new backend.
        context->SetTilingKey(1);
        return ge::GRAPH_SUCCESS;
    }
    if (hiddenDim > 65535) {
        // Explicitly mark the validated batched coarse-layout fallback.
        ChunkLossGradArch35TilingData fallback(context->GetRawTilingData()->GetData());
        fallback.set_scheduleMode(3);
        return ge::GRAPH_SUCCESS;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const bool largeDense = numTokens >= 128 && hiddenDim >= 128 && vocabSize >= 1024;
    // BF16's exact probability VF benefits from smaller CV batches in the
    // measured long-N partition. Keep enough V tasks for several core waves.
    const bool bf16 = context->GetInputDesc(0)->GetDataType() == ge::DT_BF16;
    uint32_t vocabTileSize = largeDense ? (bf16 ? 64 : 128) :
                                          (numTokens <= 4 && hiddenDim <= 4 && vocabSize >= 2048 ? 256 : 16);
    if (largeDense) {
        while (vocabTileSize > 16 && (vocabSize + vocabTileSize - 1) / vocabTileSize < 4 * platform.GetCoreNumAic()) {
            vocabTileSize /= 2;
        }
    }
    // Full G resides in two L1 slots. Split-M Fixpipe needs even logical M;
    // odd-V inputs retain the existing general path rather than reading an
    // uncomputed padded L0C row. No public input constraint is added.
    if (largeDense && numTokens <= 512 && vocabSize % 2 == 0) {
        using namespace ChunkLossGradOptions;
        uint32_t options = bf16 ? BUILD_MASK :
                                  (BUILD_MASK & (REUSE_METADATA | P_VIA_GM | REUSE_LOGITS_WEIGHT | LOGITS_GROUP_TWO |
                                                 REUSE_LOGITS_L0A));
        // Grouping trades first-fragment latency for operand reuse. Keep the
        // original path for small N/H, where screening did not show a gain.
        if (numTokens < 256 || hiddenDim < 512) {
            options &= ~(REUSE_LOGITS_WEIGHT | LOGITS_GROUP_TWO | REUSE_LOGITS_L0A);
        }
        const bool pViaGm = (options & P_VIA_GM) != 0;
        uint32_t directVt = vocabTileSize;
        if (options & BF16_VT128) {
            directVt = 128;
            while (directVt > 16 && (vocabSize + directVt - 1) / directVt < 4 * platform.GetCoreNumAic()) {
                directVt /= 2;
            }
        }
        uint64_t l1Bytes = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Bytes);
        const uint64_t gL1 = Align(uint64_t(directVt) * Align(numTokens, 16) * 2, 512);
        const uint32_t numCoreGroups = std::min<uint64_t>(platform.GetCoreNumAic(),
                                                          (vocabSize + directVt - 1) / directVt);
        // Two shared FP32 L/P slots (split-M per AIV), one G NZ scratch,
        // immutable metadata and one low-precision H256 output tile.
        const uint32_t vectorElements = (directVt / 2) * 256;
        const uint32_t outputSlots = (options & OUTPUT_DOUBLE_BUFFER) ? 2 : 1;
        const uint64_t separateL = (options & SEPARATE_L_P) ? uint64_t(directVt) * 128 * 4 : 0;
        const uint64_t vectorBytes = uint64_t(directVt) * 256 * 4 + separateL + uint64_t(directVt + 1) * 128 * 2 +
                                     Align(numTokens, 64) * 20 + uint64_t(vectorElements) * 2 * outputSlots + 64 +
                                     (pViaGm ? 64 * 4 : 0);
        if (numCoreGroups > 0 && platform.GetCoreNumAiv() >= 2 * numCoreGroups && 256 * 1024 + 2 * gL1 <= l1Bytes &&
            vectorBytes + RESERVE <= ubBytes) {
            const uint64_t system = platform.GetLibApiWorkSpaceSize();
            const uint64_t partial = pViaGm ? Align(uint64_t(directVt) * hiddenDim * 4, 512) : 0;
            const uint64_t perCore = 2 * partial;
            if (perCore > (UINT64_MAX - system) / numCoreGroups) {
                return ge::GRAPH_FAILED;
            }
            ChunkLossGradArch35TilingData tilingData;
            tilingData.set_alpha(GetChunkLossGradAlpha(context));
            tilingData.set_reserved(0);
            tilingData.set_ignoreIndex(GetChunkLossGradIgnoreIndex(context));
            tilingData.set_numTokens(numTokens);
            tilingData.set_hiddenDim(hiddenDim);
            tilingData.set_vocabSize(vocabSize);
            tilingData.set_coreNum(numCoreGroups);
            tilingData.set_tokenTile(128);
            tilingData.set_vocabTile(directVt);
            tilingData.set_vectorChunk(vectorElements);
            tilingData.set_logitsBytes(0);
            tilingData.set_logitsGradBytes(0);
            tilingData.set_partialBytes(partial);
            tilingData.set_coreWorkspaceBytes(perCore);
            tilingData.set_scheduleMode(5 | (options << MODE_SHIFT));
            tilingData.set_tokenPitch(Align(numTokens, 16));
            tilingData.set_cacheElements(Align(numTokens, 64));
            tilingData.set_residentAccum(0);
            tilingData.set_accumElements(0);
            tilingData.set_slotCount(2);
            if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
                return ge::GRAPH_FAILED;
            }
            tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
            context->SetBlockDim(numCoreGroups);
            context->SetTilingKey(2);
            context->GetWorkspaceSizes(1)[0] = system + perCore * numCoreGroups;
            return ge::GRAPH_SUCCESS;
        }
    }
    uint32_t tokenTileSize = std::min<uint64_t>(numTokens, largeDense ? 512 : 256);
    const uint64_t accumElementsPerLane = Align((uint64_t(vocabTileSize) * hiddenDim + 1) / 2, 16);
    uint32_t vectorElements = std::min<uint64_t>(accumElementsPerLane, 8192);
    bool resident = true;
    uint64_t vectorBytes = 0;
    bool fits = false;
    // Compare both accumulator modes at a token-window size BEFORE reducing
    // that size. An impossible full accumulator must not force NT down to16
    // before the usable segmented route is considered.
    for (;;) {
        const bool singleWindow = largeDense && numTokens <= tokenTileSize;
        for (uint32_t mode = 0; mode < (singleWindow ? 1U : 2U) && !fits; ++mode) {
            resident = !singleWindow && mode == 0;
            vectorElements = std::min<uint64_t>(accumElementsPerLane, singleWindow ? 16384 : 8192);
            for (;;) {
                const uint64_t cache = Align(tokenTileSize, 64), pitch = Align(tokenTileSize, 16),
                               rows = (vocabTileSize + 1) / 2;
                const uint64_t staging = rows * cache * 4 + rows * pitch * 2 + cache * 20 +
                                         uint64_t(vectorElements + 64) * 4 + uint64_t(vectorElements) * 2 + 64;
                const uint64_t accum = singleWindow ? 0 :
                                                      (resident ? (accumElementsPerLane + 64) * 4 :
                                                                  uint64_t(vectorElements + 64) * 4);
                vectorBytes = staging + accum;
                if (vectorBytes + RESERVE <= ubBytes) {
                    fits = true;
                    break;
                }
                if (vectorElements <= 1024) {
                    break;
                }
                vectorElements /= 2;
            }
        }
        if (fits) {
            break;
        }
        if (tokenTileSize <= 16) {
            return ge::GRAPH_FAILED;
        }
        tokenTileSize = (tokenTileSize + 1) / 2;
    }
    const uint32_t numCoreGroups = std::min<uint64_t>(platform.GetCoreNumAic(),
                                                      (vocabSize + vocabTileSize - 1) / vocabTileSize);
    const uint64_t tasks = (vocabSize + vocabTileSize - 1) / vocabTileSize;
    const uint64_t numTokenChunks = (numTokens + tokenTileSize - 1) / tokenTileSize;
    // Only groups with multiple work chunks can benefit from look-ahead. Slots
    // are GM L/G/P ownership, independent of the single-stage UB scratch.
    const uint32_t slots = numTokenChunks > 1 || tasks > numCoreGroups ? 2 : 1;
    const uint64_t logitsWorkspaceBytes = Align(uint64_t(vocabTileSize) * tokenTileSize * 4, 512);
    const uint64_t gradientWorkspaceBytes = Align(uint64_t(vocabTileSize) * Align(tokenTileSize, 16) * 2, 512);
    const uint64_t partialWorkspaceBytes = Align(uint64_t(vocabTileSize) * hiddenDim * 4, 512);
    const uint64_t perCore = slots * (logitsWorkspaceBytes + gradientWorkspaceBytes + partialWorkspaceBytes) +
                             (resident || numTokenChunks == 1 ? 0 : partialWorkspaceBytes);
    const uint64_t system = platform.GetLibApiWorkSpaceSize();
    if (numCoreGroups == 0 || platform.GetCoreNumAiv() < 2 * numCoreGroups ||
        perCore > (UINT64_MAX - system) / numCoreGroups) {
        return ge::GRAPH_FAILED;
    }
    ChunkLossGradArch35TilingData tilingData;
    tilingData.set_alpha(GetChunkLossGradAlpha(context));
    tilingData.set_reserved(0);
    tilingData.set_ignoreIndex(GetChunkLossGradIgnoreIndex(context));
    tilingData.set_numTokens(numTokens);
    tilingData.set_hiddenDim(hiddenDim);
    tilingData.set_vocabSize(vocabSize);
    tilingData.set_coreNum(numCoreGroups);
    tilingData.set_tokenTile(tokenTileSize);
    tilingData.set_vocabTile(vocabTileSize);
    tilingData.set_vectorChunk(vectorElements);
    tilingData.set_logitsBytes(logitsWorkspaceBytes);
    tilingData.set_logitsGradBytes(gradientWorkspaceBytes);
    tilingData.set_partialBytes(partialWorkspaceBytes);
    tilingData.set_coreWorkspaceBytes(perCore);
    tilingData.set_scheduleMode(slots == 2 ? 2 : 1);
    tilingData.set_tokenPitch(Align(tokenTileSize, 16));
    tilingData.set_cacheElements(Align(tokenTileSize, 64));
    tilingData.set_residentAccum(resident);
    tilingData.set_accumElements(resident ? accumElementsPerLane : vectorElements);
    tilingData.set_slotCount(slots);
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(numCoreGroups);
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = perCore * numCoreGroups + system;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(ChunkLossGrad).Tiling(Tiling).TilingParse<ChunkLossGradCompileInfo>(ParseChunkLossGrad);
} // namespace optiling
