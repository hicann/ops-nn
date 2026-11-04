// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// clang-format on

#include "max_pool2d_with_mask_backward_tiling.h"
#include <algorithm>

namespace optiling {
namespace {
constexpr uint64_t SCALAR_UB_BYTES = 96 * 1024;
constexpr uint64_t SCATTER_UB_BYTES = 192 * 1024;
constexpr uint64_t VECTOR_UB_BYTES = 240 * 1024;
// Element capacities match the corresponding Kernel template buffers. They
// cannot be enlarged using platform UB alone without changing those buffers.
constexpr uint64_t PIPELINE_SCATTER_ELEMENTS = 25600;
constexpr uint64_t LARGE_SCATTER_ELEMENTS = 32768;
constexpr uint64_t SMALL_SCATTER_ELEMENTS = 16384;

enum VectorLayout : uint64_t {
    BASE_LAYOUT = 0,
    SPARSE_C8 = 1,
    DENSE_C8 = 2,
    WIDE_OUTPUT = 3,
    TIGHT_ORIGINS = 4,
    FP32_K3S2 = 5,
    FP16_K4S3 = 6,
    FP16_K4S3_SMALL_INPUT = 7,
    SHARED_FP32 = 8,
    SHARED_FP16_TAIL = 9,
    REBALANCED_K3S2 = 10,
    NARROW_FP32_RAMP = 13,
    NARROW_FP16_RAMP = 14,
    FULL_WIDTH_K7 = 15
};

uint64_t SelectVectorTilingKey(uint64_t layout, ge::DataType dtype, uint64_t channelBlock, uint64_t kh, uint64_t kw,
                               uint64_t sh, uint64_t sw)
{
    // Cache variants share a layout budget but use different Kernel code.
    if (dtype == ge::DT_FLOAT && layout == WIDE_OUTPUT && kh == 5 && kw == 5 && sh == 4 && sw == 4) {
        return 20047;
    }
    if (dtype == ge::DT_FLOAT && layout == SPARSE_C8 && kh == 3 && kw == 5 && sh == 2 && sw == 3) {
        return 20049;
    }
    if (layout == BASE_LAYOUT) {
        return (dtype == ge::DT_FLOAT16 ? 20030 : 20031) + (channelBlock == 8 ? 2 : 0);
    }
    if (layout == TIGHT_ORIGINS) {
        return kh == 5 && kw == 3 && sh == 4 && sw == 2 ? 20057 :
               kh == 7 && kw == 3 && sh == 4 && sw == 2 ? 20059 :
                                                          20053;
    }
    if (layout == NARROW_FP32_RAMP) {
        return kh == 5 ? 20073 : 20069;
    }
    struct LayoutKey {
        uint64_t layout, fp16, fp32;
    };
    constexpr LayoutKey keys[] = {
        {SPARSE_C8, 20034, 20035},        {DENSE_C8, 20036, 20037},         {WIDE_OUTPUT, 20038, 20039},
        {FP32_K3S2, 20055, 20055},        {FP16_K4S3, 20056, 20056},        {FP16_K4S3_SMALL_INPUT, 20058, 20058},
        {SHARED_FP32, 20061, 20061},      {SHARED_FP16_TAIL, 20062, 20062}, {REBALANCED_K3S2, 20063, 20063},
        {NARROW_FP16_RAMP, 20068, 20068}, {FULL_WIDTH_K7, 20071, 20071}};
    for (const auto& entry : keys) {
        if (entry.layout == layout) {
            return dtype == ge::DT_FLOAT16 ? entry.fp16 : entry.fp32;
        }
    }
    return 0; // All selectable layouts must have a matching Kernel key.
}
} // namespace

class Tiling310P : public MaxPool2dWithMaskBackwardTilingBase {
public:
    explicit Tiling310P(gert::TilingContext* context) : MaxPool2dWithMaskBackwardTilingBase(context) {}
    ge::graphStatus Run()
    {
        OP_CHECK_IF(GetShapeAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Invalid 310P pooling inputs."),
                    return ge::GRAPH_FAILED);
        auto& p = maxPoolGradParams;
        auto dtype = context_->GetInputDesc(0)->GetDataType();
        OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT,
                    OP_LOGE(context_, "310P supports FP16 and FP32; native BF16 is unavailable."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(p.diDim != 1 || p.doDim != 1 || p.kd != 1 || p.sd != 1 || p.pDTop != 0 || p.dilationD != 1 ||
                        p.dilationH != 1 || p.dilationW != 1,
                    OP_LOGE(context_, "310P branch requires 2D pooling with dilation=1."), return ge::GRAPH_FAILED);
        auto platform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint64_t ub = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        const uint64_t cores = platform.GetCoreNumAic();
        OP_CHECK_IF(cores == 0 || ub < SCALAR_UB_BYTES,
                    OP_LOGE(context_, "310P platform lacks required AI cores or 96KiB UB."), return ge::GRAPH_FAILED);
        // A stripe ends on a pooling-row boundary. With no overlap/padding,
        // every argmax belongs to exactly one stripe and scatter has no conflicts.
        const bool largeScatter = p.wiDim >= 192 && ub >= VECTOR_UB_BYTES;
        const uint64_t pipelineRows = (PIPELINE_SCATTER_ELEMENTS / p.wiDim) / p.sh * p.sh;
        const bool pipelineFloat = largeScatter && dtype == ge::DT_FLOAT && p.wiDim % 8 == 0 && p.ncDim % cores == 0 &&
                                   pipelineRows > 0 && p.hiDim >= 2 * pipelineRows;
        const uint64_t scatterCapacity = pipelineFloat ?
                                             PIPELINE_SCATTER_ELEMENTS :
                                             (largeScatter ? LARGE_SCATTER_ELEMENTS : SMALL_SCATTER_ELEMENTS);
        const uint64_t scatterRows = (scatterCapacity / p.wiDim) / p.sh * p.sh;
        if (p.kh <= p.sh && p.kw <= p.sw && p.pHTop == 0 && p.pWTop == 0 && !(p.kh == 1 && p.kw == 1) &&
            p.wiDim >= 16 && scatterRows > 0 && p.hiDim * p.wiDim < (1ULL << 31) && ub >= SCATTER_UB_BYTES) {
            p.totalCnt = p.ncDim * ((p.hiDim + scatterRows - 1) / scatterRows);
            p.usedCoreNum = std::min<uint64_t>(cores, p.totalCnt);
            p.baseNc = 1;
            p.baseHo = scatterRows;
            p.baseWo = scatterCapacity;
            SetBaseTilingData();
            tilingData.set_totalUBSize(largeScatter ? VECTOR_UB_BYTES : SCATTER_UB_BYTES);
            context_->SetTilingKey(pipelineFloat ? 20045 :
                                                   (dtype == ge::DT_FLOAT16 ? 20040 : 20041) + (largeScatter ? 2 : 0));
            return GetWorkspaceSize();
        }
        // Small N*C uses C8 and a larger spatial block, preserving Normal flow.
        // Both channel blocks fit in the reserved 240 KiB including scratch.
        const bool anisotropic = p.sh == 1 && p.sw > 1 && p.hiDim * p.wiDim >= 16384;
        const uint64_t channelBlock = p.ncDim > 4 && p.hiDim * p.wiDim < 16384 ? 16 : 8;
        struct TileChoice {
            uint64_t w, h, cuts, inputPitch;
        };
        auto chooseTile = [&](uint64_t inputCapacity, uint64_t outputCapacity, bool tightBounds = false,
                              uint64_t forcedWidth = 0) {
            const bool dense96 = channelBlock == 8 && dtype == ge::DT_FLOAT && p.ncDim <= 8 && p.sh == 1 && p.sw == 1 &&
                                 p.kh <= 3 && p.kw <= 3 && p.hiDim * p.wiDim >= 65536 && p.wiDim % 96 == 0;
            uint64_t tileW = p.sh > 1 && p.sw > 1 ? 256 : (anisotropic ? 128 : (dense96 ? 96 : 64));
            if (forcedWidth)
                tileW = forcedWidth;
            uint64_t cuts = 1, tileH = 0, selectedInputPitch = 0;
            do {
                cuts = std::max<uint64_t>(1, p.wiDim / tileW);
                const uint64_t maxWidth = p.wiDim - (cuts - 1) * tileW;
                const uint64_t outputPitch = (maxWidth + 15) / 16 * 16;
                const uint64_t maxCols = (maxWidth + p.kw - 2 + (tightBounds ? 0 : p.sw - 1)) / p.sw + 1;
                const uint64_t alignment = channelBlock == 8 && dtype == ge::DT_FLOAT ? 8 : 16;
                const uint64_t inputPitch = (maxCols + alignment - 1) / alignment * alignment;
                selectedInputPitch = inputPitch;
                tileH = outputCapacity / outputPitch;
                // The index buffer stages one 32B output tail per channel/row.
                const uint64_t storeBlock = dtype == ge::DT_FLOAT ? 8 : 16;
                if (maxWidth % storeBlock != 0)
                    tileH = std::min<uint64_t>(tileH, inputCapacity / 8);
                while (tileH &&
                       ((tileH + p.kh - 2 + (tightBounds ? 0 : p.sh - 1)) / p.sh + 1) * inputPitch > inputCapacity)
                    --tileH;
                if (tileH || tileW == 64)
                    break;
                tileW /= 2;
            } while (true);
            return TileChoice{tileW, tileH, cuts, selectedInputPitch};
        };
        auto chosen = chooseTile(8192 / channelBlock, 12288 / channelBlock);
        uint64_t layout = BASE_LAYOUT;
        const bool sparseC8 = channelBlock == 8 && p.sh > 1 && p.sw > 1;
        const bool denseC8 = channelBlock == 8 && p.sh == 1 && p.sw == 1 && p.kh <= 3 && p.kw <= 3 &&
                             p.hiDim * p.wiDim >= 65536;
        const bool fp32Sparse = sparseC8 && dtype == ge::DT_FLOAT;
        const bool fp16Sparse = sparseC8 && dtype == ge::DT_FLOAT16;
        const bool k3s2 = p.kh == 3 && p.kw == 3 && p.sh == 2 && p.sw == 2;
        const bool k4s3 = p.kh == 4 && p.kw == 4 && p.sh == 3 && p.sw == 3;
        const bool k57s4 = (p.kh == 5 || p.kh == 7) && p.kw == 3 && p.sh == 4 && p.sw == 2;
        const bool k3x5 = p.kh == 3 && p.kw == 5 && p.sh == 2 && p.sw == 3 && p.hiDim * p.wiDim >= 65536;
        struct LayoutRule {
            uint64_t id;
            // Total element capacities across channelBlock lanes (not bytes).
            uint64_t inputElements, outputElements;
            bool enabled, tightBounds;
            uint64_t forcedWidth, maxRows, maxInputPitch;
            bool requireFewerTiles;
        };
        // Ordered refinements: preserve the previous priority and tie behavior.
        // Shared layouts cap rows/pitch because their tail/ramp buffers are fixed.
        const LayoutRule rules[] = {
            {sparseC8 ? SPARSE_C8 : DENSE_C8, sparseC8 ? 6144UL : 10240UL, sparseC8 ? 16384UL : 8192UL,
             sparseC8 || denseC8, false, 0, 0, 0, true},
            {WIDE_OUTPUT, 4096, 20480, fp32Sparse, false, 0, 0, 0, true},
            {TIGHT_ORIGINS, 4608, 20480, fp32Sparse && (k3x5 || k57s4), true, 0, 0, 0, true},
            {FP32_K3S2, 6144, 17408, fp32Sparse && k3s2, true, 0, 0, 0, true},
            {FP16_K4S3, 4096, 19200, fp16Sparse && k4s3, true, 0, 0, 0, true},
            {FP16_K4S3_SMALL_INPUT, 3072, 20480, fp16Sparse && k4s3, true, 0, 0, 0, true},
            {SHARED_FP32, 7168, 28672, fp32Sparse && (k3s2 || k57s4), true, 0, 32, 0, true},
            {SHARED_FP16_TAIL, 6144, 28672, fp16Sparse && p.wiDim % 16 != 0 && k4s3, true, 0, 32, 0, true},
            {REBALANCED_K3S2, 7936, 26112, fp32Sparse && k3s2, true, 0, 32, 0, true},
            {NARROW_FP32_RAMP, 5632, 30464, fp32Sparse && k57s4, true, 0, 32, 192, false},
            {NARROW_FP16_RAMP, 4608, 30464, fp16Sparse && p.wiDim % 16 != 0 && k4s3, true, 0, 32, 192, false},
            {FULL_WIDTH_K7, 6400, 30464,
             fp32Sparse && p.kh == 7 && p.kw == 3 && p.sh == 4 && p.sw == 2 && p.wiDim > 256, true, p.wiDim, 32, 264,
             false}};
        for (const auto& rule : rules) {
            if (!rule.enabled) {
                continue;
            }
            const auto candidate = chooseTile(rule.inputElements / channelBlock, rule.outputElements / channelBlock,
                                              rule.tightBounds, rule.forcedWidth);
            if (candidate.h == 0 || (rule.maxRows && candidate.h > rule.maxRows) ||
                (rule.maxInputPitch && candidate.inputPitch > rule.maxInputPitch) ||
                (rule.forcedWidth && candidate.cuts != 1)) {
                continue;
            }
            if (rule.requireFewerTiles && chosen.h &&
                ((p.hiDim + candidate.h - 1) / candidate.h) * candidate.cuts >=
                    ((p.hiDim + chosen.h - 1) / chosen.h) * chosen.cuts) {
                continue;
            }
            chosen = candidate;
            layout = rule.id;
        }
        const uint64_t tileW = chosen.w, tileH = chosen.h, cuts = chosen.cuts;
        // Store requires at least one 32-byte block, including overlapping tails.
        const uint64_t minVectorWidth = dtype == ge::DT_FLOAT ? 8 : 16;
        const bool vectorFits = p.wiDim >= minVectorWidth && p.kw <= 16 && p.sw <= 127 && tileH > 0 &&
                                p.kh * p.wiDim + p.kw < (1 << 24) && !(p.kh == 1 && p.kw == 1) && ub >= VECTOR_UB_BYTES;
        if (vectorFits) {
            p.totalCnt = ((p.ncDim + channelBlock - 1) / channelBlock) * ((p.hiDim + tileH - 1) / tileH) * cuts;
            p.usedCoreNum = std::min<uint64_t>(cores, p.totalCnt);
            p.baseNc = channelBlock;
            p.baseHo = tileH;
            p.baseWo = tileW;
            SetBaseTilingData();
            tilingData.set_totalUBSize(VECTOR_UB_BYTES);
            const uint64_t key = SelectVectorTilingKey(layout, dtype, channelBlock, p.kh, p.kw, p.sh, p.sw);
            OP_CHECK_IF(key == 0, OP_LOGE(context_, "Missing Kernel key for 310P layout."), return ge::GRAPH_FAILED);
            context_->SetTilingKey(key);
            return GetWorkspaceSize();
        }
        // The last tile absorbs a remainder, so a short tail never belongs to
        // another core. Maximum tile is 8191 elements; total UB < 96KiB.
        constexpr uint64_t tile = 4096;
        const uint64_t total = p.ncDim * p.hiDim * p.wiDim;
        p.usedCoreNum = std::min(cores, std::max<uint64_t>(1, total / tile));
        p.totalCnt = std::max<uint64_t>(1, total / tile);
        p.baseWo = tile;
        SetBaseTilingData();
        tilingData.set_totalUBSize(SCALAR_UB_BYTES);
        tilingData.set_baseWo(tile);
        tilingData.set_totalCnt(p.totalCnt);
        context_->SetTilingKey(dtype == ge::DT_FLOAT16 ? 20020 : 20021);
        return GetWorkspaceSize();
    }
};

ge::graphStatus TilingMaxPool2dWithMaskBackward310P(gert::TilingContext* context) { return Tiling310P(context).Run(); }
} // namespace optiling
