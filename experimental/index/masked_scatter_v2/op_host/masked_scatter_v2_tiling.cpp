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
 * \file masked_scatter_v2_tiling.cpp
 * \brief MaskedScatterV2 tiling (Ascend 950PR two-phase vectorized design)
 *
 * 与 A2 版的差异：
 *   1. 两阶段向量化（PhaseA 计数 -> SoftSync 广播 -> PhaseC 向量前缀+Gather+Select），
 *      替代 A2 的"重扫前序 mask + 标量 SetValue"；mask 只读 2 遍而非 O(coreNum/2) 遍。
 *   2. 支持 mask 右对齐广播（native 条件：rank<=8、innermost 扩展 stride==1、
 *      total>=8192 且 mask 物理字节数>=32；不满足的广播 shape 要求调用方预展开）。
 *   3. workspace 用作跨核计数槽（SyncAll flags + counts），必须零初始化
 *      （见 op_api L1 的 aclrtMemset / 框架 zero-init TODO）。
 */
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/masked_scatter_v2_tiling_data.h"
#include "../op_kernel/masked_scatter_v2_tiling_key.h"

namespace optiling {
using namespace Ops::NN::OpTiling;
static constexpr int64_t CHUNK_ELEMS = 4096;        // kernel 单 chunk 元素数（UB 预算 205KB）
static constexpr int64_t WS_INT32_PER_CORE = 32;    // SyncAll flags 区（int32 数/核）
static constexpr int64_t WS_RESERVE_INT32 = 512;    // counts 区预留
static constexpr int64_t BCAST_MIN_TOTAL = 8192;    // native 广播准入：>= 2 个完整 chunk
static constexpr int64_t BCAST_MIN_MASK_BYTES = 32; // native 广播准入：mask 本体 >= 1 个 32B MTE 块

struct MaskedScatterV2CompileInfo {};

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* ctx, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = ctx->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(ctx, "MaskedScatterV2: coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 解析 mask 广播：返回 0 = 同 shape；1 = 可 native 广播（填 strides）；-1 = 不支持的广播
static int32_t AnalyzeMaskLayout(gert::TilingContext* ctx, int64_t total, int32_t maskSize[8], int32_t maskStride[8],
                                 int32_t& maskRankOut, int64_t& maskNumel)
{
    auto xShape = EnsureNotScalar(ctx->GetInputShape(0)->GetStorageShape());
    auto mShape = EnsureNotScalar(ctx->GetInputShape(1)->GetStorageShape());
    const int64_t sRank = xShape.GetDimNum();
    const int64_t bRank = mShape.GetDimNum();
    maskNumel = mShape.GetShapeSize();
    if (mShape == xShape) {
        return 0;
    }
    if (bRank > sRank) {
        return -1;
    }
    for (int64_t i = 0; i < bRank; ++i) {
        const int64_t sd = sRank - bRank + i;
        const int64_t ms = mShape.GetDim(i);
        if (ms != 1 && ms != xShape.GetDim(sd)) {
            return -1;
        }
    }
    // 右对齐展开 shape 与 strides（广播维 stride=0；不满足 native 条件时 stride 置 -1 标记）
    int64_t acc = 1;
    for (int64_t d = sRank - 1; d >= 0; --d) {
        const int64_t srcIdx = d - (sRank - bRank);
        const bool bcastDim = (srcIdx >= 0) && (mShape.GetDim(srcIdx) == 1);
        maskSize[d] = static_cast<int32_t>(xShape.GetDim(d));
        maskStride[d] = bcastDim ? 0 : static_cast<int32_t>(acc);
        if (!bcastDim) {
            acc *= xShape.GetDim(d);
        }
    }
    maskRankOut = static_cast<int32_t>(sRank);
    // native 准入（与 torch 直调版一致：innermost 扩展 stride 必须为 1）
    if (sRank > 8 || total < BCAST_MIN_TOTAL || maskNumel * 1 < BCAST_MIN_MASK_BYTES || maskStride[sRank - 1] != 1) {
        return -1;
    }
    return 1;
}

static ge::graphStatus MaskedScatterV2TilingFunc(gert::TilingContext* ctx)
{
    OP_CHECK_NULL_WITH_CONTEXT(ctx, ctx);
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(ctx, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(ctx, "MaskedScatterV2: GetPlatformInfo error"), return ge::GRAPH_FAILED);

    auto xShape = EnsureNotScalar(ctx->GetInputShape(0)->GetStorageShape());
    auto updatesShape = EnsureNotScalar(ctx->GetInputShape(2)->GetStorageShape());
    const int64_t total = xShape.GetShapeSize();
    const int64_t sourceLen = updatesShape.GetShapeSize();

    // dtype / shape 校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32};
    auto inputDesc = ctx->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, inputDesc);
    OP_CHECK_IF(supportedDtype.count(inputDesc->GetDataType()) == 0,
                OP_LOGE(ctx, "MaskedScatterV2: unsupported dtype %d", static_cast<int>(inputDesc->GetDataType())),
                return ge::GRAPH_FAILED);
    auto maskDesc = ctx->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, maskDesc);
    OP_CHECK_IF(maskDesc->GetDataType() != ge::DT_BOOL, OP_LOGE(ctx, "MaskedScatterV2: mask dtype must be bool"),
                return ge::GRAPH_FAILED);

    int32_t maskSize[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int32_t maskStride[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t maskRank = 0;
    int64_t maskNumel = 0;
    const int32_t layout = AnalyzeMaskLayout(ctx, total, maskSize, maskStride, maskRank, maskNumel);
    OP_CHECK_IF(layout < 0,
                OP_LOGE(ctx, "MaskedScatterV2: mask broadcast shape not supported natively; please expand mask to x "
                             "shape first"),
                return ge::GRAPH_FAILED);

    MaskedScatterV2TilingData* tiling = ctx->GetTilingData<MaskedScatterV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(MaskedScatterV2TilingData), 0, sizeof(MaskedScatterV2TilingData)) != EOK,
                OP_LOGE(ctx, "MaskedScatterV2: set tiling data error"), return ge::GRAPH_FAILED);

    const int64_t totalChunks = (total + CHUNK_ELEMS - 1) / CHUNK_ELEMS;
    tiling->total = static_cast<int32_t>(total);
    tiling->sourceLen = static_cast<int32_t>(sourceLen);
    tiling->coreNum = static_cast<int32_t>(coreNum);
    tiling->useSync = (totalChunks > 1) ? 1 : 0;
    tiling->chunksBase = static_cast<int32_t>(totalChunks / coreNum);
    tiling->chunksRem = static_cast<int32_t>(totalChunks % coreNum);
    tiling->maskIsBcast = (layout == 1) ? 1 : 0;
    tiling->maskRank = (layout == 1) ? maskRank : static_cast<int32_t>(xShape.GetDimNum());
    tiling->maskTotal = static_cast<int32_t>(maskNumel);
    for (int64_t d = 0; d < tiling->maskRank; ++d) {
        tiling->maskSize[d] = maskSize[d];
        tiling->maskStride[d] = maskStride[d];
    }

    // workspace：SyncAll flags + counts（必须零初始化，见文件头注释 3）
    size_t* workspaceSizes = ctx->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, workspaceSizes);
    workspaceSizes[0] = static_cast<size_t>((coreNum * WS_INT32_PER_CORE + WS_RESERVE_INT32) * sizeof(int32_t));

    ctx->SetBlockDim(static_cast<uint32_t>(coreNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMaskedScatterV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaskedScatterV2)
    .Tiling(MaskedScatterV2TilingFunc)
    .TilingParse<MaskedScatterV2CompileInfo>(TilingParseForMaskedScatterV2);
} // namespace optiling
