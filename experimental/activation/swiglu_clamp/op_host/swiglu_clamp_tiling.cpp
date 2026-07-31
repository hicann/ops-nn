/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_clamp_tiling.cpp
 * \brief SwigluClamp tiling: two-level split.
 *        - Block level: split by ROW (cores always split on row boundaries so the gate/up
 *          pair of one row is never split). rowBytes = 2*N*dtypeSize is a multiple of the
 *          32B UB alignment (guaranteed by the N % ubAlignElements check), so every core
 *          starts 32B-aligned; MoE shapes (N>=1280, bf16) are additionally 512B cache-line
 *          aligned for best GM throughput.
 *        - UB level: tileM = floor(UB_SIZE_LIMIT / bufferCoefficient / N), where
 *          bufferCoefficient = 32 B/out-element (inQueueX double-buffered [tileM,2N] +
 *          outQueueY double-buffered [tileM,N] + 5 fp32 TBufs). Row-internal tiling is not
 *          implemented, so N must fit a single row in UB (rejected up front otherwise).
 *
 *        Logic is ported from the verified vllm-ascend feat/swiglustep-ascendc-fused tiling.
 */
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/swiglu_clamp_tiling_data.h"
#include "../op_kernel/swiglu_clamp_tiling_key.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr int64_t UB_ALIGN_BYTE = 32;            // intra-core UB align
constexpr int64_t UB_RESERVED_BYTES = 16 * 1024; // InitBuffer metadata/alignment reserve
// B per out-element = inQueueX(2 buf * 2N/N * dtypeSize) + outQueueY(2 buf * dtypeSize)
//                  + 5 fp32 TBufs (5*4) = 6*dtypeSize + 20  (bf16/fp16=32, fp32=44).
// Must scale with dtype, else fp32 tileM is too large and InitBuffer overflows UB.

struct SwigluClampCompileInfo {};

static int64_t GetDtypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return 4;
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_BF16:
            return 2;
        default:
            return 2;
    }
}

static ge::graphStatus TilingParseForSwigluClamp([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SwigluClampTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    SwigluClampTilingData* tiling = context->GetTilingData<SwigluClampTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SwigluClampTilingData), 0, sizeof(SwigluClampTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // ---- x: [..., 2N] row-major. M = rows (product of leading dims), N = lastDim/2 ----
    auto* xShapeDesc = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapeDesc);
    auto xShape = xShapeDesc->GetStorageShape();
    const int64_t dimNum = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(dimNum < 1, OP_LOGE(context, "x dim num must be >= 1, got %ld", dimNum), return ge::GRAPH_FAILED);
    const int64_t lastDim = xShape.GetDim(dimNum - 1); // = 2N
    const int64_t N = lastDim / 2;
    int64_t M = 1;
    for (int64_t i = 0; i < dimNum - 1; ++i) {
        M *= xShape.GetDim(i);
    }

    auto* descX = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, descX);
    const int64_t dtypeSize = GetDtypeSize(descX->GetDataType());
    const int64_t ubAlignElements = UB_ALIGN_BYTE / dtypeSize; // 16 for bf16/fp16, 8 for fp32

    // require x last dim even + N 32B-aligned (so every per-row DataCopy is aligned)
    OP_CHECK_IF(lastDim % 2 != 0 || N % ubAlignElements != 0,
                OP_LOGE(context, "swiglu_clamp: x last dim must be 2N with N %ld-aligned, got lastDim=%ld",
                        ubAlignElements, lastDim),
                return ge::GRAPH_FAILED);

    // ---- limit: Attr 0 (OPTIONAL Float, default 7.0) ----
    float limit = 7.0f;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const auto* limitAttr = attrs->GetAttrPointer<float>(0);
        if (limitAttr != nullptr) {
            limit = *limitAttr;
        }
    }
    OP_CHECK_IF(limit <= 0.0f, OP_LOGE(context, "swiglu_clamp: limit must be > 0, got %f", limit),
                return ge::GRAPH_FAILED);

    // empty input guard
    if (M == 0 || N == 0) {
        tiling->totalLength = 0;
        tiling->N = N;
        tiling->formerNum = 0;
        tiling->formerLength = 0;
        tiling->tailNum = 0;
        tiling->tailLength = 0;
        tiling->tileLength = 0;
        tiling->limit = limit;
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(0));
        return ge::GRAPH_SUCCESS;
    }

    // ---- platform: core count + real per-core UB size (queried per-SoC, not hardcoded) ----
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (totalCoreNum == 0) {
        totalCoreNum = 1;
    }
    uint64_t ubSizeTotal = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTotal);
    const int64_t UB_SIZE_LIMIT = static_cast<int64_t>(ubSizeTotal) - UB_RESERVED_BYTES;
    OP_CHECK_IF(UB_SIZE_LIMIT <= 0, OP_LOGE(context, "ub size too small: %lu", ubSizeTotal), return ge::GRAPH_FAILED);

    // ---- tileM (rows per UB tile). bufferCoefficient scales with dtype (see kernel.h): ----
    //   bf16/fp16: 32 B/out-elem, fp32: 44 B/out-elem. Single row must fit (no row-internal tiling).
    const int64_t bufferCoefficient = 6 * dtypeSize + 20;
    const int64_t maxTileElements = UB_SIZE_LIMIT / bufferCoefficient;
    OP_CHECK_IF(N > maxTileElements,
                OP_LOGE(context,
                        "swiglu_clamp: N=%ld exceeds single-row UB capacity (%ld out-elements); "
                        "row-internal tiling not implemented",
                        N, maxTileElements),
                return ge::GRAPH_FAILED);
    const int64_t tileM = maxTileElements / N; // N already 32B-aligned => any tileM stays aligned

    // ---- block level: split by ROW (whole rows per core) ----
    const int64_t rowsPerCore = (M + static_cast<int64_t>(totalCoreNum) - 1) / static_cast<int64_t>(totalCoreNum);
    int64_t usedCoreNum = (M + rowsPerCore - 1) / rowsPerCore;
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    const int64_t formerNum = usedCoreNum - 1;
    const int64_t formerLength = rowsPerCore;
    const int64_t tailNum = 1;
    const int64_t tailLength = M - formerNum * rowsPerCore;

    // ---- fill tiling struct ----
    tiling->totalLength = M;
    tiling->N = N;
    tiling->formerNum = formerNum;
    tiling->formerLength = formerLength;
    tiling->tailNum = tailNum;
    tiling->tailLength = tailLength;
    tiling->tileLength = tileM;
    tiling->limit = limit;

    // ---- workspace (elementwise: only sys workspace) ----
    size_t usrSize = 0;
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    context->SetTilingKey(GET_TPL_TILING_KEY(0));
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwigluClamp)
    .Tiling(SwigluClampTilingFunc)
    .TilingParse<SwigluClampCompileInfo>(TilingParseForSwigluClamp);

} // namespace optiling
