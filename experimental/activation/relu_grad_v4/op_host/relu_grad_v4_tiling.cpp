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
 * \file relu_grad_v4_tiling.cpp
 * \brief ReluGradV4 tiling: element-wise elementwise backprop for ReLU, multi-dtype vectorized.
 */
#include <algorithm>
#include <set>

#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/relu_grad_v4_tiling_data.h"
#include "../op_kernel/relu_grad_v4_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

// Copy-block alignment: minimum DataCopy granularity in bytes.
constexpr int64_t kCopyBlockBytes = 512;
// Alignment for Compare/Select vector operations (256 B / sizeof(float) = 64 elements).
constexpr int64_t kVectorAlignBytes = 256;
// UB buffer factor for fp32 vectorized path (queues + zero + mask with margin).
constexpr int64_t kUbCoeffFp32 = 36;
// UB buffer factor for fp16/bf16 cast-vectorized path.
constexpr int64_t kUbCoeffHalf = 36;

struct ReluGradV4CompileInfo {};

// Per-core tiling parameters passed to GetWorkspaceSize / SetTilingData.
struct CoreTilingPlan {
    int64_t headCoreCount = 0; // number of "head" cores (larger block)
    int64_t headBlockLen = 0;  // element count per head core
    int64_t tailBlockLen = 0;  // element count for the tail core
    int64_t tileLen = 1;       // tile granularity within each core
    uint32_t activeCores = 1;
};

// Scalar shape sentinel used when input shape is 0-dim (scalar tensor).
static const gert::Shape kScalarFallbackShape = {1};

static const gert::Shape& NormalizeShape(const gert::Shape& shape)
{
    return (shape.GetDimNum() == 0) ? kScalarFallbackShape : shape;
}

static ge::graphStatus QueryPlatformInfo(gert::TilingContext* ctx, uint64_t& ubBytes, int64_t& coreCount)
{
    auto* info = ctx->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, info);
    auto plat = platform_ascendc::PlatformAscendC(info);
    coreCount = static_cast<int64_t>(plat.GetCoreNumAiv());
    OP_CHECK_IF(coreCount <= 0, OP_LOGE(ctx, "coreCount is invalid"), return ge::GRAPH_FAILED);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    OP_CHECK_IF(ubBytes == 0, OP_LOGE(ctx, "ubBytes is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ExtractShapeAndDtype(gert::TilingContext* ctx, int64_t& elemCount, ge::DataType& dtype)
{
    auto* gShape = ctx->GetInputShape(0);
    auto* fShape = ctx->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, gShape);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, fShape);

    const gert::Shape& gStore = NormalizeShape(gShape->GetStorageShape());
    const gert::Shape& fStore = NormalizeShape(fShape->GetStorageShape());
    OP_CHECK_IF(gStore != fStore, OP_LOGE(ctx, "shape mismatch between gradients and features"),
                return ge::GRAPH_FAILED);
    elemCount = gStore.GetShapeSize();

    auto* gDesc = ctx->GetInputDesc(0);
    auto* fDesc = ctx->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, gDesc);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, fDesc);
    dtype = gDesc->GetDataType();

    const std::set<ge::DataType> kSupportedTypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                                                    ge::DT_INT8,  ge::DT_UINT8,   ge::DT_INT32};
    OP_CHECK_IF(kSupportedTypes.count(dtype) == 0, OP_LOGE(ctx, "unsupported gradients dtype"),
                return ge::GRAPH_FAILED);
    // mask is always uint8 regardless of gradients dtype (ReluGradV4: backprops = mask ? gradients : 0).
    OP_CHECK_IF(fDesc->GetDataType() != ge::DT_UINT8, OP_LOGE(ctx, "mask input must be uint8"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AllocSysWorkspace(gert::TilingContext* ctx)
{
    auto plat = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint32_t sysWs = plat.GetLibApiWorkSpaceSize();
    size_t* ws = ctx->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, ws);
    ws[0] = sysWs;
    return ge::GRAPH_SUCCESS;
}

static void ZeroOutTilingData(ReluGradV4TilingData* td)
{
    td->formerNum = 0;
    td->formerLength = 0;
    td->tailLength = 0;
    td->tileLength = 1;
}

static bool IsFloatingType(ge::DataType dt) { return dt == ge::DT_FLOAT || dt == ge::DT_FLOAT16 || dt == ge::DT_BF16; }

// Returns true for int types that can take the vectorized Compare+Select path.
static bool IsVecSupportedInt(ge::DataType dt) { return dt == ge::DT_INT8 || dt == ge::DT_UINT8 || dt == ge::DT_INT32; }

// Per-element UB byte cost for each dtype path.
static int64_t CalcUbCoefficient(ge::DataType dt, int64_t dtBytes)
{
    if (dt == ge::DT_FLOAT) {
        return kUbCoeffFp32;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return kUbCoeffHalf;
    }
    if (dt == ge::DT_INT32) {
        // mask-multiply path: queues + featFp32 + multFp32 + one + zero + multInt + mask
        return 56;
    }
    if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        // two-step cast vec path: int8 queues (3x2) + half {feat,grad} (2x2) + fp32 {feat,grad,out,zero} (4x4)
        // + mask (1) = ~27 bytes/elem, with margin
        return 26;
    }
    return dtBytes * 6; // int64 / double scalar path
}

// Minimum element count to satisfy DataCopy and vector-op alignment.
static int64_t CalcAlignCount(ge::DataType dt, int64_t dtBytes)
{
    if (dtBytes == 0) {
        return 1;
    }
    int64_t baseAlign = 32 / dtBytes;
    if (baseAlign <= 0) {
        baseAlign = 1;
    }
    if (IsFloatingType(dt) || IsVecSupportedInt(dt)) {
        int64_t vecAlign = kVectorAlignBytes / static_cast<int64_t>(sizeof(float));
        baseAlign = std::max<int64_t>(baseAlign, vecAlign);
    }
    return baseAlign;
}

static ge::graphStatus ComputeCorePlan(gert::TilingContext* ctx, int64_t totalElems, ge::DataType dt, uint64_t ubBytes,
                                       int64_t cores, CoreTilingPlan& plan)
{
    uint32_t rawDtBytes = 0;
    ge::TypeUtils::GetDataTypeLength(dt, rawDtBytes);
    if (rawDtBytes == 0) {
        OP_LOGE(ctx, "GetDataTypeLength returned 0 for gradients dtype");
        return ge::GRAPH_FAILED;
    }
    int64_t dtBytes = static_cast<int64_t>(rawDtBytes);
    if (dtBytes == 0 || cores == 0) {
        OP_LOGE(ctx, "invalid dtBytes(%ld) or cores(%ld)", dtBytes, cores);
        return ge::GRAPH_FAILED;
    }

    int64_t alignUnit = std::max<int64_t>(1, kCopyBlockBytes / dtBytes);
    int64_t elemsPerCore = (totalElems + cores - 1) / cores;
    int64_t alignedPerCore = ((elemsPerCore + alignUnit - 1) / alignUnit) * alignUnit;
    int64_t usedCores = std::max<int64_t>(1, (totalElems + alignedPerCore - 1) / alignedPerCore);

    plan.headCoreCount = usedCores - 1;
    plan.headBlockLen = alignedPerCore;
    plan.tailBlockLen = totalElems - plan.headCoreCount * plan.headBlockLen;
    plan.activeCores = static_cast<uint32_t>(usedCores);

    int64_t ubCoeff = CalcUbCoefficient(dt, dtBytes);
    if (ubCoeff == 0) {
        OP_LOGE(ctx, "CalcUbCoefficient returned 0");
        return ge::GRAPH_FAILED;
    }
    int64_t alignCount = CalcAlignCount(dt, dtBytes);
    if (alignCount == 0) {
        OP_LOGE(ctx, "CalcAlignCount returned 0");
        return ge::GRAPH_FAILED;
    }
    int64_t maxTile = static_cast<int64_t>(ubBytes) / ubCoeff;
    plan.tileLen = (maxTile / alignCount) * alignCount;
    if (plan.tileLen <= 0) {
        plan.tileLen = alignCount;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FlushTilingData(gert::TilingContext* ctx, const CoreTilingPlan& plan, ReluGradV4TilingData* td)
{
    td->formerNum = plan.headCoreCount;
    td->formerLength = plan.headBlockLen;
    td->tailLength = plan.tailBlockLen;
    td->tileLength = plan.tileLen;
    ctx->SetBlockDim(plan.activeCores);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluGradV4TilingFunc(gert::TilingContext* ctx)
{
    uint64_t ubBytes = 0;
    int64_t coreCount = 0;
    OP_CHECK_IF(QueryPlatformInfo(ctx, ubBytes, coreCount) != ge::GRAPH_SUCCESS,
                OP_LOGE(ctx, "QueryPlatformInfo failed"), return ge::GRAPH_FAILED);

    int64_t elemCount = 0;
    ge::DataType dtype;
    OP_CHECK_IF(ExtractShapeAndDtype(ctx, elemCount, dtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(ctx, "ExtractShapeAndDtype failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(AllocSysWorkspace(ctx) != ge::GRAPH_SUCCESS, OP_LOGE(ctx, "AllocSysWorkspace failed"),
                return ge::GRAPH_FAILED);

    auto* td = ctx->GetTilingData<ReluGradV4TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, td);
    OP_CHECK_IF(memset_s(td, sizeof(ReluGradV4TilingData), 0, sizeof(ReluGradV4TilingData)) != EOK,
                OP_LOGE(ctx, "memset_s tiling data failed"), return ge::GRAPH_FAILED);

    uint32_t dtKey = static_cast<uint32_t>(dtype);
    if (elemCount <= 0) {
        ZeroOutTilingData(td);
        ctx->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(ctx, dtKey);
        return ge::GRAPH_SUCCESS;
    }

    CoreTilingPlan plan;
    OP_CHECK_IF(ComputeCorePlan(ctx, elemCount, dtype, ubBytes, coreCount, plan) != ge::GRAPH_SUCCESS,
                OP_LOGE(ctx, "ComputeCorePlan failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(FlushTilingData(ctx, plan, td) != ge::GRAPH_SUCCESS, OP_LOGE(ctx, "FlushTilingData failed"),
                return ge::GRAPH_FAILED);
    ASCENDC_TPL_SEL_PARAM(ctx, dtKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReluGradV4([[maybe_unused]] gert::TilingParseContext* ctx)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReluGradV4).Tiling(ReluGradV4TilingFunc).TilingParse<ReluGradV4CompileInfo>(TilingParseForReluGradV4);
} // namespace optiling
