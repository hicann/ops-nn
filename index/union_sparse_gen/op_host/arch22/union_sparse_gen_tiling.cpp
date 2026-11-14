/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "op_host/tiling_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/union_sparse_gen_tiling_data.h"
#include "../union_sparse_gen_contract.h"
#include "../union_sparse_gen_dfx.h"
#include "../union_sparse_gen_tiling_validation.h"
#include "../../op_kernel/arch22/union_sparse_gen_tiling_key.h"
#include "../../op_kernel/arch22/union_sparse_gen_membership_plan.h"

namespace Contract = UnionSparseGenContract;
namespace Plan = UnionSparseGenA2Plan;

namespace optiling {
namespace {
constexpr const char* kOpName = "UnionSparseGen";

using UnionSparseGenTilingValidation::ReadTilingSpec;
using UnionSparseGenTilingValidation::TilingSpec;
} // namespace

ge::graphStatus InitializeSemantic(gert::TilingContext* context, const TilingSpec& spec, int64_t coreNum,
                                   UnionSparseGenTilingData*& tiling)
{
    const auto* attrs = context->GetAttrs();
    const auto* sparseShape = context->GetInputShape(Contract::kSparseIdxInput);
    const auto* qShape = context->GetOptionalInputShape(Contract::kCuQInput);
    const int64_t heads = sparseShape->GetStorageShape().GetDim(Contract::kHeadAxis);
    const int64_t totalQ = sparseShape->GetStorageShape().GetDim(Contract::kQueryAxis);
    const int64_t topK = sparseShape->GetStorageShape().GetDim(Contract::kSparseEntryAxis);
    const int64_t batch = qShape ? qShape->GetStorageShape().GetDim(Contract::kLengthAxis) - 1 :
                                   context->GetOptionalInputShape(Contract::kUsedQInput)
                                       ->GetStorageShape()
                                       .GetDim(Contract::kLengthAxis);
    const int64_t totalWork = heads * spec.groups;
    const int64_t usedCore = std::max<int64_t>(1, std::min(coreNum, std::max<int64_t>(1, totalWork)));
    tiling = context->GetTilingData<UnionSparseGenTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    if (memset_s(tiling, sizeof(*tiling), 0, sizeof(*tiling)) != EOK) {
        return ge::GRAPH_FAILED;
    }
    const auto* sortedAttr = attrs->GetAttrPointer<bool>(Contract::kSorted);
    const auto* residualAttr = attrs->GetAttrPointer<int64_t>(Contract::kResidual);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortedAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, residualAttr);
    tiling->isInputIdxSorted = *sortedAttr ? 1 : 0;
    tiling->residualBlockMode = *residualAttr;
    tiling->hasCuSqlenKv = context->GetOptionalInputDesc(Contract::kCuKvInput) != nullptr ? 1 : 0;
    tiling->batchSize = batch;
    tiling->headCount = heads;
    tiling->totalQ = totalQ;
    tiling->topK = topK;
    tiling->numGroups = spec.groups;
    tiling->maxLogicalBlocks = spec.logicalBlocks;
    tiling->unionCapacity = spec.capacity;
    tiling->unionSize = spec.unionSize;
    tiling->maskType = spec.maskType;
    tiling->blockY = spec.blockY;
    tiling->usedCoreNum = usedCore;
    tiling->workPerCore = totalWork / usedCore;
    tiling->workTail = totalWork % usedCore;
    tiling->useInt32 = 0;
    tiling->hasSeqUsedQ = context->GetOptionalInputDesc(Contract::kUsedQInput) != nullptr ? 1 : 0;
    tiling->hasSeqUsedKv = context->GetOptionalInputDesc(Contract::kUsedKvInput) != nullptr ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

#ifndef UNION_SPARSE_GEN_SIMD_INPUT_WINDOW
#define UNION_SPARSE_GEN_SIMD_INPUT_WINDOW 512
#endif
#ifndef UNION_SPARSE_GEN_SIMD_OUTPUT_KIB
#define UNION_SPARSE_GEN_SIMD_OUTPUT_KIB 16
#endif
#ifndef UNION_SPARSE_GEN_SIMD_ADAPTIVE_OUTPUT
#define UNION_SPARSE_GEN_SIMD_ADAPTIVE_OUTPUT 1
#endif
struct SimdBuffers {
    uint64_t window, outputBytes, inputRows, metadataBytes;
    bool batchedVector, extended;
};

SimdBuffers InitialBuffers(const gert::TilingContext* context, const UnionSparseGenTilingData* tiling)
{
    const int64_t blockY = tiling->blockY, unionSize = tiling->unionSize;
    const int64_t topK = tiling->topK, maxLogicalBlocks = tiling->maxLogicalBlocks;
    const bool sorted = tiling->isInputIdxSorted != 0;
    const int64_t residual = tiling->residualBlockMode;
    const auto* qShape = context->GetOptionalInputShape(Contract::kCuQInput);
    // Account for all TBufs, compiler/private state and two output tiles. Input
    // windows and output tiles shrink before any legal public shape is rejected.
    // Select a compiled resource layout, not a shape-specific algorithm.
    // Batched vector preparation is enabled for the calibrated eight-row
    // merge topology once its input upper bound can fill a 512-entry batch.
    // Short bounded domains need no proposal scratch at that work scale.
    const int64_t inputBound = unionSize * std::min<int64_t>(topK, Plan::kCostInputCap);
    const bool boundedWork = maxLogicalBlocks <= Plan::kSmallMembershipDomainBlocks &&
                             inputBound <= Plan::kMinBatchedEntries;
    const bool extended = !sorted || residual != 0;
    const bool batchedVector = !extended && unionSize == Plan::kMinGroupWidth &&
                               inputBound >= Plan::kMinBatchedEntries && !boundedWork;
    uint64_t window = UNION_SPARSE_GEN_SIMD_INPUT_WINDOW;
    uint64_t outputBytes = UNION_SPARSE_GEN_SIMD_OUTPUT_KIB * uint64_t(Plan::kKiB);
    // A2 complete-output windows show that Y128 padding loses bandwidth with
    // 16KiB writes. Keep32KiB there; smaller-Y resident merging benefits from
    // the extra algorithm UB made available by16KiB output slots.
    if (UNION_SPARSE_GEN_SIMD_ADAPTIVE_OUTPUT) {
        // Wider groups need the saved input UB for complete resident proposal
        // buffers. This changes internal resource allocation, not legality.
        if (unionSize > Plan::kMinGroupWidth) {
            window = std::min<uint64_t>(window, uint64_t(Plan::kWideGroupInputWindow));
        } else if (blockY != Plan::kNarrowBlockY) {
            window = Plan::kVectorInputWindow;
        }
        if (unionSize > Plan::kMinGroupWidth || blockY == Contract::kSupportedBlockY || topK <= Plan::kSmallKMax) {
            outputBytes = std::max<uint64_t>(outputBytes, Plan::kLargeOutputBytes);
        }
    }
    // Direct merging reserves all supported rows and uses no batched vector
    // scratch. Keep the input allocation identical on host and device.
    if (!batchedVector) {
        window = Plan::kDirectInputWindow;
        outputBytes = Plan::kLargeOutputBytes;
    }
    // New membership paths trade output tile size for domain table capacity.
    // This resource choice leaves the calibrated sorted mode0 plan unchanged.
    if (extended) {
        outputBytes = Plan::kSmallOutputBytes;
    }
    const uint64_t inputRows = batchedVector ? static_cast<uint64_t>(unionSize) : uint64_t(Plan::kMaxGroupWidth);
    const uint64_t metadataBytes = UnionSparseGenMetadataBudgetBytes(qShape != nullptr, tiling->hasSeqUsedQ != 0,
                                                                     tiling->hasSeqUsedKv != 0) +
                                   uint64_t(Plan::kCountBytes);
    return {window, outputBytes, inputRows, metadataBytes, batchedVector, extended};
}

uint64_t FixedBufferBytes(const SimdBuffers& buffers, int64_t blockY)
{
    return Plan::kCompilerReserveBytes + buffers.metadataBytes + buffers.inputRows * buffers.window * Plan::kWordBytes +
           Plan::kOutputSlots * buffers.outputBytes +
           Plan::kOutputSlots * (buffers.outputBytes / (blockY * Plan::kWordBytes)) *
               (Plan::kPairWords * Plan::kWordBytes) +
           (blockY == Plan::kGatherBlockY ? buffers.outputBytes : 0U);
}

ge::graphStatus FitBuffers(UnionSparseGenTilingData* tiling, uint64_t ubSize, SimdBuffers& buffers)
{
    const int64_t blockY = tiling->blockY;
    const uint64_t minOutputBytes = std::max<uint64_t>(Plan::kMinOutputBytes,
                                                       blockY * Plan::kDmaWords * Plan::kWordBytes);
    while (FixedBufferBytes(buffers, blockY) > ubSize && buffers.outputBytes > minOutputBytes) {
        buffers.outputBytes /= Plan::kBudgetShrinkFactor;
    }
    while (FixedBufferBytes(buffers, blockY) > ubSize && buffers.window > uint64_t(Plan::kDmaWords)) {
        buffers.window /= Plan::kBudgetShrinkFactor;
    }
    if (FixedBufferBytes(buffers, blockY) > ubSize) {
        OP_LOGE(kOpName, "UB cannot fit the minimum SIMD dataflow");
        return ge::GRAPH_FAILED;
    }
    tiling->simdInputWindow = buffers.window;
    tiling->simdTileBlocks = buffers.outputBytes / (blockY * Plan::kWordBytes);
    tiling->simdAlgorithmBytes = ((ubSize - FixedBufferBytes(buffers, blockY)) / Plan::kDmaBlockBytes) *
                                 Plan::kDmaBlockBytes;
    if (buffers.extended && tiling->simdAlgorithmBytes < Plan::kMinMembershipBytes) {
        OP_LOGE(kOpName, "A2 membership algorithm UB=%ld bytes, minimum required=1280", tiling->simdAlgorithmBytes);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void SelectPlan(gert::TilingContext* context, const UnionSparseGenTilingData* tiling, const SimdBuffers& buffers)
{
    const int64_t blockY = tiling->blockY, unionSize = tiling->unionSize;
    const int64_t topK = tiling->topK, maxLogicalBlocks = tiling->maxLogicalBlocks;
    const int64_t residual = tiling->residualBlockMode;
    uint64_t key = !buffers.batchedVector ? UNION_SPARSE_GEN_A2_KEY_ORDERED_GENERAL :
                                            (blockY == Plan::kNarrowBlockY ? UNION_SPARSE_GEN_A2_KEY_ORDERED_BATCH_512 :
                                                                             UNION_SPARSE_GEN_A2_KEY_ORDERED_BATCH_256);
    const char* plan = !buffers.batchedVector ?
                           "ORDERED_GENERAL" :
                           (blockY == Plan::kNarrowBlockY ? "ORDERED_BATCH_512" : "ORDERED_BATCH_256");
    if (buffers.extended) {
        const bool sortFits = UnionSparseGenA2Plan::MembershipSortFits(topK, unionSize, residual,
                                                                       tiling->simdAlgorithmBytes);
        const bool tableOnly = !sortFits || maxLogicalBlocks <= UnionSparseGenA2Plan::kSmallMembershipDomainBlocks;
        key = tableOnly ? UNION_SPARSE_GEN_A2_KEY_MEMBERSHIP_TABLE : UNION_SPARSE_GEN_A2_KEY_MEMBERSHIP_ADAPTIVE;
        plan = tableOnly ? "MEMBERSHIP_TABLE" : "MEMBERSHIP_ADAPTIVE";
    }
    context->SetTilingKey(key);
    OP_LOGI(kOpName, "A2 plan=%s key=%lu inputWindow=%ld tileBlocks=%ld algorithmBytes=%ld", plan, key,
            tiling->simdInputWindow, tiling->simdTileBlocks, tiling->simdAlgorithmBytes);
}

static ge::graphStatus TilingUnionSparseGen(gert::TilingContext* context)
{
    TilingSpec spec;
    if (ReadTilingSpec(context, spec) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const int64_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum <= 0 || ubSize == 0) {
        OP_LOGE(kOpName, "target AIV/UB resource is unavailable");
        return ge::GRAPH_FAILED;
    }
    UnionSparseGenTilingData* tiling = nullptr;
    if (InitializeSemantic(context, spec, coreNum, tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto buffers = InitialBuffers(context, tiling);
    if (FitBuffers(tiling, ubSize, buffers) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    // No GM bitset or high-level API scratch is used by the SIMD backend.
    workspaces[0] = 0;
    context->SetBlockDim(static_cast<uint32_t>(tiling->usedCoreNum));
    SelectPlan(context, tiling, buffers);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareUnionSparseGen(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UnionSparseGen)
    .Tiling(TilingUnionSparseGen)
    .TilingParse<UnionSparseGenCompileInfo>(TilingPrepareUnionSparseGen);
} // namespace optiling
