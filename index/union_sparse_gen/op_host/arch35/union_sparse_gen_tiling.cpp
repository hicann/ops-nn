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
#include "../../op_kernel/arch35/union_sparse_gen_a5_tiling_data.h"
#include "../../op_kernel/arch35/union_sparse_gen_a5_tiling_key.h"
#include "tiling/sort/sort_tiling_intf.h"
#include <vector>
#include "../union_sparse_gen_contract.h"
#include "../union_sparse_gen_dfx.h"
#include "../union_sparse_gen_tiling_validation.h"

namespace Contract = UnionSparseGenContract;
namespace Plan = UnionSparseGenA5Plan;

namespace optiling {
namespace {
constexpr const char* kOpName = "UnionSparseGen";

bool ProductFitsInt32(std::initializer_list<int64_t> factors)
{
    constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
    int64_t product = 1;
    for (int64_t factor : factors) {
        if (factor < 0 || (factor != 0 && product > kInt32Max / factor)) {
            return false;
        }
        product *= factor;
    }
    return product <= kInt32Max;
}

using UnionSparseGenTilingValidation::ReadTilingSpec;
using UnionSparseGenTilingValidation::TilingSpec;
} // namespace

ge::graphStatus InitializeSemantic(gert::TilingContext* context, const TilingSpec& spec, int64_t coreNum,
                                   UnionSparseGenA5TilingData*& payload)
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
    const int64_t usedCore = std::max<int64_t>(1, std::min(coreNum, totalWork));
    const bool useInt32 = ProductFitsInt32({heads, totalQ, topK}) &&
                          ProductFitsInt32({heads, spec.groups, spec.capacity, spec.blockY}) &&
                          ProductFitsInt32({spec.logicalBlocks, spec.blockY});

    payload = context->GetTilingData<UnionSparseGenA5TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, payload);
    auto* tiling = &payload->semantic;
    if (memset_s(payload, sizeof(*payload), 0, sizeof(*payload)) != EOK) {
        OP_LOGE(kOpName, "failed to clear tiling data");
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
    tiling->useInt32 = useInt32 ? 1 : 0;
    tiling->hasSeqUsedQ = context->GetOptionalInputDesc(Contract::kUsedQInput) != nullptr ? 1 : 0;
    tiling->hasSeqUsedKv = context->GetOptionalInputDesc(Contract::kUsedKvInput) != nullptr ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

bool NeedsMembership(const UnionSparseGenA5TilingData* payload, int64_t coreNum, uint64_t ubSize)
{
    const auto* tiling = &payload->semantic;
    const int64_t maxLogicalBlocks = tiling->maxLogicalBlocks;
    const int64_t capacity = tiling->unionCapacity;
    const int64_t unionSize = tiling->unionSize;
    const int64_t residual = tiling->residualBlockMode;
    const int64_t blockY = tiling->blockY;
    const int64_t totalWork = tiling->headCount * tiling->numGroups;
    const bool sorted = tiling->isInputIdxSorted != 0;
    // Underfilled launches with large mask rows cannot amortize the serial
    // subgroup merge/writer. Use the existing block-cooperative table only
    // when its complete logical domain and compaction buffers fit in UB.
    const int64_t fullTableBlocks = (maxLogicalBlocks + Plan::kTableBlockAlignment - 1) / Plan::kTableBlockAlignment *
                                    Plan::kTableBlockAlignment;
    const int64_t fullCompact = (std::min<int64_t>(capacity, Plan::kThreadNum * Plan::kBitsPerWord) + Plan::kDmaWords -
                                 1) /
                                Plan::kDmaWords * Plan::kDmaWords;
    const int64_t cooperativeBytes = fullTableBlocks * (unionSize + Plan::kPresenceBitsPerBlock) / Plan::kBitsPerByte +
                                     fullCompact * Plan::kCompactRecordBytes + kUnionSparseGenA5StateBytes +
                                     int64_t(Plan::kCompilerReserveBytes);
    const bool cooperativeOrdered = sorted && residual == 0 && totalWork > 0 && totalWork < coreNum &&
                                    capacity * blockY >= Plan::kThreadNum * Plan::kThreadNum &&
                                    uint64_t(cooperativeBytes) <= ubSize;
    const bool useMembership = !sorted || residual != 0 || cooperativeOrdered;

    return useMembership;
}

ge::graphStatus PlanPackedMemory(UnionSparseGenA5TilingData* payload, int64_t available)
{
    const auto* tiling = &payload->semantic;
    const int64_t maxLogicalBlocks = tiling->maxLogicalBlocks;
    const int64_t capacity = tiling->unionCapacity;
    const int64_t unionSize = tiling->unionSize;
    // One presence word per thread can emit at most32 selected blocks.
    int64_t compact = std::min<int64_t>(capacity, Plan::kThreadNum * Plan::kBitsPerWord);
    compact = (compact + Plan::kDmaWords - 1) / Plan::kDmaWords * Plan::kDmaWords;
    while (compact * Plan::kCompactRecordBytes + Plan::kMinTableBytes > available && compact > Plan::kBitsPerWord) {
        compact = std::max<int64_t>(Plan::kBitsPerWord, (compact / Plan::kBudgetShrinkFactor + Plan::kDmaWords - 1) /
                                                            Plan::kDmaWords * Plan::kDmaWords);
    }
    const int64_t tableBytes = available - compact * Plan::kCompactRecordBytes;
    const int64_t domainLimit = (tableBytes * Plan::kBitsPerByte / (unionSize + Plan::kPresenceBitsPerBlock) /
                                 Plan::kTableBlockAlignment) *
                                Plan::kTableBlockAlignment;
    if (domainLimit < Plan::kTableBlockAlignment) {
        OP_LOGE(kOpName, "A5 membership domain limit=%ld must be >=256; available UB=%ld compact=%ld", domainLimit,
                available, compact);
        return ge::GRAPH_FAILED;
    }
    payload->compactCapacity = compact;
    payload->tableBlockCapacity = std::min<int64_t>(
        (maxLogicalBlocks + Plan::kTableBlockAlignment - 1) / Plan::kTableBlockAlignment * Plan::kTableBlockAlignment,
        domainLimit);
    payload->tableScratchBytes = payload->tableBlockCapacity * (unionSize + Plan::kPresenceBitsPerBlock) /
                                     Plan::kBitsPerByte +
                                 compact * Plan::kCompactRecordBytes;
    return ge::GRAPH_SUCCESS;
}

void TryRadixMemory(UnionSparseGenA5TilingData* payload, int64_t available, UnionSparseGenA5Algorithm& algorithm)
{
    const auto* tiling = &payload->semantic;
    const int64_t topK = tiling->topK;
    const int64_t maxLogicalBlocks = tiling->maxLogicalBlocks;
    const int64_t unionSize = tiling->unionSize;
    const int64_t residual = tiling->residualBlockMode;
    const int64_t records = unionSize * (topK + residual);
    // This lower bound avoids querying Sort for capacities that cannot fit.
    if (records <= available / Plan::kRadixRecordBytes) {
        const int64_t allocated = (records + Plan::kDmaWords - 1) / Plan::kDmaWords * Plan::kDmaWords;
        AscendC::SortConfig config{AscendC::SortType::RADIX_SORT, false, true, true};
        uint32_t maximum = 0, minimum = 0;
        AscendC::GetSortMaxMinTmpSize(ge::Shape(std::vector<int64_t>{records}), ge::DT_UINT32, ge::DT_UINT32, true,
                                      config, maximum, minimum);
        const int64_t temporary = (int64_t(maximum) + Plan::kDmaBlockBytes - 1) / Plan::kDmaBlockBytes *
                                  Plan::kDmaBlockBytes;
        if (allocated * Plan::kRadixRecordBytes + temporary <= available) {
            payload->radixCapacity = allocated;
            payload->radixTmpBytes = temporary;
            payload->radixScratchBytes = allocated * Plan::kRadixRecordBytes + temporary;
            // Prefer the measured packed backend when the whole domain is resident.
            // For fragmented domains, bounded RadixSort avoids repeated input scans.
            algorithm = maxLogicalBlocks <= payload->tableBlockCapacity ? UnionSparseGenA5Algorithm::PackedMembership :
                                                                          UnionSparseGenA5Algorithm::RadixMembership;
        }
    }
}

ge::graphStatus ChooseAlgorithm(UnionSparseGenA5TilingData* payload, int64_t coreNum, uint64_t ubSize,
                                UnionSparseGenA5Algorithm& algorithm)
{
    if (!NeedsMembership(payload, coreNum, ubSize)) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t stateBytes = kUnionSparseGenA5StateBytes;
    if (ubSize <= Plan::kCompilerReserveBytes + stateBytes + uint64_t(Plan::kMinTableBytes)) {
        OP_LOGE(kOpName, "A5 membership requires UB > %lu bytes, got %lu",
                Plan::kCompilerReserveBytes + stateBytes + uint64_t(Plan::kMinTableBytes), ubSize);
        return ge::GRAPH_FAILED;
    }
    const int64_t available = int64_t(ubSize - Plan::kCompilerReserveBytes) - stateBytes;
    if (PlanPackedMemory(payload, available) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    algorithm = UnionSparseGenA5Algorithm::PackedMembership;
    TryRadixMemory(payload, available, algorithm);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReserveDynamicBuffers(gert::TilingContext* context, const UnionSparseGenA5TilingData* payload,
                                      UnionSparseGenA5Algorithm algorithm)
{
    if (algorithm != UnionSparseGenA5Algorithm::WarpMerge) {
        // TPipe offsets alone do not reserve SIMT shared memory at launch.
        // Reserve exactly the selected bounded buffers and retain the cache budget.
        const int64_t scratch = algorithm == UnionSparseGenA5Algorithm::RadixMembership ? payload->radixScratchBytes :
                                                                                          payload->tableScratchBytes;
        if (context->SetDynUBufSize(uint32_t(scratch + kUnionSparseGenA5StateBytes)) != ge::GRAPH_SUCCESS) {
            OP_LOGE(kOpName, "A5 dynamic UB reservation failed for %ld bytes", scratch + kUnionSparseGenA5StateBytes);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void SelectPlan(gert::TilingContext* context, const UnionSparseGenA5TilingData* payload, uint64_t ubSize,
                UnionSparseGenA5Algorithm algorithm)
{
    const int64_t unionSize = payload->semantic.unionSize;
    const char* planName = "ORDERED_SUBGROUP_8";
    uint64_t tilingKey = UNION_SPARSE_GEN_A5_KEY_ORDERED_SUBGROUP_8;
    if (algorithm == UnionSparseGenA5Algorithm::WarpMerge) {
        // Group width changes subgroup scheduling and the compiled SIMT state.
        if (unionSize == Plan::kHalfWarpLanes) {
            tilingKey = UNION_SPARSE_GEN_A5_KEY_ORDERED_SUBGROUP_16;
            planName = "ORDERED_SUBGROUP_16";
        } else if (unionSize == Plan::kWarpLanes) {
            tilingKey = UNION_SPARSE_GEN_A5_KEY_ORDERED_SUBGROUP_32;
            planName = "ORDERED_SUBGROUP_32";
        }
    } else if (algorithm == UnionSparseGenA5Algorithm::PackedMembership) {
        planName = "PACKED_TABLE";
        tilingKey = UNION_SPARSE_GEN_A5_KEY_PACKED_TABLE;
    } else if (algorithm == UnionSparseGenA5Algorithm::RadixMembership) {
        planName = "RADIX_SORT";
        tilingKey = UNION_SPARSE_GEN_A5_KEY_RADIX_SORT;
    }
    context->SetTilingKey(tilingKey);
    OP_LOGI(kOpName,
            "A5 plan=%s key=%lu UB=%lu radixCapacity=%ld radixTmp=%ld radixBytes=%ld tableBlocks=%ld compact=%ld "
            "tableBytes=%ld stateBytes=%ld",
            planName, tilingKey, ubSize, payload->radixCapacity, payload->radixTmpBytes, payload->radixScratchBytes,
            payload->tableBlockCapacity, payload->compactCapacity, payload->tableScratchBytes,
            kUnionSparseGenA5StateBytes);
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
    int64_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) {
        OP_LOGE(kOpName, "AIV core count is zero");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0U) {
        OP_LOGE(kOpName, "UB size is zero");
        return ge::GRAPH_FAILED;
    }

    UnionSparseGenA5TilingData* payload = nullptr;
    if (InitializeSemantic(context, spec, coreNum, payload) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t workspaceSize = platform.GetLibApiWorkSpaceSize();
    auto algorithm = UnionSparseGenA5Algorithm::WarpMerge;
    if (ChooseAlgorithm(payload, coreNum, ubSize, algorithm) != ge::GRAPH_SUCCESS ||
        ReserveDynamicBuffers(context, payload, algorithm) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SelectPlan(context, payload, ubSize, algorithm);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    // Every supported backend uses UB plus the library system workspace.
    // No GM bitset or sorted GM intermediate is allocated.
    workspaces[0] = static_cast<size_t>(workspaceSize);
    context->SetBlockDim(static_cast<uint32_t>(payload->semantic.usedCoreNum));
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
