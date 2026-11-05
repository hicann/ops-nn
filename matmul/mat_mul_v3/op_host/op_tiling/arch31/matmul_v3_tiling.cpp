/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_v3_tiling.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "../../../op_kernel/mat_mul_v3_tiling_data.h"
#include "../../../op_kernel/arch31/mat_mul_v3_l1_policy.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t kBlockDim = 1U;
constexpr size_t kDefaultWorkspaceBytes = 32U;
constexpr int64_t kDefaultOpImplMode = 1;
constexpr int64_t kBiasTileColumns = 16;
// The Kirin9030 Matmul API path loses correctness when a single invocation spans
// N >= 65536. Keep each invocation inside the largest device-verified range;
// MatMulV3's outer N loop writes the tiles directly into the final output.
constexpr int64_t kMaxSingleCoreN = 32768;

int32_t BaseMNSize(int64_t extent, int64_t otherExtent)
{
    if (extent >= 128 && otherExtent >= 128) {
        return 128;
    }
    return extent >= 64 ? 64 : 32;
}

int32_t BaseKSize(int64_t extent) { return extent >= 64 ? 64 : 32; }

int32_t Al1StripRows(int64_t m, int64_t n, int64_t k, bool transB, bool hasBias)
{
    using namespace matmul_v3_arch31;
    if (hasBias || !transB || m % kCubeAlign != 0 || (n + kAl1BaseN - 1) / kAl1BaseN < kMinReuseNTiles) {
        return 0;
    }
    const int64_t kAligned = (k + kCubeAlign - 1) / kCubeAlign * kCubeAlign;
    int64_t rows = kAl1StripBudgetBytes / (kAligned * kFp16Bytes);
    rows = rows / kCubeAlign * kCubeAlign;
    if (rows < kCubeAlign || rows >= kMaxStripRows) {
        return 0;
    }
    const int64_t mAligned = (m + kCubeAlign - 1) / kCubeAlign * kCubeAlign;
    return mAligned <= rows ? static_cast<int32_t>(mAligned) : 0;
}

bool UseSplitKAl1(int64_t m, int64_t n, int64_t k, bool transB, bool hasBias, int32_t baseM, int32_t baseN,
                  int32_t baseK)
{
    using namespace matmul_v3_arch31;
    if (hasBias || !transB || m != baseM || baseM != kMaxStripRows || baseN != kAl1BaseN || baseK != kAl1BaseK ||
        m % kCubeAlign != 0 || n % kCubeAlign != 0 || k % baseK != 0 ||
        (n + kAl1BaseN - 1) / kAl1BaseN < kMinReuseNTiles) {
        return false;
    }
    const int64_t fullABytes = m * k * kFp16Bytes;
    const int64_t stripK = kAl1StripBudgetBytes / (m * kFp16Bytes) / baseK * baseK;
    return fullABytes > kAl1StripBudgetBytes && stripK >= baseK && stripK < k;
}

template <typename T>
bool IsNd(const T* desc)
{
    return desc != nullptr && ge::GetPrimaryFormat(desc->GetStorageFormat()) == ge::FORMAT_ND;
}
} // namespace

template <typename Context>
bool IsMatmulV3Arch31Context(Context* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return false;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto socVersion = platform.GetSocVersion();
    // Kirin9030/KirinX90 共用 arch31 tiling 与内核
    return socVersion == platform_ascendc::SocVersion::KIRIN9030 ||
           socVersion == platform_ascendc::SocVersion::KIRINX90;
}

bool IsMatmulV3Arch31(gert::TilingContext* context) { return IsMatmulV3Arch31Context(context); }

bool IsMatmulV3Arch31(gert::TilingParseContext* context) { return IsMatmulV3Arch31Context(context); }

ge::graphStatus MatmulV3Arch31Tiling(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* x1Shape = context->GetInputShape(0);
    const auto* x2Shape = context->GetInputShape(1);
    const auto* x1Desc = context->GetInputDesc(0);
    const auto* x2Desc = context->GetInputDesc(1);
    const auto* yDesc = context->GetOutputDesc(0);
    const auto* attrs = context->GetAttrs();
    if (x1Shape == nullptr || x2Shape == nullptr || x1Desc == nullptr || x2Desc == nullptr || yDesc == nullptr ||
        attrs == nullptr || x1Shape->GetStorageShape().GetDimNum() != 2U ||
        x2Shape->GetStorageShape().GetDimNum() != 2U || x1Desc->GetDataType() != ge::DT_FLOAT16 ||
        x2Desc->GetDataType() != ge::DT_FLOAT16 || yDesc->GetDataType() != ge::DT_FLOAT16 || !IsNd(x1Desc) ||
        !IsNd(x2Desc) || !IsNd(yDesc)) {
        return ge::GRAPH_FAILED;
    }

    const bool* transAAttr = attrs->GetAttrPointer<bool>(0);
    const bool* transBAttr = attrs->GetAttrPointer<bool>(1);
    const int64_t* offsetXAttr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* opImplModeAttr = attrs->GetAttrPointer<int64_t>(3);
    const bool transA = transAAttr != nullptr && *transAAttr;
    const bool transB = transBAttr != nullptr && *transBAttr;
    const int64_t offsetX = offsetXAttr == nullptr ? 0 : *offsetXAttr;
    const int64_t opImplMode = opImplModeAttr == nullptr ? kDefaultOpImplMode : *opImplModeAttr;
    if (transA || offsetX != 0 || opImplMode != kDefaultOpImplMode) {
        return ge::GRAPH_FAILED;
    }

    const auto& x1 = x1Shape->GetStorageShape();
    const auto& x2 = x2Shape->GetStorageShape();
    const int64_t m = x1.GetDim(0);
    const int64_t k = x1.GetDim(1);
    const int64_t kb = x2.GetDim(transB ? 1U : 0U);
    const int64_t n = x2.GetDim(transB ? 0U : 1U);
    if (m <= 0 || n <= 0 || k <= 0 || k != kb || m > std::numeric_limits<int32_t>::max() ||
        n > std::numeric_limits<int32_t>::max() || k > std::numeric_limits<int32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    const auto* biasShape = context->GetOptionalInputShape(2);
    const auto* biasDesc = context->GetOptionalInputDesc(2);
    const bool hasBias = biasShape != nullptr && biasDesc != nullptr;
    if (hasBias && (biasDesc->GetDataType() != ge::DT_FLOAT16 || !IsNd(biasDesc) ||
                    biasShape->GetStorageShape().GetDimNum() != 1U || biasShape->GetStorageShape().GetDim(0) != n)) {
        return ge::GRAPH_FAILED;
    }

    MatmulTilingData plan{};
    plan.matmulTiling.usedCoreNum = 1;
    plan.matmulTiling.M = static_cast<int32_t>(m);
    plan.matmulTiling.N = static_cast<int32_t>(n);
    plan.matmulTiling.Ka = static_cast<int32_t>(k);
    plan.matmulTiling.Kb = static_cast<int32_t>(k);
    plan.matmulTiling.singleCoreM = static_cast<int32_t>(m);
    const int64_t singleCoreN = hasBias ? std::min(n, kBiasTileColumns) : std::min(n, kMaxSingleCoreN);
    plan.matmulTiling.singleCoreN = static_cast<int32_t>(singleCoreN);
    plan.matmulTiling.singleCoreK = static_cast<int32_t>(k);
    plan.matmulTiling.baseM = BaseMNSize(m, n);
    plan.matmulTiling.baseN = BaseMNSize(n, m);
    if (!hasBias && !transB && n >= 65535) {
        // The wide-row ND2NZ fallback copies each K row separately. Use the
        // largest B tile that still fits double-buffered 32 KiB L0B, halving
        // those row-copy calls for narrow-M large-N GEMV shapes.
        plan.matmulTiling.baseN = 128;
    }
    plan.matmulTiling.baseK = BaseKSize(k);
    plan.matmulTiling.depthA1 = 2;
    plan.matmulTiling.depthB1 = 2;
    plan.matmulTiling.stepM = 1;
    plan.matmulTiling.stepN = 1;
    plan.matmulTiling.isBias = hasBias ? 1 : 0;
    plan.matmulTiling.stepKa = 1;
    plan.matmulTiling.stepKb = 1;
    if (!hasBias && plan.matmulTiling.baseM == 128 && plan.matmulTiling.baseN == 128 && (k == 256 || k == 512)) {
        const int32_t fullKSteps = static_cast<int32_t>(k / plan.matmulTiling.baseK);
        plan.matmulTiling.stepKa = fullKSteps;
        plan.matmulTiling.stepKb = fullKSteps;
        plan.matmulTiling.depthA1 = 2 * fullKSteps;
        plan.matmulTiling.depthB1 = 2 * fullKSteps;
    }
    plan.matmulTiling.dbL0A = 2;
    plan.matmulTiling.dbL0B = 2;
    plan.matmulTiling.dbL0C = 1;
    plan.matmulTiling.BatchNum = 1;

    const int32_t al1StripRows = Al1StripRows(m, n, k, transB, hasBias);
    if (al1StripRows > 0) {
        plan.matmulTiling.baseM = al1StripRows;
        plan.matmulTiling.baseN = static_cast<int32_t>(matmul_v3_arch31::kAl1BaseN);
        plan.matmulTiling.baseK = static_cast<int32_t>(matmul_v3_arch31::kAl1BaseK);
        plan.matmulTiling.mxTypePara = matmul_v3_arch31::kAl1StripMarker;
    }
    const bool useSplitKAl1 = al1StripRows == 0 && UseSplitKAl1(m, n, k, transB, hasBias, plan.matmulTiling.baseM,
                                                                plan.matmulTiling.baseN, plan.matmulTiling.baseK);
    if (useSplitKAl1) {
        const int64_t stripK = matmul_v3_arch31::kAl1StripBudgetBytes / (m * matmul_v3_arch31::kFp16Bytes) /
                               plan.matmulTiling.baseK * plan.matmulTiling.baseK;
        plan.matmulTiling.stepKa = static_cast<int32_t>(stripK / plan.matmulTiling.baseK);
        plan.matmulTiling.mxTypePara = matmul_v3_arch31::kSplitKAl1Marker;
    }
    plan.tileL2cacheTiling.mTileCntL2 = 1U;
    plan.tileL2cacheTiling.nTileCntL2 = 1U;
    plan.tileL2cacheTiling.calOrder = 1U;
    plan.matmulRunInfo.transA = 0U;
    plan.matmulRunInfo.transB = transB ? 1U : 0U;
    plan.l2cacheUseInfo.l2CacheFlag = 1U;

    static_assert(sizeof(plan) == 280U, "MatMulV3 Arch31 tiling ABI drift");
    auto* raw = context->GetRawTilingData();
    if (raw == nullptr || raw->GetData() == nullptr || raw->GetCapacity() < sizeof(plan)) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw->GetData(), &plan, sizeof(plan));
    raw->SetDataSize(sizeof(plan));
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t outputElements = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
    if (useSplitKAl1 && outputElements > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = useSplitKAl1 ? static_cast<size_t>(outputElements * sizeof(uint16_t)) : kDefaultWorkspaceBytes;
    context->SetBlockDim(kBlockDim);
    context->SetTilingKey(0U);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
