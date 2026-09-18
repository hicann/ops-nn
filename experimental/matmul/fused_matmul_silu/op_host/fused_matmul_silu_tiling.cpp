/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <cstring>
#include <limits>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fused_matmul_silu_tiling_data.h"

using namespace matmul_tiling;

namespace optiling {
namespace {

constexpr uint64_t kSysWorkspaceSize = 16UL * 1024UL * 1024UL;
constexpr uint64_t kBlockK = 64UL;
constexpr uint64_t kMaxSupportedK = 4096UL;
constexpr uint64_t kVectorBytesPerElem = 18UL;
constexpr uint64_t kVectorTileAlign = 16UL;
constexpr uint32_t kBatchScheduleMode = 1U;
constexpr int32_t kTilingApiSuccess = 0;

struct CompileInfo {};

struct PlatformInfo {
    uint64_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
};

struct ShapeInfo {
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
};

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : value / alignment * alignment;
}

bool IsTilingApiSuccess(int32_t status) { return status == kTilingApiSuccess; }

ge::graphStatus CheckTilingApiStatus(gert::TilingContext* context, int32_t status, const char* apiName)
{
    if (!IsTilingApiSuccess(status)) {
        OP_LOGE(context->GetNodeName(), "matmul tiling %s failed, ret=%d", apiName, status);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReadPlatform(gert::TilingContext* context, PlatformInfo& info)
{
    auto* rawPlatform = context->GetPlatformInfo();
    if (rawPlatform == nullptr) {
        OP_LOGE(context->GetNodeName(), "platform info is null");
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(rawPlatform);
    info.aicNum = platform.GetCoreNumAic();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, info.l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, info.l0cSize);
    if (info.aicNum == 0 || info.ubSize == 0 || info.l1Size == 0 || info.l0cSize == 0) {
        OP_LOGE(context->GetNodeName(), "invalid platform resource");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReadTensorInfo(gert::TilingContext* context, const gert::StorageShape*& xShape,
                               const gert::StorageShape*& weightShape, const gert::StorageShape*& biasShape,
                               const gert::CompileTimeTensorDesc*& xDesc,
                               const gert::CompileTimeTensorDesc*& weightDesc,
                               const gert::CompileTimeTensorDesc*& biasDesc, const gert::CompileTimeTensorDesc*& yDesc)
{
    xShape = context->GetInputShape(0);
    weightShape = context->GetInputShape(1);
    biasShape = context->GetInputShape(2);
    xDesc = context->GetInputDesc(0);
    weightDesc = context->GetInputDesc(1);
    biasDesc = context->GetInputDesc(2);
    yDesc = context->GetOutputDesc(0);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr || xDesc == nullptr ||
        weightDesc == nullptr || biasDesc == nullptr || yDesc == nullptr) {
        OP_LOGE(context->GetNodeName(), "shape or descriptor is null");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTensorDesc(gert::TilingContext* context, const gert::CompileTimeTensorDesc* xDesc,
                                const gert::CompileTimeTensorDesc* weightDesc,
                                const gert::CompileTimeTensorDesc* biasDesc, const gert::CompileTimeTensorDesc* yDesc)
{
    auto xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()));
    auto wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(weightDesc->GetStorageFormat()));
    auto bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(biasDesc->GetStorageFormat()));
    auto yFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat()));
    if (xFormat != ge::FORMAT_ND || wFormat != ge::FORMAT_ND || bFormat != ge::FORMAT_ND || yFormat != ge::FORMAT_ND) {
        OP_LOGE(context->GetNodeName(), "FusedMatmulSilu only supports ND format");
        return ge::GRAPH_FAILED;
    }
    if (xDesc->GetDataType() != ge::DT_BF16 || weightDesc->GetDataType() != ge::DT_BF16 ||
        biasDesc->GetDataType() != ge::DT_BF16 || yDesc->GetDataType() != ge::DT_BF16) {
        OP_LOGE(context->GetNodeName(), "FusedMatmulSilu only supports BF16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateAndReadShape(gert::TilingContext* context, ShapeInfo& shape)
{
    const gert::StorageShape* xShape = nullptr;
    const gert::StorageShape* weightShape = nullptr;
    const gert::StorageShape* biasShape = nullptr;
    const gert::CompileTimeTensorDesc* xDesc = nullptr;
    const gert::CompileTimeTensorDesc* weightDesc = nullptr;
    const gert::CompileTimeTensorDesc* biasDesc = nullptr;
    const gert::CompileTimeTensorDesc* yDesc = nullptr;
    if (ReadTensorInfo(context, xShape, weightShape, biasShape, xDesc, weightDesc, biasDesc, yDesc) !=
            ge::GRAPH_SUCCESS ||
        CheckTensorDesc(context, xDesc, weightDesc, biasDesc, yDesc) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const auto& x = xShape->GetStorageShape();
    const auto& w = weightShape->GetStorageShape();
    const auto& b = biasShape->GetStorageShape();
    if (x.GetDimNum() != 2 || w.GetDimNum() != 2 || b.GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "expected x/weight/bias ranks 2/2/1");
        return ge::GRAPH_FAILED;
    }

    const int64_t m = x.GetDim(0);
    const int64_t k = x.GetDim(1);
    const int64_t n = w.GetDim(0);
    const int64_t weightK = w.GetDim(1);
    const int64_t biasN = b.GetDim(0);
    // Dynamic-shape execution enters tiling with concrete storage dimensions. Reject an unresolved
    // dimension before converting it to uint64_t, which would otherwise turn -1 into a large value.
    if (m <= 0 || n <= 0 || k <= 0 || weightK <= 0 || biasN <= 0 || weightK != k || biasN != n) {
        OP_LOGE(context->GetNodeName(), "invalid or mismatched shape");
        return ge::GRAPH_FAILED;
    }
    shape.m = static_cast<uint64_t>(m);
    shape.k = static_cast<uint64_t>(k);
    shape.n = static_cast<uint64_t>(n);
    if (shape.k % kBlockK != 0 || shape.k > kMaxSupportedK) {
        OP_LOGE(context->GetNodeName(), "FusedMatmulSilu requires K aligned to 64 and no greater than 4096");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateMatmulTilingInput(gert::TilingContext* context, const PlatformInfo& platform,
                                          const ShapeInfo& shape, platform_ascendc::PlatformAscendC*& ascendcPlatform)
{
    ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        OP_LOGE(context->GetNodeName(), "failed to get AscendC platform instance");
        return ge::GRAPH_FAILED;
    }
    if (platform.aicNum > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        shape.m > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        shape.n > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        shape.k > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "shape or core count exceeds MatmulTiling API range");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfigureMatmulTiling(gert::TilingContext* context, const PlatformInfo& platform,
                                      const ShapeInfo& shape, MultiCoreMatmulTiling& matmulTiling,
                                      FusedMatmulSiluTilingData& tilingData)
{
    if (CheckTilingApiStatus(context, matmulTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false),
                             "SetAType") != ge::GRAPH_SUCCESS ||
        CheckTilingApiStatus(context, matmulTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, true),
                             "SetBType") != ge::GRAPH_SUCCESS ||
        CheckTilingApiStatus(context, matmulTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16),
                             "SetCType") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Bias is added by the AIV post-process so it follows the same BF16-to-FP32 path as SiLU.
    if (CheckTilingApiStatus(context, matmulTiling.SetBias(false), "SetBias") != ge::GRAPH_SUCCESS ||
        CheckTilingApiStatus(context, matmulTiling.SetDim(static_cast<int32_t>(platform.aicNum)), "SetDim") !=
            ge::GRAPH_SUCCESS ||
        CheckTilingApiStatus(context,
                             matmulTiling.SetShape(static_cast<int32_t>(shape.m), static_cast<int32_t>(shape.n),
                                                   static_cast<int32_t>(shape.k)),
                             "SetShape") != ge::GRAPH_SUCCESS ||
        CheckTilingApiStatus(context,
                             matmulTiling.SetOrgShape(static_cast<int32_t>(shape.m), static_cast<int32_t>(shape.n),
                                                      static_cast<int32_t>(shape.k)),
                             "SetOrgShape") != ge::GRAPH_SUCCESS ||
        // MultiCoreMatmulTiling is initialized from PlatformAscendCManager. Keep that platform's
        // buffer configuration instead of overriding it with a second platform descriptor.
        CheckTilingApiStatus(context, matmulTiling.SetBufferSpace(), "SetBufferSpace") != ge::GRAPH_SUCCESS ||
        CheckTilingApiStatus(context, matmulTiling.GetTiling(tilingData.matmulTiling), "GetTiling") !=
            ge::GRAPH_SUCCESS ||
        tilingData.matmulTiling.usedCoreNum == 0) {
        OP_LOGE(context->GetNodeName(), "failed to build multi-core matmul tiling");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfigureVectorTiling(gert::TilingContext* context, const PlatformInfo& platform,
                                      const ShapeInfo& shape, FusedMatmulSiluTilingData& tilingData)
{
    // The post-process owns three BF16 and three FP32 UB buffers, or 18 bytes per element.
    // Use the available UB capacity so one MTE round handles the largest safe vector tile.
    uint64_t vectorTileElems = AlignDown(platform.ubSize / kVectorBytesPerElem, kVectorTileAlign);
    if (vectorTileElems == 0) {
        OP_LOGE(context->GetNodeName(), "UB resource is insufficient for SiLU post-process");
        return ge::GRAPH_FAILED;
    }
    tilingData.params.m = static_cast<uint32_t>(shape.m);
    tilingData.params.n = static_cast<uint32_t>(shape.n);
    tilingData.params.k = static_cast<uint32_t>(shape.k);
    tilingData.params.usedCoreNum = tilingData.matmulTiling.usedCoreNum;
    tilingData.params.vectorTileElems = static_cast<uint32_t>(vectorTileElems);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildMatmulTiling(gert::TilingContext* context, const PlatformInfo& platform, const ShapeInfo& shape,
                                  FusedMatmulSiluTilingData& tilingData)
{
    platform_ascendc::PlatformAscendC* ascendcPlatform = nullptr;
    if (ValidateMatmulTilingInput(context, platform, shape, ascendcPlatform) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    MultiCoreMatmulTiling matmulTiling(*ascendcPlatform);
    if (ConfigureMatmulTiling(context, platform, shape, matmulTiling, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ConfigureVectorTiling(context, platform, shape, tilingData);
}

ge::graphStatus SetWorkspace(gert::TilingContext* context, const FusedMatmulSiluTilingData& tilingData)
{
    auto* rawPlatform = context->GetPlatformInfo();
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (rawPlatform == nullptr || workspace == nullptr) {
        OP_LOGE(context->GetNodeName(), "platform info or workspace is null");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(rawPlatform);
    workspace[0] = kSysWorkspaceSize + platform.GetLibApiWorkSpaceSize();
    context->SetBlockDim(tilingData.params.usedCoreNum);
    context->SetScheduleMode(kBatchScheduleMode);
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus PrepareForFusedMatmulSilu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForFusedMatmulSilu(gert::TilingContext* context)
{
    if (context == nullptr || context->GetRawTilingData() == nullptr || context->GetWorkspaceSizes(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    PlatformInfo platform;
    ShapeInfo shape;
    FusedMatmulSiluTilingData tilingData{};
    if (ReadPlatform(context, platform) != ge::GRAPH_SUCCESS ||
        ValidateAndReadShape(context, shape) != ge::GRAPH_SUCCESS ||
        BuildMatmulTiling(context, platform, shape, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto* raw = context->GetRawTilingData();
    auto ret = memcpy_s(raw->GetData(), raw->GetCapacity(), &tilingData, sizeof(FusedMatmulSiluTilingData));
    if (ret != EOK) {
        OP_LOGE(context->GetNodeName(), "copy tiling data failed");
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(sizeof(FusedMatmulSiluTilingData));
    if (SetWorkspace(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedMatmulSilu).Tiling(TilingForFusedMatmulSilu).TilingParse<CompileInfo>(PrepareForFusedMatmulSilu);

} // namespace optiling
