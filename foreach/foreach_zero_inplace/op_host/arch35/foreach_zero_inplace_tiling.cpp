/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/foreach_zero_inplace_tiling_data.h"
#include "../../op_kernel/arch35/foreach_zero_inplace_tiling_key.h"

namespace optiling {
namespace {

constexpr uint32_t kDataBlockBytes = 32;
constexpr uint32_t kBufferNum = 2;
constexpr int32_t kInputIndex = 0;
constexpr size_t kMaxTensorRank = 8;

struct ForeachZeroInplaceCompileInfo {};

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "AIV core count is invalid"), return ge::GRAPH_FAILED);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= sizeof(ForeachZeroInplaceTilingData), OP_LOGE(context, "UB size is insufficient"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetTilingKey(gert::TilingContext* context, ge::DataType dtype, uint64_t& tilingKey,
                             uint32_t& elementBytes)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            tilingKey = GET_TPL_TILING_KEY(FOREACH_ZERO_INPLACE_SCH_MODE_FLOAT16);
            elementBytes = sizeof(uint16_t);
            return ge::GRAPH_SUCCESS;
        case ge::DT_FLOAT:
            tilingKey = GET_TPL_TILING_KEY(FOREACH_ZERO_INPLACE_SCH_MODE_FLOAT);
            elementBytes = sizeof(float);
            return ge::GRAPH_SUCCESS;
        case ge::DT_INT32:
            tilingKey = GET_TPL_TILING_KEY(FOREACH_ZERO_INPLACE_SCH_MODE_INT32);
            elementBytes = sizeof(int32_t);
            return ge::GRAPH_SUCCESS;
        case ge::DT_BF16:
            tilingKey = GET_TPL_TILING_KEY(FOREACH_ZERO_INPLACE_SCH_MODE_BF16);
            elementBytes = sizeof(uint16_t);
            return ge::GRAPH_SUCCESS;
        case ge::DT_INT16:
            tilingKey = GET_TPL_TILING_KEY(FOREACH_ZERO_INPLACE_SCH_MODE_INT16);
            elementBytes = sizeof(int16_t);
            return ge::GRAPH_SUCCESS;
        default:
            OP_LOGE(context, "ForeachZeroInplace does not support dtype %d", static_cast<int32_t>(dtype));
            return ge::GRAPH_FAILED;
    }
}

bool LocateFlatOffset(const ForeachZeroInplaceTilingData* tiling, int64_t flatOffset, uint16_t& tensorIndex,
                      int64_t& tensorOffset)
{
    if (flatOffset < 0) {
        return false;
    }
    int64_t consumed = 0;
    for (int32_t i = 0; i < tiling->tensorNum; ++i) {
        const int64_t elements = tiling->tensorElements[i];
        if (flatOffset >= consumed && flatOffset - consumed < elements) {
            tensorIndex = static_cast<uint16_t>(i);
            tensorOffset = flatOffset - consumed;
            return true;
        }
        if (elements < 0 || consumed > std::numeric_limits<int64_t>::max() - elements) {
            return false;
        }
        consumed += elements;
    }
    return false;
}

ge::graphStatus AssignDataToCores(gert::TilingContext* context, ForeachZeroInplaceTilingData* tiling,
                                  int64_t totalElements, int64_t availableCores, uint32_t elementBytes)
{
    if (totalElements == 0) {
        tiling->needCoreNum = 1;
        tiling->tensorStart[0] = 1;
        tiling->tensorEnd[0] = 0;
        tiling->tensorEndOffset[0] = -1;
        return ge::GRAPH_SUCCESS;
    }

    const int64_t elementsPerBlock = kDataBlockBytes / elementBytes;
    const int64_t blockCount = totalElements / elementsPerBlock + (totalElements % elementsPerBlock != 0 ? 1 : 0);
    int64_t coreCount = std::min<int64_t>(availableCores, blockCount);
    coreCount = std::min<int64_t>(coreCount, FOREACH_ZERO_INPLACE_MAX_CORE_NUM);
    tiling->needCoreNum = static_cast<int32_t>(coreCount);

    const int64_t blocksPerCore = blockCount / coreCount;
    const int64_t extraBlocks = blockCount % coreCount;
    int64_t flatStart = 0;
    for (int32_t core = 0; core < tiling->needCoreNum; ++core) {
        const int64_t coreBlocks = blocksPerCore + (core < extraBlocks ? 1 : 0);
        const int64_t remainingElements = totalElements - flatStart;
        const int64_t wholeRemainingBlocks = remainingElements / elementsPerBlock;
        const int64_t flatEnd = coreBlocks > wholeRemainingBlocks ? totalElements :
                                                                    flatStart + coreBlocks * elementsPerBlock;
        OP_CHECK_IF(!LocateFlatOffset(tiling, flatStart, tiling->tensorStart[core], tiling->tensorStartOffset[core]),
                    OP_LOGE(context, "failed to locate core %d start offset", core), return ge::GRAPH_FAILED);
        OP_CHECK_IF(!LocateFlatOffset(tiling, flatEnd - 1, tiling->tensorEnd[core], tiling->tensorEndOffset[core]),
                    OP_LOGE(context, "failed to locate core %d end offset", core), return ge::GRAPH_FAILED);
        flatStart = flatEnd;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

static ge::graphStatus Tiling4ForeachZeroInplace(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t availableCores = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, availableCores) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);
    auto instances = computeNodeInfo->GetInputInstanceInfo(kInputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, instances);
    const uint64_t tensorNum = static_cast<uint64_t>(instances->GetInstanceNum());
    OP_CHECK_IF(
        tensorNum == 0 || tensorNum > FOREACH_ZERO_INPLACE_MAX_TENSOR_NUM,
        OP_LOGE(context, "x tensor count %lu must be in [1, %d]", tensorNum, FOREACH_ZERO_INPLACE_MAX_TENSOR_NUM),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetComputeNodeOutputNum() != tensorNum,
                OP_LOGE(context, "x input and ref output counts must match"), return ge::GRAPH_FAILED);

    auto tiling = context->GetTilingData<ForeachZeroInplaceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ForeachZeroInplaceTilingData), 0, sizeof(ForeachZeroInplaceTilingData)) != EOK,
                OP_LOGE(context, "set tiling data failed"), return ge::GRAPH_FAILED);

    ge::DataType dtype = ge::DT_UNDEFINED;
    int64_t totalTensorElements = 0;
    tiling->tensorNum = static_cast<int32_t>(tensorNum);
    for (int32_t i = 0; i < tiling->tensorNum; ++i) {
        auto desc = context->GetDynamicInputDesc(kInputIndex, i);
        auto shape = context->GetDynamicInputShape(kInputIndex, i);
        auto outputDesc = context->GetOutputDesc(i);
        auto outputShape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
        const ge::DataType currentDtype = desc->GetDataType();
        OP_CHECK_IF(outputDesc->GetDataType() != currentDtype,
                    OP_LOGE(context, "x[%d] input and ref output dtypes must match", i), return ge::GRAPH_FAILED);
        if (i == 0) {
            dtype = currentDtype;
        } else {
            OP_CHECK_IF(currentDtype != dtype, OP_LOGE(context, "x tensor dtypes must match"), return ge::GRAPH_FAILED);
        }
        const auto& storageShape = shape->GetStorageShape();
        const auto& originShape = shape->GetOriginShape();
        const bool isScalarStorage = originShape.GetDimNum() == 0 && storageShape.GetDimNum() == 1 &&
                                     storageShape.GetDim(0) == 1;
        OP_CHECK_IF(originShape != storageShape && !isScalarStorage,
                    OP_LOGE(context, "x[%d] origin and storage shapes must match except for scalar storage [1]", i),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(outputShape->GetStorageShape() != storageShape || outputShape->GetOriginShape() != originShape,
                    OP_LOGE(context, "x[%d] input and ref output shapes must match", i), return ge::GRAPH_FAILED);
        OP_CHECK_IF(originShape.GetDimNum() > kMaxTensorRank,
                    OP_LOGE(context, "x[%d] rank must not exceed %zu", i, kMaxTensorRank), return ge::GRAPH_FAILED);
        bool hasZeroDim = false;
        for (size_t dim = 0; dim < storageShape.GetDimNum(); ++dim) {
            OP_CHECK_IF(storageShape.GetDim(dim) < 0, OP_LOGE(context, "x[%d] has a negative dimension", i),
                        return ge::GRAPH_FAILED);
            hasZeroDim = hasZeroDim || storageShape.GetDim(dim) == 0;
        }
        const int64_t elements = hasZeroDim ? 0 : storageShape.GetShapeSize();
        OP_CHECK_IF(elements < 0, OP_LOGE(context, "x[%d] has an invalid element count", i), return ge::GRAPH_FAILED);
        OP_CHECK_IF(totalTensorElements > std::numeric_limits<int64_t>::max() - elements,
                    OP_LOGE(context, "x tensor element count overflows int64"), return ge::GRAPH_FAILED);
        tiling->tensorElements[i] = elements;
        totalTensorElements += elements;
    }

    uint64_t tilingKey = 0;
    uint32_t elementBytes = 0;
    OP_CHECK_IF(GetTilingKey(context, dtype, tilingKey, elementBytes) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetTilingKey failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        AssignDataToCores(context, tiling, totalTensorElements, availableCores, elementBytes) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "AssignDataToCores failed"), return ge::GRAPH_FAILED);

    const uint64_t availableUb = ubSize - sizeof(ForeachZeroInplaceTilingData);
    const uint64_t ubChunkBytes = Ops::Base::FloorAlign(availableUb / kBufferNum,
                                                        static_cast<uint64_t>(kDataBlockBytes));
    OP_CHECK_IF(ubChunkBytes > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "usable UB size exceeds tiling field capacity"), return ge::GRAPH_FAILED);
    tiling->ubChunkBytes = static_cast<uint32_t>(ubChunkBytes);
    OP_CHECK_IF(tiling->ubChunkBytes < kDataBlockBytes, OP_LOGE(context, "usable UB size is insufficient"),
                return ge::GRAPH_FAILED);

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(static_cast<uint32_t>(tiling->needCoreNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ForeachZeroInplace([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachZeroInplace)
    .Tiling(Tiling4ForeachZeroInplace)
    .TilingParse<ForeachZeroInplaceCompileInfo>(TilingPrepare4ForeachZeroInplace);

} // namespace optiling
