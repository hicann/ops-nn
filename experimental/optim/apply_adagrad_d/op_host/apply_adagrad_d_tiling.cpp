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
 * \file apply_adagrad_d_tiling.cpp
 * \brief Tiling for ApplyAdagradD (experimental, ascend910b)
 *
 */

#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/apply_adagrad_d_tiling_data.h"
#include "../op_kernel/apply_adagrad_d_tiling_key.h"
#include <limits>

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr uint32_t BUFFER_NUM      = 2;    // double-buffer
constexpr uint32_t WS_SYS_SIZE     = 512U;
constexpr uint32_t UB_FACTOR       = 18;
constexpr uint64_t UB_BLOCK_BYTES  = 32ULL;

struct ApplyAdagradDCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context->GetNodeName(), "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeInfo(gert::TilingContext* context,
                                    int64_t& totalElems,
                                    ge::DataType& dataType,
                                    bool& updateSlots)
{
    // Total elements from var input (index 0)
    auto inputVar = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputVar);
    totalElems = inputVar->GetStorageShape().GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context->GetNodeName(), "unsupported dtype: %d", static_cast<int32_t>(dataType));
        return ge::GRAPH_FAILED;
    }

    // update_slots attribute (index 0), default true
    updateSlots = true;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* attrPtr = attrs->GetAttrPointer<bool>(0);
        if (attrPtr != nullptr) {
            updateSlots = *attrPtr;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApplyAdagradDTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("ApplyAdagradD", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    // 1. Platform info
    uint64_t ubSize = 0;
    int64_t  coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Shape / attr info
    int64_t     totalElems  = 0;
    ge::DataType dataType;
    bool        updateSlots = true;
    OP_CHECK_IF(GetShapeInfo(context, totalElems, dataType, updateSlots) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetShapeInfo error"), return ge::GRAPH_FAILED);

    // 3. Workspace
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // Handle empty input
    if (totalElems <= 0) {
        ApplyAdagradDTilingData* tiling = context->GetTilingData<ApplyAdagradDTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        OP_CHECK_IF(memset_s(tiling, sizeof(ApplyAdagradDTilingData), 0, sizeof(ApplyAdagradDTilingData)) != EOK,
                    OP_LOGE(context->GetNodeName(), "memset tiling data error"), return ge::GRAPH_FAILED);
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }

    // 4. Compute tiling parameters (following celu_v2 style)
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    if (typeLength == 0) {
        OP_LOGE(context->GetNodeName(), "typeLength is 0, dataType: %d", static_cast<int32_t>(dataType));
        return ge::GRAPH_FAILED;
    }

    // tileDataNum: how many T-elements fit in one tile
    uint64_t blockSize = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context));
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context->GetNodeName(), "blockSize is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockSize != UB_BLOCK_BYTES,
                OP_LOGE(context->GetNodeName(), "blockSize is not 32B, blockSize: %lu", blockSize),
                return ge::GRAPH_FAILED);

    // ubSize / (BUFFER_NUM * UB_FACTOR * typeLength), rounded down to UB block boundary
    uint64_t inputBytes        = static_cast<uint64_t>(typeLength);
    uint64_t tileBlockNum_ub   = ubSize / (blockSize * static_cast<uint64_t>(BUFFER_NUM) *
                                           static_cast<uint64_t>(UB_FACTOR));
    uint64_t tileBlockNum = (tileBlockNum_ub == 0) ? 1ULL : tileBlockNum_ub;

    uint64_t tileDataNum = (tileBlockNum * blockSize) / inputBytes;
    if (tileDataNum == 0ULL) tileDataNum = 1ULL;

    // Total UB blocks in the input (var tensor)
    uint64_t inputLengthBytes = static_cast<uint64_t>(totalElems) * inputBytes;
    uint64_t blocksTotal      = (inputLengthBytes + blockSize - 1ULL) / blockSize;

    uint64_t coreNum64 = static_cast<uint64_t>(coreNum);
    if (coreNum64 > blocksTotal) coreNum64 = blocksTotal;
    if (coreNum64 == 0ULL)       coreNum64 = 1ULL;
    uint32_t finalCoreNum = static_cast<uint32_t>(coreNum64);

    uint64_t everyCoreBlocks = blocksTotal / coreNum64;
    uint64_t tailBlockNum    = blocksTotal % coreNum64;

    // small-core (gets everyCoreBlocks blocks)
    uint64_t smallCoreDataNum  = (everyCoreBlocks * blockSize) / inputBytes;
    uint64_t smallTileNum      = everyCoreBlocks / tileBlockNum;
    uint64_t finalSmallTileNum = ((everyCoreBlocks % tileBlockNum) == 0) ? smallTileNum : (smallTileNum + 1);
    int64_t  smallTail_i       = static_cast<int64_t>(smallCoreDataNum) -
                                  static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(smallTileNum);
    uint64_t smallTailDataNum  = (smallTail_i <= 0) ? tileDataNum : static_cast<uint64_t>(smallTail_i);

    // big-core (gets everyCoreBlocks+1 blocks)
    uint64_t bigEveryBlocks   = everyCoreBlocks + 1ULL;
    uint64_t bigCoreDataNum   = (bigEveryBlocks * blockSize) / inputBytes;
    uint64_t bigTileNum       = bigEveryBlocks / tileBlockNum;
    uint64_t finalBigTileNum  = ((bigEveryBlocks % tileBlockNum) == 0) ? bigTileNum : (bigTileNum + 1);
    int64_t  bigTail_i        = static_cast<int64_t>(bigCoreDataNum) -
                                 static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(bigTileNum);
    uint64_t bigTailDataNum   = (bigTail_i <= 0) ? tileDataNum : static_cast<uint64_t>(bigTail_i);

    const uint64_t int64Max = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    OP_CHECK_IF(tileDataNum > int64Max || smallCoreDataNum > int64Max || bigCoreDataNum > int64Max ||
                    smallTailDataNum > int64Max || bigTailDataNum > int64Max ||
                    finalSmallTileNum > int64Max || finalBigTileNum > int64Max || tailBlockNum > int64Max,
                OP_LOGE(context->GetNodeName(), "tiling value exceeds int64 range"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tileDataNum > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
                    smallCoreDataNum > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
                    bigCoreDataNum > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                OP_LOGE(context->GetNodeName(), "tiling value exceeds kernel uint32 range"), return ge::GRAPH_FAILED);

    // 5. Write tiling data
    ApplyAdagradDTilingData* tiling = context->GetTilingData<ApplyAdagradDTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ApplyAdagradDTilingData), 0, sizeof(ApplyAdagradDTilingData)) != EOK,
                OP_LOGE(context->GetNodeName(), "memset tiling data error"), return ge::GRAPH_FAILED);

    tiling->smallCoreDataNum  = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum    = static_cast<int64_t>(bigCoreDataNum);
    tiling->tileDataNum       = static_cast<int64_t>(tileDataNum);
    tiling->smallTailDataNum  = static_cast<int64_t>(smallTailDataNum);
    tiling->bigTailDataNum    = static_cast<int64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
    tiling->finalBigTileNum   = static_cast<int64_t>(finalBigTileNum);
    tiling->tailBlockNum      = static_cast<int64_t>(tailBlockNum);
    tiling->totalDataNum      = totalElems;
    tiling->updateSlots       = updateSlots ? 1 : 0;

    context->SetBlockDim(finalCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));

    OP_LOGI(context->GetNodeName(), "ApplyAdagradD tiling: coreNum=%u tileDataNum=%llu updateSlots=%d",
            finalCoreNum, static_cast<unsigned long long>(tileDataNum), (int)updateSlots);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForApplyAdagradD([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyAdagradD).Tiling(ApplyAdagradDTilingFunc).TilingParse<ApplyAdagradDCompileInfo>(TilingParseForApplyAdagradD);

} // namespace optiling
