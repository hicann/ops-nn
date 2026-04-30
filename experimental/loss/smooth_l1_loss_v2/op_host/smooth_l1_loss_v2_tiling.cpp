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
 * \file smooth_l1_loss_v2_tiling.cpp
 * \brief SmoothL1LossV2 tiling
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/smooth_l1_loss_v2_tiling_data.h"
#include "../op_kernel/smooth_l1_loss_v2_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;
using Ops::Base::GetUbBlockSize;

#define UB_DATA_NUM_FLOAT_NONE 5U
#define UB_DATA_NUM_FLOAT_REDUCE 4U
#define UB_DATA_NUM_OTHER_NONE 6U
#define UB_DATA_NUM_OTHER_REDUCE 5U
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_SPLIT_NUM_NONE = 4096;
constexpr uint32_t TILE_SPLIT_NUM_REDUCE = 2048;
constexpr uint32_t SINGLE_CORE_THRESHOLD = 8192;
constexpr uint32_t SYNC_WORKSPACE_REGION_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0;
struct SmoothL1LossV2CompileInfo {
};

static ge::graphStatus TilingParseForSmoothL1LossV2([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // Prefer AIV core number in runtime, and fall back to generic core number for compatibility.
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = ascendcPlatform.GetCoreNum();
    }
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, int64_t coreNum, int reductionType)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    if (reductionType != 0) {
        OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
        uint32_t blockSize = GetUbBlockSize(context);
        size_t syncSize = static_cast<size_t>(coreNum) * blockSize * SYNC_WORKSPACE_REGION_NUM;
        // Reserve 8 float lanes per core for cross-core reduction staging.
        size_t reduceSize = static_cast<size_t>(coreNum) * 8U * sizeof(float);
        usrSize = syncSize + reduceSize;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static int GetReductionType(gert::TilingContext* context)
{
    return static_cast<int>(smooth_l1_loss_v2_host::GetReductionTypeFromContext(context, true));
}

static float GetSigmaAttr(gert::TilingContext* context)
{
    return smooth_l1_loss_v2_host::GetSigmaFromContext(context);
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t ubSize, uint64_t& inputNum, uint64_t& inputBytes, uint64_t& tileBlockNum,
    uint64_t& tileDataNum, uint64_t& inputLengthAlgin32, int reductionType)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    uint32_t blockSize = GetUbBlockSize(context);

    inputNum = inputShape->GetStorageShape().GetShapeSize();
    auto labelShape = context->GetInputShape(1);
    OP_CHECK_IF(labelShape == nullptr, OP_LOGE(context, "label shape is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputShape->GetStorageShape() != labelShape->GetStorageShape(),
        OP_LOGE(context, "predict and label shapes must match"), return ge::GRAPH_FAILED);

    auto predictDesc = context->GetInputDesc(0);
    auto labelDesc = context->GetInputDesc(1);
    OP_CHECK_IF(predictDesc == nullptr, OP_LOGE(context, "predict desc is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(labelDesc == nullptr, OP_LOGE(context, "label desc is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        predictDesc->GetDataType() != labelDesc->GetDataType(), OP_LOGE(context, "predict and label dtypes must match"),
        return ge::GRAPH_FAILED);

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(predictDesc->GetDataType(), typeLength);
    if (inputNum == 0) {
        OP_LOGE(context, "inputNum is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t inputLength = inputNum * typeLength;
    inputBytes = typeLength;
    uint64_t ubDataNumber = 0;
    if (predictDesc->GetDataType() == ge::DT_FLOAT) {
        ubDataNumber = (reductionType == 0) ? UB_DATA_NUM_FLOAT_NONE : UB_DATA_NUM_FLOAT_REDUCE;
    } else {
        ubDataNumber = (reductionType == 0) ? UB_DATA_NUM_OTHER_NONE : UB_DATA_NUM_OTHER_REDUCE;
    }
    if (blockSize == 0 || ubDataNumber == 0 || BUFFER_NUM == 0) {
        OP_LOGE(context, "blockSize, ubDataNumber, or bufferNum is 0");
        return ge::GRAPH_FAILED;
    }
    if (inputBytes == 0) {
        OP_LOGE(context, "inputBytes is 0");
        return ge::GRAPH_FAILED;
    }
    tileBlockNum = (ubSize / blockSize) / ubDataNumber / BUFFER_NUM;
    tileDataNum = (tileBlockNum * blockSize) / inputBytes;
    if (tileBlockNum == 0 || tileDataNum == 0) {
        OP_LOGE(context, "tileBlockNum or tileDataNum is 0");
        return ge::GRAPH_FAILED;
    }
    inputLengthAlgin32 = (((inputLength + blockSize - 1) / blockSize) * blockSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    gert::TilingContext* context, uint64_t inputLengthAlgin32, int64_t coreNum, uint64_t tileBlockNum,
    uint64_t inputBytes, uint64_t tileDataNum, uint32_t blockSize, uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum,
    uint64_t& smallTailDataNum, uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum, uint64_t& finalBigTileNum,
    uint64_t& tailBlockNum)
{
    if (blockSize == 0 || coreNum == 0 || tileBlockNum == 0 || inputBytes == 0) {
        OP_LOGE(
            context, "CalculateCoreBlockNums invalid params: blockSize=%u coreNum=%ld tileBlockNum=%lu inputBytes=%lu",
            blockSize, coreNum, tileBlockNum, inputBytes);
        return ge::GRAPH_FAILED;
    }
    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / blockSize / coreNum;
    tailBlockNum = (inputLengthAlgin32 / blockSize) % coreNum;
    smallCoreDataNum = everyCoreInputBlockNum * blockSize / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * blockSize / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

struct SmoothL1LossV2BasicInfo {
    uint64_t ubSize = 0;
    int64_t platformCoreNum = 0;
    int reductionType = 0;
    float sigma = 0.0f;
    uint64_t inputNum = 0;
    uint64_t inputBytes = 0;
    uint64_t tileBlockNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t inputLengthAlgin32 = 0;
};

static ge::graphStatus PrepareTilingInputs(
    gert::TilingContext* context, SmoothL1LossV2TilingData*& tiling, SmoothL1LossV2BasicInfo& info)
{
    tiling = context->GetTilingData<SmoothL1LossV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SmoothL1LossV2TilingData), 0, sizeof(SmoothL1LossV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    ge::graphStatus ret = GetPlatformInfo(context, info.ubSize, info.platformCoreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    info.reductionType = GetReductionType(context);
    OP_CHECK_IF(info.reductionType < 0, OP_LOGE(context, "Invalid reduction attr"), return ge::GRAPH_FAILED);
    info.sigma = GetSigmaAttr(context);

    ret = GetShapeAttrsInfo(
        context, info.ubSize, info.inputNum, info.inputBytes, info.tileBlockNum, info.tileDataNum,
        info.inputLengthAlgin32, info.reductionType);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static int64_t DecideLaunchCoreNum(const SmoothL1LossV2BasicInfo& info)
{
    uint64_t tileSplitNum = (info.reductionType == 0) ? TILE_SPLIT_NUM_NONE : TILE_SPLIT_NUM_REDUCE;
    uint64_t calcCoreNum = info.inputNum / tileSplitNum;
    if (info.inputNum % tileSplitNum) {
        calcCoreNum += 1;
    }

    if (info.reductionType == 0) {
        // Non-reduction path is launch-latency sensitive for short shapes.
        if (info.inputNum <= 128) {
            calcCoreNum = 1;
        } else if (info.inputNum <= 1024) {
            calcCoreNum = 2;
        } else if (info.inputNum <= 8192) {
            calcCoreNum = 8;
        } else if (info.inputNum <= 32768) {
            calcCoreNum = 12;
        }

        // Keep fp32 short-shape parallelism lighter to avoid split overhead.
        if (info.inputBytes == sizeof(float) && info.inputNum <= 8192 && calcCoreNum > 4) {
            calcCoreNum = 4;
        }
    } else {
        // Reduction path: keep tiny shapes on one core, enable dynamic multicore for larger shapes.
        if (info.inputNum <= SINGLE_CORE_THRESHOLD) {
            calcCoreNum = 1;
        } else if (info.inputNum <= 131072) {
            calcCoreNum = std::min<uint64_t>(calcCoreNum, 16);
        } else if (info.inputNum <= 524288) {
            calcCoreNum = std::min<uint64_t>(calcCoreNum, 32);
        }
    }

    if (calcCoreNum == 0) {
        calcCoreNum = 1;
    }
    int64_t coreNum = (calcCoreNum < static_cast<uint64_t>(info.platformCoreNum)) ? static_cast<int64_t>(calcCoreNum) :
                                                                                    info.platformCoreNum;
    return (coreNum <= 0) ? 1 : coreNum;
}

static ge::graphStatus FinalizeTilingResult(
    gert::TilingContext* context, SmoothL1LossV2TilingData* tiling, const SmoothL1LossV2BasicInfo& info,
    int64_t coreNum)
{
    // 统一通过 workspace + core0 汇总完成 sum/mean，避免全局原子带来的抖动。
    context->SetNeedAtomic(false);

    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
    uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;
    ge::graphStatus ret = CalculateCoreBlockNums(
        context, info.inputLengthAlgin32, coreNum, info.tileBlockNum, info.inputBytes, info.tileDataNum,
        GetUbBlockSize(context), smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum,
        finalBigTileNum, tailBlockNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ge::GRAPH_FAILED);

    tiling->smallCoreDataNum = static_cast<uint64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint64_t>(info.tileDataNum);
    tiling->smallTailDataNum = static_cast<uint64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<uint64_t>(tailBlockNum);
    tiling->totalDataNum = static_cast<uint64_t>(info.inputNum);
    tiling->blockBytes = GetUbBlockSize(context);
    tiling->sigma = info.sigma;
    tiling->reduction = info.reductionType;

    OP_CHECK_IF(
        GetWorkspaceSize(context, coreNum, info.reductionType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
    uint64_t tilingKey = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SmoothL1LossV2TilingFunc(gert::TilingContext* context)
{
    SmoothL1LossV2TilingData* tiling = nullptr;
    SmoothL1LossV2BasicInfo info{};

    ge::graphStatus ret = PrepareTilingInputs(context, tiling, info);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "PrepareTilingInputs error"), return ge::GRAPH_FAILED);

    int64_t coreNum = DecideLaunchCoreNum(info);

    ret = FinalizeTilingResult(context, tiling, info, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "FinalizeTilingResult error"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(SmoothL1LossV2)
    .Tiling(SmoothL1LossV2TilingFunc)
    .TilingParse<SmoothL1LossV2CompileInfo>(TilingParseForSmoothL1LossV2);
} // namespace optiling
