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
 * \file hinge_loss_tiling.cpp
 * \brief HingeLoss算子的tiling(分块)策略实现，支持多核切分和多数据类型
 */

#include <algorithm>
#include <limits>
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "../op_kernel/hinge_loss_tiling_data.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t BUFFER_NUM = 2;

#define UB_NUM_FLOAT 15U
#define UB_NUM_OTHER 25U

static const gert::Shape g_vec_1_shape = {1};

struct HingeLossCompileInfo {};

inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static int64_t GetTypeSize(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return 2;
    }
    return 4;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    auto inputPredict = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputPredict);
    auto inputShapePredict = EnsureNotScalar(inputPredict->GetStorageShape());

    auto inputTarget = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputTarget);
    (void)EnsureNotScalar(inputTarget->GetStorageShape());

    auto outLoss = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outLoss);
    (void)EnsureNotScalar(outLoss->GetStorageShape());

    totalIdx = inputShapePredict.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HingeLossTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalIdx;
    ge::DataType dataType;

    OP_CHECK_IF(GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    HingeLossTilingData* tiling = context->GetTilingData<HingeLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    OP_CHECK_IF(memset_s(tiling, sizeof(HingeLossTilingData), 0, sizeof(HingeLossTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Total element count
    int64_t totalLength = totalIdx;
    OP_CHECK_IF(totalLength < 0, OP_LOGE(context, "totalLength is negative"), return ge::GRAPH_FAILED);

    // Element size in bytes for the input dtype
    int64_t typeSize = GetTypeSize(dataType);

    // Compute tile capacity from UB size
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    uint64_t ubDataNumber = (dataType == ge::DT_FLOAT) ? UB_NUM_FLOAT : UB_NUM_OTHER;
    uint64_t tileBlockNum = (static_cast<uint64_t>(ubSize) / static_cast<uint64_t>(ubBlockSize)) / ubDataNumber;
    int64_t tileDataNum = static_cast<int64_t>((tileBlockNum * static_cast<uint64_t>(ubBlockSize)) /
                                               static_cast<uint64_t>(typeSize));
    if (tileDataNum == 0) {
        tileDataNum = 1;
    }

    // Split logical elements evenly. The leading tailBlockNum cores process one
    // more element than the remaining cores.
    int64_t usedCoreNum = (totalLength == 0) ? 1 : std::min(totalLength, coreNum);
    int64_t smallCoreDataNum = totalLength / usedCoreNum;
    int64_t tailBlockNum = totalLength % usedCoreNum;
    int64_t bigCoreDataNum = smallCoreDataNum + ((tailBlockNum > 0) ? 1 : 0);

    int64_t finalBigTileNum = (bigCoreDataNum > 0) ? Ops::Base::CeilDiv(bigCoreDataNum, tileDataNum) : 0;
    int64_t finalSmallTileNum = (smallCoreDataNum > 0) ? Ops::Base::CeilDiv(smallCoreDataNum, tileDataNum) : 0;
    int64_t smallTailDataNum = (smallCoreDataNum > 0) ? smallCoreDataNum - (finalSmallTileNum - 1) * tileDataNum : 0;
    int64_t bigTailDataNum = (bigCoreDataNum > 0) ? bigCoreDataNum - (finalBigTileNum - 1) * tileDataNum : 0;

    constexpr int64_t uint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    OP_CHECK_IF(smallCoreDataNum > uint32Max || bigCoreDataNum > uint32Max || finalBigTileNum > uint32Max ||
                    finalSmallTileNum > uint32Max || tileDataNum > uint32Max || smallTailDataNum > uint32Max ||
                    bigTailDataNum > uint32Max || tailBlockNum > uint32Max,
                OP_LOGE(context, "tiling value exceeds uint32_t range"), return ge::GRAPH_FAILED);

    tiling->smallCoreDataNum = static_cast<uint32_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint32_t>(bigCoreDataNum);
    tiling->finalBigTileNum = static_cast<uint32_t>(finalBigTileNum);
    tiling->finalSmallTileNum = static_cast<uint32_t>(finalSmallTileNum);
    tiling->tileDataNum = static_cast<uint32_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<uint32_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint32_t>(bigTailDataNum);
    tiling->tailBlockNum = static_cast<uint32_t>(tailBlockNum);

    context->SetBlockDim(static_cast<int32_t>(usedCoreNum));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHingeLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HingeLoss).Tiling(HingeLossTilingFunc).TilingParse<HingeLossCompileInfo>(TilingParseForHingeLoss);
} // namespace optiling
