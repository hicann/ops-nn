/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hinge_loss_grad_tiling.cpp
 * \brief HingeLossGrad算子的tiling(分块)策略实现，支持多核切分和多数据类型
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "../op_kernel/hinge_loss_grad_tiling_data.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t BUFFER_NUM = 2;

#define UB_NUM_FLOAT  15U
#define UB_NUM_OTHER  25U

static const gert::Shape g_vec_1_shape = {1};

struct HingeLossGradCompileInfo {
};

inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}

static int64_t GetTypeSize(ge::DataType dtype) {
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

    auto inputGradOutput = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGradOutput);
    (void)EnsureNotScalar(inputGradOutput->GetStorageShape());

    auto outGradInput = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outGradInput);
    (void)EnsureNotScalar(outGradInput->GetStorageShape());

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

static ge::graphStatus HingeLossGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t totalIdx;
    ge::DataType dataType;

    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    HingeLossGradTilingData* tiling = context->GetTilingData<HingeLossGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    OP_CHECK_IF(
        memset_s(tiling, sizeof(HingeLossGradTilingData), 0, sizeof(HingeLossGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    // Total element count
    int64_t totalLength = totalIdx;

    // Element size in bytes for the input dtype
    int64_t typeSize = GetTypeSize(dataType);

    // Compute tile capacity from UB size
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    uint64_t ubDataNumber = (dataType == ge::DT_FLOAT) ? UB_NUM_FLOAT : UB_NUM_OTHER;
    uint64_t tileBlockNum = (static_cast<uint64_t>(ubSize) / static_cast<uint64_t>(ubBlockSize)) / ubDataNumber;
    int64_t tileDataNum = static_cast<int64_t>((tileBlockNum * static_cast<uint64_t>(ubBlockSize)) / static_cast<uint64_t>(typeSize));
    if (tileDataNum == 0) {
        tileDataNum = 1;
    }

    // Split work across small and big cores
    int64_t usedCoreNum;
    int64_t bigCoreDataNum;
    int64_t smallCoreDataNum;
    int64_t tailBlockNum;

    if (totalLength <= coreNum * tileDataNum) {
        bigCoreDataNum = 0;
        tailBlockNum = 0;
        smallCoreDataNum = Ops::Base::CeilDiv(totalLength, coreNum);
        usedCoreNum = Ops::Base::CeilDiv(totalLength, smallCoreDataNum);
    } else {
        bigCoreDataNum = tileDataNum;
        int64_t remainAfterBig = totalLength - bigCoreDataNum * coreNum;
        if (remainAfterBig > 0 && remainAfterBig <= coreNum) {
            tailBlockNum = remainAfterBig;
            smallCoreDataNum = bigCoreDataNum + 1;
        } else {
            tailBlockNum = 0;
            smallCoreDataNum = bigCoreDataNum;
        }
        usedCoreNum = coreNum;
    }

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = (bigCoreDataNum > 0) ? Ops::Base::CeilDiv(bigCoreDataNum, tileDataNum) : 0;
    tiling->finalSmallTileNum = Ops::Base::CeilDiv(smallCoreDataNum, tileDataNum);
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallCoreDataNum - (tiling->finalSmallTileNum - 1) * tileDataNum;
    tiling->bigTailDataNum = (bigCoreDataNum > 0) ? bigCoreDataNum - (tiling->finalBigTileNum - 1) * tileDataNum : 0;
    tiling->tailBlockNum = tailBlockNum;

    context->SetBlockDim(static_cast<int32_t>(usedCoreNum));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHingeLossGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HingeLossGrad).Tiling(HingeLossGradTilingFunc).TilingParse<HingeLossGradCompileInfo>(TilingParseForHingeLossGrad);
} // namespace optiling
