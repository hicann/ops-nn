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
 * \file apply_came_part1_tiling.cc
 * \brief
 */
#include <algorithm>
#include <cmath>
#include <limits>
#include <securec.h>
#include "error_util.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "apply_came_part1_tiling_arch35.h"

namespace optiling {

const std::string OP_NAME = "ApplyCamePart1";

constexpr int32_t GRAD_DIM_NUM = 2;
constexpr int64_t GAIN_COEFFICIENT = 2;

constexpr int64_t N_NUM_PER_CORE = 64;
constexpr int64_t M_NUM_PER_CORE = 64;
constexpr int64_t BLOCK_SIZE = 8;
constexpr int32_t BATCH_MODE = 1;
constexpr int64_t DIRECT_BATCH_ALIGNMENT = 8;
constexpr int64_t MAX_DIRECT_MATRIX_ELEMENTS = 8192;
constexpr size_t MIN_USER_WORKSPACE_BYTES = 32;
constexpr size_t WORKSPACE_ALIGNMENT_BYTES = 512;

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t valueNum = 0;
    if (factor == 0) {
        OP_LOGW(OP_NAME.c_str(), "Tiling4ApplyCamePart1 factor is 0");
        return value;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

inline static int64_t CeilAlign(int64_t value, int64_t factor) { return CeilDiv(value, factor) * factor; }

inline static bool CheckedCeilAlign(size_t value, size_t factor, size_t& alignedValue)
{
    if (factor == 0) {
        return false;
    }
    const size_t remainder = value % factor;
    if (remainder == 0) {
        alignedValue = value;
        return true;
    }
    const size_t padding = factor - remainder;
    if (value > std::numeric_limits<size_t>::max() - padding) {
        return false;
    }
    alignedValue = value + padding;
    return true;
}

inline static bool CheckedMultiply(size_t lhs, size_t rhs, size_t& result)
{
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline static bool CheckedAdd(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

// n方向如果 (n / t + 1) > 48，那么n方向进行t *= 2，直至满足小于48的场景，此时t为最终单核上有效的nNormalCoreNum_
inline static int64_t CeilMul(int64_t InputNum, int64_t normalValue, int64_t totalCore)
{
    if (normalValue == 0) {
        return normalValue;
    }
    if (InputNum < normalValue) {
        return InputNum;
    }
    while (CeilDiv(InputNum, normalValue) > totalCore) {
        normalValue *= GAIN_COEFFICIENT;
    }
    return normalValue;
}

static ge::graphStatus ApplyCamePart1SetTilingData(gert::TilingContext* context, ApplyCamePart1TilingData& tilingData)
{
    gert::TilingData* rawTilingData = context->GetRawTilingData();
    OP_LOGE_IF(rawTilingData == nullptr, ge::GRAPH_FAILED, context->GetNodeType(), "GetRawTilingData failed.");
    OP_TILING_CHECK(
        sizeof(tilingData) > rawTilingData->GetCapacity(),
        VECTOR_INNER_ERR_REPORT_TILIING(context, "actual tiling data size %zu > context tiling data size %zu",
                                        sizeof(tilingData), rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &tilingData, sizeof(tilingData));
    OP_TILING_CHECK(ret != EOK, VECTOR_INNER_ERR_REPORT_TILIING(context, "copy tiling data failed"),
                    return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData));
    if (tilingData.get_directBatchMode() == 0) {
        context->SetScheduleMode(BATCH_MODE);
    }
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcNdimTiling(const gert::TilingContext* context, ApplyCamePart1TilingData& tilingData)
{
    OP_LOGD(context->GetNodeName(), "TilingApplyCamePart1 Enter CalcNdimTiling.");
    int64_t nNormalCoreNum = std::max(CeilMul(tilingData.get_N(), N_NUM_PER_CORE, tilingData.get_totalCoreNum()),
                                      static_cast<int64_t>(1));
    int64_t nTailCoreNum = tilingData.get_N() % nNormalCoreNum;
    if (nTailCoreNum == 0) {
        nTailCoreNum = nNormalCoreNum;
    }
    int64_t nCoreNum = CeilDiv(tilingData.get_N(), nNormalCoreNum);
    int64_t nLoopNormCore = CeilDiv(nNormalCoreNum, N_NUM_PER_CORE);
    int64_t nLoopTailCore = CeilDiv(nTailCoreNum, N_NUM_PER_CORE);
    tilingData.set_nNormalCoreNum(nNormalCoreNum);
    tilingData.set_nTailCoreNum(nTailCoreNum);
    tilingData.set_nCoreNum(nCoreNum);
    tilingData.set_nLoopNormCore(nLoopNormCore);
    tilingData.set_nLoopTailCore(nLoopTailCore);
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcBatchSplit(const gert::TilingContext* context, ApplyCamePart1TilingData& tilingData)
{
    OP_LOGD(context->GetNodeName(), "TilingApplyCamePart1 Enter CalcBatchSplit.");
    const int64_t sliceCoreNum = tilingData.get_nCoreNum() * tilingData.get_mCoreNum();
    OP_TILING_CHECK(
        sliceCoreNum <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingApplyCamePart1 slice core num is invalid."),
        return ge::GRAPH_FAILED);
    // Assign independent batch groups to disjoint workspace regions.
    int64_t batchCoreNum = 1;
    if (tilingData.get_totalCoreNum() > sliceCoreNum && tilingData.get_batchCount() > 1) {
        batchCoreNum = std::min(tilingData.get_totalCoreNum() / sliceCoreNum, tilingData.get_batchCount());
    }
    tilingData.set_batchCoreNum(batchCoreNum);
    tilingData.set_usedCoreNum(sliceCoreNum * batchCoreNum);
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcMdimTiling(const gert::TilingContext* context, ApplyCamePart1TilingData& tilingData)
{
    OP_LOGD(context->GetNodeName(), "TilingApplyCamePart1 Enter CalcMdimTiling.");
    const int64_t nCoreNum = tilingData.get_nCoreNum();
    OP_TILING_CHECK(
        nCoreNum <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingApplyCamePart1 nCoreNum is invalid."),
        return ge::GRAPH_FAILED);

    const int64_t columnTileCount = CeilDiv(tilingData.get_M(), M_NUM_PER_CORE);
    const int64_t availableMCoreNum = std::max<int64_t>(1, tilingData.get_totalCoreNum() / nCoreNum);
    int64_t mCoreNum = std::min(columnTileCount, availableMCoreNum);
    const int64_t columnsPerCore = CeilDiv(tilingData.get_M(), mCoreNum);
    const int64_t mNormalCoreNum = CeilAlign(columnsPerCore, M_NUM_PER_CORE);
    mCoreNum = CeilDiv(tilingData.get_M(), mNormalCoreNum);
    const int64_t mTailCoreNum = tilingData.get_M() - mNormalCoreNum * (mCoreNum - 1);

    tilingData.set_mLoopNumCore(CeilDiv(mNormalCoreNum, M_NUM_PER_CORE));
    tilingData.set_mNormalCoreNum(mNormalCoreNum);
    tilingData.set_mTailCoreNum(mTailCoreNum);
    tilingData.set_mCoreNum(mCoreNum);
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus UpdateTilingMsg(const gert::TilingContext* context, ApplyCamePart1TilingData& tilingData)
{
    // set coreNum
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t totalCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    tilingData.set_totalCoreNum(totalCoreNum);

    auto N = tilingData.get_N();
    auto M = tilingData.get_M();
    OP_TILING_CHECK((M == 0 || N == 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingApplyCamePart1 m or n is 0."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckParamsShape(const gert::TilingContext* context)
{
    auto inputShape0 = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, inputShape0);
    auto gradShape = inputShape0->GetStorageShape();
    OPS_ERR_IF(gradShape.GetShapeSize() == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the gradShape of input should not be empty tensor"),
               return ge::GRAPH_FAILED);

    auto gradDimNum = gradShape.GetDimNum();
    OP_TILING_CHECK(gradDimNum < GRAD_DIM_NUM,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "grad shape dim must be at least 2, but got %zu", gradDimNum),
                    return ge::GRAPH_FAILED);

    auto epsShapeInput = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epsShapeInput);
    auto epsShape = epsShapeInput->GetStorageShape();
    OP_TILING_CHECK(
        epsShape.GetShapeSize() != 1,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "eps must contain exactly one element, but got %s",
                                        Ops::Base::ToString(epsShape).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool ShouldUseDirectBatch(const ApplyCamePart1TilingData& tilingData)
{
    const int64_t n = tilingData.get_N();
    const int64_t m = tilingData.get_M();
    if (n <= 0 || m <= 0 || n > MAX_DIRECT_MATRIX_ELEMENTS / m) {
        return false;
    }
    const int64_t matrixElements = n * m;
    if (matrixElements > MAX_DIRECT_MATRIX_ELEMENTS) {
        return false;
    }
    const bool singleVectorTile = n <= N_NUM_PER_CORE && m <= M_NUM_PER_CORE;
    const int64_t alignedBatchGroupCount = CeilDiv(tilingData.get_batchCount(), DIRECT_BATCH_ALIGNMENT);
    const bool enoughIndependentBatchGroups = alignedBatchGroupCount >= tilingData.get_totalCoreNum();
    return singleVectorTile || enoughIndependentBatchGroups;
}

static ge::graphStatus CalculateDirectBatchWorkspace(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t workspaceSize = 0;
    OP_TILING_CHECK(
        !CheckedAdd(sysWorkspaceSize, MIN_USER_WORKSPACE_BYTES, workspaceSize),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 direct workspace size overflows."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !CheckedCeilAlign(workspaceSize, WORKSPACE_ALIGNMENT_BYTES, workspaceSize),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 direct workspace alignment overflows."),
        return ge::GRAPH_FAILED);
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    userWorkspaceSize[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateApplyCamePart1Workspace(gert::TilingContext* context,
                                                        ApplyCamePart1TilingData& tilingData)
{
    const size_t nCoreNum = static_cast<size_t>(tilingData.get_nCoreNum());
    const size_t mCoreNum = static_cast<size_t>(tilingData.get_mCoreNum());
    const size_t nLoopNormCore = static_cast<size_t>(tilingData.get_nLoopNormCore());
    const size_t nLoopTailCore = static_cast<size_t>(tilingData.get_nLoopTailCore());
    const size_t mLoopNumCore = static_cast<size_t>(tilingData.get_mLoopNumCore());
    OP_TILING_CHECK(
        nCoreNum == 0 || mCoreNum == 0 || nLoopNormCore == 0 || nLoopTailCore == 0 || mLoopNumCore == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 workspace dimensions are invalid."),
        return ge::GRAPH_FAILED);

    size_t normalLoopCount = 0;
    size_t nLoopCount = 0;
    size_t rElements = 0;
    size_t rcElements = 0;
    const size_t totalColumnTileCount = static_cast<size_t>(CeilDiv(tilingData.get_M(), M_NUM_PER_CORE));
    OP_TILING_CHECK(
        !CheckedMultiply(nCoreNum - 1, nLoopNormCore, normalLoopCount) ||
            !CheckedAdd(normalLoopCount, nLoopTailCore, nLoopCount) ||
            !CheckedMultiply(nLoopCount, mCoreNum, rElements) ||
            !CheckedMultiply(nLoopCount, totalColumnTileCount, rcElements),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 workspace element count overflows."),
        return ge::GRAPH_FAILED);

    constexpr size_t rcSlotSize = 8;
    const size_t rcReducePartialCount = static_cast<size_t>(CeilDiv(tilingData.get_N(), static_cast<int64_t>(8192)));
    OP_TILING_CHECK(rcElements < rcReducePartialCount,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "ApplyCamePart1 workspace cannot hold scalar reduction partials."),
                    return ge::GRAPH_FAILED);

    size_t rcStorageElements = 0;
    size_t rWorkspaceBytes = 0;
    size_t cWorkspaceBytes = 0;
    OP_TILING_CHECK(
        !CheckedMultiply(rcElements, rcSlotSize, rcStorageElements) ||
            !CheckedMultiply(rElements, static_cast<size_t>(N_NUM_PER_CORE * sizeof(float)), rWorkspaceBytes) ||
            !CheckedMultiply(rcElements, static_cast<size_t>(M_NUM_PER_CORE * sizeof(float)), cWorkspaceBytes),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 workspace byte size overflows."),
        return ge::GRAPH_FAILED);

    size_t rcAlignedElements = 0;
    size_t rWorkspaceSize = 0;
    size_t cWorkspaceSize = 0;
    size_t rcWorkspaceSize = 0;
    OP_TILING_CHECK(
        !CheckedCeilAlign(rcStorageElements, 128U, rcAlignedElements) ||
            !CheckedCeilAlign(rWorkspaceBytes, 512U, rWorkspaceSize) ||
            !CheckedCeilAlign(cWorkspaceBytes, 512U, cWorkspaceSize) ||
            !CheckedMultiply(rcAlignedElements, 2U * sizeof(float), rcWorkspaceSize),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 workspace alignment overflows."),
        return ge::GRAPH_FAILED);

    size_t syncWorkspaceSize = 0;
    OP_TILING_CHECK(
        !CheckedMultiply(static_cast<size_t>(tilingData.get_totalCoreNum()), 32U, syncWorkspaceSize) ||
            !CheckedAdd(syncWorkspaceSize, 32U, syncWorkspaceSize),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 sync workspace size overflows."),
        return ge::GRAPH_FAILED);

    const size_t batchCoreNum = static_cast<size_t>(tilingData.get_batchCoreNum());
    size_t totalRcWorkspaceSize = 0;
    size_t totalRWorkspaceSize = 0;
    size_t totalCWorkspaceSize = 0;
    OP_TILING_CHECK(
        batchCoreNum == 0 || !CheckedMultiply(rcWorkspaceSize, batchCoreNum, totalRcWorkspaceSize) ||
            !CheckedMultiply(rWorkspaceSize, batchCoreNum, totalRWorkspaceSize) ||
            !CheckedMultiply(cWorkspaceSize, batchCoreNum, totalCWorkspaceSize),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 batch workspace size overflows."),
        return ge::GRAPH_FAILED);

    size_t userSize = 0;
    OP_TILING_CHECK(
        !CheckedAdd(syncWorkspaceSize, totalRcWorkspaceSize, userSize) ||
            !CheckedAdd(userSize, totalRWorkspaceSize, userSize) ||
            !CheckedAdd(userSize, totalCWorkspaceSize, userSize),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 total workspace byte size overflows."),
        return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t totalWorkspaceSize = 0;
    OP_TILING_CHECK(
        !CheckedAdd(userSize, sysWorkspaceSize, totalWorkspaceSize),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 system workspace size overflows."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !CheckedCeilAlign(totalWorkspaceSize, WORKSPACE_ALIGNMENT_BYTES, totalWorkspaceSize),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 workspace alignment overflows."),
        return ge::GRAPH_FAILED);
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    userWorkspaceSize[0] = totalWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InitializeApplyCamePart1Tiling(gert::TilingContext* context,
                                                      ApplyCamePart1TilingData& tilingData, ge::DataType& dataType)
{
    OP_TILING_CHECK(CheckParamsShape(context) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1Tiling check shape failed."),
                    return ge::GRAPH_FAILED);

    auto input = context->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, input);
    auto epsInput = context->GetInputDesc(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epsInput);
    OP_TILING_CHECK(epsInput->GetDataType() != ge::DT_FLOAT,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 eps must be float32."),
                    return ge::GRAPH_FAILED);
    dataType = input->GetDataType();
    int64_t inputElementBytes = 0;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        inputElementBytes = 2;
        OP_LOGI(context->GetNodeName(), "Current dataType is [float16] or [bfloat16].");
    } else if (dataType == ge::DT_FLOAT) {
        inputElementBytes = sizeof(float);
        OP_LOGI(context->GetNodeName(), "Current dataType is [float32].");
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "grad", Ops::Base::ToString(dataType).c_str(),
                                  "float16, bfloat16 or float32");
        return ge::GRAPH_FAILED;
    }

    auto storageInputShape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, storageInputShape);
    auto inputShape = storageInputShape->GetStorageShape();
    OP_TILING_CHECK(
        (inputShape.GetDimNum() < GRAD_DIM_NUM),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 input dimNum=(%lu) must be at least 2.",
                                        inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    auto N = inputShape.GetDim(inputShape.GetDimNum() - 2);
    auto M = inputShape.GetDim(inputShape.GetDimNum() - 1);
    int64_t inputNum = 1;
    for (size_t i = 0; i < inputShape.GetDimNum(); ++i) {
        int64_t dim = inputShape.GetDim(i);
        OP_TILING_CHECK(
            dim <= 0 || inputNum > std::numeric_limits<int64_t>::max() / dim,
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 input shape product overflows."),
            return ge::GRAPH_FAILED);
        inputNum *= dim;
    }
    constexpr int64_t kOutputElementBytes = sizeof(float);
    const int64_t maxTensorBytes = std::numeric_limits<int64_t>::max();
    const int64_t sumGradRElements = inputNum / M;
    const int64_t sumGradCElements = inputNum / N;
    const int64_t batchCount = inputNum / (N * M);
    OP_TILING_CHECK(
        inputNum > maxTensorBytes / inputElementBytes || sumGradRElements > maxTensorBytes / kOutputElementBytes ||
            sumGradCElements > maxTensorBytes / kOutputElementBytes ||
            batchCount > maxTensorBytes / kOutputElementBytes,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 tensor byte size overflows."),
        return ge::GRAPH_FAILED);
    tilingData.set_N(N);
    tilingData.set_M(M);
    tilingData.set_batchCount(batchCount);

    OP_TILING_CHECK(UpdateTilingMsg(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "UpdateTilingMsg fail."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateApplyCamePart1Tiling(gert::TilingContext* context, ApplyCamePart1TilingData& tilingData)
{
    if (ShouldUseDirectBatch(tilingData)) {
        tilingData.set_directBatchMode(1);
        tilingData.set_nNormalCoreNum(tilingData.get_N());
        tilingData.set_nTailCoreNum(tilingData.get_N());
        tilingData.set_nCoreNum(1);
        tilingData.set_nLoopNormCore(1);
        tilingData.set_nLoopTailCore(1);
        tilingData.set_mNormalCoreNum(tilingData.get_M());
        tilingData.set_mTailCoreNum(0);
        tilingData.set_mCoreNum(1);
        tilingData.set_mLoopNumCore(CeilDiv(tilingData.get_M(), M_NUM_PER_CORE));
        const int64_t alignedBatchGroupCount = CeilDiv(tilingData.get_batchCount(), DIRECT_BATCH_ALIGNMENT);
        const int64_t candidateCoreNum = std::min(tilingData.get_totalCoreNum(), alignedBatchGroupCount);
        const int64_t groupsPerCore = CeilDiv(alignedBatchGroupCount, candidateCoreNum);
        const int64_t activeCoreNum = CeilDiv(alignedBatchGroupCount, groupsPerCore);
        tilingData.set_batchCoreNum(std::max<int64_t>(1, activeCoreNum));
        tilingData.set_usedCoreNum(tilingData.get_batchCoreNum());
        return CalculateDirectBatchWorkspace(context);
    }

    tilingData.set_directBatchMode(0);
    OP_TILING_CHECK(CalcNdimTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CalcNdimTiling fail."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CalcMdimTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CalcMdimTiling fail."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CalcBatchSplit(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CalcBatchSplit fail."),
                    return ge::GRAPH_FAILED);

    return CalculateApplyCamePart1Workspace(context, tilingData);
}

static void LogApplyCamePart1Tiling(const ApplyCamePart1TilingData& tilingData, ge::DataType dataType)
{
    OP_LOGI("[ApplyCamePart1]", "[nCoreNum]: %ld", tilingData.get_nCoreNum());
    OP_LOGI("[ApplyCamePart1]", "[mCoreNum]: %ld", tilingData.get_mCoreNum());
    OP_LOGI("[ApplyCamePart1]", "[batchCount]: %ld", tilingData.get_batchCount());
    OP_LOGI("[ApplyCamePart1]", "[batchCoreNum]: %ld", tilingData.get_batchCoreNum());
    OP_LOGI("[ApplyCamePart1]", "[usedCoreNum]: %ld", tilingData.get_usedCoreNum());
    OP_LOGI("[ApplyCamePart1]", "[directBatchMode]: %ld", tilingData.get_directBatchMode());
    OP_LOGI("[ApplyCamePart1]", "[totalCoreNum]: %ld", tilingData.get_totalCoreNum());
    OP_LOGI("[ApplyCamePart1]", "[N]: %ld", tilingData.get_N());
    OP_LOGI("[ApplyCamePart1]", "[M]: %ld", tilingData.get_M());
    OP_LOGI("[ApplyCamePart1]", "[nNormalCoreNum]: %ld", tilingData.get_nNormalCoreNum());
    OP_LOGI("[ApplyCamePart1]", "[nTailCoreNum]: %ld", tilingData.get_nTailCoreNum());
    OP_LOGI("[ApplyCamePart1]", "[nLoopTailCore]: %ld", tilingData.get_nLoopTailCore());
    OP_LOGI("[ApplyCamePart1]", "[nLoopNormCore]: %ld", tilingData.get_nLoopNormCore());
    OP_LOGI("[ApplyCamePart1]", "[mNormalCoreNum]: %ld", tilingData.get_mNormalCoreNum());
    OP_LOGI("[ApplyCamePart1]", "[mTailCoreNum]: %ld", tilingData.get_mTailCoreNum());
    OP_LOGI("[ApplyCamePart1]", "[mLoopNumCore]: %ld", tilingData.get_mLoopNumCore());
    OP_LOGI("[ApplyCamePart1]", "[dataType]: %d", static_cast<int32_t>(dataType));
}

ge::graphStatus TilingApplyCamePart1(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingApplyCamePart1 running begin.");
    ApplyCamePart1TilingData tilingData;
    ge::DataType dataType = ge::DT_UNDEFINED;
    OP_TILING_CHECK(InitializeApplyCamePart1Tiling(context, tilingData, dataType) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Initialize tiling data failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CalculateApplyCamePart1Tiling(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Calculate tiling data failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ApplyCamePart1SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 set tilingData fail."),
                    return ge::GRAPH_FAILED);
    LogApplyCamePart1Tiling(tilingData, dataType);

    context->SetBlockDim(tilingData.get_usedCoreNum());
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForApplyCamePart1(gert::TilingParseContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK(
        (totalCoreNum <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingPrepare4ApplyCamePart1 fail to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_TILING_CHECK(
        (ubSizePlatForm <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingPrepare4ApplyCamePart1 fail to get ub size."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 向框架注册入口函数
IMPL_OP_OPTILING(ApplyCamePart1)
    .Tiling(TilingApplyCamePart1)
    .TilingParse<ApplyCamePart1CompileInfo>(TilingPrepareForApplyCamePart1);
} // namespace optiling
