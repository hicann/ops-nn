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
#include "../../op_kernel/arch35/apply_came_part1_tiling_key.h"

namespace optiling {

const std::string OP_NAME = "ApplyCamePart1";

constexpr int32_t GRAD_DIM_NUM = 2;
constexpr int64_t GAIN_COEFFICIENT = 2;

constexpr int64_t N_NUM_PER_CORE = 64;
constexpr int64_t M_NUM_PER_CORE = 64;
constexpr int64_t BLOCK_SIZE = 8;
constexpr int32_t BATCH_MODE = 1;
constexpr uint64_t FLOAT16_TILING_KEY = 1;
constexpr uint64_t BF16_TILING_KEY = 27;

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
    context->SetScheduleMode(BATCH_MODE);
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

static inline ge::graphStatus CalcMdimTiling(const gert::TilingContext* context, ApplyCamePart1TilingData& tilingData)
{
    OP_LOGD(context->GetNodeName(), "TilingApplyCamePart1 Enter CalcMdimTiling.");
    int64_t mNormalCoreNum = 1;
    int64_t mTailCoreNum = 0;
    int64_t mCoreNum = 1;
    int64_t mLoopNumCore = 1;
    auto upperBound = tilingData.get_totalCoreNum();
    auto lowerBound = tilingData.get_totalCoreNum() / 2;
    if (((tilingData.get_nCoreNum() <= upperBound) && (tilingData.get_nCoreNum() > lowerBound)) ||
        (tilingData.get_M() <= M_NUM_PER_CORE) || (tilingData.get_nNormalCoreNum() <= N_NUM_PER_CORE)) {
        mNormalCoreNum = tilingData.get_M();
        mLoopNumCore = CeilDiv(mNormalCoreNum, M_NUM_PER_CORE);
        tilingData.set_mLoopNumCore(mLoopNumCore);
        tilingData.set_mNormalCoreNum(mNormalCoreNum);
        tilingData.set_mTailCoreNum(0);
        tilingData.set_mCoreNum(1);
    } else {
        // CeilMul currently keeps nCoreNum above totalCoreNum / 2 whenever
        // this branch could be selected. The m-axis split remains explicit
        // here so the layout stays extensible if the n-axis policy changes.
        int64_t mShapeLength = tilingData.get_M();
        OP_TILING_CHECK(
            tilingData.get_nCoreNum() == 0,
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingApplyCamePart1 get_nCoreNum is 0."),
            return ge::GRAPH_FAILED);
        mCoreNum = tilingData.get_totalCoreNum() / tilingData.get_nCoreNum(); // 向下取整
        OP_TILING_CHECK(mCoreNum == 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingApplyCamePart1 mCoreNum is 0."),
                        return ge::GRAPH_FAILED);
        mNormalCoreNum = std::max(CeilAlign(mShapeLength / mCoreNum, M_NUM_PER_CORE), static_cast<int64_t>(1));
        mTailCoreNum = mShapeLength % mNormalCoreNum;
        mLoopNumCore = CeilDiv(mNormalCoreNum, M_NUM_PER_CORE);
        tilingData.set_mLoopNumCore(mLoopNumCore);
        tilingData.set_mNormalCoreNum(mNormalCoreNum);
        tilingData.set_mTailCoreNum(mTailCoreNum);
        tilingData.set_mCoreNum(mCoreNum);
    }
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

static bool CheckParamsShape(const gert::TilingContext* context)
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
                    return false);

    auto epsShapeInput = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epsShapeInput);
    auto epsShape = epsShapeInput->GetStorageShape();
    const bool epsIsOneElement = epsShape.GetDimNum() == 0 ||
                                 (epsShape.GetDimNum() == 1 &&
                                  (epsShape.GetDim(0) == 1 || epsShape.GetDim(0) == ge::UNKNOWN_DIM));
    OP_TILING_CHECK(!epsIsOneElement,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "eps must be a scalar or a 1-element tensor, but got %s",
                                                    Ops::Base::ToString(epsShape).c_str()),
                    return false);

    return true;
}

static ge::graphStatus CalculateApplyCamePart1Workspace(gert::TilingContext* context,
                                                        ApplyCamePart1TilingData& tilingData)
{
    const auto usedCoreNum = tilingData.get_nCoreNum() * tilingData.get_mCoreNum();
    tilingData.set_usedCoreNum(usedCoreNum);
    const uint64_t maxSize = std::numeric_limits<size_t>::max();
    const uint64_t nCoreNum = static_cast<uint64_t>(tilingData.get_nCoreNum());
    const uint64_t mCoreNum = static_cast<uint64_t>(tilingData.get_mCoreNum());
    const uint64_t mLoopNumCore = static_cast<uint64_t>(tilingData.get_mLoopNumCore());
    OP_TILING_CHECK(nCoreNum == 0 || mCoreNum == 0 || tilingData.get_nLoopNormCore() <= 0 || mLoopNumCore == 0 ||
                        nCoreNum - 1 > maxSize / static_cast<uint64_t>(tilingData.get_nLoopNormCore()),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 workspace size overflows."),
                    return ge::GRAPH_FAILED);
    const uint64_t normalLoopCount = (nCoreNum - 1U) * static_cast<uint64_t>(tilingData.get_nLoopNormCore());
    const uint64_t tailLoopCount = static_cast<uint64_t>(tilingData.get_nLoopTailCore());
    OP_TILING_CHECK(normalLoopCount > maxSize - tailLoopCount,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 workspace size overflows."),
                    return ge::GRAPH_FAILED);
    const uint64_t nLoopCount = normalLoopCount + tailLoopCount;
    OP_TILING_CHECK(nLoopCount > maxSize / mCoreNum,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 workspace size overflows."),
                    return ge::GRAPH_FAILED);
    const uint64_t rElements = nLoopCount * mCoreNum;
    OP_TILING_CHECK(rElements > maxSize / mLoopNumCore,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 workspace size overflows."),
                    return ge::GRAPH_FAILED);
    const uint64_t rcElements = rElements * mLoopNumCore;
    constexpr uint64_t rcSlotSize = 8;
    const uint64_t rcReducePartialCount = (static_cast<uint64_t>(tilingData.get_N()) + 8192U - 1U) / 8192U;
    OP_TILING_CHECK(rcElements < rcReducePartialCount,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "ApplyCamePart1 workspace cannot hold scalar reduction partials."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        rcElements > maxSize / (M_NUM_PER_CORE * sizeof(float)) ||
            rElements > maxSize / (N_NUM_PER_CORE * sizeof(float)) || rcElements > maxSize - 127U ||
            rElements > maxSize - 127U,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 workspace byte size overflows."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rcElements > maxSize / rcSlotSize || rcElements * rcSlotSize > maxSize - 127U,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 RC slot size overflows."),
                    return ge::GRAPH_FAILED);
    const size_t rcStorageElements = static_cast<size_t>(rcElements * rcSlotSize);
    const size_t rcAlignedElements = CeilAlign(rcStorageElements, 128);
    OP_TILING_CHECK(
        rcAlignedElements > maxSize / (2U * sizeof(float)),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 RC workspace byte size overflows."),
        return ge::GRAPH_FAILED);
    const size_t rcWorkspaceSize = rcAlignedElements * 2U * sizeof(float);
    const size_t rWorkspaceSize = CeilAlign(static_cast<size_t>(rElements * N_NUM_PER_CORE * sizeof(float)), 512);
    const size_t cWorkspaceSize = CeilAlign(static_cast<size_t>(rcElements * M_NUM_PER_CORE * sizeof(float)), 512);
    const size_t syncWorkspaceSize = static_cast<size_t>(tilingData.get_totalCoreNum()) * 32U + 32U;
    OP_TILING_CHECK(
        rcWorkspaceSize > maxSize - rWorkspaceSize || rWorkspaceSize > maxSize - rcWorkspaceSize - cWorkspaceSize ||
            syncWorkspaceSize > maxSize - rcWorkspaceSize - rWorkspaceSize - cWorkspaceSize,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1 total workspace byte size overflows."),
        return ge::GRAPH_FAILED);
    const size_t userSize = syncWorkspaceSize + rcWorkspaceSize + rWorkspaceSize + cWorkspaceSize;
    constexpr size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    userWorkspaceSize[0] = userSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InitializeApplyCamePart1Tiling(gert::TilingContext* context,
                                                      ApplyCamePart1TilingData& tilingData, ge::DataType& dataType)
{
    OP_TILING_CHECK(!CheckParamsShape(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyCamePart1Tiling check shape failed."),
                    return ge::GRAPH_FAILED);

    auto input = context->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, input);
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
    tilingData.set_N(N);
    tilingData.set_M(M);
    tilingData.set_batchCount(inputNum / (N * M));

    OP_TILING_CHECK(UpdateTilingMsg(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "UpdateTilingMsg fail."),
                    return ge::GRAPH_FAILED);

    dataType = input->GetDataType();
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        OP_LOGI(context->GetNodeName(), "Current dataType is [float16] or [bfloat16].");
    } else if (dataType == ge::DT_FLOAT) {
        OP_LOGI(context->GetNodeName(), "Current dataType is [float32].");
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "grad", Ops::Base::ToString(dataType).c_str(),
                                  "float16, bfloat16 or float32");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateApplyCamePart1Tiling(gert::TilingContext* context, ApplyCamePart1TilingData& tilingData)
{
    OP_TILING_CHECK(CalcNdimTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CalcNdimTiling fail."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CalcMdimTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CalcMdimTiling fail."),
                    return ge::GRAPH_FAILED);

    return CalculateApplyCamePart1Workspace(context, tilingData);
}

static void LogApplyCamePart1Tiling(const ApplyCamePart1TilingData& tilingData, ge::DataType dataType)
{
    OP_LOGI("[ApplyCamePart1]", "[nCoreNum]: %ld", tilingData.get_nCoreNum());
    OP_LOGI("[ApplyCamePart1]", "[mCoreNum]: %ld", tilingData.get_mCoreNum());
    OP_LOGI("[ApplyCamePart1]", "[usedCoreNum]: %ld", tilingData.get_usedCoreNum());
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
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));
    // Keep the host selector aligned with the canonical dtype keys used by the
    // Ascend950 kernel objects shipped in the package.
    uint64_t canonicalKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        canonicalKey = FLOAT16_TILING_KEY;
    } else if (dataType == ge::DT_BF16) {
        canonicalKey = BF16_TILING_KEY;
    }
    context->SetTilingKey(canonicalKey);
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
