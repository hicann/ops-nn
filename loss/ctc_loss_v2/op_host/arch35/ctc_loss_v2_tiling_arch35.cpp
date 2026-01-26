/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ctc_loss_v2_tiling_arch35.cpp
 * \brief ctc_loss_v2_tiling_arch35
 */
#include <cmath>
#include <climits>
#include "log/log.h"
#include "util/math_util.h"
#include "loss/ctc_loss_v2/op_kernel/arch35/ctc_loss_v2_tiling_key.h"
#include "ctc_loss_v2_tiling_arch35.h"
#include "register/op_impl_registry.h"

using namespace std;
using namespace ge;
using namespace AscendC;

namespace optiling {
const std::string OP_NAME = "CTCLossV2";
const std::string OP_GRAD_NAME = "CTCLossV2Grad";
static const size_t LOG_PROBS_IDX = 0;
static const size_t TARGETS_IDX = 1;
static const size_t INPUT_LENGTHS_IDX = 2;
static const size_t TARGET_LENGTHS_IDX = 3;

static const size_t INT64_SIZE = 8;
static const size_t DIM0 = 0;
static const size_t DIM1 = 1;
static const size_t DIM2 = 2;
static const size_t DIM3 = 3;
static const size_t FLOAT32_SIZE = 4;
static const size_t BLOCK = 32;
constexpr int32_t SIMT_RESERVED_SIZE = 32 * 1024;
constexpr int32_t THREAD_NUM_64 = 64;
constexpr int32_t THREAD_NUM_512 = 512;
constexpr int32_t THREAD_NUM_1024 = 1024;
constexpr uint32_t SCHEDULE_MODE = 1; // batchmode

static const vector<DataType> SUPPORTED_DTYPE = {DT_FLOAT, DT_FLOAT16, DT_BF16};

inline static ge::graphStatus CTCLossV2SetTilingData(
    gert::TilingContext* context, CTCLossV2TilingData4AscendC& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static inline void PrintTilingData(CTCLossV2TilingData4AscendC& tilingData)
{
    OP_LOGI("CTCLossV2", "maxInputLength: %ld", tilingData.get_maxInputLength());
    OP_LOGI("CTCLossV2", "maxTargetLength: %ld", tilingData.get_maxTargetLength());
    OP_LOGI("CTCLossV2", "lpInputStride: %ld", tilingData.get_lpInputStride());
    OP_LOGI("CTCLossV2", "lpBatchStride: %ld", tilingData.get_lpBatchStride());
    OP_LOGI("CTCLossV2", "lpCharStride: %ld", tilingData.get_lpCharStride());
    OP_LOGI("CTCLossV2", "laBatchStride: %ld", tilingData.get_laBatchStride());
    OP_LOGI("CTCLossV2", "laInputStride: %ld", tilingData.get_laInputStride());
    OP_LOGI("CTCLossV2", "laTargetStride: %ld", tilingData.get_laTargetStride());
    OP_LOGI("CTCLossV2", "tgTargetStride: %ld", tilingData.get_tgTargetStride());
    OP_LOGI("CTCLossV2", "batchSize: %ld", tilingData.get_batchSize());
    OP_LOGI("CTCLossV2", "blank: %ld", tilingData.get_blank());
    OP_LOGI("CTCLossV2", "blockDimX: %ld", tilingData.get_blockDimX());
    OP_LOGI("CTCLossV2", "blockDimY: %ld", tilingData.get_blockDimY());
    OP_LOGI("CTCLossV2", "targetsDim: %ld", tilingData.get_targetsDim());
    OP_LOGI("CTCLossV2", "tgBatchStride: %ld", tilingData.get_tgBatchStride());
    OP_LOGI("CTCLossV2", "workspaceSize: %ld", tilingData.get_workspaceSize());
    OP_LOGI("CTCLossV2", "gridY: %ld", tilingData.get_gridY());
    OP_LOGI("CTCLossV2", "usedCoreNum: %ld", tilingData.get_usedCoreNum());
}

template <typename T>
inline static int64_t GetMaxTargetLength(const gert::Tensor* targetLengths, int64_t batchSize)
{
    int64_t maxTargetLength = 0;
    auto targetLengthsData = targetLengths->GetData<T>();
    for (int64_t i = 0; i < batchSize; i++) {
        if (maxTargetLength < targetLengthsData[i]) {
            maxTargetLength = targetLengthsData[i];
        }
    }
    return maxTargetLength;
}

template <typename T>
inline static bool IsInputeLengthValid(const gert::Tensor* inputLengths, int64_t batchSize, int64_t maxInputLength)
{
    auto inputLengthsData = inputLengths->GetData<T>();
    for (int64_t b = 0; b < batchSize; b++) {
        if (inputLengthsData[b] > maxInputLength) {
            return false;
        }
    }
    return true;
}

inline static bool IsLargeSize(
    int64_t logProbsDimSize0, int64_t logProbsDimSize1, int64_t logProbsDimSize2, int64_t maxTargetLength)
{
    return (logProbsDimSize0 > INT32_MAX) || (logProbsDimSize1 > INT32_MAX) || (logProbsDimSize2 > INT32_MAX) ||
           (maxTargetLength > INT32_MAX) || (logProbsDimSize0 * logProbsDimSize1 > INT32_MAX) ||
           (logProbsDimSize0 * logProbsDimSize1 * logProbsDimSize2 > INT32_MAX) ||
           (logProbsDimSize0 * logProbsDimSize1 * (DIM2 * maxTargetLength + DIM1) > INT32_MAX);
}

ge::graphStatus Tiling4CTCLossV2ForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4CTCLossV2ForAscendC running begin");
    CTCLossV2TilingData4AscendC tilingData;
    // set lpInputStride, lpBatchStride, lpCharStride
    auto logProbs = context->GetInputTensor(LOG_PROBS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, logProbs);
    auto logProbsDesc = context->GetInputDesc(LOG_PROBS_IDX);
    DataType logProbsDtype = logProbsDesc->GetDataType();
    OP_CHECK_IF(
        find(SUPPORTED_DTYPE.begin(), SUPPORTED_DTYPE.end(), logProbsDtype) == SUPPORTED_DTYPE.end(),
        OP_LOGE(
            context->GetNodeName(), "CTCLossV2 dtype is not support, should be one of {float32, float16, bfloat16}."),
        return ge::GRAPH_FAILED);
    auto logProbsShape = logProbs->GetStorageShape();
    size_t logProbsDimNum = logProbsShape.GetDimNum();
    int64_t logProbsDimSize0 = 0;
    int64_t logProbsDimSize1 = 0;
    int64_t logProbsDimSize2 = 0;
    // log_prob = (T, C)
    if (logProbsDimNum == DIM2) {
        logProbsDimSize0 = logProbsShape.GetDim(DIM0);
        logProbsDimSize1 = 1; // N = 1
        logProbsDimSize2 = logProbsShape.GetDim(DIM1);
        // log_prob = (T, N, C)
    } else if (logProbsDimNum == DIM3) {
        logProbsDimSize0 = logProbsShape.GetDim(DIM0);
        logProbsDimSize1 = logProbsShape.GetDim(DIM1);
        logProbsDimSize2 = logProbsShape.GetDim(DIM2);
    }
    tilingData.set_lpInputStride(logProbsDimSize1 * logProbsDimSize2);
    tilingData.set_lpBatchStride(logProbsDimSize2);
    tilingData.set_lpCharStride(1);
    // set batchSize
    int64_t batchSize = logProbsDimSize1;
    tilingData.set_batchSize(batchSize);
    // set maxInputLength
    int64_t maxInputLength = logProbsDimSize0;
    auto inputLengths = context->GetInputTensor(INPUT_LENGTHS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLengths);
    auto inputLengthsDesc = context->GetInputDesc(INPUT_LENGTHS_IDX);
    DataType inputLengthsDtype = inputLengthsDesc->GetDataType();
    if (inputLengthsDtype == DT_INT32 && !IsInputeLengthValid<int32_t>(inputLengths, batchSize, maxInputLength)) {
        return ge::GRAPH_FAILED;
    } else if (
        inputLengthsDtype == DT_INT64 && !IsInputeLengthValid<int64_t>(inputLengths, batchSize, maxInputLength)) {
        return ge::GRAPH_FAILED;
    }
    tilingData.set_maxInputLength(maxInputLength);
    // set maxTargetLength
    int64_t maxTargetLength = 0;
    auto targetLengths = context->GetInputTensor(TARGET_LENGTHS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetLengths);
    auto targetLengthsDesc = context->GetInputDesc(TARGET_LENGTHS_IDX);
    DataType targetLengthsDtype = targetLengthsDesc->GetDataType();
    if (targetLengthsDtype == DT_INT32) {
        maxTargetLength = GetMaxTargetLength<int32_t>(targetLengths, batchSize);
    } else {
        maxTargetLength = GetMaxTargetLength<int64_t>(targetLengths, batchSize);
    }
    tilingData.set_maxTargetLength(maxTargetLength);

    // set targetDim、tgBatchStride
    int64_t targetsDim = 0;
    int64_t tgBatchStride = 0;
    auto targets = context->GetInputTensor(TARGETS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, targets);
    auto targetsShape = targets->GetStorageShape();
    targetsDim = targetsShape.GetDimNum();
    tilingData.set_targetsDim(targetsDim);
    if (targetsDim == DIM1) {
        tgBatchStride = 1;
    } else if (targetsDim == DIM2) {
        tgBatchStride = targetsShape.GetDim(DIM1);
    }
    tilingData.set_tgBatchStride(tgBatchStride);
    // set laBatchStride, laInputStride, laTargetStride
    int64_t dim1 = logProbsDimSize0;
    int64_t dim2 = 2 * maxTargetLength + 1;
    tilingData.set_laBatchStride(dim1 * dim2);
    tilingData.set_laInputStride(dim2);
    tilingData.set_laTargetStride(1);
    // set tgTargetStride
    int64_t tgTargetStride = 1;
    tilingData.set_tgTargetStride(tgTargetStride);
    // set blank
    auto attrs = context->GetAttrs();
    const int64_t* blankPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, blankPtr);
    int64_t blank = *blankPtr;
    tilingData.set_blank(blank);

    // set Local Memory Size and Total Core Num
    uint64_t ubSize = 0;
    auto compileInfo = reinterpret_cast<const CTCLossV2ForCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    context->SetLocalMemorySize(ubSize - SIMT_RESERVED_SIZE);
    context->SetScheduleMode(SCHEDULE_MODE);

    // set TilingKey
#ifdef DAVID_FPGA
    int64_t MAX_THREAD = THREAD_NUM_64;
#else
    int64_t MAX_THREAD = THREAD_NUM_512;
#endif
    size_t EXTRA_WORKSPACE_SIZE = 0;
    uint64_t threadTypeInt32 = CTC_LOSS_V2_TPL_KEY_FALSE;
    uint64_t isFP32 = CTC_LOSS_V2_TPL_KEY_FALSE;
    if (logProbsDtype == DT_FLOAT) {
        isFP32 = CTC_LOSS_V2_TPL_KEY_TRUE;
    } else {
        EXTRA_WORKSPACE_SIZE =
            Ops::Base::CeilAlign((DIM2 * maxTargetLength + 1) * FLOAT32_SIZE * batchSize * logProbsDimSize0, BLOCK);
    }
    if (!IsLargeSize(logProbsDimSize0, logProbsDimSize1, logProbsDimSize2, maxTargetLength)) {
        MAX_THREAD = THREAD_NUM_1024;
        threadTypeInt32 = CTC_LOSS_V2_TPL_KEY_TRUE;
    }
    int32_t blockDimX = 0;
    int32_t blockDimY = 0;
    int32_t threadsTarget = MAX_THREAD;
    int32_t threadsBatch = 0;
    if (threadsTarget >= dim2) {
        threadsTarget = DIM2 * maxTargetLength + 1;
    }
    int32_t threadsMaxBatch = MAX_THREAD / threadsTarget;
    OP_CHECK_IF(
        totalCoreNum == 0, OP_LOGE("Tiling4CTCLossV2ForAscendC", "totalCoreNum is zero."), return ge::GRAPH_FAILED);
    int32_t threadsPerBlockBatch = (logProbsDimSize1 + totalCoreNum - 1) / totalCoreNum;
    blockDimX = threadsTarget;
    blockDimY = threadsPerBlockBatch < threadsMaxBatch ? threadsPerBlockBatch : threadsMaxBatch;
    threadsBatch = blockDimY;
    tilingData.set_blockDimX(blockDimX);
    tilingData.set_blockDimY(blockDimY);
    OP_LOGI(context->GetNodeName(), "isFP32 is %lu, threadTypeInt32 is %lu", isFP32, threadTypeInt32);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(isFP32, threadTypeInt32);
    OP_LOGI(context->GetNodeName(), "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    context->SetScheduleMode(1);
    int32_t gridY = (batchSize + threadsBatch - 1) / threadsBatch;
    tilingData.set_gridY(gridY);
    int32_t usedCoreNum = min(gridY, totalCoreNum);
    tilingData.set_usedCoreNum(usedCoreNum);
    context->SetBlockDim(usedCoreNum);

    // set workspace
    size_t SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYSTEM_WORKSPACE_SIZE + EXTRA_WORKSPACE_SIZE;
    tilingData.set_workspaceSize(currentWorkspace[0]);
    OP_CHECK_IF(
        CTCLossV2SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "CTCLossV2SetTilingData set tiling data fail."), return ge::GRAPH_FAILED);
    PrintTilingData(tilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetTotalCoreNum(gert::TilingParseContext* context)
{
    auto compileInfo4AscendC = context->GetCompiledInfo<CTCLossV2ForCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo4AscendC);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo4AscendC->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo4AscendC->totalCoreNum <= 0),
        OP_LOGE(context->GetNodeName(), "TilingPrepare4CTCLossV2 fail to get core num."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4CTCLossV2(gert::TilingContext* context)
{
    OP_LOGD(OP_NAME.c_str(), "Tiling4CTCLossV2 rt2.0 is running.");

    return Tiling4CTCLossV2ForAscendC(context);
}

static ge::graphStatus TilingPrepare4CTCLossV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CTCLossV2)
    .InputsDataDependency({INPUT_LENGTHS_IDX, TARGET_LENGTHS_IDX})
    .Tiling(Tiling4CTCLossV2)
    .TilingParse<CTCLossV2CompileInfo>(TilingPrepare4CTCLossV2);
} // namespace optiling