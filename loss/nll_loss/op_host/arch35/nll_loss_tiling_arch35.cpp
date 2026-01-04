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
 * \file nll_loss_tiling_arch35.cpp
 * \brief nll_loss_tiling_arch35
 */

#include "nll_loss_tiling_arch35.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_util.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/nll_loss_tiling_key.h"

using namespace Ops::Base;

namespace optiling {

constexpr uint32_t THREAD_DIM = 512;
constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;
constexpr int64_t SIMT_TILINGKEY = 0;
constexpr int64_t SIMD_TILINGKEY = 1;
constexpr uint32_t REDUCTION_MEAN = 1;
constexpr uint32_t REDUCTION_SUM = 2;
constexpr uint32_t REDUCTION_NONE = 0;
constexpr uint32_t INPUT_ONE_DIM = 1;
constexpr uint32_t INPUT_TWO_DIM = 2;
constexpr uint32_t INPUT_FOUR_DIM = 4;
constexpr uint32_t WEIGHT_INDEX = 2;
constexpr uint32_t DATA_NUM_SINGLE_CORE = 256;
constexpr uint32_t ALIGN_SIZE = 128;
constexpr uint32_t POW = 2;
constexpr uint32_t MIX_MODE_THRESHOD = 524288;
constexpr uint32_t NUMBER_TWO = 2;
constexpr uint32_t NUMBER_THREE = 3;
constexpr uint32_t DCACHE_SIZE = 32 * 1024;
const gert::Shape g_vec_1_shape = {1};

std::map<ge::DataType, uint32_t> xTypeKeyMap = {
    {ge::DT_FLOAT, 0},
    {ge::DT_BF16, 1},
    {ge::DT_FLOAT16, 2},
};

std::map<ge::DataType, uint32_t> targetTypeKeyMap = {
    {ge::DT_INT32, 10},
    {ge::DT_INT64, 20},
    {ge::DT_UINT8, 30},
};

static inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}

template <typename T>
static T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

static void SetTilingData(NLLLossACTilingData& tiling, NLLLossACTilingParam& tilingParam)
{
    tiling.set_ignoreIndex(tilingParam.ignoreIndex);
    tiling.set_reduction(tilingParam.reduction);
    tiling.set_xDimN(tilingParam.xDimN);
    tiling.set_xDims(tilingParam.xDims);
    tiling.set_xDimC(tilingParam.xDimC);
    tiling.set_xDimH(tilingParam.xDimH);
    tiling.set_xDimW(tilingParam.xDimW);
    tiling.set_coreNum(tilingParam.coreNum);
    tiling.set_isWeightPresent(tilingParam.isWeightPresent);
    tiling.set_coreLoopCnt(tilingParam.coreLoopCnt);
    tiling.set_calNumInterBlock(tilingParam.calNumInterBlock);

    tiling.set_mainReduceSize(tilingParam.mainReduceSize);
    tiling.set_usedLogicCore(tilingParam.usedLogicCore);
    tiling.set_loopInCore(tilingParam.loopInCore);
    tiling.set_fullIndex(tilingParam.fullIndex);
    tiling.set_tailSize(tilingParam.tailSize);
    tiling.set_padSize(tilingParam.padSize);
    tiling.set_loopNum1(tilingParam.loopNum1);
    tiling.set_tailNum1(tilingParam.tailNum1);
    tiling.set_tailMoveSize(tilingParam.tailMoveSize);
    tiling.set_tailMainReduceSize(tilingParam.tailMainReduceSize);
    tiling.set_tailRemainSize(tilingParam.tailRemainSize);
    tiling.set_loopNum3(tilingParam.loopNum3);
    tiling.set_tailNum3(tilingParam.tailNum3);
    tiling.set_dealingNumOneCore(tilingParam.dealingNumOneCore);
    tiling.set_frontCore(tilingParam.frontCore);
    tiling.set_dealingNumOnce(tilingParam.dealingNumOnce);

    OP_LOGI(
        "NLLLoss",
        "tilingData: ignoreIndex: %ld, reduction: %u, xDimN: %ld, xDims: %ld, xDimC: %u, xDimH: %ld, xDimW: %ld, "
        "coreNum: %u, isWeightPresent: "
        "%u, coreLoopCnt: %u, calNumInterBlock: %d, "
        "mainReduceSize: %ld, usedLogicCore: %u, loopInCore: %d, fullIndex: %d, tailSize: %d, padSize: %d, loopNum1: "
        "%d, tailNum1: %d, tailMoveSize: %d, "
        "tailMainReduceSize: %d, tailRemainSize: %d, loopNum3: %d, tailNum3: %d",
        tilingParam.ignoreIndex, tilingParam.reduction, tilingParam.xDimN, tilingParam.xDims, tilingParam.xDimC,
        tilingParam.xDimH, tilingParam.xDimW, tilingParam.coreNum, tilingParam.isWeightPresent, tilingParam.coreLoopCnt,
        tilingParam.calNumInterBlock, tilingParam.mainReduceSize, tilingParam.usedLogicCore, tilingParam.loopInCore,
        tilingParam.fullIndex, tilingParam.tailSize, tilingParam.padSize, tilingParam.loopNum1, tilingParam.tailNum1,
        tilingParam.tailMoveSize, tilingParam.tailMainReduceSize, tilingParam.tailRemainSize, tilingParam.loopNum3,
        tilingParam.tailNum3);
}

static uint32_t LargestPowerOfTwo(uint32_t n)
{
    if (n == 1) {
        return 1;
    }

    uint32_t power = 1;
    while (power * POW <= n) {
        power *= POW;
    }
    return power;
}

static void Tiling4NLLLossSimt(gert::TilingContext* context, NLLLossACTilingParam& tilingParams)
{
    OP_LOGD(context->GetNodeName(), "Tiling4NLLLossSimt is begin");
    tilingParams.usedThread = std::min(tilingParams.maxThread, THREAD_DIM);
    tilingParams.tilingKey = SIMT_TILINGKEY;
    if (tilingParams.xDims == INPUT_ONE_DIM || tilingParams.xDims == INPUT_TWO_DIM) {
        tilingParams.coreNumSum = CeilDiv(tilingParams.xDimN, static_cast<int64_t>(tilingParams.usedThread));
    } else if (tilingParams.xDims == INPUT_FOUR_DIM) {
        tilingParams.coreNumSum = CeilDiv(
            tilingParams.xDimN * tilingParams.xDimH * tilingParams.xDimW,
            static_cast<int64_t>(tilingParams.usedThread));
    }
    tilingParams.coreNum = std::min(tilingParams.coreNumSum, tilingParams.maxCoreNum);
    tilingParams.dealingNumOneCore = (tilingParams.xDimN * tilingParams.xDimH * tilingParams.xDimW) / static_cast<int64_t>(tilingParams.coreNum);
    tilingParams.frontCore = (tilingParams.xDimN * tilingParams.xDimH * tilingParams.xDimW) % static_cast<int64_t>(tilingParams.coreNum);
    tilingParams.dealingNumOnce = tilingParams.ubSize / tilingParams.dtypeSize;
    tilingParams.fullIndex = tilingParams.coreNumSum % tilingParams.coreNum;
    tilingParams.calNumInterBlock = LargestPowerOfTwo(tilingParams.coreNum);
}

static void Tiling4NLLLossSimd(gert::TilingContext* context, NLLLossACTilingParam& tilingParams)
{
    tilingParams.tilingKey = SIMD_TILINGKEY;
    tilingParams.mainReduceSize = DATA_NUM_SINGLE_CORE;
    if (tilingParams.xDims == INPUT_ONE_DIM || tilingParams.xDims == INPUT_TWO_DIM) {
        tilingParams.usedLogicCore = CeilDiv(tilingParams.xDimN, tilingParams.mainReduceSize);
        tilingParams.tailSize = tilingParams.xDimN % tilingParams.mainReduceSize;
    } else if (tilingParams.xDims == INPUT_FOUR_DIM) {
        tilingParams.usedLogicCore =
            CeilDiv(tilingParams.xDimN * tilingParams.xDimH * tilingParams.xDimW, tilingParams.mainReduceSize);
        tilingParams.tailSize =
            tilingParams.xDimN * tilingParams.xDimH * tilingParams.xDimW % tilingParams.mainReduceSize;
    }
    if (tilingParams.tailSize == 0) {
        tilingParams.tailSize = tilingParams.mainReduceSize;
    }
    tilingParams.loopInCore = tilingParams.usedLogicCore / tilingParams.maxCoreNum;
    tilingParams.fullIndex = tilingParams.usedLogicCore % tilingParams.maxCoreNum;
    if (tilingParams.fullIndex == 0) {
        tilingParams.loopInCore -= 1;
        tilingParams.fullIndex = tilingParams.maxCoreNum;
    }
    uint32_t remain = tilingParams.tailSize % ALIGN_SIZE;
    tilingParams.padSize = remain == 0 ? 0 : ALIGN_SIZE - remain;
    tilingParams.loopNum1 = std::pow(POW, (int32_t)(log2(tilingParams.loopInCore + 1)));
    tilingParams.tailNum1 = tilingParams.loopInCore + 1 - tilingParams.loopNum1;
    tilingParams.tailMoveSize =
        tilingParams.tailNum1 == 0 ? tilingParams.tailSize : tilingParams.mainReduceSize + tilingParams.tailSize;
    tilingParams.tailMainReduceSize = std::pow(POW, (int32_t)(log2(tilingParams.tailMoveSize + tilingParams.padSize)));
    tilingParams.tailRemainSize = tilingParams.tailMoveSize + tilingParams.padSize - tilingParams.tailMainReduceSize;
    tilingParams.loopNum3 = std::pow(POW, (int32_t)(log2(tilingParams.loopInCore)));
    tilingParams.tailNum3 = tilingParams.loopInCore - tilingParams.loopNum3;

    tilingParams.coreNum =
        tilingParams.usedLogicCore > tilingParams.maxCoreNum ? tilingParams.maxCoreNum : tilingParams.usedLogicCore;
}

static ge::graphStatus Tiling4NLLLossAC(gert::TilingContext* context, uint32_t maxCoreNum, uint32_t maxThread)
{
    OP_LOGD(context->GetNodeName(), "Tiling4NLLLossAC is begin");
    NLLLossACTilingData tiling;
    NLLLossACTilingParam tilingParams;
    size_t usrSize = 0;

    tilingParams.maxCoreNum = maxCoreNum;
    tilingParams.maxThread = maxThread;
    OP_LOGD("NLLLoss", "maxCoreNum: %u", tilingParams.maxCoreNum);

    uint64_t ubSizePlatForm{0};
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    tilingParams.ubSize = ubSizePlatForm - DCACHE_SIZE;
    auto xTensorShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensorShape);
    const gert::Shape& xShape = xTensorShape->GetStorageShape();
    auto targetTensorShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetTensorShape);
    const gert::Shape& targetShape = EnsureNotScalar(targetTensorShape->GetStorageShape());

    if (xShape.GetDimNum() == INPUT_ONE_DIM || xShape.GetDimNum() == INPUT_TWO_DIM) {
        auto nsize = xShape.GetDimNum() == 1 ? 1 : xShape.GetDim(0);
        OP_CHECK_IF(
            (targetShape.GetDim(0) < nsize),
            OP_LOGE(context->GetNodeName(), "The target size should be greater or equal to N!"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            targetShape.GetDimNum() != 1, OP_LOGE(context->GetNodeName(), "Input tensor target should be 1D"),
            return ge::GRAPH_FAILED);
    } else if (xShape.GetDimNum() == INPUT_FOUR_DIM) {
        auto nsize = xShape.GetDim(0);
        auto hsize = xShape.GetDim(NUMBER_TWO);
        auto wsize = xShape.GetDim(NUMBER_THREE);

        OP_CHECK_IF(
            (xShape.GetShapeSize() <= 0),
            OP_LOGE(context->GetNodeName(), "Input x shape size should be greater than 0!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (targetShape.GetDim(0) < nsize),
            OP_LOGE(context->GetNodeName(), "The target dim0 size should be equal to N!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (targetShape.GetDim(1) < hsize),
            OP_LOGE(context->GetNodeName(), "The target dim1 size should be equal to H!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (targetShape.GetDim(NUMBER_TWO) < wsize),
            OP_LOGE(context->GetNodeName(), "The target dim2 size should be equal to W!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            targetShape.GetDimNum() != 3, OP_LOGE(context->GetNodeName(), "Input tensor target should be 3D"),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE("NLLLoss", "Input x dims should be 1D, 2D or 4D!");
        return ge::GRAPH_FAILED;
    }

    auto xTensorType = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensorType);
    auto xType = xTensorType->GetDataType();
    tilingParams.xDims = xShape.GetDimNum();

    if (tilingParams.xDims == INPUT_FOUR_DIM) {
        tilingParams.xDimN = xShape.GetDim(0);
        tilingParams.xDimC = xShape.GetDim(1);
        tilingParams.xDimH = xShape.GetDim(NUMBER_TWO);
        tilingParams.xDimW = xShape.GetDim(NUMBER_THREE);
    } else if (tilingParams.xDims == INPUT_TWO_DIM) {
        tilingParams.xDimN = xShape.GetDim(0);
        tilingParams.xDimC = xShape.GetDim(1);
    } else {
        tilingParams.xDimN = 1;
        tilingParams.xDimC = xShape.GetDim(0);
    }
    OP_CHECK_IF(
        xTypeKeyMap.count(xType) == 0,
        OP_LOGE(
            context->GetNodeName(), "xType not support this data type, just support float32, bfloat16 and float16."),
        return ge::GRAPH_FAILED);
    tilingParams.dtypeSize = sizeof(xType);

    auto targetTensorType = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetTensorType);
    auto targetType = targetTensorType->GetDataType();
    OP_CHECK_IF(
        targetTypeKeyMap.count(targetType) == 0,
        OP_LOGE(context->GetNodeName(), "targetType not support this data type, just support int64, int32 and uint8."),
        return ge::GRAPH_FAILED);

    auto weightShape = context->GetOptionalInputShape(WEIGHT_INDEX);
    if (weightShape == nullptr) {
        OP_LOGI("NLLLoss", "Weight is null");
        tilingParams.isWeightPresent = 0;
    } else {
        OP_LOGI("NLLLoss", "Weight is not null");
        tilingParams.isWeightPresent = 1;
    }

    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto* reduction = attrs->GetAttrPointer<char>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, reduction);
    if (strcmp(reduction, "none") == 0) {
        tilingParams.reduction = REDUCTION_NONE;
    } else if (strcmp(reduction, "sum") == 0) {
        tilingParams.reduction = REDUCTION_SUM;
    } else if (strcmp(reduction, "mean") == 0) {
        tilingParams.reduction = REDUCTION_MEAN;
    } else {
        OP_LOGE(context->GetNodeName(), "Invalid reduction mode, only support mean, sum and none!");
        return ge::GRAPH_FAILED;
    }

    const auto* ignoreIndex = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ignoreIndex);
    tilingParams.ignoreIndex = *ignoreIndex;

    if ((tilingParams.xDims == 1) || (tilingParams.reduction == 0) ||
        ((tilingParams.xDimN < MIX_MODE_THRESHOD) && (tilingParams.xDims == INPUT_TWO_DIM)) ||
        (tilingParams.xDims == INPUT_FOUR_DIM)) {
        Tiling4NLLLossSimt(context, tilingParams);
    } else {
        Tiling4NLLLossSimd(context, tilingParams);
    }

    SetTilingData(tiling, tilingParams);
    context->SetBlockDim(tilingParams.coreNum);
    const uint64_t tilingKeyTemp = GET_TPL_TILING_KEY(tilingParams.tilingKey, tilingParams.xDims, tilingParams.reduction);
    context->SetTilingKey(tilingKeyTemp);
    context->SetLocalMemorySize(ubSizePlatForm - DCACHE_SIZE);
    context->SetScheduleMode(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    if (tilingParams.reduction == REDUCTION_SUM || tilingParams.reduction == REDUCTION_MEAN) {
        usrSize = tilingParams.maxCoreNum * sizeof(float) * NUMBER_TWO;
    }
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ASCENDC_TOOLS_WORKSPACE + usrSize;
    OP_LOGD(context->GetNodeName(), "Tiling4NLLLossAC is end");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NLLLossTiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "NLLLossTiling running begin");
    auto compileInfo = reinterpret_cast<const NLLLossCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context->GetNodeName(), "NLLLossTiling tik compile_info is Null! Running simt tiling!");
    uint32_t maxCoreNum = compileInfo->maxCoreNum;
    uint32_t maxThread = compileInfo->maxThread;
    return Tiling4NLLLossAC(context, maxCoreNum, maxThread);
}

static ge::graphStatus TilingPrepareForNLLLoss(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare is running.");
    auto compileInfo = context->GetCompiledInfo<NLLLossCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context->GetNodeName(), "AscendC SIMT Graph Success!");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->maxCoreNum <= 0), OP_LOGE(context->GetNodeName(), "The core num is invalid."),
        return ge::GRAPH_FAILED);
    compileInfo->maxThread = 2048U;
    OP_CHECK_IF(
        (compileInfo->maxThread <= 0), OP_LOGE(context->GetNodeName(), "The thread num is invalid."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
// register tiling inferface of the Nllloss op
IMPL_OP_OPTILING(NLLLoss).Tiling(NLLLossTiling).TilingParse<NLLLossCompileInfo>(TilingPrepareForNLLLoss);

} // namespace optiling