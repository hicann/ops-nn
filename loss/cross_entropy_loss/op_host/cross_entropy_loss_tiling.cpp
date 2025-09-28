/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file cross_entropy_loss_tiling.cc
 * \brief
 */

#include "cross_entropy_loss_tiling.h"

namespace optiling {
constexpr uint32_t INPUT_DATA_IDX = 0;
constexpr uint32_t INPUT_TARGET_IDX = 1;
constexpr uint32_t INPUT_WEIGHT_IDX = 2;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t NUM_1 = 1;
constexpr uint32_t NUM_2 = 2;
constexpr uint32_t NUM_3 = 3;
constexpr uint32_t NUM_4 = 4;
constexpr uint32_t NUM_8 = 8;
constexpr uint32_t NUM_64 = 64;
constexpr uint32_t DOUBLE_BUFFER = 1;
constexpr uint32_t ATTR_REDUCTION_IDX = 0;
constexpr uint32_t ATTR_IGNORE_INDEX_IDX = 1;
constexpr uint32_t ATTR_LABEL_SMOOTHING_IDX = 2;
constexpr uint32_t BLOCK_32 = 32;
constexpr uint32_t BLOCK_512 = 512;
constexpr uint32_t DTYPE_LEN_FP32 = 4;
constexpr uint32_t DTYPE_LEN_HALF = 2;
constexpr uint32_t FP32_128_REPEAT = 8192; // 64 * 128
constexpr uint32_t FP32_PER_REPEAT = 64;
constexpr uint32_t REDUCTION_NONE = 0;
constexpr uint32_t REDUCTION_MEAN = 1;
constexpr uint32_t REDUCTION_SUM = 2;
constexpr int64_t DEFAULT_IGNORE_INDEX = -100LL;
uint32_t WORKSPACE_16MB_SIZE = 16 * 1024 * 1024;

// tiling key
constexpr uint32_t TILING_KEY_BF16 = 1;
constexpr uint32_t TILING_KEY_FP16 = 2;
constexpr uint32_t TILING_KEY_FP32 = 3;

CrossEntropyLossTilingData crossEntropyLossTiling;

static void PrintInfo(const gert::TilingContext* context)
{
    auto nodeName = context;
    OP_LOGD(nodeName, ">>>>>>>>>>>>> cross_entropy_loss tiling data begin <<<<<<<<<<<<<");
    OP_LOGD(nodeName, "targetNum = %lu.", crossEntropyLossTiling.get_targetNum());
    OP_LOGD(nodeName, "frontCoreNum = %lu.", crossEntropyLossTiling.get_frontCoreNum());
    OP_LOGD(nodeName, "tailCoreNum = %lu.", crossEntropyLossTiling.get_tailCoreNum());
    OP_LOGD(nodeName, "frontBatchNum = %lu.", crossEntropyLossTiling.get_frontBatchNum());
    OP_LOGD(nodeName, "tailBatchNum = %lu.", crossEntropyLossTiling.get_tailBatchNum());
    OP_LOGD(nodeName, "inputUbSize = %lu.", crossEntropyLossTiling.get_inputUbSize());
    OP_LOGD(nodeName, "castTmpBufByte = %lu.", crossEntropyLossTiling.get_castTmpBufByte());
    OP_LOGD(nodeName, "lnTmpBufSize = %lu.", crossEntropyLossTiling.get_lnTmpBufSize());
    OP_LOGD(nodeName, "weightTmpBufSize = %lu.", crossEntropyLossTiling.get_weightTmpBufSize());
    OP_LOGD(nodeName, "totalTmpBufByte = %lu.", crossEntropyLossTiling.get_totalTmpBufByte());
    OP_LOGD(nodeName, "ubLoopNum = %lu.", crossEntropyLossTiling.get_ubLoopNum());
    OP_LOGD(nodeName, "ubTailNum = %lu.", crossEntropyLossTiling.get_ubTailNum());
    OP_LOGD(nodeName, "vecLoopNum = %lu.", crossEntropyLossTiling.get_vecLoopNum());
    OP_LOGD(nodeName, "vecTailNum = %lu.", crossEntropyLossTiling.get_vecTailNum());
    OP_LOGD(nodeName, "tailVecLoopNum = %lu.", crossEntropyLossTiling.get_tailVecLoopNum());
    OP_LOGD(nodeName, "tailVecTailNum = %lu.", crossEntropyLossTiling.get_tailVecTailNum());
    OP_LOGD(nodeName, "defaultWeight = %u.", crossEntropyLossTiling.get_defaultWeight());
    OP_LOGD(nodeName, ">>>>>>>>>>>>> cross_entropy_loss tiling data end <<<<<<<<<<<<<");
}

template <typename T>
static inline uint64_t CeilDiv(uint64_t num, T div)
{
    return div == 0 ? div : (num + div - 1) / div;
}

template <typename T>
static inline uint64_t GetDiv(uint64_t num, T div)
{
    return div == 0 ? div : num / div;
}

template <typename T>
static inline uint64_t GetRemainder(uint64_t num, T div)
{
    return div == 0 ? div : num % div;
}

static void SplitByRepeat(uint64_t len, uint64_t& loopNum, uint64_t& tailNum)
{
    if (len >= FP32_128_REPEAT) {
        loopNum = CeilDiv(len, FP32_128_REPEAT);
        tailNum = GetRemainder(len, FP32_128_REPEAT);
        if (tailNum != 0) {
            loopNum -= 1;
        }
    } else {
        loopNum = 0;
        tailNum = len;
    }
}

static bool IsRegbaseSocVersion4CELOSS(platform_ascendc::SocVersion version)
{
    const static std::set<platform_ascendc::SocVersion> regbaseSocVersions = {
        platform_ascendc::SocVersion::ASCEND910_95
    };

    return regbaseSocVersions.find(version) != regbaseSocVersions.end();
}

bool IsRegbaseSocVersion4CELOSS(const gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    return IsRegbaseSocVersion4CELOSS(socVersion);
}

static void CoresSplitTiling(gert::TilingContext* context)
{
    uint64_t batchSize = context->GetInputShape(INPUT_DATA_IDX)->GetStorageShape().GetDim(DIM_0);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();

    uint64_t frontCoreNum;
    uint64_t tailCoreNum;
    uint64_t frontBatchNum;
    uint64_t tailBatchNum;
    uint64_t usedCoreNum;
    if (batchSize < totalCoreNum) {
        frontCoreNum = batchSize;
        tailCoreNum = 0;
        frontBatchNum = NUM_1;
        tailBatchNum = 0;
        usedCoreNum = frontCoreNum;
    } else {
        uint64_t batchRemainder = GetRemainder(batchSize, totalCoreNum);
        uint64_t batchDiv = GetDiv(batchSize, totalCoreNum);
        frontCoreNum = batchRemainder == 0 ? totalCoreNum : batchRemainder;
        frontBatchNum = CeilDiv(batchSize, totalCoreNum);
        tailCoreNum = totalCoreNum - frontCoreNum;
        tailBatchNum = batchRemainder == 0 ? 0 : batchDiv;
        usedCoreNum = totalCoreNum;
    }
    crossEntropyLossTiling.set_frontCoreNum(frontCoreNum);
    crossEntropyLossTiling.set_tailCoreNum(tailCoreNum);
    crossEntropyLossTiling.set_frontBatchNum(frontBatchNum);
    crossEntropyLossTiling.set_tailBatchNum(tailBatchNum);
    context->SetBlockDim(usedCoreNum);
}

static void UbSplitTiling(const gert::TilingContext* context)
{
    uint64_t maxUbSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);
    auto inputDtype = context->GetInputDesc(INPUT_DATA_IDX)->GetDataType();
    // ub align
    uint64_t frontBatchNum = crossEntropyLossTiling.get_frontBatchNum();
    uint64_t lnTmpBufByte = CeilDiv(frontBatchNum * DTYPE_LEN_FP32, BLOCK_32) * BLOCK_32;
    uint64_t weightTmpBufByte = lnTmpBufByte;
    uint64_t smoothingLossBufByte = lnTmpBufByte;
    uint64_t reduceCalcTmpBufByte = BLOCK_32;
    uint64_t reduceResTmpBufByte = BLOCK_32;
    uint64_t workLocalByte = FP32_128_REPEAT / NUM_8 * DTYPE_LEN_FP32; // 4096
    uint64_t reserveUb = lnTmpBufByte + weightTmpBufByte + smoothingLossBufByte + reduceCalcTmpBufByte +
                         reduceResTmpBufByte + workLocalByte;
    // label smoothing
    bool needWeightBuf =
        crossEntropyLossTiling.get_labelSmoothing() > 0 && context->GetOptionalInputDesc(INPUT_WEIGHT_IDX) != nullptr;
    uint64_t inputUbByte = 0;
    uint64_t castTmpBufByte = 0;
    uint64_t probOutBufByte = 0;
    uint64_t weight4SmoothingBufByte = 0;
    uint64_t inputUbSize = 0;
    uint64_t targetNum = crossEntropyLossTiling.get_targetNum();
    uint64_t oneBatchByte = 0;
    uint64_t weightBuf = 0;
    if (inputDtype == ge::DT_FLOAT) {
        weightBuf = needWeightBuf ? NUM_1 : 0; // labelSmoothing计算需要搬入weight[C]
        inputUbByte = ((maxUbSize - reserveUb) / (DOUBLE_BUFFER + weightBuf)) / BLOCK_512 * BLOCK_512;
        oneBatchByte = CeilDiv(targetNum * NUM_4, BLOCK_32) * BLOCK_32;
        inputUbByte = inputUbByte >= oneBatchByte ? oneBatchByte : inputUbByte;
        castTmpBufByte = 0;
        probOutBufByte = 0;
        weight4SmoothingBufByte = needWeightBuf ? inputUbByte : 0;
        inputUbSize = GetDiv(inputUbByte, NUM_4);
    } else {
        weightBuf = needWeightBuf ? NUM_2 : 0;
        inputUbByte = ((maxUbSize - reserveUb) / (DOUBLE_BUFFER + NUM_1 + NUM_2 + weightBuf)) / BLOCK_512 * BLOCK_512;
        oneBatchByte = CeilDiv(targetNum * NUM_2, BLOCK_32) * BLOCK_32;
        inputUbByte =
            inputUbByte >= oneBatchByte ? oneBatchByte : inputUbByte; // 小shape一次搬入一个batch，暂不考虑多载的情况。
        castTmpBufByte = NUM_2 * inputUbByte;
        probOutBufByte = inputUbByte;
        weight4SmoothingBufByte = needWeightBuf ? castTmpBufByte : 0;
        inputUbSize = GetDiv(inputUbByte, NUM_2);
    }
    uint64_t totalTmpBufByte = reserveUb + castTmpBufByte + probOutBufByte + weight4SmoothingBufByte;
    crossEntropyLossTiling.set_inputUbSize(inputUbSize);
    crossEntropyLossTiling.set_castTmpBufByte(castTmpBufByte);
    crossEntropyLossTiling.set_lnTmpBufSize(lnTmpBufByte / DTYPE_LEN_FP32);
    crossEntropyLossTiling.set_weightTmpBufSize(weightTmpBufByte / DTYPE_LEN_FP32);
    crossEntropyLossTiling.set_weight4SmoothingBufSize(weight4SmoothingBufByte / DTYPE_LEN_FP32);
    crossEntropyLossTiling.set_totalTmpBufByte(totalTmpBufByte);
}

static void VecCalcTiling()
{
    uint64_t targetNum = crossEntropyLossTiling.get_targetNum();
    uint64_t ubLoopNum;
    uint64_t ubTailNum;
    uint64_t inputUbSize = crossEntropyLossTiling.get_inputUbSize();
    if (targetNum > inputUbSize) {
        ubLoopNum = CeilDiv(targetNum, inputUbSize);
        ubTailNum = GetRemainder(targetNum, inputUbSize);
        if (ubTailNum != 0) {
            ubLoopNum -= 1;
        }
    } else { // 小C的模板后续补充
        ubLoopNum = 0;
        ubTailNum = targetNum;
    }
    crossEntropyLossTiling.set_ubLoopNum(ubLoopNum);
    crossEntropyLossTiling.set_ubTailNum(ubTailNum);

    // splited by 128 repeat
    uint64_t vecLoopNum;
    uint64_t vecTailNum;
    SplitByRepeat(inputUbSize, vecLoopNum, vecTailNum);
    crossEntropyLossTiling.set_vecLoopNum(vecLoopNum);
    crossEntropyLossTiling.set_vecTailNum(vecTailNum);

    uint64_t tailVecLoopNum;
    uint64_t tailVecTailNum;
    SplitByRepeat(ubTailNum, tailVecLoopNum, tailVecTailNum);
    crossEntropyLossTiling.set_tailVecLoopNum(tailVecLoopNum);
    crossEntropyLossTiling.set_tailVecTailNum(tailVecTailNum);
}

static ge::graphStatus GetTilingAttr(gert::TilingContext* context)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto input = context->GetInputDesc(INPUT_DATA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    auto inputDtype = input->GetDataType();
    uint64_t tilingKey = 0;
    if (inputDtype == ge::DT_BF16) {
        tilingKey += TILING_KEY_BF16;
    } else if (inputDtype == ge::DT_FLOAT16) {
        tilingKey += TILING_KEY_FP16;
    } else if (inputDtype == ge::DT_FLOAT) {
        tilingKey += TILING_KEY_FP32;
    }
    context->SetTilingKey(tilingKey);

    const char* reductionStr = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
    uint64_t reduction = REDUCTION_MEAN; // default mode
    if (strcmp(reductionStr, "mean") == 0) {
        reduction = REDUCTION_MEAN;
    } else if (strcmp(reductionStr, "sum") == 0) {
        reduction = REDUCTION_SUM;
    } else if (strcmp(reductionStr, "none") == 0) {
        reduction = REDUCTION_NONE;
    } else {
        OP_LOGE(context, "Reduction should be in ['none', 'mean', 'sum']");
        return ge::GRAPH_FAILED;
    }
    crossEntropyLossTiling.set_reduction(reduction);

    int64_t ignoreIndex = DEFAULT_IGNORE_INDEX;
    const int64_t* ignoreIndexAttr = attrs->GetAttrPointer<int64_t>(ATTR_IGNORE_INDEX_IDX);
    ignoreIndex = ignoreIndexAttr == nullptr ? DEFAULT_IGNORE_INDEX : *ignoreIndexAttr;
    crossEntropyLossTiling.set_ignoreIndex(ignoreIndex);

    float labelSmoothing = 0.0;
    const float* labelSmoothingAttr = attrs->GetAttrPointer<float>(ATTR_LABEL_SMOOTHING_IDX);
    labelSmoothing = labelSmoothingAttr == nullptr ? 0.0 : *labelSmoothingAttr;
    if (labelSmoothing < 0.0 || labelSmoothing > 1.0) {
        OP_LOGE(context, "labelSmoothing should be in [0.0, 1.0]");
        return ge::GRAPH_FAILED;
    }
    crossEntropyLossTiling.set_labelSmoothing(labelSmoothing);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDtype(gert::TilingContext* context)
{
    auto nodeName = context;

    auto input = context->GetInputDesc(INPUT_DATA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    auto inputDtype = input->GetDataType();

    input = context->GetInputDesc(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    auto targetDtype = input->GetDataType();

    bool validDtype = inputDtype == ge::DT_BF16 || inputDtype == ge::DT_FLOAT || inputDtype == ge::DT_FLOAT16;
    OP_CHECK_IF(
        !validDtype,
        OP_LOGE(nodeName, "Input dtype should be in the support list:[BF16, FLOAT, FLOAT16]."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (targetDtype != ge::DT_INT64), OP_LOGE(nodeName, "Target dtype only supports INT64."),
        return ge::GRAPH_FAILED);
    // optional input
    auto weight = context->GetOptionalInputDesc(INPUT_WEIGHT_IDX);
    if (weight != nullptr) {
        auto weightDtype = weight->GetDataType();
        OP_CHECK_IF(
            (weightDtype != ge::DT_FLOAT),
            OP_LOGE(nodeName, "Weight dtype only supports FP32."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape(gert::TilingContext* context)
{
    auto nodeName = context;
    auto input = context->GetInputShape(INPUT_DATA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    auto inputShape = input->GetStorageShape();

    auto target = context->GetInputShape(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, target);
    auto targetShape = target->GetStorageShape();
    OP_CHECK_IF(
        (targetShape.GetDim(0) != inputShape.GetDim(DIM_0)),
        OP_LOGE(nodeName, "The dim 0 of input should be equal to the size of target."),
        return ge::GRAPH_FAILED);

    auto weight = context->GetInputShape(INPUT_WEIGHT_IDX);
    if (weight != nullptr) {
        auto weightShape = weight->GetStorageShape();
        OP_CHECK_IF(
            (weightShape.GetDim(0) != inputShape.GetDim(DIM_1)),
            OP_LOGE(nodeName, "The dim 1 of input should be equal to the size of weight."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingData(gert::TilingContext* context)
{
    uint64_t targetNum = context->GetInputShape(INPUT_DATA_IDX)->GetStorageShape().GetDim(DIM_1);
    crossEntropyLossTiling.set_targetNum(targetNum);
    CoresSplitTiling(context);
    UbSplitTiling(context);
    VecCalcTiling();
    // default weight is all 1
    auto weightDesc = context->GetOptionalInputDesc(INPUT_WEIGHT_IDX);
    bool defaultWeight = (weightDesc == nullptr) ? 1 : 0;
    crossEntropyLossTiling.set_defaultWeight(defaultWeight);
    // workspace
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t usrWorkspace = DTYPE_LEN_FP32 * NUM_3 * NUM_64;
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_16MB_SIZE + usrWorkspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4CrossEntropyLoss(gert::TilingContext* context)
{
    auto nodeName = context;
    OP_LOGD(nodeName, "Tiling4CrossEntropyLoss begin");
    bool regBase = IsRegbaseSocVersion4CELOSS(context);
    OP_LOGD(nodeName, "regBase is %d", regBase);
    if (regBase) {
        OP_LOGD(nodeName, "enter Tiling4CrossEntropyLossRegbase");
        return Tiling4CrossEntropyLossRegbase(context);
    }

    OP_CHECK_IF(
        GetTilingAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "GetTilingAttr failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckInputDtype(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckInputDtype failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckInputShape(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckInputShape failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetTilingData(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "GetTilingData failed."),
        return ge::GRAPH_FAILED);
    crossEntropyLossTiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(crossEntropyLossTiling.GetDataSize());
    PrintInfo(context);
    OP_LOGD(nodeName, "Tiling4CrossEntropyLoss end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParse4CrossEntropyLoss(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("CrossEntropyLoss", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

struct CrossEntropyLossCompileInfo {
};

IMPL_OP_OPTILING(CrossEntropyLoss)
    .Tiling(Tiling4CrossEntropyLoss)
    .TilingParse<CrossEntropyLossCompileInfo>(TilingParse4CrossEntropyLoss);
} // namespace optiling