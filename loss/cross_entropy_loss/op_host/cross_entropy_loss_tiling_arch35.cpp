/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file cross_entropy_loss_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include <cstring>
#include <cmath>
#include <bitset>
#include "cross_entropy_loss_tiling.h"
#include "../op_kernel/arch35/cross_entropy_loss_tiling_data.h"
#include "../op_kernel/arch35/cross_entropy_loss_tiling_key.h"

namespace optiling {
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr int64_t CONST_3 = 3;   // loss smooth weight总共需要3块
constexpr int64_t CONST_64 = 64; // n方向处理固定一次是64个数
constexpr int64_t CONST_96 = 96; // clearUb大小，固定是3*8*4
constexpr uint32_t INPUT_DATA_IDX = 0;
constexpr uint32_t INPUT_TARGET_IDX = 1;
constexpr uint32_t INPUT_WEIGHT_IDX = 2;
constexpr uint32_t ATTR_REDUCTION_IDX = 0;
constexpr uint32_t ATTR_IGNORE_INDEX_IDX = 1;
constexpr uint32_t ATTR_LABEL_SMOOTHING_IDX = 2;

constexpr int64_t DEFAULT_IGNORE_INDEX = -100LL;
constexpr int64_t BLOCK_32 = 32;   // 不应该写死
constexpr int64_t BLOCK_256 = 256; // 不应该写死
constexpr int64_t DTYPE_LEN_FP16 = 2;
constexpr int64_t DTYPE_LEN_FP32 = 4;
constexpr int64_t DTYPE_LEN_INT64 = 8;
constexpr uint64_t REDUCTION_NONE = 0;
constexpr uint64_t REDUCTION_MEAN = 1;
constexpr uint64_t REDUCTION_SUM = 2;
constexpr int64_t AR_DB_NUM = 2; // 需要开db且shape为AR的的ub数量
constexpr int64_t AR_NUM = 2;    // 不需要开db且shape为AR的的ub数量
constexpr int64_t A_NUM = 6;     // shape为A的的ub数量
constexpr int64_t SUM_NUM = 6;   // redution为sum和mean时所需要reduceSum的数量
constexpr int64_t DOUBLE_BUFFER_2 = 2;
constexpr int64_t DOUBLE_BUFFER_1 = 1;
constexpr size_t WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);

struct BaseTilingData {
    int64_t realCoreNum;
    int64_t frontCoreNum;
    int64_t frontCoreNSize;
    int64_t nLoopTimesB;
    int64_t onceNSize;
    int64_t onceNAlign;
    int64_t nTailNumB;
    int64_t nLoopTimesT;
    int64_t nTailNumT;
    int64_t ignoreIndex;
    int64_t cLoopTimes;
    int64_t cOnceNum;
    int64_t cOnceNumTail;
    int64_t onceNSizeT;
    int64_t N;
    int64_t C;
    int64_t kTimesTail;
    int64_t kTimes;
    float labelSmoothing;
    float labelSmooth1;
    float labelSmoothC;
    int64_t cacheStart;
    int64_t db;
};

int64_t findClosestPowerOfTwo(uint64_t a)
{
    int64_t k = 0;
    while (static_cast<bool>(a >>= 1ULL) && a != 0ULL) {
        k++;
    }
    return k;
}

int64_t countOnes(uint64_t num)
{
    int64_t count = 0;
    while (num != 0ULL) {
        count += static_cast<int64_t>(num & 1ULL); // 检查最低位是否为1
        num >>= 1ULL;
    }
    return count - 1;
}

class CrossEntropyLossRegbaseTiling
{
public:
    explicit CrossEntropyLossRegbaseTiling(gert::TilingContext* context) : context_(context){};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckInputShape();
    ge::graphStatus CalTilingData();
    ge::graphStatus GetAttrTilingData();
    bool IsAllC();
    int64_t CeilDiv(int64_t a, int64_t b) const;
    int64_t CalculateTail(int64_t a, int64_t b, int64_t c) const;
    int64_t CeilDivMul(int64_t a, int64_t b) const;
    void GetTilingKey();
    void ComputeNotAllC();
    void ComputeAllC(int64_t aNum, int64_t perBlock, int64_t blockNum);
    void ComputeUbTiling(
        int64_t reserveUb, int64_t isWeightQue, int64_t isWeightUb, int64_t isCleanUb, int64_t isSmoothUb);
    void FillTilingData();
    void PrintTilingData() const;

private:
    BaseTilingData BaseTiling_;
    uint64_t schId = static_cast<uint64_t>(0);
    uint64_t reduction = static_cast<uint64_t>(0);
    uint64_t isWeight = static_cast<uint64_t>(0);
    uint64_t labelS = static_cast<uint64_t>(0);
    uint64_t ignorex = static_cast<uint64_t>(0);

    int64_t coreNum = 0;
    uint64_t ubSize = 0;
    int64_t dtypeX = 0;
    int64_t dtypeTarget = 0;
    gert::TilingContext* context_ = nullptr;
    CrossEntropyLossRegBaseTilingData* tilingData_{nullptr};
};

ge::graphStatus CrossEntropyLossRegbaseTiling::CheckInputDtype()
{
    auto input = context_->GetInputDesc(INPUT_DATA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input);
    auto inputDtype = input->GetDataType();

    input = context_->GetInputDesc(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input);
    auto targetDtype = input->GetDataType();

    bool validDtype = inputDtype == ge::DT_BF16 || inputDtype == ge::DT_FLOAT || inputDtype == ge::DT_FLOAT16;
    OP_CHECK_IF(
        !validDtype,
        OP_LOGE(
            context_, "Input dtype should be in the support list:[BF16, FLOAT, FLOAT16]."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (targetDtype != ge::DT_INT64 && targetDtype != ge::DT_INT32),
        OP_LOGE(context_, "Target dtype only supports INT64 or INT32."),
        return ge::GRAPH_FAILED);
    dtypeTarget = targetDtype == ge::DT_INT64 ? DTYPE_LEN_INT64 : DTYPE_LEN_FP32;
    // optional input
    auto weight = context_->GetOptionalInputDesc(INPUT_WEIGHT_IDX);
    if (weight != nullptr) {
        isWeight = static_cast<uint64_t>(1);
        auto weightDtype = weight->GetDataType();
        OP_CHECK_IF(
            (weightDtype != ge::DT_FLOAT),
            OP_LOGE(context_, "Weight dtype only supports FP32."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void CrossEntropyLossRegbaseTiling::ComputeAllC(int64_t aNum, int64_t perBlock, int64_t blockNum)
{
    BaseTiling_.cOnceNum = (BaseTiling_.C + perBlock - 1) / perBlock * perBlock;
    BaseTiling_.onceNSize = aNum > blockNum ? blockNum : aNum;
    BaseTiling_.nLoopTimesB = blockNum / BaseTiling_.onceNSize;
    BaseTiling_.nLoopTimesT = BaseTiling_.frontCoreNSize / BaseTiling_.onceNSize;
    auto nTailNumB = blockNum % BaseTiling_.onceNSize;
    BaseTiling_.nLoopTimesB = nTailNumB == 0 ? BaseTiling_.nLoopTimesB - 1 : BaseTiling_.nLoopTimesB;
    BaseTiling_.nTailNumB = nTailNumB == 0 ? BaseTiling_.onceNSize : nTailNumB;
    auto nTailNumT = BaseTiling_.frontCoreNSize % BaseTiling_.onceNSize;
    BaseTiling_.nLoopTimesT = nTailNumT == 0 ? BaseTiling_.nLoopTimesT - 1 : BaseTiling_.nLoopTimesT;
    BaseTiling_.nTailNumT = nTailNumT == 0 ? BaseTiling_.onceNSize : nTailNumT;
}

bool CrossEntropyLossRegbaseTiling::IsAllC()
{
    int64_t blockNum = Ops::Base::CeilDiv(BaseTiling_.N, BaseTiling_.realCoreNum);
    int64_t perBlock = BLOCK_32 / dtypeX;
    int64_t cAligned = (BaseTiling_.C + perBlock - 1) / perBlock * perBlock;
    int64_t allUbSize = static_cast<int64_t>(ubSize);
    int64_t arBuf_db2 = DOUBLE_BUFFER_2 * AR_DB_NUM * cAligned * dtypeX + AR_NUM * cAligned * DTYPE_LEN_FP32;
    int64_t aBuf_db2 = dtypeX + dtypeTarget + A_NUM * DTYPE_LEN_FP32;
    int64_t aNum_db2 = (allUbSize - cAligned * DTYPE_LEN_FP32 - SUM_NUM * BLOCK_32) / (arBuf_db2 + aBuf_db2);
    OP_LOGI(
        context_,
        "blockNum:%ld, perBlock:%ld, cAligned:%ld, allUbSize:%ld, arBuf_db2:%ld, aBuf_db2:%ld, aNum_db2:%ld", blockNum,
        perBlock, cAligned, allUbSize, arBuf_db2, aBuf_db2, aNum_db2);
    if (aNum_db2 < perBlock) {
        int64_t arBuf_db1 = DOUBLE_BUFFER_1 * AR_DB_NUM * cAligned * dtypeX + AR_NUM * cAligned * DTYPE_LEN_FP32;
        int64_t aBuf_db1 = dtypeX + dtypeTarget + A_NUM * DTYPE_LEN_FP32;
        int64_t aNum_db1 = (allUbSize - cAligned * DTYPE_LEN_FP32 - SUM_NUM * BLOCK_32) / (arBuf_db1 + aBuf_db1);
        OP_LOGI(
            context_,
            "blockNum:%ld, perBlock:%ld, cAligned:%ld, allUbSize:%ld, arBuf_db1:%ld, aBuf_db1:%ld, aNum_db1:%ld",
            blockNum, perBlock, cAligned, allUbSize, arBuf_db1, aBuf_db1, aNum_db1);
        if (aNum_db1 < perBlock) {
            return false;
        } else {
            ComputeAllC(aNum_db1, perBlock, blockNum);
            BaseTiling_.db = DOUBLE_BUFFER_1;
            return true;
        }
    } else {
        ComputeAllC(aNum_db2, perBlock, blockNum);
        BaseTiling_.db = DOUBLE_BUFFER_2;
        return true;
    }
}

ge::graphStatus CrossEntropyLossRegbaseTiling::CheckInputShape()
{
    auto input = context_->GetInputShape(INPUT_DATA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input);
    auto inputShape = input->GetStorageShape();
    OP_CHECK_IF(
        inputShape.GetDimNum() != DIM_2,
        OP_LOGE(context_, "The dim of input0 should be equal to 2."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputShape.GetDim(DIM_0) != 0 && inputShape.GetDim(DIM_1) == 0,
        OP_LOGE(context_, "When N is not empty, C is empty, not support."),
        return ge::GRAPH_FAILED);
    auto target = context_->GetInputShape(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, target);
    auto targetShape = target->GetStorageShape();
    OP_CHECK_IF(
        targetShape.GetDimNum() != DIM_1,
        OP_LOGE(context_, "The dim of target should be equal to 1."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (targetShape.GetDim(0) != inputShape.GetDim(DIM_0)),
        OP_LOGE(
            context_, "The dim 0 of input0 should be equal to the size of target."),
        return ge::GRAPH_FAILED);

    auto weight = context_->GetOptionalInputShape(INPUT_WEIGHT_IDX);
    if (weight != nullptr) {
        auto weightShape = weight->GetStorageShape();
        OP_CHECK_IF(
            (weightShape.GetDim(0) != inputShape.GetDim(DIM_1)),
            OP_LOGE(
                context_, "The dim 1 of input should be equal to the size of weight."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossEntropyLossRegbaseTiling::GetAttrTilingData()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* reductionStr = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
    if (strcmp(reductionStr, "mean") == 0) {
        reduction = REDUCTION_MEAN;
    } else if (strcmp(reductionStr, "sum") == 0) {
        reduction = REDUCTION_SUM;
    } else if (strcmp(reductionStr, "none") == 0) {
        reduction = REDUCTION_NONE;
    } else {
        OP_LOGE(context_, "Reduction should be in ['none', 'mean', 'sum']");
        return ge::GRAPH_FAILED;
    }
    int64_t ignoreIndex = DEFAULT_IGNORE_INDEX;
    const int64_t* ignoreIndexAttr = attrs->GetAttrPointer<int64_t>(ATTR_IGNORE_INDEX_IDX);
    ignoreIndex = ignoreIndexAttr == nullptr ? DEFAULT_IGNORE_INDEX : *ignoreIndexAttr;
    BaseTiling_.ignoreIndex = ignoreIndex;
    ignorex = ignoreIndex >= 0 ? static_cast<uint64_t>(1) : static_cast<uint64_t>(0);
    float labelSmoothing = 0.0;
    const float* labelSmoothingAttr = attrs->GetAttrPointer<float>(ATTR_LABEL_SMOOTHING_IDX);
    labelSmoothing = labelSmoothingAttr == nullptr ? 0.0 : *labelSmoothingAttr;
    if (labelSmoothing < 0.0 || labelSmoothing > 1.0) {
        OP_LOGE(context_, "labelSmoothing should be in [0.0, 1.0]");
        return ge::GRAPH_FAILED;
    }
    labelS = labelSmoothing > 0.0f ? REDUCTION_MEAN : static_cast<uint64_t>(0);
    BaseTiling_.labelSmoothing = labelSmoothing;
    BaseTiling_.labelSmooth1 = 1 - labelSmoothing;
    BaseTiling_.labelSmoothC = BaseTiling_.C == 0 ? 0.0 : labelSmoothing / BaseTiling_.C;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossEntropyLossRegbaseTiling::CalTilingData()
{
    auto logProbShape = context_->GetInputShape(INPUT_DATA_IDX);
    auto inputDtype = context_->GetInputDesc(INPUT_DATA_IDX)->GetDataType();
    BaseTiling_.N = logProbShape->GetStorageShape().GetDim(DIM_0);
    BaseTiling_.C = logProbShape->GetStorageShape().GetDim(DIM_1);
    BaseTiling_.realCoreNum = (BaseTiling_.N > coreNum) || (BaseTiling_.N == 0) ? coreNum : BaseTiling_.N;
    BaseTiling_.frontCoreNSize = BaseTiling_.N / BaseTiling_.realCoreNum;
    BaseTiling_.frontCoreNum = BaseTiling_.N - BaseTiling_.frontCoreNSize * BaseTiling_.realCoreNum;
    dtypeX = inputDtype == ge::DT_FLOAT ? DTYPE_LEN_FP32 : DTYPE_LEN_FP16;

    OP_CHECK_IF(
        GetAttrTilingData() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "Get attr tilingData failed"),
        return ge::GRAPH_FAILED);
    GetTilingKey();
    return ge::GRAPH_SUCCESS;
}

void CrossEntropyLossRegbaseTiling::GetTilingKey()
{
    if (BaseTiling_.N == 0 || BaseTiling_.C == 0) {
        OP_LOGI(context_, "enter empty sch");
        BaseTiling_.realCoreNum = 1;
        BaseTiling_.db = DOUBLE_BUFFER_1;
        schId = static_cast<uint64_t>(1);
        isWeight = static_cast<uint64_t>(0);
        labelS = static_cast<uint64_t>(0);
        ignorex = static_cast<uint64_t>(0);
    } else if (IsAllC()) {     // 是否能走全载模板，和朱磊对齐
        schId = REDUCTION_SUM; // 预留给全载模板的
    } else {
        schId = static_cast<uint64_t>(0);
        ComputeNotAllC(); // 具体计算非全载模板的tilingdata值
    }
    OP_LOGI(
        context_, "schId is %lu, reduction is %lu, isWeight is %lu, labelS is %lu, ignorex is %lu",
        schId, reduction, isWeight, labelS, ignorex);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId, reduction, isWeight, labelS, ignorex);
    OP_LOGI(context_, "tilingKey is %ld", tilingKey);
    context_->SetTilingKey(tilingKey);
}

int64_t CrossEntropyLossRegbaseTiling::CeilDiv(int64_t a, int64_t b) const
{
    if (b == 0) {
        return b;
    }
    int64_t result = a / b;
    if (a % b == 0) {
        return result;
    } else {
        return result + 1;
    }
}

int64_t CrossEntropyLossRegbaseTiling::CalculateTail(int64_t a, int64_t b, int64_t c) const
{
    if (a % b == 0) {
        return 0;
    } else {
        return a - b * (c - 1);
    }
}

int64_t CrossEntropyLossRegbaseTiling::CeilDivMul(int64_t a, int64_t b) const
{
    return ((a + b - 1) / b) * b;
}

void CrossEntropyLossRegbaseTiling::ComputeUbTiling(
    int64_t reserveUb, int64_t isWeightQue, int64_t isWeightUb, int64_t isCleanUb, int64_t isSmoothUb)
{
    int64_t nUb = CONST_64 * DTYPE_LEN_FP32;
    int64_t needNub =
        reserveUb + nUb + isCleanUb * CONST_96 + isSmoothUb * nUb + isWeightUb * nUb + CONST_64 * dtypeTarget;
    int64_t queUb = DTYPE_LEN_FP16 * dtypeX + isWeightQue * DTYPE_LEN_FP32 + DTYPE_LEN_FP32;
    int64_t cUbParams = queUb + DTYPE_LEN_FP32 + isSmoothUb * DTYPE_LEN_FP32;
    int64_t onceC = (static_cast<int64_t>(ubSize) - needNub) / cUbParams;
    int64_t oneBlock = BLOCK_32 / dtypeX;
    BaseTiling_.cOnceNum = (onceC / oneBlock) * oneBlock;
    // 此时计算下cacheUb，看是否够用，不够用需要重新计算下
    int64_t tailUb = static_cast<int64_t>(ubSize) - needNub - cUbParams * BaseTiling_.cOnceNum;

    BaseTiling_.cLoopTimes = CeilDiv(BaseTiling_.C, BaseTiling_.cOnceNum);

    int64_t log2k = findClosestPowerOfTwo(static_cast<uint64_t>(BaseTiling_.cLoopTimes));
    // 找到最近的整数n
    BaseTiling_.kTimes = static_cast<int64_t>(1ULL << log2k);
    BaseTiling_.kTimesTail = BaseTiling_.cLoopTimes - BaseTiling_.kTimes;
    int64_t result = (BaseTiling_.kTimes - 1) ^ (BaseTiling_.kTimes);
    BaseTiling_.cacheStart = countOnes(static_cast<uint64_t>(result));
    int64_t cacheUb = (BaseTiling_.cacheStart + 1) * DTYPE_LEN_INT64 * DTYPE_LEN_FP32;
    OP_LOGI(
        context_, "needNub is %ld, cUbParams is %ld, tailUb is %ld, cacheStart %ld", needNub, cUbParams,
        tailUb, BaseTiling_.cacheStart);
    while (cacheUb > tailUb) { // 这里注意，如果剩余ub不够cacheUb,那么上面的需要重新计算
        BaseTiling_.cOnceNum = BaseTiling_.cOnceNum - oneBlock;
        tailUb = static_cast<int64_t>(ubSize) - needNub - cUbParams * BaseTiling_.cOnceNum;

        BaseTiling_.cLoopTimes = CeilDiv(BaseTiling_.C, BaseTiling_.cOnceNum);
        log2k = findClosestPowerOfTwo(static_cast<uint64_t>(BaseTiling_.cLoopTimes));
        // 找到最近的整数n
        BaseTiling_.kTimes = static_cast<int64_t>(1ULL << log2k);
        BaseTiling_.kTimesTail = BaseTiling_.cLoopTimes - BaseTiling_.kTimes;
        result = (BaseTiling_.kTimes - 1) ^ (BaseTiling_.kTimes);
        BaseTiling_.cacheStart = countOnes(static_cast<uint64_t>(result));
        cacheUb = (BaseTiling_.cacheStart + 1) * DTYPE_LEN_INT64 * DTYPE_LEN_FP32;
        OP_LOGI(
            context_, "cOnceNum %ld, tailUb %ld, cacheStart %ld, cacheUb %ld", BaseTiling_.cOnceNum,
            tailUb, BaseTiling_.cacheStart, cacheUb);
    }
    BaseTiling_.cOnceNumTail = CalculateTail(BaseTiling_.C, BaseTiling_.cOnceNum, BaseTiling_.cLoopTimes);
}

void CrossEntropyLossRegbaseTiling::ComputeNotAllC()
{
    int64_t nFrontCore = BaseTiling_.frontCoreNSize + 1;
    int64_t nTailCore = BaseTiling_.frontCoreNSize;
    // 计算n方向的tiling参数
    BaseTiling_.onceNSize = nFrontCore >= CONST_64 ? CONST_64 : nFrontCore;
    BaseTiling_.onceNSizeT = nTailCore >= CONST_64 ? CONST_64 : nTailCore;
    BaseTiling_.nLoopTimesB = CeilDiv(nFrontCore, BaseTiling_.onceNSize);
    BaseTiling_.nTailNumB = CalculateTail(nFrontCore, BaseTiling_.onceNSize, BaseTiling_.nLoopTimesB);
    BaseTiling_.nLoopTimesT = CeilDiv(nTailCore, BaseTiling_.onceNSizeT);
    BaseTiling_.nTailNumT = CalculateTail(nTailCore, BaseTiling_.onceNSizeT, BaseTiling_.nLoopTimesT);
    // 计算C方向tiling参数
    int64_t reserveUb = CONST_64;
    int64_t isWeightQue = 0;
    int64_t isWeightUb = 0;
    int64_t isCleanUb = 0;
    int64_t isSmoothUb = 0;
    if (isWeight == REDUCTION_MEAN && (labelS != REDUCTION_NONE)) {
        isWeightQue = 1;
    }
    if ((isWeight == REDUCTION_MEAN) || (reduction == REDUCTION_MEAN && isWeight == REDUCTION_SUM)) {
        isWeightUb = 1;
    }
    if (labelS != REDUCTION_NONE) {
        isSmoothUb = 1;
    }
    if (reduction != REDUCTION_NONE) {
        isCleanUb = 1;
    }
    ComputeUbTiling(reserveUb, isWeightQue, isWeightUb, isCleanUb, isSmoothUb);
}

void CrossEntropyLossRegbaseTiling::FillTilingData()
{
    tilingData_->realCoreNum = (BaseTiling_.realCoreNum);
    tilingData_->frontCoreNum = BaseTiling_.frontCoreNum;
    tilingData_->frontCoreNSize = BaseTiling_.frontCoreNSize;
    tilingData_->nLoopTimesB = BaseTiling_.nLoopTimesB;
    tilingData_->onceNSize = BaseTiling_.onceNSize;
    tilingData_->nTailNumB = BaseTiling_.nTailNumB;
    tilingData_->nLoopTimesT = BaseTiling_.nLoopTimesT;
    tilingData_->nTailNumT = BaseTiling_.nTailNumT;
    tilingData_->ignoreIndex = BaseTiling_.ignoreIndex;
    tilingData_->cLoopTimes = BaseTiling_.cLoopTimes;
    tilingData_->cOnceNum = BaseTiling_.cOnceNum;
    tilingData_->cOnceNumTail = BaseTiling_.cOnceNumTail;
    tilingData_->onceNSizeT = BaseTiling_.onceNSizeT;
    tilingData_->C = BaseTiling_.C;
    tilingData_->kTimesTail = BaseTiling_.kTimesTail;
    tilingData_->kTimes = BaseTiling_.kTimes;
    tilingData_->labelSmooth1 = BaseTiling_.labelSmooth1;
    tilingData_->labelSmoothC = BaseTiling_.labelSmoothC;
    tilingData_->cacheStart = BaseTiling_.cacheStart;
    tilingData_->db = BaseTiling_.db;
}

void CrossEntropyLossRegbaseTiling::PrintTilingData() const
{
    auto nodeName = context_;
    OP_LOGD(nodeName, "cross_entropy_loss regbase tiling data begin print.");
    OP_LOGD(nodeName, "realCoreNum = %ld.", tilingData_->realCoreNum);
    OP_LOGD(nodeName, "frontCoreNum = %ld.", tilingData_->frontCoreNum);
    OP_LOGD(nodeName, "frontCoreNSize = %ld.", tilingData_->frontCoreNSize);
    OP_LOGD(nodeName, "nLoopTimesB = %ld.", tilingData_->nLoopTimesB);
    OP_LOGD(nodeName, "onceNSize = %ld.", tilingData_->onceNSize);
    OP_LOGD(nodeName, "nTailNumB = %ld.", tilingData_->nTailNumB);
    OP_LOGD(nodeName, "nLoopTimesT = %ld.", tilingData_->nLoopTimesT);
    OP_LOGD(nodeName, "nTailNumT = %ld.", tilingData_->nTailNumT);
    OP_LOGD(nodeName, "ignoreIndex = %ld.", tilingData_->ignoreIndex);
    OP_LOGD(nodeName, "cLoopTimes = %ld.", tilingData_->cLoopTimes);
    OP_LOGD(nodeName, "cOnceNum = %ld.", tilingData_->cOnceNum);
    OP_LOGD(nodeName, "cOnceNumTail = %ld.", tilingData_->cOnceNumTail);
    OP_LOGD(nodeName, "onceNSizeT = %ld.", tilingData_->onceNSizeT);
    OP_LOGD(nodeName, "C = %ld.", tilingData_->C);
    OP_LOGD(nodeName, "kTimesTail = %ld.", tilingData_->kTimesTail);
    OP_LOGD(nodeName, "kTimes = %ld.", tilingData_->kTimes);
    OP_LOGD(nodeName, "cacheStart = %ld.", tilingData_->cacheStart);
    OP_LOGD(nodeName, "db = %ld.", tilingData_->db);
    OP_LOGD(nodeName, "labelSmooth1 = %f.", tilingData_->labelSmooth1);
    OP_LOGD(nodeName, "labelSmoothC = %f.", tilingData_->labelSmoothC);
}

ge::graphStatus CrossEntropyLossRegbaseTiling::Init()
{
    OP_LOGD(context_, "Enter CrossEntropyLossRegbaseTiling init.");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(context_, "coreNum must greater than zero, but is %ld", coreNum),
        return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize <= 0UL,
        OP_LOGE(context_, "ubSize must greater than zero, but is %lu", ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGI(context_, "coreNum is %ld, ubSize is %lu", coreNum, ubSize);
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<CrossEntropyLossRegBaseTilingData>();
        OP_CHECK_IF(
            tilingData_ == nullptr,
            OP_LOGE(context_, "get tilingdata ptr failed"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        (memset_s(
             tilingData_, sizeof(CrossEntropyLossRegBaseTilingData), 0, sizeof(CrossEntropyLossRegBaseTilingData)) !=
         EOK),
        OP_LOGE(context_, "memset tilingdata failed"), return ge::GRAPH_FAILED);
    OP_LOGD(context_, "Exit CrossEntropyLossRegbaseTiling init.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossEntropyLossRegbaseTiling::DoTiling()
{
    OP_LOGD(context_, "Enter CrossEntropyLossRegbaseTiling DoTiling");

    OP_CHECK_IF(
        CheckInputDtype() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "CheckInputParams is failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckInputShape() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "CheckInputShapes is failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalTilingData() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "CalTilingData failed"), return ge::GRAPH_FAILED);
    FillTilingData();
    PrintTilingData();
    context_->SetBlockDim(tilingData_->realCoreNum);
    context_->SetScheduleMode(1);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    int64_t extraWorkSpace = coreNum * BLOCK_32 * DTYPE_LEN_FP32 * CONST_3;
    int64_t workSpaceSize = reduction == static_cast<uint64_t>(0) ? 0 : extraWorkSpace;
    workspaces[0] = WORKSPACE_SIZE + workSpaceSize;
    OP_LOGI(context_, "End dotiling, workspaces is %ld", extraWorkSpace);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4CrossEntropyLossRegbase(gert::TilingContext* context)
{
    OP_LOGD(context, "Start Tiling4CrossEntropyLossRegbase.");

    CrossEntropyLossRegbaseTiling tilingImpl = CrossEntropyLossRegbaseTiling(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4CrossEntropyLossForAscendC init failed.");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4CrossEntropyLossForAscendC do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "end Tiling4CrossEntropyLossRegbase");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling