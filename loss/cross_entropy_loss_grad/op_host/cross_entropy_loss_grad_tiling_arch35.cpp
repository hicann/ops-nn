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
 * \file cross_entropy_loss_grad_regbase_tiling.cc
 * \brief cross_entropy_loss_grad_regbase_tiling source file
 */
#include <iostream>
#include "cross_entropy_loss_grad_tiling_arch35.h"
#include "error_util.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"

namespace optiling {
constexpr uint32_t SYS_WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr uint64_t INPUT_GRAD_LOSS_IDX = 0;
constexpr uint64_t INPUT_LOG_PROB_IDX = 1;
constexpr uint64_t INPUT_TARGET_IDX = 2;
constexpr uint64_t INPUT_WEIGHT_IDX = 3;
constexpr uint64_t REDUCTION_ATTR_IDX = 0;
constexpr uint64_t IGNORE_INDEX_ATTR_IDX = 1;
constexpr uint64_t LABEL_SMOOTHING_ATTR_IDX = 2;
constexpr size_t REQUIRES_INPUT_NUM = 3;
constexpr size_t INPUT_NUM = 6;

constexpr uint64_t FP16_BF16_DTYPE_SIZE = 2;
constexpr uint64_t FP32_INT32_DTYPE_SIZE = 4;
constexpr uint64_t INT64_DTYPE_SIZE = 8;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t ALIGN_SIZE = 512;
constexpr int64_t FP32_ALIGN_NUM = 128;
constexpr uint64_t SPLIT_BUFFER = 12;
constexpr uint64_t WEIGHT_SPLIT_BUFFER = 16;
constexpr int64_t CONST_SIXTY_THREE = 63;
constexpr int64_t BLOCK_B32 = 8;
constexpr int64_t FULL_LOAD_BUFFER_B32 = 3;
constexpr int64_t FULL_LOAD_BUFFER_B64 = 4;
constexpr int64_t FULL_LOAD_BUFFER_F32 = 1;
constexpr int64_t FULL_LOAD_BUFFER_F16 = 2;
constexpr int64_t INPUT_NC_BUFFER_NUM = 2;
constexpr int64_t FULL_LOAD_N_TMP_BUFFER = 2;

constexpr uint64_t TILING_KEY_NONE = 0;
constexpr uint64_t TILING_KEY_MEAN = 1;
constexpr uint64_t TILING_KEY_SUM = 2;
constexpr uint64_t TILING_KEY_TRUE = 1;
constexpr uint64_t TILING_KEY_FALSE = 0;
constexpr int64_t CONST1 = 1;
constexpr int64_t CONST2 = 2;
static const gert::Shape g_vec_1_shape = {1};

void CrossEntropyLossGradRegbaseTiling::PrintInfo() {
    auto nodeName = tilingContext;
    OP_LOGD(nodeName, "Start to print CrossEntropyLossGradRegbase tiling data");
    OP_LOGD(nodeName, "reduction: %ld", ceLossGradRegbaseTiling->reduction);
    OP_LOGD(nodeName, "ignoreIndex: %ld", ceLossGradRegbaseTiling->ignoreIndex);
    OP_LOGD(nodeName, "labelSmoothing: %f", ceLossGradRegbaseTiling->labelSmoothing);
    OP_LOGD(nodeName, "rowVal: %ld", ceLossGradRegbaseTiling->rowVal);
    OP_LOGD(nodeName, "colVal: %ld", ceLossGradRegbaseTiling->colVal);
    OP_LOGD(nodeName, "frontCoreNum: %ld", ceLossGradRegbaseTiling->frontCoreNum);
    OP_LOGD(nodeName, "tailCoreNum: %ld", ceLossGradRegbaseTiling->tailCoreNum);
    OP_LOGD(nodeName, "usedCoreNum: %ld", ceLossGradRegbaseTiling->usedCoreNum);
    OP_LOGD(nodeName, "frontRowNum: %ld", ceLossGradRegbaseTiling->frontRowNum);
    OP_LOGD(nodeName, "tailRowNum: %ld", ceLossGradRegbaseTiling->tailRowNum);
    OP_LOGD(nodeName, "alignColLoopNum: %ld", ceLossGradRegbaseTiling->alignColLoopNum);
    OP_LOGD(nodeName, "alignNCNum: %ld", ceLossGradRegbaseTiling->alignNCNum);
    OP_LOGD(nodeName, "colLoop: %ld", ceLossGradRegbaseTiling->colLoop);
    OP_LOGD(nodeName, "colLoopNumTail: %ld", ceLossGradRegbaseTiling->colLoopNumTail);
    OP_LOGD(nodeName, "targetSize: %ld", ceLossGradRegbaseTiling->targetSize);
    OP_LOGD(nodeName, "targetCastSize: %ld", ceLossGradRegbaseTiling->targetCastSize);
    OP_LOGD(nodeName, "smoothSize: %ld", ceLossGradRegbaseTiling->smoothSize);
    OP_LOGD(nodeName, "gradLossSize: %ld", ceLossGradRegbaseTiling->gradLossSize);
    OP_LOGD(nodeName, "ignoreSize: %ld", ceLossGradRegbaseTiling->ignoreSize);
    OP_LOGD(nodeName, "targetWeightSize: %ld", ceLossGradRegbaseTiling->targetWeightSize);
    OP_LOGD(nodeName, "tBuf2Size: %ld", ceLossGradRegbaseTiling->tBuf2Size);
    OP_LOGD(nodeName, "tBuf3Size: %ld", ceLossGradRegbaseTiling->tBuf3Size);
    OP_LOGD(nodeName, "kTimes: %ld", ceLossGradRegbaseTiling->kTimes);
    OP_LOGD(nodeName, "cOnceNum: %ld", ceLossGradRegbaseTiling->cOnceNum);
    OP_LOGD(nodeName, "cOnceNumTail: %ld", ceLossGradRegbaseTiling->cOnceNumTail);
    OP_LOGD(nodeName, "kTimesTail: %ld", ceLossGradRegbaseTiling->kTimesTail);
    OP_LOGD(nodeName, "cOnceNumTailAlign: %ld", ceLossGradRegbaseTiling->cOnceNumTailAlign);
    OP_LOGD(nodeName, "cacheStart: %ld", ceLossGradRegbaseTiling->cacheStart);

    OP_LOGD(nodeName, "Print CrossEntropyLossGradRegbase tiling data end");
}

int64_t CrossEntropyLossGradRegbaseTiling::FindNearestPower2(const int64_t value)
{
    if (value < CONST2) {
        return CONST1;
    } else {
        const int64_t pow = CONST_SIXTY_THREE - __builtin_clzll(static_cast<uint64_t>(value));
        return (CONST1 << pow);
    }
}

int64_t CrossEntropyLossGradRegbaseTiling::CalLog2(int64_t value)
{
    int64_t res = 0;
    while (value > CONST1) {
        value = value >> CONST1;
        res++;
    }
    return res;
}

int64_t CrossEntropyLossGradRegbaseTiling::GetMod(int64_t l_value, int64_t r_value) {
  if (r_value == 0) {
    return l_value;
  }
  return l_value % r_value;
}

const gert::Shape &EnsureNotScalar4CELoss(const gert::Shape &inShape) {
  if (inShape.IsScalar()) {
    return g_vec_1_shape;
  }
  return inShape;
}

void CrossEntropyLossGradRegbaseTiling::InitSplitInfo() {
    // 分核
    int64_t frontCoreNum = GetMod(static_cast<int64_t>(rowVal), static_cast<int64_t>(totalCoreNum)) != 0 ? GetMod(static_cast<int64_t>(rowVal), static_cast<int64_t>(totalCoreNum)) : totalCoreNum;
    int64_t tailCoreNum = rowVal <= totalCoreNum ? 0 : totalCoreNum - frontCoreNum;
    int64_t numBlocks = frontCoreNum + tailCoreNum;
    int64_t frontRowNum = Ops::Base::CeilDiv(rowVal, totalCoreNum);
    int64_t tailRowNum = Ops::Base::FloorDiv(rowVal, totalCoreNum);
    ceLossGradRegbaseTiling->frontCoreNum = frontCoreNum;
    ceLossGradRegbaseTiling->tailCoreNum = tailCoreNum;
    ceLossGradRegbaseTiling->usedCoreNum = numBlocks;
    ceLossGradRegbaseTiling->frontRowNum = frontRowNum;
    ceLossGradRegbaseTiling->tailRowNum = tailRowNum;
    tilingContext->SetBlockDim(static_cast<uint64_t>(numBlocks));
}

ge::graphStatus CrossEntropyLossGradRegbaseTiling::GetPlatform() {
    auto platformInfo = tilingContext->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUbSize);
    totalCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF((totalCoreNum <= 0),
                    OP_LOGE(tilingContext, "Failed to get core num."),
                    return ge::GRAPH_FAILED);
    vectorLength = static_cast<uint64_t>(Ops::Base::GetVRegSize(tilingContext));
    OP_CHECK_IF((vectorLength <= FP32_INT32_DTYPE_SIZE),
        OP_LOGE(tilingContext, "vector length is invalid."), return ge::GRAPH_FAILED);

    blockSize = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(tilingContext));
    OP_CHECK_IF((blockSize <= 0), OP_LOGE(tilingContext, "block size is invalid."),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUbSize);
    OP_LOGD(tilingContext, "Get total ub size:%lu", totalUbSize);
    OP_CHECK_IF((totalUbSize <= 0),
                    OP_LOGE(tilingContext, "Failed to get ub size."),
                    return ge::GRAPH_FAILED);

    if (ceLossGradRegbaseTiling == nullptr) {
        ceLossGradRegbaseTiling = tilingContext->GetTilingData<CrossEntropyLossGradRegbaseTilingData>();
        OP_CHECK_IF(ceLossGradRegbaseTiling == nullptr,
            OP_LOGE(tilingContext, "get tilingdata ptr failed"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((memset_s(ceLossGradRegbaseTiling, sizeof(CrossEntropyLossGradRegbaseTilingData), 0,
        sizeof(CrossEntropyLossGradRegbaseTilingData)) != EOK),
        OP_LOGE(tilingContext, "memset tilingdata failed"), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext, "Exit CrossEntropyLossGradRegbaseTiling init.");
    return ge::GRAPH_SUCCESS;
}

bool CrossEntropyLossGradRegbaseTiling::CELossGradInnerCoreFullLoad() {
    alignColLoopNum = Ops::Base::CeilAlign(colVal, static_cast<int64_t>(BLOCK_SIZE / dtypeSize));
    int64_t queBuffer = 0;
    int64_t tmpBuffer = static_cast<int64_t>(alignColLoopNum * sizeof(float));
    int64_t tmpABuffer = static_cast<int64_t>(FULL_LOAD_N_TMP_BUFFER * sizeof(float)) + static_cast<int64_t>(targetDtypeSize);
    int64_t ncBufferNum = INPUT_NC_BUFFER_NUM * alignColLoopNum * static_cast<int64_t>(dtypeSize) + alignColLoopNum * static_cast<int64_t>(FP32_INT32_DTYPE_SIZE); // input+output+tmpbuf
    if (ceLossGradTilingKey.reduction == TILING_KEY_NONE) {
        tmpABuffer += static_cast<int64_t>(dtypeSize);   //gradloss
    } else if (ceLossGradTilingKey.reduction == TILING_KEY_MEAN) {
        tBuf3Size = static_cast<int64_t>(Ops::Base::CeilAlign(totalCoreNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE));
        tBuf2Size = static_cast<int64_t>(BLOCK_SIZE);
    }
    if (ceLossGradTilingKey.isWeight == TILING_KEY_TRUE) {
        queBuffer += static_cast<int64_t>(alignColLoopNum * sizeof(float)); // weightbuf
        tmpABuffer += static_cast<int64_t>(FP32_INT32_DTYPE_SIZE);    // weightgather
        if(ceLossGradTilingKey.labelS == TILING_KEY_TRUE) {
            ncBufferNum += static_cast<int64_t>(alignColLoopNum * sizeof(float));   // sumtmpbuf
            tmpABuffer += static_cast<int64_t>(FP32_INT32_DTYPE_SIZE);    // sumbuf
        }
    }
    if(ceLossGradTilingKey.labelS == TILING_KEY_TRUE) {
        tmpABuffer += static_cast<int64_t>(FP32_INT32_DTYPE_SIZE); //smoothbuf
    }
    int64_t ubNSize = static_cast<int64_t>(totalUbSize) - static_cast<int64_t>(queBuffer + tmpBuffer + tBuf2Size + tBuf2Size + tBuf3Size);
    int64_t theoreticalMaxN = ubNSize / ncBufferNum;
    int64_t theoreticalNAlign = Ops::Base::CeilAlign(theoreticalMaxN, static_cast<int64_t>(BLOCK_SIZE / dtypeSize));
    int64_t allowedN = (ubNSize - tmpABuffer * theoreticalNAlign) / ncBufferNum;

    int64_t finalN = std::min(theoreticalMaxN, allowedN);
    int64_t finalNAlign = Ops::Base::CeilAlign(finalN, static_cast<int64_t>(BLOCK_SIZE / dtypeSize));

    int64_t finalTotal = ncBufferNum * finalN + tmpABuffer * finalNAlign;
    if (finalTotal > ubNSize) {
        finalN = ubNSize / (ncBufferNum + tmpABuffer);
        finalNAlign = Ops::Base::CeilAlign(finalN, static_cast<int64_t>(BLOCK_SIZE / dtypeSize));
        finalTotal = ncBufferNum * finalN + tmpABuffer * finalNAlign;
    }
    if (finalTotal > ubNSize || finalN < 1) {
        return false;
    }

    alignNCNum = finalN * alignColLoopNum;
    targetSize = Ops::Base::CeilAlign(finalNAlign * targetDtypeSize, BLOCK_SIZE);
    uint64_t peerLoopNNum = Ops::Base::FloorDiv(static_cast<uint64_t>(targetSize), targetDtypeSize);
    targetCastSize = Ops::Base::CeilAlign(peerLoopNNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
    ignoreSize = targetCastSize;
    ceLossGradTilingKey.schId = static_cast<uint32_t>(CROSS_ENTROPY_LOSS_GRAD_MODE_1);
    smoothSize = ceLossGradTilingKey.labelS == TILING_KEY_TRUE ? targetCastSize : 0;
    targetWeightSize = ceLossGradTilingKey.isWeight == TILING_KEY_TRUE ? targetCastSize : 0;
    gradLossSize = ceLossGradTilingKey.reduction == TILING_KEY_NONE ? Ops::Base::CeilAlign(peerLoopNNum * dtypeSize, BLOCK_SIZE) : 0;
    return true;
}

void CrossEntropyLossGradRegbaseTiling::CELossGradCoreSplitInfo() {
    uint64_t frontRowNum = static_cast<uint64_t>(ceLossGradRegbaseTiling->frontRowNum);
    frontRowNum = frontRowNum > vectorLength ? vectorLength : frontRowNum;
    targetSize = Ops::Base::CeilAlign(frontRowNum * INT64_DTYPE_SIZE, BLOCK_SIZE);
    if (targetDtypeSize == FP32_INT32_DTYPE_SIZE) {
        targetSize = Ops::Base::CeilAlign(frontRowNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
    }
    targetCastSize = Ops::Base::CeilAlign(frontRowNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
    ignoreSize = Ops::Base::CeilAlign(frontRowNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
    smoothSize = ceLossGradRegbaseTiling->labelSmoothing >= 0 ? targetCastSize : 0;

    auto weightTensor = tilingContext->GetOptionalInputTensor(INPUT_WEIGHT_IDX);
    float labelSmooth = ceLossGradRegbaseTiling->labelSmoothing;
    if (weightTensor == nullptr) {  // weight为None
        OP_LOGD(tilingContext, "weight is None");
        bufferNum = SPLIT_BUFFER;
    } else {
        OP_LOGD(tilingContext, "weight is not None");
        targetWeightSize = Ops::Base::CeilAlign(frontRowNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
        bufferNum = labelSmooth > 0 ? WEIGHT_SPLIT_BUFFER + targetSize : WEIGHT_SPLIT_BUFFER;
    }
    if (ceLossGradTilingKey.reduction == TILING_KEY_NONE) {
        gradLossSize = Ops::Base::CeilAlign(frontRowNum * dtypeSize, BLOCK_SIZE);
    } else if (ceLossGradTilingKey.reduction == TILING_KEY_MEAN) {
        tBuf2Size = static_cast<int64_t>(BLOCK_SIZE);
        tBuf3Size = Ops::Base::CeilAlign(std::max(static_cast<uint64_t>(totalCoreNum), frontRowNum) * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
    }
    colLoopNum = Ops::Base::FloorDiv((totalUbSize - targetSize - targetCastSize - gradLossSize - tBuf2Size -
                                tBuf3Size - ignoreSize - targetWeightSize - smoothSize),
                                bufferNum);
    if (colLoopNum < colVal) {
        alignColLoopNum = Ops::Base::FloorAlign(colLoopNum, (FP32_ALIGN_NUM));
    } else {
        alignColLoopNum = Ops::Base::FloorAlign(colVal, (FP32_ALIGN_NUM));
    }
    colLoop = alignColLoopNum == 0 ? 0 : Ops::Base::FloorDiv(colVal, alignColLoopNum);
    colLoopNumTail = GetMod(static_cast<int64_t>(colVal), static_cast<int64_t>(alignColLoopNum));
    alignNCNum = alignColLoopNum;
    if (weightTensor != nullptr && labelSmooth > 0) {
      // 二分
      kTimes = FindNearestPower2(colLoop);
      int64_t kNums = CalLog2(kTimes);
      cOnceNum = alignColLoopNum;
      int64_t cNumTail = colVal - kTimes * cOnceNum;
      kTimesTail = cOnceNum == 0 ? 0 : Ops::Base::CeilDiv(cNumTail, cOnceNum);
      cOnceNumTail = cOnceNum == 0 ? cNumTail : GetMod(static_cast<int64_t>(cNumTail), static_cast<int64_t>(cOnceNum));
      cOnceNumTailAlign = Ops::Base::CeilAlign(cOnceNumTail, BLOCK_B32);
      cacheStart = kNums * static_cast<int64_t>(vectorLength / sizeof(float));
    }

    ceLossGradTilingKey.schId = static_cast<uint32_t>(CROSS_ENTROPY_LOSS_GRAD_MODE_0);
}

ge::graphStatus CrossEntropyLossGradRegbaseTiling::SetTilingData() {
    ceLossGradRegbaseTiling->rowVal = rowVal;
    ceLossGradRegbaseTiling->colVal = colVal;
    ceLossGradRegbaseTiling->alignColLoopNum = alignColLoopNum;
    ceLossGradRegbaseTiling->alignNCNum = alignNCNum;
    ceLossGradRegbaseTiling->colLoop = colLoop;
    ceLossGradRegbaseTiling->colLoopNumTail = colLoopNumTail;
    ceLossGradRegbaseTiling->targetSize = targetSize;
    ceLossGradRegbaseTiling->targetCastSize = targetCastSize;
    ceLossGradRegbaseTiling->smoothSize = smoothSize;
    ceLossGradRegbaseTiling->gradLossSize = gradLossSize;
    ceLossGradRegbaseTiling->ignoreSize = ignoreSize;
    ceLossGradRegbaseTiling->targetWeightSize = targetWeightSize;
    ceLossGradRegbaseTiling->tBuf2Size = tBuf2Size;
    ceLossGradRegbaseTiling->tBuf3Size = tBuf3Size;
    ceLossGradRegbaseTiling->kTimes = kTimes;
    ceLossGradRegbaseTiling->cOnceNum = cOnceNum;
    ceLossGradRegbaseTiling->cOnceNumTail = cOnceNumTail;
    ceLossGradRegbaseTiling->kTimesTail = kTimesTail;
    ceLossGradRegbaseTiling->cOnceNumTailAlign = cOnceNumTailAlign;
    ceLossGradRegbaseTiling->cacheStart = cacheStart;
    size_t usrSize = static_cast<size_t>(totalCoreNum * FP32_INT32_DTYPE_SIZE * BLOCK_SIZE);
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = usrSize + SYS_WORKSPACE_SIZE;

    PrintInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossEntropyLossGradRegbaseTiling::GetTilingInput() {
    const gert::StorageShape* gradLossShape = tilingContext->GetInputShape(INPUT_GRAD_LOSS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradLossShape);
    const gert::StorageShape* logProbShape = tilingContext->GetInputShape(INPUT_LOG_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, logProbShape);
    const gert::StorageShape* targetShape = tilingContext->GetInputShape(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, targetShape);
    const gert::Shape& gradLossStrorageShape = EnsureNotScalar4CELoss(gradLossShape->GetStorageShape());
    const gert::Shape& logProbStrorageShape = EnsureNotScalar4CELoss(logProbShape->GetStorageShape());
    const gert::Shape& targetStrorageShape = EnsureNotScalar4CELoss(targetShape->GetStorageShape());
    OP_CHECK_IF((logProbStrorageShape.GetDimNum() != 2),
                    OP_LOGE(tilingContext, "input log_prob dim num must be 2."),
                    return ge::GRAPH_FAILED);
    if (ceLossGradTilingKey.reduction == TILING_KEY_NONE) {
        OP_CHECK_IF(
            (gradLossStrorageShape.GetDim(0) != logProbStrorageShape.GetDim(0) ||
             gradLossStrorageShape.GetDimNum() != 1),
            OP_LOGE(tilingContext,
                                            "The dim 0 of log_prob should be equal to the size of grad_loss and "
                                            "grad_loss dimNum should be equle to 1 when reduction is none."),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (gradLossStrorageShape.GetDim(0) != 1 || gradLossStrorageShape.GetDimNum() != 1),
            OP_LOGE(
                tilingContext,
                "grad_loss should be a scalar or shape should be equal to [1] when reduction is sum or mean."),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((targetStrorageShape.GetDimNum() != 1 || targetStrorageShape.GetDim(0) != logProbStrorageShape.GetDim(0)),
                    OP_LOGE(tilingContext,
                                                    "The dim 0 of target should be equal to the size of log_prob dim 1 and target dim num should be 1."),
                    return ge::GRAPH_FAILED);

    auto weightTensor = tilingContext->GetOptionalInputTensor(INPUT_WEIGHT_IDX);
    if (weightTensor != nullptr) {
        ceLossGradTilingKey.isWeight = TILING_KEY_TRUE;
        auto weightShape = EnsureNotScalar4CELoss(weightTensor->GetStorageShape());
        OP_CHECK_IF((weightShape.GetDimNum() != 1),
                        OP_LOGE(tilingContext,
                                                        "The dim 0 of log_prob should be equal to the size of target."),
                        return ge::GRAPH_FAILED);
        OP_CHECK_IF((weightShape.GetDim(0) != logProbShape->GetStorageShape().GetDim(1)),
                        OP_LOGE(tilingContext,
                                                        "The dim 1 of logProb should be equal to the size of weight."),
                        return ge::GRAPH_FAILED);
    }
    rowVal = logProbShape->GetStorageShape().GetDim(0);
    colVal = logProbShape->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossEntropyLossGradRegbaseTiling::GetTilingAttr() {
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    auto reduction = attrs->GetAttrPointer<char>(REDUCTION_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, reduction);
    if (strcmp(reduction, "none") == 0) {
        OP_LOGD(tilingContext, "Attr reduction is none.");
        ceLossGradTilingKey.reduction = TILING_KEY_NONE;
    } else if (strcmp(reduction, "mean") == 0) {
        OP_LOGD(tilingContext, "Attr reduction is mean.");
        ceLossGradTilingKey.reduction  = TILING_KEY_MEAN;
    } else if (strcmp(reduction, "sum") == 0) {
        OP_LOGD(tilingContext, "Attr reduction is sum.");
        ceLossGradTilingKey.reduction = TILING_KEY_SUM;
    } else {
      OP_LOGE(tilingContext, "reduction is not in none, mean or sum.");
      return ge::GRAPH_FAILED;
    }
    ceLossGradRegbaseTiling->reduction = static_cast<int64_t>(ceLossGradTilingKey.reduction);
    auto ignoreIndex = attrs->GetAttrPointer<int64_t>(IGNORE_INDEX_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, ignoreIndex);
    int64_t ignore = static_cast<int64_t>(*ignoreIndex);
    ceLossGradRegbaseTiling->ignoreIndex = ignore;
    if (ignore >= 0) {
        ceLossGradTilingKey.isIgnore = TILING_KEY_TRUE;
    }
    auto labelSmoothing = attrs->GetAttrPointer<float>(LABEL_SMOOTHING_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, labelSmoothing);
    float labelSmooth = static_cast<float>(*labelSmoothing);
    if (labelSmooth >=0 && labelSmooth <= 1) {
      ceLossGradRegbaseTiling->labelSmoothing = labelSmooth;
    } else {
      OP_LOGE(tilingContext, "the value range of labelSmoothing needs to be within [0, 1].");
      return ge::GRAPH_FAILED;
    }
    if (labelSmooth > 0) {
        ceLossGradTilingKey.labelS = TILING_KEY_TRUE;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossEntropyLossGradRegbaseTiling::DoEmptyTensorTiling() {
    uint64_t schEmpty = CROSS_ENTROPY_LOSS_GRAD_MODE_2;
    uint64_t zero = 0;
    OP_LOGI(tilingContext, "schId is %ld", schEmpty);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schEmpty, zero, zero, zero, zero);
    OP_LOGI(tilingContext, "tilingKey is %ld", tilingKey);
    tilingContext->SetTilingKey(tilingKey);

    size_t* workSpaces = tilingContext->GetWorkspaceSizes(1);
    workSpaces[0] = SYS_WORKSPACE_SIZE;
    tilingContext->SetBlockDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossEntropyLossGradRegbaseTiling::CheckDtype() {
    auto logProbDesc = tilingContext->GetInputDesc(INPUT_LOG_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, logProbDesc);
    auto dataType = logProbDesc->GetDataType();
    if (dataType == ge::DT_BF16) {
      dtypeSize = ge::GetSizeByDataType(dataType);
    } else if (dataType == ge::DT_FLOAT16) {
      dtypeSize = ge::GetSizeByDataType(dataType);
    } else if (dataType == ge::DT_FLOAT) {
      dtypeSize = ge::GetSizeByDataType(dataType);
    } else {
      OP_LOGE(tilingContext, "input log_prob dtype only support float32, float16, bfloat16.");
      return ge::GRAPH_FAILED;
    }

    auto gradLossDesc = tilingContext->GetInputDesc(INPUT_GRAD_LOSS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradLossDesc);
    auto gradLossDataType = gradLossDesc->GetDataType();
    OP_CHECK_IF(gradLossDataType != dataType,
                    OP_LOGE(tilingContext, "input grad_loss and log_prob dtype should be same."),
                    return ge::GRAPH_FAILED);

    auto weightDesc = tilingContext->GetOptionalInputDesc(INPUT_WEIGHT_IDX);
    if (weightDesc != nullptr) {
        auto weightDtype = weightDesc->GetDataType();
        OP_CHECK_IF(weightDtype != ge::DT_FLOAT,
                        OP_LOGE(tilingContext,
                                                        "optional input weight dtype only support float32."),
                        return ge::GRAPH_FAILED);
    }
    auto targetDesc = tilingContext->GetInputDesc(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, targetDesc);
    auto targetDataType = targetDesc->GetDataType();
    if (targetDataType == ge::DT_INT64) {
      targetDtypeSize = ge::GetSizeByDataType(targetDataType);
    } else if (targetDataType == ge::DT_INT32) {
        targetDtypeSize = ge::GetSizeByDataType(targetDataType);
    } else {
      OP_LOGE(tilingContext, "input target dtype only support int32 or int64.");
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossEntropyLossGradRegbaseTiling::RunTiling() {
    if (tilingContext == nullptr) {
        OP_LOGE("CrossEntropyLossGrad", "Tiling Context is nullptr");
    }
    OP_LOGD(tilingContext, "CrossEntropyLossGradRegbaseTiling RunTiling enter.");

    OP_CHECK_IF(GetPlatform() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext, "GetPlatform failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetTilingAttr() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext, "GetTilingAttr failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetTilingInput() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext, "GetTilingInput failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext, "CheckDtype failed."),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* inputShape = tilingContext->GetInputShape(INPUT_LOG_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputShape);
    const gert::Shape& inputStorageShape = inputShape->GetStorageShape();
    if (inputStorageShape.GetShapeSize() == 0){
      OP_LOGD(tilingContext, "input has empty tensor");
      OP_CHECK_IF(DoEmptyTensorTiling() != ge::GRAPH_SUCCESS,
                      OP_LOGE(tilingContext, "DoEmptyTensorTiling failed."),
                      return ge::GRAPH_FAILED);
      return ge::GRAPH_SUCCESS;
    }

    // 核间切分
    InitSplitInfo();
    OP_LOGD("CrossEntropyLossGradRegbaseTiling InitSplitInfo finished");

    // 核内切分
    if (!CELossGradInnerCoreFullLoad()){
        CELossGradCoreSplitInfo();
    }
    OP_LOGD("CrossEntropyLossGradRegbaseTiling CELossGradCoreSplitInfo finished");

    OP_LOGI(tilingContext, "schId is %u, reduction is %u, isWeight is %u, labelS is %u, isIgnore is %u", ceLossGradTilingKey.schId, ceLossGradTilingKey.reduction,
    ceLossGradTilingKey.isWeight, ceLossGradTilingKey.labelS, ceLossGradTilingKey.isIgnore);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(ceLossGradTilingKey.schId, ceLossGradTilingKey.reduction, ceLossGradTilingKey.isWeight, ceLossGradTilingKey.labelS, ceLossGradTilingKey.isIgnore);
    OP_LOGI(tilingContext, "tilingKey is %lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);

    OP_CHECK_IF(SetTilingData() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext, "SetTilingData failed."),
                    return ge::GRAPH_FAILED);
    OP_LOGD("CrossEntropyLossGradRegbaseTiling tiling end");
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
