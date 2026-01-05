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
 * \file ctc_loss_v2_grad_tiling_arch35.cpp
 * \brief
 */

#include "log/log.h"
#include "ctc_loss_v2_grad_tiling_arch35.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {
const std::string OP_GRAD_NAME = "CTCLossV2Grad";
constexpr int64_t T_DIM = 0;
constexpr int64_t N_DIM = 1;
constexpr int64_t C_DIM = 2;
constexpr int64_t BATCH_DIM = 0;
constexpr int64_t ALPHA_T_DIM = 1;
constexpr int64_t ALPHA_DIM = 2;

constexpr int64_t S_COE = 2;

constexpr int64_t INPUT_GRAD_OUT_IDX = 0;
constexpr int64_t INPUT_LOG_PROB_IDX = 1;
constexpr int64_t INPUT_TRAGETS_IDX = 2;
constexpr int64_t INPUT_INPUT_LENGTHS_IDX = 3;
constexpr int64_t INPUT_TARGET_LENGTHS_IDX = 4;
constexpr int64_t INPUT_LOSS_IDX = 5;
constexpr int64_t INPUT_LOG_ALPHA_IDX = 6;
constexpr int64_t LOSS_DIM_NUM = 1;
constexpr int64_t GRAD_DIM_NUM = 3;

constexpr int64_t FLOAT_DSIZE = 4;

constexpr int64_t GRAD_OUT_DIM_NUM = 1;
constexpr int64_t GRAD_OUT_DIM_INDEX = 1;
constexpr int64_t LOG_PROBS_DIM_NUM = 3;
constexpr int64_t TARGETS_DIM_NUM_ONE = 1;
constexpr int64_t TARGETS_DIM_NUM_TWO = 2;
constexpr int64_t TARGETS_N_DIM_INDEX = 0;
constexpr int64_t TARGETS_S_DIM_INDEX = 1;
constexpr int64_t INPUT_LENGTHS_DIM_NUM = 1;
constexpr int64_t INPUT_LENGTHS_N_DIM_INDEX = 0;
constexpr int64_t TARGET_LENGTHS_DIM_NUM = 1;
constexpr int64_t TARGET_LENGTHS_N_DIM_INDEX = 0;
constexpr int64_t OUTPUT_GRAD_INDEX = 0;
constexpr int64_t LOG_ALPHA_DIM_NUM = 3;

constexpr int64_t MAX_THREAD_NUM_1024 = 1024;
constexpr int64_t MAX_THREAD_NUM_256 = 256;
constexpr int64_t MAX_THREAD_NUM_512 = 512;
constexpr int64_t INIT_GM_NUM = 3;
constexpr int64_t MAX_UB_SIZE = 221184;

constexpr size_t TILING_KEY_INT32 = 0;
constexpr size_t TILING_KEY_INT64 = 1;

static const size_t INPUT_LENGTHS_IDX = 2;
static const size_t TARGET_LENGTHS_IDX = 3;

const std::string OP_NAME = "CTCLossV2Grad";

class CTCLossV2GradTiling4AscendC {
public:
    explicit CTCLossV2GradTiling4AscendC(gert::TilingContext* context) : context_(context){};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    bool CheckShapeInfo();
    void PrintTilingData() const;

private:
    bool InitParmaGradOut();
    bool InitLogAlpha();
    bool InitParamLogProbs();
    bool InitTargets();
    bool InitInputLengths();
    bool InitTargetLengths();
    bool InitNegLogLikelihood();
    bool InitGrad();
    void InitLogBetaThreadParams(int64_t maxThreadNum);
    void InitUpdateLcabThreadParams(int64_t maxThreadNum);
    void InitCalGradThreadParams(int64_t maxThreadNum);
    void InitGmParams();
    bool IsLargeSize() const;
    int64_t GetThreadNum();
    ge::graphStatus InitSimtParams();

private:
    CTCLossV2GradTilingData4AscnedC tilingData;
    gert::TilingContext* context_ = nullptr;
    int64_t coreUsed = 0;
    int64_t coreNum = 0;
    int64_t alphaLength = 0;

    int64_t maxInputLength = 1;
    int64_t sDimRange = 1;
    int64_t maxTargetLength = 1;
    int64_t symbolSet = 1;

    int64_t batchSize = 0;
    int64_t targetsDimNum = 1;
    int64_t targetsNum = 1;
    int64_t BLANK = 0;
    int64_t zeroInfinity = 0;
    int64_t gradOutN = 0;
    int64_t targetsN = 0;
    int64_t inputLengthsN = 0;
    int64_t targetLengthsN = 0;
    int64_t lossN = 0;
    int64_t logAlphaN = 0;
    int64_t gradN = 0;
    int64_t gradT = 0;
    int64_t gradC = 0;
    int64_t logAlphaT = 0;
    int64_t blockDimX = 0;
    int64_t blockDimY = 0;

    int64_t initGradGmStartBlock = 0;
    int64_t initGradGmEndBlock = 0;
    int64_t initGradGmSizePerBlock = 0;
    int64_t initGradGmSizeLastBlock = 0;

    int64_t initLogBetaGmStartBlock = 0;
    int64_t initLogBetaGmEndBlock = 0;
    int64_t initLogBetaGmSizePerBlock = 0;
    int64_t initLogBetaGmSizeLastBlock = 0;

    int64_t initTempGradGmStartBlock = 0;
    int64_t initTempGradGmEndBlock = 0;
    int64_t initTempGradGmSizePerBlock = 0;
    int64_t initTempGradGmSizeEndBlock = 0;

    int64_t logBetaThreadNum;
    int64_t updateLcabThreadNum;
    int64_t calGradThreadNum;
};

void CTCLossV2GradTiling4AscendC::InitGmParams()
{
    int64_t perBlockNum = coreNum / INIT_GM_NUM;
    initGradGmStartBlock = 0;
    initGradGmEndBlock = perBlockNum - 1;
    initGradGmSizePerBlock = (batchSize * symbolSet * maxInputLength) / perBlockNum;
    initGradGmSizeLastBlock = initGradGmSizePerBlock + (batchSize * symbolSet * maxInputLength) % perBlockNum;

    initLogBetaGmStartBlock = perBlockNum;
    initLogBetaGmEndBlock = S_COE * perBlockNum - 1;
    initLogBetaGmSizePerBlock = (batchSize * alphaLength * maxInputLength) / perBlockNum;
    initLogBetaGmSizeLastBlock = initLogBetaGmSizePerBlock + (batchSize * alphaLength * maxInputLength) % perBlockNum;

    initTempGradGmStartBlock = S_COE * perBlockNum;
    initTempGradGmEndBlock = coreNum - 1;
    initTempGradGmSizePerBlock = (batchSize * symbolSet * maxInputLength) / (coreNum - S_COE * perBlockNum);
    initTempGradGmSizeEndBlock =
        initTempGradGmSizePerBlock + (batchSize * symbolSet * maxInputLength) % (coreNum - S_COE * perBlockNum);
}

void CTCLossV2GradTiling4AscendC::InitCalGradThreadParams(const int64_t maxThreadNum)
{
    int64_t totalThreadNum = batchSize * maxInputLength * symbolSet;
    int64_t threadNumPerBlock = (totalThreadNum + coreNum - 1) / coreNum;
    calGradThreadNum = threadNumPerBlock < maxThreadNum ? threadNumPerBlock : maxThreadNum;
}

void CTCLossV2GradTiling4AscendC::InitUpdateLcabThreadParams(const int64_t maxThreadNum)
{
    int64_t totalThreadNum = batchSize * maxInputLength;
    int64_t threadNumPerBlock = (totalThreadNum + coreNum - 1) / coreNum;
    updateLcabThreadNum = threadNumPerBlock < maxThreadNum ? threadNumPerBlock : maxThreadNum;
}

void CTCLossV2GradTiling4AscendC::InitLogBetaThreadParams(const int64_t maxThreadNum)
{
    int64_t threadsTarget = maxThreadNum;
    if (threadsTarget >= S_COE * maxTargetLength + 1) {
        threadsTarget = S_COE * maxTargetLength + 1;
    }
    // 计算blockDimY的值
    int64_t threadsMaxBatch = maxThreadNum / threadsTarget;
    int64_t threadsPerBlockBatch = (batchSize + coreNum - 1) / coreNum;
    blockDimX = threadsTarget;
    blockDimY = threadsPerBlockBatch < threadsMaxBatch ? threadsPerBlockBatch : threadsMaxBatch;
    logBetaThreadNum = blockDimX * blockDimY;
}

bool CTCLossV2GradTiling4AscendC::IsLargeSize() const
{
    return (batchSize > INT32_MAX) || (symbolSet > INT32_MAX) || (maxInputLength > INT32_MAX) ||
           (alphaLength > INT32_MAX) || (batchSize * symbolSet > INT32_MAX) ||
           (batchSize * logAlphaT * alphaLength > INT32_MAX);
}

int64_t CTCLossV2GradTiling4AscendC::GetThreadNum()
{
    if (IsLargeSize()) {
        return MAX_THREAD_NUM_512;
    }
    return MAX_THREAD_NUM_1024;
}
ge::graphStatus CTCLossV2GradTiling4AscendC::InitSimtParams()
{
    OP_LOGD(context_->GetNodeName(), "CTCLossV2Grad SIMT tiling running");
    int64_t maxThreadNum = GetThreadNum();
    InitLogBetaThreadParams(maxThreadNum);
    InitUpdateLcabThreadParams(maxThreadNum);
    InitCalGradThreadParams(maxThreadNum);
    InitGmParams();

    size_t sysWorkspaceSize = static_cast<size_t>(16 * 1024 * 1024);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize +
                          static_cast<size_t>(maxInputLength * batchSize * alphaLength * FLOAT_DSIZE) +
                          static_cast<size_t>(maxInputLength * batchSize * symbolSet * FLOAT_DSIZE);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CTCLossV2GradTiling4AscendC::RunKernelTiling()
{
    if (IsLargeSize()) {
        context_->SetTilingKey(TILING_KEY_INT64);
    } else {
        context_->SetTilingKey(TILING_KEY_INT32);
    }
    context_->SetBlockDim(coreNum);
    context_->SetScheduleMode(1);   // 全核同步
    tilingData.set_alphaLength(alphaLength);
    tilingData.set_maxInputLength(maxInputLength);
    tilingData.set_symbolSet(symbolSet);
    tilingData.set_batchSize(batchSize);
    tilingData.set_sDimRange(sDimRange);
    tilingData.set_targetsDimNum(targetsDimNum);
    tilingData.set_targetsNum(targetsNum);
    tilingData.set_BLANK(BLANK);
    tilingData.set_zeroInfinity(zeroInfinity);
    tilingData.set_blockDimX(blockDimX);
    tilingData.set_blockDimY(blockDimY);
    tilingData.set_logAlphaT(logAlphaT);
    tilingData.set_initGradGmStartBlock(initGradGmStartBlock);
    tilingData.set_initGradGmEndBlock(initGradGmEndBlock);
    tilingData.set_initGradGmSizePerBlock(initGradGmSizePerBlock);
    tilingData.set_initGradGmSizeLastBlock(initGradGmSizeLastBlock);
    tilingData.set_initLogBetaGmStartBlock(initLogBetaGmStartBlock);
    tilingData.set_initLogBetaGmEndBlock(initLogBetaGmEndBlock);
    tilingData.set_initLogBetaGmSizePerBlock(initLogBetaGmSizePerBlock);
    tilingData.set_initLogBetaGmSizeLastBlock(initLogBetaGmSizeLastBlock);
    tilingData.set_initTempGradGmStartBlock(initTempGradGmStartBlock);
    tilingData.set_initTempGradGmEndBlock(initTempGradGmEndBlock);
    tilingData.set_initTempGradGmSizePerBlock(initTempGradGmSizePerBlock);
    tilingData.set_initTempGradGmSizeEndBlock(initTempGradGmSizeEndBlock);
    tilingData.set_logBetaThreadNum(logBetaThreadNum);
    tilingData.set_updateLcabThreadNum(updateLcabThreadNum);
    tilingData.set_calGradThreadNum(calGradThreadNum);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context_->SetLocalMemorySize(MAX_UB_SIZE);
    return ge::GRAPH_SUCCESS;
}

void CTCLossV2GradTiling4AscendC::PrintTilingData() const
{
    auto nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "Start printing");
    OP_LOGD(nodeName, "alphaLength is %ld.", alphaLength);
    OP_LOGD(nodeName, "maxInputLength is %ld.", maxInputLength);
    OP_LOGD(nodeName, "symbolSet is %ld.", symbolSet);
    OP_LOGD(nodeName, "batchSize is %ld.", batchSize);
    OP_LOGD(nodeName, "targetsDimNum is %ld.", targetsDimNum);
    OP_LOGD(nodeName, "sDimRange is %ld.", sDimRange);
    OP_LOGD(nodeName, "targetsNum is %ld.", targetsNum);
    OP_LOGD(nodeName, "BLANK is %ld.", BLANK);
    OP_LOGD(nodeName, "zeroInfinity is %ld.", zeroInfinity);
    OP_LOGD(nodeName, "blockDimX is %ld.", blockDimX);
    OP_LOGD(nodeName, "blockDimY is %ld.", blockDimY);
    OP_LOGD(nodeName, "logAlphaT is %ld.", logAlphaT);
    OP_LOGD(nodeName, "coreNum is %ld.", coreNum);
    OP_LOGD(nodeName, "initGradGmStartBlock  is %ld.", initGradGmStartBlock);
    OP_LOGD(nodeName, "initGradGmEndBlock is %ld.", initGradGmEndBlock);
    OP_LOGD(nodeName, "initGradGmSizePerBlock is %ld.", initGradGmSizePerBlock);
    OP_LOGD(nodeName, "initGradGmSizeLastBlock is %ld.", initGradGmSizeLastBlock);
    OP_LOGD(nodeName, "initLogBetaGmStartBlock is %ld.", initLogBetaGmStartBlock);
    OP_LOGD(nodeName, "initLogBetaGmEndBlock is %ld.", initLogBetaGmEndBlock);
    OP_LOGD(nodeName, "initLogBetaGmSizePerBlock is %ld.", initLogBetaGmSizePerBlock);
    OP_LOGD(nodeName, "initLogBetaGmSizeLastBlock is %ld.", initLogBetaGmSizeLastBlock);
    OP_LOGD(nodeName, "initTempGradGmStartBlock is %ld.", initTempGradGmStartBlock);
    OP_LOGD(nodeName, "initTempGradGmEndBlock is %ld.", initTempGradGmEndBlock);
    OP_LOGD(nodeName, "initTempGradGmSizePerBlock is %ld.", initTempGradGmSizePerBlock);
    OP_LOGD(nodeName, "initTempGradGmSizeEndBlock is %ld.", initTempGradGmSizeEndBlock);
    OP_LOGD(nodeName, "logBetaThreadNum is %ld.", logBetaThreadNum);
    OP_LOGD(nodeName, "updateLcabThreadNum is %ld.", updateLcabThreadNum);
    OP_LOGD(nodeName, "calGradThreadNum is %ld.", calGradThreadNum);
    OP_LOGD(nodeName, "End printing");
    OP_LOGD(nodeName, "CTCLossV2Grad tiling end running");
}

ge::graphStatus CTCLossV2GradTiling4AscendC::Init()
{
    OP_LOGD(context_->GetNodeName(), "CTCLossV2Grad tiling starts running");
    auto compileInfo = static_cast<const CTCLossV2GradForCompileInfo*>(context_->GetCompileInfo());

    coreNum = compileInfo->totalCoreNum;

    if (!CheckShapeInfo()) {
        return ge::GRAPH_FAILED;
    }

    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const auto* blankPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blankPtr);
    BLANK = *blankPtr;
    OP_CHECK_IF(
        BLANK < 0 || BLANK >= symbolSet,
        OP_LOGE(context_->GetNodeName(), "BLANK is out of the range, please input the right value."),
        return ge::GRAPH_FAILED);
    const auto* zeroInfinityPtr = attrs->GetAttrPointer<int64_t>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, zeroInfinityPtr);
    zeroInfinity = *zeroInfinityPtr;
    return InitSimtParams();
}

bool CTCLossV2GradTiling4AscendC::InitGrad()
{
    auto const gradShape = context_->GetOutputShape(OUTPUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradShape);
    auto const gradStorageShape = gradShape->GetStorageShape();
    OP_CHECK_IF(
        gradStorageShape.GetDimNum() != GRAD_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "Check grad shape failed, the dims of grad not equal 3."), return false);
    gradN = gradStorageShape.GetDim(N_DIM);
    gradT = gradStorageShape.GetDim(T_DIM);
    gradC = gradStorageShape.GetDim(C_DIM);
    return true;
}

bool CTCLossV2GradTiling4AscendC::InitLogAlpha()
{
    auto const logAlphaShape = context_->GetInputShape(INPUT_LOG_ALPHA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, logAlphaShape);
    auto const logAlphaShapeVal = logAlphaShape->GetStorageShape();
    OP_CHECK_IF(
        logAlphaShapeVal.GetDimNum() != LOG_ALPHA_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "Check log_alpha shape failed, the dims of log_alpha not equal 3."),
        return false);
    logAlphaN = logAlphaShapeVal.GetDim(BATCH_DIM);
    logAlphaT = logAlphaShapeVal.GetDim(ALPHA_T_DIM);
    alphaLength = logAlphaShapeVal.GetDim(ALPHA_DIM);
    maxTargetLength = (alphaLength - 1) / S_COE;
    return true;
}

bool CTCLossV2GradTiling4AscendC::InitNegLogLikelihood()
{
    auto const lossShape = context_->GetInputShape(INPUT_LOSS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, lossShape);
    auto const lossShapeVal = lossShape->GetStorageShape();
    OP_CHECK_IF(
        lossShapeVal.GetDimNum() != LOSS_DIM_NUM,
        OP_LOGE(
            context_->GetNodeName(),
            "Check neg_log_likelihood shape failed, the dims of neg_log_likelihood not equal 1."),
        return false);
    lossN = lossShapeVal.GetDim(BATCH_DIM);
    return true;
}

bool CTCLossV2GradTiling4AscendC::InitTargetLengths()
{
    auto const targetLengthsShape = context_->GetInputShape(INPUT_TARGET_LENGTHS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, targetLengthsShape);
    auto const targetLengthsStorageShape = targetLengthsShape->GetStorageShape();
    OP_CHECK_IF(
        targetLengthsStorageShape.GetDimNum() != TARGET_LENGTHS_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "Check target_lengths shape failed, the dims of target_lengths not equal 1."),
        return false);
    targetLengthsN = targetLengthsStorageShape.GetDim(TARGET_LENGTHS_N_DIM_INDEX);
    return true;
}

bool CTCLossV2GradTiling4AscendC::InitInputLengths()
{
    auto const inputLengthsShape = context_->GetInputShape(INPUT_INPUT_LENGTHS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLengthsShape);
    auto const inputLengthsStorageShape = inputLengthsShape->GetStorageShape();
    OP_CHECK_IF(
        inputLengthsStorageShape.GetDimNum() != INPUT_LENGTHS_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "Check input_lengths shape failed, the dims of input_lengths not equal 1."),
        return false);
    inputLengthsN = inputLengthsStorageShape.GetDim(INPUT_LENGTHS_N_DIM_INDEX);
    return true;
}

bool CTCLossV2GradTiling4AscendC::InitTargets()
{
    auto const targetsShape = context_->GetInputShape(INPUT_TRAGETS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, targetsShape);
    auto const targetsStorageShape = targetsShape->GetStorageShape();
    targetsDimNum = targetsStorageShape.GetDimNum();
    OP_CHECK_IF(
        targetsDimNum != TARGETS_DIM_NUM_ONE && targetsDimNum != TARGETS_DIM_NUM_TWO,
        OP_LOGE(context_->GetNodeName(), "Check targets shape failed, the dims of targets not equal 1 or 2."),
        return false);
    targetsN = targetsNum = targetsStorageShape.GetDim(TARGETS_N_DIM_INDEX);
    if (targetsDimNum > TARGETS_DIM_NUM_ONE) {
        targetsNum = targetsNum * targetsStorageShape.GetDim(TARGETS_S_DIM_INDEX);
        sDimRange = targetsStorageShape.GetDim(TARGETS_S_DIM_INDEX);
    }
    return true;
}

bool CTCLossV2GradTiling4AscendC::InitParamLogProbs()
{
    auto const logProbsShape = context_->GetInputShape(INPUT_LOG_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, logProbsShape);
    auto const logProbsStorageShape = logProbsShape->GetStorageShape();
    OP_CHECK_IF(
        logProbsStorageShape.GetDimNum() != GRAD_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "Check log_probs shape failed, the dims of log_probs not equal 3."),
        return false);
    maxInputLength = logProbsStorageShape.GetDim(T_DIM);
    batchSize = logProbsStorageShape.GetDim(N_DIM);
    symbolSet = logProbsStorageShape.GetDim(C_DIM);
    return true;
}

bool CTCLossV2GradTiling4AscendC::InitParmaGradOut()
{
    auto const gradOutShape = context_->GetInputShape(INPUT_GRAD_OUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradOutShape);
    auto const gradOutStoreShape = gradOutShape->GetStorageShape();
    OP_CHECK_IF(
        gradOutStoreShape.GetDimNum() != GRAD_OUT_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "Check grad_out shape failed, the dims of grad_out not equal 1."),
        return false);
    gradOutN = gradOutStoreShape.GetDim(GRAD_OUT_DIM_INDEX);
    return true;
}

bool CTCLossV2GradTiling4AscendC::CheckShapeInfo()
{
    auto nodeName = context_->GetNodeName();
    if (!InitParamLogProbs() || !InitParmaGradOut() || !InitTargets() || !InitInputLengths()) {
        return false;
    }
    if (!InitTargetLengths() || !InitNegLogLikelihood() || !InitGrad() || !InitLogAlpha()) {
        return false;
    }
    OP_LOGD(nodeName, "gradOutN is %ld.", gradOutN);
    OP_LOGD(nodeName, "batchSize is %ld.", batchSize);
    OP_LOGD(nodeName, "inputLengthsN is %ld.", inputLengthsN);
    OP_LOGD(nodeName, "targetLengthsN is %ld.", targetLengthsN);
    OP_LOGD(nodeName, "lossN is %ld.", lossN);
    OP_LOGD(nodeName, "logAlphaN is %ld.", logAlphaN);
    OP_LOGD(nodeName, "gradN is %ld.", gradN);
    OP_LOGD(nodeName, "targetsDimNum is %ld.", targetsDimNum);
    OP_LOGD(nodeName, "targetsN is %ld.", targetsN);
    bool NCheck = gradOutN == batchSize && batchSize == inputLengthsN && inputLengthsN == targetLengthsN &&
                  targetLengthsN == lossN && lossN == logAlphaN && logAlphaN == gradN;
    NCheck = targetsDimNum > 1 ? (NCheck && (batchSize == targetsN)) : NCheck;
    OP_CHECK_IF(!NCheck, OP_LOGE(nodeName, "Check batchSize failed."), return false);
    OP_LOGD(nodeName, "maxInputLength is %ld.", maxInputLength);
    OP_LOGD(nodeName, "gradT is %ld.", gradT);
    OP_LOGD(nodeName, "logAlphaT is %ld.", logAlphaT);
    bool TCheck = maxInputLength == gradT && gradT == logAlphaT;
    OP_CHECK_IF(!TCheck, OP_LOGE(nodeName, "Check max time failed."), return false);

    bool CCheck = symbolSet == gradC;
    OP_CHECK_IF(!CCheck, OP_LOGE(nodeName, "Check symbolSet failed."), return false);
    return true;
}

ge::graphStatus Tiling4CTCLossV2GradForAscendC(gert::TilingContext* context)
{
    CTCLossV2GradTiling4AscendC tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CTCLossV2GradTiling4AscendC init failed!");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

static ge::graphStatus Tiling4CTCLossV2Grad(gert::TilingContext* context)
{
    OP_LOGI(OP_GRAD_NAME, "ascendc value is true, start Tiling4CTCLossV2GradForAscendC");
    return Tiling4CTCLossV2GradForAscendC(context);
}

static ge::graphStatus TilingPrepare4CTCLossV2Grad([[maybe_unused]] gert::TilingParseContext* context)
{
    auto compileInfo4AscendC = context->GetCompiledInfo<CTCLossV2GradForCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo4AscendC);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo4AscendC->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo4AscendC->totalCoreNum <= 0),
        OP_LOGE(context->GetNodeName(), "TilingPrepare4CTCLossV2Grad fail to get core num."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CTCLossV2Grad)
    .Tiling(Tiling4CTCLossV2Grad)
    .TilingParse<CTCLossV2GradForCompileInfo>(TilingPrepare4CTCLossV2Grad)
    .InputsDataDependency({INPUT_LENGTHS_IDX, TARGET_LENGTHS_IDX});
} // namespace optiling