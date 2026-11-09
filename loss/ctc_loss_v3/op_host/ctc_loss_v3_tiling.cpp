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
 * \file ctc_loss_v3_tiling.cpp
 * \brief
 */
#include "ctc_loss_v3_tiling.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace std;

namespace optiling {
namespace {
ge::graphStatus ReportTilingFailure(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}
} // namespace
constexpr uint64_t RESERVE_SAPCE = 1.5 * 1024;
constexpr int64_t SELECT_SPACE = 8 * 1024;
constexpr uint32_t FLOAT_DTYPE_BYTES = 4;
constexpr int64_t T_DIM = 0;
constexpr int64_t N_DIM = 1;
constexpr int64_t C_DIM = 2;
constexpr int64_t BATCH_DIM = 0;
constexpr int64_t S_DIM = 2;
constexpr int64_t EIGHT_ALIGN = 8;
constexpr int64_t INPUT_LOG_PROB_IDX = 0;
constexpr int64_t INPUT_TRAGETS_IDX = 1;
constexpr int64_t INPUT_INPUT_LENGTHS_IDX = 2;
constexpr int64_t INPUT_TARGET_LENGTHS_IDX = 3;
constexpr int64_t LOSS_DIM_NUM = 1;
constexpr int64_t PROB_DIM_NUM = 3;
constexpr int64_t TARGETS_DIM_NUM = 2;
constexpr int64_t MAX_SYMBOL_SET_LEN_V3 = 200000;
constexpr int64_t MAX_LABEL_LEN = 1000;
constexpr int64_t MAX_BATCH = 8180;
constexpr int64_t LAGRE_BATCH = 1500;
constexpr int64_t NUM_PER_REPEAT = 64;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t FLOAT_DSIZE = 4;
constexpr int64_t INT64_DSIZE = 8;
constexpr int64_t MASKNUM_PER_BYTES = 8;
constexpr int64_t WORKSPACE_ALPHA_NUM = 3;
constexpr int64_t ZERO_INFINITY_IDX = 2;
constexpr int64_t NUM_SEVEN = 7;
constexpr int64_t NUM_SIX = 6;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_TWELVE = 12;
const std::string OP_NAME = "CTCLossV3";

class CTCLossV3Tiling {
public:
    explicit CTCLossV3Tiling(gert::TilingContext* ctx) : context(ctx) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    bool CheckShapeInfo();
    bool CheckShapeInfoForN();
    bool CheckShapeInfoForCT();
    bool CheckDtypeInfo() const;
    void PrintTilingData() const;
    void TilingDataPrint();
    bool GetMaxTargetLengths();

private:
    CTCLossV3TilingData tilingData;
    gert::TilingContext* context = nullptr;
    int64_t useWorkspaceSize = 0;
    int64_t coreUsed = 48;
    int64_t sliceLength = 1;
    int64_t sliceLengthTail = 1;
    int64_t probSliceNum = 1;
    int64_t alphaLength = 0;

    int64_t timeStep = 1;
    int64_t targetsDimLength = 0;
    int64_t maxTargetLength = 1;
    int64_t symbleSet = 1;

    int64_t batchSize = 192 * 1024;
    int64_t targetsDimNum = 1;
    int64_t targetsNum = 1;
    int64_t taskPerCore = 1;
    int64_t taskTailCore = 1;
    int64_t blankAtr = 0;
    int64_t zeroInfinity = 0;
    int64_t targetDsize = 0;
    int64_t logProbDszie = 0;
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
};

bool CTCLossV3Tiling::CheckDtypeInfo() const
{
    auto nodeName = context->GetNodeName();
    auto logProbsDesc = context->GetInputDesc(INPUT_LOG_PROB_IDX);
    OP_CHECK_IF(
        logProbsDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "logProbsDesc",
                                                 "failed to get the input desc of log_probs from the tiling context"),
        return false);
    auto logProbsDtype = logProbsDesc->GetDataType();
    if (logProbsDtype != ge::DT_FLOAT16 && logProbsDtype != ge::DT_FLOAT && logProbsDtype != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "log_probs", Ops::Base::ToString(logProbsDtype).c_str(),
                                  "fp16, bf16 or fp32");
        return false;
    }
    auto targetsDesc = context->GetInputDesc(INPUT_TRAGETS_IDX);
    OP_CHECK_IF(
        targetsDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "targetsDesc",
                                                 "failed to get the input desc of targets from the tiling context"),
        return false);
    auto targetsDtype = targetsDesc->GetDataType();
    if (targetsDtype != ge::DataType::DT_INT32 && targetsDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "targets", Ops::Base::ToString(targetsDtype).c_str(), "int32 or int64");
        return false;
    }
    auto inputLengthsDesc = context->GetInputDesc(INPUT_INPUT_LENGTHS_IDX);
    OP_CHECK_IF(inputLengthsDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "inputLengthsDesc",
                    "failed to get the input desc of input_lengths from the tiling context"),
                return false);
    auto inputLengthsDtype = inputLengthsDesc->GetDataType();
    if (inputLengthsDtype != ge::DataType::DT_INT32 && inputLengthsDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "input_lengths", Ops::Base::ToString(inputLengthsDtype).c_str(),
                                  "int32 or int64");
        return false;
    }
    auto targetLengthsDesc = context->GetInputDesc(INPUT_TARGET_LENGTHS_IDX);
    OP_CHECK_IF(targetLengthsDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "targetLengthsDesc",
                    "failed to get the input desc of target_lengths from the tiling context"),
                return false);
    auto targetLengthsDtype = targetLengthsDesc->GetDataType();
    if (targetLengthsDtype != ge::DataType::DT_INT32 && targetLengthsDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "target_lengths", Ops::Base::ToString(targetLengthsDtype).c_str(),
                                  "int32 or int64");
        return false;
    }
    auto lossDesc = context->GetOutputDesc(0);
    OP_CHECK_IF(lossDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "lossDesc",
                    "failed to get the output desc of neg_log_likelihood from the tiling context"),
                return false);
    auto lossDtype = lossDesc->GetDataType();
    if (lossDtype != ge::DT_FLOAT16 && lossDtype != ge::DT_FLOAT && lossDtype != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "neg_log_likelihood", Ops::Base::ToString(lossDtype).c_str(),
                                  "fp16, bf16 or fp32");
        return false;
    }
    auto logAlphaDesc = context->GetOutputDesc(1);
    OP_CHECK_IF(
        logAlphaDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "logAlphaDesc",
                                                 "failed to get the output desc of log_alpha from the tiling context"),
        return false);
    auto logAlphaDtype = logAlphaDesc->GetDataType();
    if (logAlphaDtype != ge::DT_FLOAT16 && logAlphaDtype != ge::DT_FLOAT && logAlphaDtype != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "log_alpha", Ops::Base::ToString(logAlphaDtype).c_str(),
                                  "fp16, bf16 or fp32");
        return false;
    }
    if (logProbsDtype != lossDtype || lossDtype != logAlphaDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            nodeName, "log_probs, neg_log_likelihood and log_alpha",
            Ops::Base::ToString(logProbsDtype) + ", " + Ops::Base::ToString(lossDtype) + ", " +
                Ops::Base::ToString(logAlphaDtype),
            "The dtypes of log_probs, neg_log_likelihood and log_alpha must be the same");
        return false;
    }
    return true;
}

bool CTCLossV3Tiling::CheckShapeInfoForN()
{
    // Check the shape info for some inputs, focusing mainly on batchSize.
    auto nodeName = context->GetNodeName();
    auto const inputLengthsShape = context->GetInputShape(INPUT_INPUT_LENGTHS_IDX);
    OP_CHECK_IF(inputLengthsShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "inputLengthsShape",
                    "failed to get the input shape of input_lengths from the tiling context"),
                return false);
    auto const inputLengthsShapeVal = inputLengthsShape->GetStorageShape();
    OP_CHECK_IF(inputLengthsShapeVal.GetDimNum() != LOSS_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "input_lengths",
                                                         std::to_string(inputLengthsShapeVal.GetDimNum()).c_str(),
                                                         "the shape dim of input_lengths should be 1"),
                return false);
    inputLengthsN = inputLengthsShapeVal.GetDim(BATCH_DIM);
    auto const targetLengthsShape = context->GetInputShape(INPUT_TARGET_LENGTHS_IDX);
    OP_CHECK_IF(targetLengthsShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "targetLengthsShape",
                    "failed to get the input shape of target_lengths from the tiling context"),
                return false);
    auto const targetLengthsShapeVal = targetLengthsShape->GetStorageShape();
    OP_CHECK_IF(targetLengthsShapeVal.GetDimNum() != LOSS_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "target_lengths",
                                                         std::to_string(targetLengthsShapeVal.GetDimNum()).c_str(),
                                                         "the shape dim of target_lengths should be 1"),
                return false);
    targetLengthsN = targetLengthsShapeVal.GetDim(BATCH_DIM);
    auto const lossShape = context->GetOutputShape(0);
    OP_CHECK_IF(lossShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "lossShape",
                    "failed to get the output shape of neg_log_likelihood from the tiling context"),
                return false);
    auto const lossShapeVal = lossShape->GetStorageShape();
    OP_CHECK_IF(lossShapeVal.GetDimNum() != LOSS_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "neg_log_likelihood",
                                                         std::to_string(lossShapeVal.GetDimNum()).c_str(),
                                                         "the shape dim of neg_log_likelihood should be 1"),
                return false);
    lossN = lossShapeVal.GetDim(BATCH_DIM);
    return true;
}

bool CTCLossV3Tiling::CheckShapeInfoForCT()
{
    // Check the shape information for some inputs and outputs, focusing mainly on symbleSet and time.
    auto nodeName = context->GetNodeName();
    auto const logProbsShape = context->GetInputShape(INPUT_LOG_PROB_IDX);
    OP_CHECK_IF(
        logProbsShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "logProbsShape",
                                                 "failed to get the input shape of log_probs from the tiling context"),
        return false);
    auto const logProbsShapeVal = logProbsShape->GetStorageShape();
    OP_CHECK_IF(logProbsShapeVal.GetDimNum() != PROB_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "log_probs",
                                                         std::to_string(logProbsShapeVal.GetDimNum()).c_str(),
                                                         "the shape dim of log_probs should be 3"),
                return false);
    timeStep = logProbsShapeVal.GetDim(T_DIM);
    batchSize = logProbsShapeVal.GetDim(N_DIM);
    symbleSet = logProbsShapeVal.GetDim(C_DIM);
    auto const targetsShape = context->GetInputShape(INPUT_TRAGETS_IDX);
    OP_CHECK_IF(
        targetsShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "targetsShape",
                                                 "failed to get the input shape of targets from the tiling context"),
        return false);
    auto const targetsShapeVal = targetsShape->GetStorageShape();
    targetsDimNum = targetsShapeVal.GetDimNum();
    OP_CHECK_IF(targetsDimNum != LOSS_DIM_NUM && targetsDimNum != TARGETS_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "targets", std::to_string(targetsDimNum).c_str(),
                                                         "the shape dim of targets should be 1 or 2"),
                return false);
    targetsN = targetsNum = targetsShapeVal.GetDim(BATCH_DIM);
    auto ret = GetMaxTargetLengths();
    if (!ret) {
        return false;
    }
    if (targetsDimNum > LOSS_DIM_NUM) {
        targetsNum = targetsNum * targetsShapeVal.GetDim(1);
        targetsDimLength = targetsShapeVal.GetDim(1);
    } else {
        targetsDimLength = maxTargetLength;
    }

    auto const logAlphaShape = context->GetOutputShape(1);
    OP_CHECK_IF(
        logAlphaShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "logAlphaShape",
                                                 "failed to get the output shape of log_alpha from the tiling context"),
        return false);
    auto const logAlphaShapeVal = logAlphaShape->GetStorageShape();
    OP_CHECK_IF(logAlphaShapeVal.GetDimNum() != PROB_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "log_alpha",
                                                         std::to_string(logAlphaShapeVal.GetDimNum()).c_str(),
                                                         "the shape dim of log_alpha should be 3"),
                return false);
    logAlphaN = logAlphaShapeVal.GetDim(T_DIM);
    logAlphaT = logAlphaShapeVal.GetDim(N_DIM);
    alphaLength = logAlphaShapeVal.GetDim(C_DIM);
    return true;
}

bool CTCLossV3Tiling::GetMaxTargetLengths()
{
    const gert::Tensor* targetLengthsTensor = context->GetInputTensor(INPUT_TARGET_LENGTHS_IDX);
    auto targetLengthsDesc = context->GetInputDesc(INPUT_TARGET_LENGTHS_IDX);
    OP_CHECK_IF(targetLengthsDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "targetLengthsDesc",
                    "failed to get the input desc of target_lengths from the tiling context"),
                return false);
    auto targetLengthsDtype = targetLengthsDesc->GetDataType();
    if (targetLengthsDtype == ge::DataType::DT_INT64) {
        const int64_t* targetLengthsValue = targetLengthsTensor->GetData<int64_t>();
        OP_CHECK_IF(targetLengthsValue == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "target_lengths",
                        "failed to get the const input data of target_lengths from the tiling context"),
                    return false);
        for (int64_t i = 0; i < batchSize; i++) {
            maxTargetLength = std::max(maxTargetLength, static_cast<int64_t>(targetLengthsValue[i]));
        }
    } else if (targetLengthsDtype == ge::DataType::DT_INT32) {
        int32_t maxTargetLengthTmp = 1;
        const int32_t* targetLengthsValue = targetLengthsTensor->GetData<int32_t>();
        OP_CHECK_IF(targetLengthsValue == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "target_lengths",
                        "failed to get the const input data of target_lengths from the tiling context"),
                    return false);
        for (int64_t i = 0; i < batchSize; i++) {
            maxTargetLengthTmp = std::max(maxTargetLengthTmp, static_cast<int32_t>(targetLengthsValue[i]));
        }
        maxTargetLength = maxTargetLengthTmp;
    }
    return true;
}

bool CTCLossV3Tiling::CheckShapeInfo()
{
    // Check the shape information for all the inputs and outputs.
    auto nodeName = context->GetNodeName();
    if (!CheckShapeInfoForN() || !CheckShapeInfoForCT()) {
        return false;
    }

    bool NCheck = batchSize == inputLengthsN && inputLengthsN == targetLengthsN && targetLengthsN == lossN &&
                  lossN == logAlphaN;
    NCheck = targetsDimNum > 1 ? (NCheck && (batchSize == targetsN)) : NCheck;
    OP_CHECK_IF(
        !NCheck,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            nodeName, "log_probs, targets, input_lengths, target_lengths, neg_log_likelihood and log_alpha",
            std::to_string(batchSize) + ", " + std::to_string(targetsN) + ", " + std::to_string(inputLengthsN) + ", " +
                std::to_string(targetLengthsN) + ", " + std::to_string(lossN) + ", " + std::to_string(logAlphaN),
            "The batch dims of log_probs, targets, input_lengths, target_lengths, neg_log_likelihood and "
            "log_alpha must be the same"),
        return false);

    OP_CHECK_IF(logAlphaN > MAX_BATCH,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "log_alpha", std::to_string(logAlphaN),
                                                      "the batch size should be less than or equal to 8180, "
                                                      "AICPU is recommended"),
                return false);

    bool TCheck = timeStep == logAlphaT;
    OP_CHECK_IF(!TCheck,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "log_probs and log_alpha",
                                                       std::to_string(timeStep) + ", " + std::to_string(logAlphaT),
                                                       "The time dims of log_probs and log_alpha must be the same"),
                return false);
    OP_CHECK_IF(symbleSet > MAX_SYMBOL_SET_LEN_V3 || maxTargetLength > MAX_LABEL_LEN,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    nodeName, "log_probs and target_lengths",
                    std::to_string(symbleSet) + ", " + std::to_string(maxTargetLength),
                    "the num classes of log_probs should be less than 200000 and the max value of target_lengths "
                    "should be less than 1000, AICPU is recommended"),
                return false);
    return true;
}

ge::graphStatus CTCLossV3Tiling::Init()
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(context, "CTCLossV3 tiling starts running");
    auto compileInfo = static_cast<const CTCLossV3CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "compileInfo",
                                                         "failed to get the compile info from the tiling context"),
                return ge::GRAPH_FAILED);

    uint32_t coreNum = compileInfo->coreNum;
    int64_t ubSize = compileInfo->ubSizePlatForm - RESERVE_SAPCE - SELECT_SPACE;
    if (!CheckDtypeInfo() || !CheckShapeInfo()) {
        return ge::GRAPH_FAILED;
    }
    auto targetDtype = context->GetInputDesc(INPUT_TRAGETS_IDX)->GetDataType();
    auto logProbDtype = context->GetInputDesc(INPUT_LOG_PROB_IDX)->GetDataType();
    targetDsize = GetSizeByDataType(targetDtype);
    logProbDszie = GetSizeByDataType(logProbDtype);

    auto* attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "attrs",
                                                         "failed to get attrs from the tiling context"),
                return ge::GRAPH_FAILED);
    const auto* blankPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_IF(blankPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "blankPtr",
                                                         "failed to get the blank attribute from the tiling context"),
                return ge::GRAPH_FAILED);
    blankAtr = *blankPtr;
    OP_CHECK_IF(blankAtr < 0 || blankAtr >= symbleSet,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "blank", std::to_string(blankAtr),
                                                      "blank should be in the range [0, num_classes)"),
                return false);
    const auto* zeroInfinityPtr = attrs->GetAttrPointer<int64_t>(ZERO_INFINITY_IDX);
    OP_CHECK_IF(
        zeroInfinityPtr == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "zeroInfinityPtr",
                                                 "failed to get the zero_infinity attribute from the tiling context"),
        return ge::GRAPH_FAILED);
    zeroInfinity = *zeroInfinityPtr;
    alphaLength = NUM_TWO * maxTargetLength + 1; // alphaLength = 2 * maxTargetLength + 1
    int64_t alphaTailSizeAlign = (alphaLength + NUM_SEVEN) / EIGHT_ALIGN * EIGHT_ALIGN;
    coreUsed = batchSize > coreNum ? coreNum : batchSize;
    taskPerCore = batchSize / coreNum;
    taskTailCore = batchSize % coreNum;

    int64_t alphaLengthAlign = Ops::Base::CeilAlign(alphaTailSizeAlign, NUM_PER_REPEAT);
    int64_t maxTargetLengthAlign = Ops::Base::CeilAlign(maxTargetLength, NUM_PER_REPEAT);
    int64_t targetLengthPlusAlign = Ops::Base::CeilAlign(maxTargetLength + 1, NUM_PER_REPEAT);
    // Calculate the ubsize needed for targetsLengths and alphaLength
    // all the numbers mean the related ub numbers
    int64_t needSize = alphaTailSizeAlign * FLOAT_DSIZE * NUM_TWO + alphaLengthAlign * FLOAT_DSIZE * NUM_FOUR +
                       Ops::Base::CeilAlign(alphaLengthAlign / MASKNUM_PER_BYTES, BLOCK_SIZE) * NUM_FOUR +
                       maxTargetLengthAlign * FLOAT_DSIZE * NUM_SEVEN +
                       targetLengthPlusAlign * INT64_DSIZE * FLOAT_DSIZE + BLOCK_SIZE * NUM_TWELVE;
    needSize = batchSize <= LAGRE_BATCH ? (needSize + batchSize * (targetDsize * NUM_TWO + INT64_DSIZE)) : needSize;
    sliceLength = (ubSize - needSize) / INT64_DSIZE; // NeedSize is smaller than ubSize
    if (sliceLength > symbleSet) {
        sliceLength = symbleSet;
    }
    sliceLength = Ops::Base::CeilAlign(sliceLength, NUM_PER_REPEAT);
    probSliceNum = sliceLength == 0 ? 0 : (symbleSet - 1 + sliceLength) / sliceLength;
    sliceLengthTail = symbleSet - (probSliceNum - 1) * sliceLength;
    size_t sysWorkspaceSize = compileInfo->sysWorkspaceSize;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t mustWorkspace = (alphaTailSizeAlign + INT64_DSIZE) * FLOAT_DSIZE * WORKSPACE_ALPHA_NUM * batchSize;
    currentWorkspace[0] = batchSize <= LAGRE_BATCH ? sysWorkspaceSize + mustWorkspace :
                                                     sysWorkspaceSize + mustWorkspace + batchSize * INT64_DSIZE;
    PrintTilingData();
    OP_LOGD(context, "CTCLossV3 tiling end running");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CTCLossV3Tiling::RunKernelTiling()
{
    context->SetTilingKey(0); // only one tilingKey now
    context->SetBlockDim(coreUsed);
    tilingData.set_sliceLength(sliceLength);
    tilingData.set_sliceLengthTail(sliceLengthTail);
    tilingData.set_probSliceNum(probSliceNum);
    tilingData.set_maxTargetLength(maxTargetLength);
    tilingData.set_targetsDimLength(targetsDimLength);
    tilingData.set_timeStep(timeStep);
    tilingData.set_symbleSet(symbleSet);
    tilingData.set_batchSize(batchSize);
    tilingData.set_targetsDimNum(targetsDimNum);
    tilingData.set_targetsNum(targetsNum);
    tilingData.set_taskPerCore(taskPerCore);
    tilingData.set_taskTailCore(taskTailCore);
    tilingData.set_blank(blankAtr);
    tilingData.set_zeroInfinity(zeroInfinity);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void CTCLossV3Tiling::PrintTilingData() const
{
    OP_LOGD(context, "Start printing");
    OP_LOGD(context, "sliceLength is %ld.", sliceLength);
    OP_LOGD(context, "sliceLengthTail is %ld.", sliceLengthTail);
    OP_LOGD(context, "probSliceNum is %ld.", probSliceNum);
    OP_LOGD(context, "maxTargetLength is %ld.", maxTargetLength);
    OP_LOGD(context, "timeStep is %ld.", timeStep);
    OP_LOGD(context, "symbleSet is %ld.", symbleSet);
    OP_LOGD(context, "batchSize is %ld.", batchSize);
    OP_LOGD(context, "targetsDimNum is %ld.", targetsDimNum);
    OP_LOGD(context, "targetsDimLength is %ld.", targetsDimLength);
    OP_LOGD(context, "targetsNum is %ld.", targetsNum);
    OP_LOGD(context, "taskPerCore is %ld.", taskPerCore);
    OP_LOGD(context, "taskTailCore is %ld.", taskTailCore);
    OP_LOGD(context, "blank is %ld.", blankAtr);
    OP_LOGD(context, "zeroInfinity is %ld.", zeroInfinity);
    OP_LOGD(context, "End printing");
}

static ge::graphStatus TilingFunc4CTCLossV3(gert::TilingContext* context)
{
    CTCLossV3Tiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepare4CTCLossV3(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ReportTilingFailure(OP_NAME.c_str(), "context is nullptr.");
    }
    auto nodeName = context->GetNodeName();
    OP_LOGD(context, "TilingPrepare4CTCLossV3 start.");
    auto compileInfo = context->GetCompiledInfo<CTCLossV3CompileInfo>();
    OP_CHECK_IF(
        compileInfo == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "compileInfo",
                                                 "failed to get the compile info from the tiling parse context"),
        return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platformInfo",
                                                 "failed to get the platform info from the tiling parse context"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), ReportTilingFailure(nodeName, "Failed to get core num."),
                return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF((compileInfo->sysWorkspaceSize <= 0U),
                ReportTilingFailure(nodeName, "sysWorkspaceSize should be greater than zero"), return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), ReportTilingFailure(nodeName, "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "TilingPrepare4CTCLossV3 end.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CTCLossV3)
    .Tiling(TilingFunc4CTCLossV3)
    .TilingParse<CTCLossV3CompileInfo>(TilingPrepare4CTCLossV3)
    .TilingInputsDataDependency({3});
} // namespace optiling
