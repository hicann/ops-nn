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
 * \file scaled_masked_softmax_grad_v2_tiling.cpp
 * \brief
 */
#include "scaled_masked_softmax_grad_v2_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"
#include "error_util.h"

namespace {
constexpr uint64_t INPUT_Y_GRAD_IDX = 0;
constexpr uint64_t INPUT_Y_IDX = 1;
constexpr uint64_t INPUT_MASK_IDX = 2;
constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;
constexpr uint64_t DIM_3 = 3;
constexpr uint64_t DIM_NUM = 4;
constexpr uint64_t REQUIRED_INPUT_NUM = 2;
constexpr uint64_t REQUIRED_OUTPUT_NUM = 1;
constexpr uint64_t ATTR_0 = 0;
constexpr uint64_t ATTR_1 = 1;
constexpr uint64_t ALIGNED_NUM = 64;
constexpr uint64_t SIZE_2 = 2;
constexpr uint64_t SIZE_4 = 4;
constexpr uint64_t SELECT_MAX_SIZE = 18 * 1024;
constexpr uint64_t LAST_DIM_MAX_SIZE = 4096;
constexpr uint64_t LAST_DIM_MAX_SIZE_D = 8192;
constexpr uint64_t MAX_NORM_HEAD_DIM = 1024;
constexpr uint64_t TILING_KEY_FP16 = 1;
constexpr uint64_t TILING_KEY_FP32 = 2;
constexpr uint64_t MASK_MODE_BNSD = 0;
constexpr uint64_t MASK_MODE_1NSD = 1;
constexpr uint64_t MASK_MODE_B1SD = 2;
constexpr uint64_t MASK_MODE_11SD = 3;
constexpr uint64_t NORM_HEADDIM_TILING_KEY = 1000;
constexpr uint64_t LARGE_HEADDIM_TILING_KEY = 2000;
} // namespace

namespace optiling {
class ScaledMaskedSoftmaxGradV2Tiling {
public:
    explicit ScaledMaskedSoftmaxGradV2Tiling(gert::TilingContext* tilingContext) : context(tilingContext) {};
    ge::graphStatus DoTiling();
    void PrintInfo();

private:
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckCoreInfo();
    ge::graphStatus CalcTilingParams();
    ge::graphStatus CalcLargeHeadDimInfo();
    ge::graphStatus CalcNormHeadDimInfo();
    ge::graphStatus SetMaskMode();
    ge::graphStatus SetTilingKey() const;
    ge::graphStatus SetAttrParams();
    ge::graphStatus SetTilingParams();

private:
    ScaledMaskedSoftmaxGradV2TilingData tiling;
    gert::TilingContext* context = nullptr;
    ge::DataType dataType;

    int64_t coreNum;
    uint64_t batch;
    uint64_t channel;
    uint64_t seqLength;
    uint64_t headDim;
    uint64_t maskBatch;
    uint64_t maskChannel;
    uint64_t ubSize;
    uint64_t usedUbSize;
    uint64_t totalLinePerHeadCore;
    uint64_t paddedHeadDim;
    uint64_t selectSize;
    uint64_t maxLinePerLoop;
    uint64_t dataSize;
};

void ScaledMaskedSoftmaxGradV2Tiling::PrintInfo()
{
    OP_LOGD(context, "----------------Start to print ScaledMaskedSoftmaxGradV2 tiling data.----------------");
    OP_LOGD(context, ">>> usedCoreNum:                     %lu", tiling.get_usedCoreNum());
    OP_LOGD(context, ">>> batch:                           %lu", tiling.get_batch());
    OP_LOGD(context, ">>> channel:                         %lu", tiling.get_channel());
    OP_LOGD(context, ">>> seqLength:                       %lu", tiling.get_seqLength());
    OP_LOGD(context, ">>> headDim:                         %lu", tiling.get_headDim());
    OP_LOGD(context, ">>> totalLine:                       %lu", tiling.get_totalLine());
    OP_LOGD(context, ">>> paddedHeadDim:                   %lu", tiling.get_paddedHeadDim());
    OP_LOGD(context, ">>> totalLinePerHeadCore:            %lu", tiling.get_totalLinePerHeadCore());
    OP_LOGD(context, ">>> totalLinePerTailCore:            %lu", tiling.get_totalLinePerTailCore());
    OP_LOGD(context, ">>> maxLinePerLoop:                  %lu", tiling.get_maxLinePerLoop());
    OP_LOGD(context, ">>> tailLinePerHeadCore:             %lu", tiling.get_tailLinePerHeadCore());
    OP_LOGD(context, ">>> tailLinePerTailCore:             %lu", tiling.get_tailLinePerTailCore());
    OP_LOGD(context, ">>> headCoreNum:                     %lu", tiling.get_headCoreNum());
    OP_LOGD(context, ">>> maskMoveMode:                    %lu", tiling.get_maskMoveMode());
    OP_LOGD(context, ">>> selectSize:                      %lu", tiling.get_selectSize());
    OP_LOGD(context, ">>> scale:                           %f", tiling.get_scale());
    OP_LOGD(context, ">>> fixedTriuMask:                   %u", tiling.get_fixedTriuMask());
    OP_LOGD(context, "----------------Print ScaledMaskedSoftmaxGradV2 tiling data end.<<<<<<<<<<<<<<<<");
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CheckInputShape()
{
    // check y_grad
    auto yGradShapePtr = context->GetInputShape(INPUT_Y_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yGradShapePtr);
    auto yGradShape = yGradShapePtr->GetStorageShape();
    int64_t yGradDimNum = yGradShape.GetDimNum();
    OP_CHECK_IF(
        (yGradDimNum != DIM_NUM), OP_LOGE(context->GetNodeName(), "yGradDimNum must be 4."),
        return ge::GRAPH_FAILED);
    batch = yGradShape.GetDim(DIM_0);
    channel = yGradShape.GetDim(DIM_1);
    seqLength = yGradShape.GetDim(DIM_2);
    headDim = yGradShape.GetDim(DIM_3);
    OP_CHECK_IF(
        (batch <= 0 || channel <= 0 || seqLength <= 0 || headDim <= 0),
        OP_LOGE(context->GetNodeName(), "The length of yGradDim must be greater than 0."),
        return ge::GRAPH_FAILED);

    auto PlatformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, PlatformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(PlatformInfo);
    uint64_t lastDimLimit = LAST_DIM_MAX_SIZE;
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
        lastDimLimit = LAST_DIM_MAX_SIZE_D;
    }

    OP_CHECK_IF(
        (headDim > lastDimLimit),
        OP_LOGE(
            context->GetNodeName(), "The length yGrad dim 3 must be less than or equal to 4096."),
        return ge::GRAPH_FAILED);

    // check y
    auto yShapePtr = context->GetInputShape(INPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    OP_CHECK_IF(
        (yShape != yGradShape),
        OP_LOGE(context->GetNodeName(), "yShape should be same as yGradShape"),
        return ge::GRAPH_FAILED);

    // check mask
    auto maskShapePtr = context->GetInputShape(INPUT_MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShapePtr);
    auto maskShape = maskShapePtr->GetStorageShape();
    int64_t maskDimNum = maskShape.GetDimNum();
    OP_CHECK_IF(
        (maskDimNum != DIM_NUM), OP_LOGE(context->GetNodeName(), "maskDimNum must be 4."),
        return ge::GRAPH_FAILED);

    // check input and mask dim
    maskBatch = maskShape.GetDim(DIM_0);
    maskChannel = maskShape.GetDim(DIM_1);
    uint64_t maskSeqLength = maskShape.GetDim(DIM_2);
    uint64_t maskHeadDim = maskShape.GetDim(DIM_3);
    OP_CHECK_IF(
        (seqLength != maskSeqLength || headDim != maskHeadDim),
        OP_LOGE(
            context->GetNodeName(), "The last two dims of mask must be equal to the last two dims of yGrad."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (maskBatch <= 0 || maskChannel <= 0),
        OP_LOGE(context->GetNodeName(), "The length of maskDim must be greater than 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((batch != maskBatch && maskBatch != 1) || (channel != maskChannel && maskChannel != 1)),
        OP_LOGE(context->GetNodeName(), "mask shape must be broadcast to yGrad shape."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CheckInputDtype()
{
    auto inputYGrad = context->GetInputDesc(INPUT_Y_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputYGrad);
    auto yGradDtype = inputYGrad->GetDataType();
    OP_CHECK_IF(
        (yGradDtype != ge::DT_FLOAT16 && yGradDtype != ge::DT_FLOAT && yGradDtype != ge::DT_BF16),
        OP_LOGE(context->GetNodeName(), "yGradDtype should be FP16/BF16/FP32"),
        return ge::GRAPH_FAILED);

    dataType = yGradDtype;
    if (dataType == ge::DT_FLOAT) {
        dataSize = SIZE_4;
    } else {
        dataSize = SIZE_2;
    }

    // check y
    auto inputY = context->GetInputDesc(INPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto yDtype = inputY->GetDataType();
    OP_CHECK_IF(
        (yDtype != yGradDtype),
        OP_LOGE(context->GetNodeName(), "yDtype should be same as yGradDtype"),
        return ge::GRAPH_FAILED);

    auto inputMask = context->GetInputDesc(INPUT_MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputMask);
    auto maskDtype = inputMask->GetDataType();
    OP_CHECK_IF(
        (maskDtype != ge::DT_BOOL), OP_LOGE(context->GetNodeName(), "maskDtype should be bool"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CheckCoreInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_LOGD(context, "Start to check core info.");
    OP_LOGD(context, "Get total core num:%ld", coreNum);
    OP_CHECK_IF(
        (coreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "Get total ub size:%lu", ubSize);
    OP_LOGD(context, "Check core info ends.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CalcLargeHeadDimInfo()
{
    selectSize = SELECT_MAX_SIZE;
    usedUbSize =
        paddedHeadDim * (REQUIRED_INPUT_NUM * dataSize + SIZE_4 * (dataSize == SIZE_2 ? REQUIRED_INPUT_NUM : 0));
    maxLinePerLoop = (ubSize - selectSize) / usedUbSize;
    maxLinePerLoop = std::min(maxLinePerLoop, totalLinePerHeadCore);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CalcNormHeadDimInfo()
{
    std::vector<int64_t> shapeVec = {1, static_cast<int64_t>(paddedHeadDim)};
    ge::Shape srcShape(shapeVec);
    uint64_t oneLineSoftmaxGradSize = AscendC::GetSoftMaxGradMaxTmpSize(srcShape, SIZE_4, false, false);
    // inQ*2+outQ + cast*2 + maskQ + api_tmp_buf
    usedUbSize = paddedHeadDim * ((REQUIRED_INPUT_NUM + REQUIRED_OUTPUT_NUM) * dataSize +
                                  SIZE_4 * (dataSize == SIZE_2 ? REQUIRED_INPUT_NUM : 0)) +
                 paddedHeadDim + oneLineSoftmaxGradSize;
    maxLinePerLoop = ubSize / usedUbSize;
    uint64_t maxLocalWorkSpaceSize = oneLineSoftmaxGradSize * maxLinePerLoop;
    selectSize = std::max(maxLocalWorkSpaceSize, SELECT_MAX_SIZE);
    if (selectSize > maxLocalWorkSpaceSize) {
        maxLinePerLoop = (ubSize - selectSize) / (usedUbSize - oneLineSoftmaxGradSize);
    }
    maxLinePerLoop = std::min(maxLinePerLoop, totalLinePerHeadCore);
    shapeVec = {static_cast<int64_t>(maxLinePerLoop), static_cast<int64_t>(paddedHeadDim)};
    srcShape = ge::Shape(shapeVec);
    AscendC::SoftMaxGradTilingFunc(srcShape, SIZE_4, selectSize, tiling.softmaxGradTilingData, false);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CalcTilingParams()
{
    uint64_t totalLine = batch * channel * seqLength;
    uint64_t usedCoreNum = std::min(totalLine, static_cast<uint64_t>(coreNum));
    totalLinePerHeadCore = Ops::Base::CeilDiv(totalLine, usedCoreNum);
    uint64_t totalLinePerTailCore = 0;
    uint64_t headCoreNum = 0;
    if (usedCoreNum != 0) {
        totalLinePerTailCore = totalLine / usedCoreNum;
        headCoreNum = totalLine % usedCoreNum;
    }

    paddedHeadDim = Ops::Base::CeilDiv(headDim, ALIGNED_NUM) * ALIGNED_NUM;
    if (paddedHeadDim > MAX_NORM_HEAD_DIM) {
        CalcLargeHeadDimInfo();
    } else {
        CalcNormHeadDimInfo();
    }

    uint64_t tailLinePerHeadCore = totalLinePerHeadCore % maxLinePerLoop;
    uint64_t tailLinePerTailCore = totalLinePerTailCore % maxLinePerLoop;
    if (tailLinePerHeadCore == 0) {
        tailLinePerHeadCore = maxLinePerLoop;
    }
    if (tailLinePerTailCore == 0) {
        tailLinePerTailCore = maxLinePerLoop;
    }

    // set tiling data
    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_totalLine(totalLine);
    tiling.set_totalLinePerTailCore(totalLinePerTailCore);
    tiling.set_tailLinePerHeadCore(tailLinePerHeadCore);
    tiling.set_tailLinePerTailCore(tailLinePerTailCore);
    tiling.set_headCoreNum(headCoreNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::SetMaskMode()
{
    uint64_t maskMoveMode = 0u;
    if (batch == maskBatch && channel == maskChannel) {
        maskMoveMode = MASK_MODE_BNSD;
    } else if (1 == maskBatch && channel == maskChannel) {
        maskMoveMode = MASK_MODE_1NSD;
    } else if (batch == maskBatch && 1 == maskChannel) {
        maskMoveMode = MASK_MODE_B1SD;
    } else if (1 == maskBatch && 1 == maskChannel) {
        maskMoveMode = MASK_MODE_11SD;
    }

    tiling.set_maskMoveMode(maskMoveMode);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::SetTilingKey() const
{
    uint64_t tilingKey = (paddedHeadDim <= MAX_NORM_HEAD_DIM ? NORM_HEADDIM_TILING_KEY : LARGE_HEADDIM_TILING_KEY);
    if (dataType == ge::DT_FLOAT16) {
        tilingKey += TILING_KEY_FP16;
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey += TILING_KEY_FP32;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::SetAttrParams()
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* scale = attrs->GetAttrPointer<float>(ATTR_0);
    const bool* fixedTriuMask = attrs->GetAttrPointer<bool>(ATTR_1);

    tiling.set_scale(*scale);
    tiling.set_fixedTriuMask(static_cast<uint32_t>(*fixedTriuMask));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::SetTilingParams()
{
    tiling.set_batch(batch);
    tiling.set_channel(channel);
    tiling.set_seqLength(seqLength);
    tiling.set_headDim(headDim);
    tiling.set_paddedHeadDim(paddedHeadDim);
    tiling.set_maxLinePerLoop(maxLinePerLoop);
    tiling.set_totalLinePerHeadCore(totalLinePerHeadCore);
    tiling.set_selectSize(selectSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::DoTiling()
{
    OP_LOGD(context, "Start running Tiling4ScaledMaskedSoftmaxGradV2.");
    OP_CHECK_IF(
        (CheckInputShape() != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "InputShape is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (CheckInputDtype() != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "InputShape is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (CheckCoreInfo() != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "CoreNum or ubSize is invalid."),
        return ge::GRAPH_FAILED);

    CalcTilingParams();

    SetTilingParams();
    SetMaskMode();
    SetTilingKey();
    SetAttrParams();

    PrintInfo();

    // 固定写法
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ScaledMaskedSoftmaxGradV2(gert::TilingContext* context)
{
    ScaledMaskedSoftmaxGradV2Tiling tilingObj(context);
    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepare4ScaledMaskedSoftmaxGradV2(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScaledMaskedSoftmaxGradV2)
    .Tiling(Tiling4ScaledMaskedSoftmaxGradV2)
    .TilingParse<ScaledMaskedSoftmaxGradV2CompileInfo>(TilingPrepare4ScaledMaskedSoftmaxGradV2);
} // namespace optiling
