/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_cross_entropy_with_logits_tiling.cc
 * \brief softmax_cross_entropy_with_logits_tiling
 */

#include <iostream>
#include <cstring>
#include "register/op_def_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../op_kernel/arch35/softmax_cross_entropy_with_logits_tiling_data.h"
#include "../op_kernel/arch35/softmax_cross_entropy_with_logits_tiling_key.h"

namespace optiling {
constexpr uint32_t INPUT_FEATURES_IDX = 0;
constexpr uint32_t INPUT_LABELS_IDX = 1;
constexpr uint32_t OUTPUT_LOSS_IDX = 0;
constexpr uint32_t OUTPUT_BACKPROP_IDX = 1;
constexpr uint32_t INPUT_NUM = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t PING_PANG_AR_NUM = 3;
constexpr int64_t PING_PANG_A_NUM = 1;
constexpr int64_t TEMP_BUF_AR_NUM = 4;
constexpr int64_t TEMP_BUF_A_NUM = 3;
constexpr uint32_t DIM_NUM = 2;
constexpr int32_t AR_UB_NUMS_B32 = 4;
constexpr int32_t AR_UB_NUMS_B16 = 5;
constexpr int32_t A_UB_NUMS = 4;

constexpr int64_t CONST_CACHE = 63;
constexpr int64_t DTYPE_LEN_FP16 = 2;
constexpr int64_t DTYPE_LEN_FP32 = 4;
constexpr uint64_t SCHID_FULL_LOAD = 0;
constexpr uint64_t SCHID_SPLIT_R = 1;
constexpr uint64_t SCHID_EMPTY = 2;
constexpr uint64_t NEED_BROC = 1;
constexpr uint64_t IS_SCALAR = 2;
constexpr size_t WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);

struct BaseTilingData {
    int64_t realCoreNum = 0;
    int64_t a = 0;               // a轴
    int64_t r = 0;               // r轴
    int64_t blockFactor = 0;     // a轴分核，主核数据量
    int64_t tailBlockFactor = 0; // a轴分核，尾核数据量
    int64_t rUbNumFactor = 0;    // R轴切分，一次UB可以放下的数据量，全载模板下等于r，注意32b对齐
    int64_t aUbNumFactor = 0;    // A轴切分，一次UB可以放下的数据量，非全载模板下等于1，注意32b对齐
    int64_t aLoopTimes = 0;      // 主核A方向循环搬移数据的次数
    int64_t aLoopTimesT = 0;     // 尾核A方向循环搬移数据的次数
    int64_t aLoopTail = 0;       // 主核A方向尾块的数据量
    int64_t aLoopTailT = 0;      // 尾核A方向尾块的数据量
    int64_t rLoopTime = 0;       // 不能全载时，R轴反向的循环次数
    int64_t rLoopTile = 0;       // 不能全载时，R轴反向的尾块数据量
    int64_t rLoopTileAlign = 0;  // 不能全载时，R轴反向的尾块数据量
    int64_t kTimesTail = 0;      // 不能全载时，完全二分累加，存在主尾块相加的次数
    int64_t kTimes = 0;          // 不能全载时，完全二分累加，2的k次方内循环次数
    int64_t tilingKey = 0;
};

class SoftmaxCrossEntropyWithLogitsRegbaseTiling {
public:
    explicit SoftmaxCrossEntropyWithLogitsRegbaseTiling(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckInputShape();
    bool IsElewiseShape(const gert::Shape& shape1, const gert::Shape& shape2);

    ge::graphStatus CalTilingData();
    ge::graphStatus CalTilingDataRSplit();
    ge::graphStatus DoDimCollapse();
    void GetTilingKey();
    void FillTilingData();
    void PrintTilingData();
    int64_t GetMod(int64_t l_value, int64_t r_value);

private:
    BaseTilingData baseTiling_;
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
    uint64_t featuresBroc = 0;
    uint64_t labelsBroc = 0;
    int64_t featuresBrcDim = 0;
    int64_t labelsBrcDim = 0;
    uint64_t schIdNum = 0;
    int64_t vfLenB32 = 0;
    int64_t ubBlockSize = 0;
    int64_t cacheLineSize = 0;

    gert::TilingContext* context_ = nullptr;
    SoftmaxCrossEntropyWithLogitsTilingData* tilingData_{nullptr};
};

ge::graphStatus SoftmaxCrossEntropyWithLogitsRegbaseTiling::CheckInputDtype()
{
    auto input = context_->GetInputDesc(INPUT_FEATURES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input);
    auto featuresDtype = input->GetDataType();

    input = context_->GetInputDesc(INPUT_LABELS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input);
    auto labelsDtype = input->GetDataType();

    OP_CHECK_IF(
        (labelsDtype != featuresDtype),
        OP_LOGE(context_->GetNodeName(), "input labels and features dtype should be same."), return ge::GRAPH_FAILED);
    bool validDtype = featuresDtype == ge::DT_BF16 || featuresDtype == ge::DT_FLOAT || featuresDtype == ge::DT_FLOAT16;
    OP_CHECK_IF(!validDtype,
                OP_LOGE(
                    context_->GetNodeName(), "input dtype is only support float16, bfloat16, float32.");
                , return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool SoftmaxCrossEntropyWithLogitsRegbaseTiling::IsElewiseShape(const gert::Shape& shape1, const gert::Shape& shape2)
{
    if (shape1.GetDimNum() != shape2.GetDimNum()) {
        return false;
    }

    for (size_t i = 0; i < shape1.GetDimNum(); i++) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            return false;
        }
    }
    return true;
}

int64_t SoftmaxCrossEntropyWithLogitsRegbaseTiling::GetMod(int64_t l_value, int64_t r_value) {
    if (r_value == 0) {
        return l_value;
    }
    return l_value % r_value;
}

ge::graphStatus SoftmaxCrossEntropyWithLogitsRegbaseTiling::CheckInputShape()
{
    auto featuresShape = context_->GetInputShape(INPUT_FEATURES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, featuresShape);
    auto featuresStorageShape = featuresShape->GetStorageShape();
    auto labelsShape = context_->GetInputShape(INPUT_LABELS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, labelsShape);
    auto labelsStorageShape = labelsShape->GetStorageShape();

    auto output = context_->GetOutputShape(OUTPUT_BACKPROP_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, output);
    auto outputShape = output->GetStorageShape();
    if (!IsElewiseShape(outputShape, featuresStorageShape)) {
        featuresBroc = NEED_BROC;
        if (featuresStorageShape.GetShapeSize() == 1) {
            featuresBroc = IS_SCALAR;
        }
        OP_LOGD(context_, "feature need broadcast.");
    }
    if (!IsElewiseShape(outputShape, labelsStorageShape)) {
        labelsBroc = NEED_BROC;
        if (labelsStorageShape.GetShapeSize() == 1) {
            labelsBroc = IS_SCALAR;
        }
        OP_LOGD(context_, "labels need broadcast.");
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxCrossEntropyWithLogitsRegbaseTiling::CalTilingData()
{
    auto featuresDtype = context_->GetInputDesc(INPUT_FEATURES_IDX)->GetDataType();
    int64_t useCoreNum = baseTiling_.a > coreNum ? coreNum : baseTiling_.a;
    int64_t blockCalNum = Ops::Base::CeilDiv(baseTiling_.a, useCoreNum);
    baseTiling_.realCoreNum = Ops::Base::CeilDiv(baseTiling_.a, blockCalNum);
    baseTiling_.blockFactor = blockCalNum;
    baseTiling_.tailBlockFactor = baseTiling_.a - blockCalNum * (baseTiling_.realCoreNum - 1);
    int64_t inputByte = featuresDtype == ge::DT_FLOAT ? DTYPE_LEN_FP32 : DTYPE_LEN_FP16;
    int64_t perBlock = ubBlockSize / inputByte;

    int64_t rAligned = (baseTiling_.r + perBlock - 1) / perBlock * perBlock;
    baseTiling_.rUbNumFactor = rAligned;
    int64_t arAllUbSize = static_cast<int64_t>(ubSize);
    int64_t arBuf =
        (DOUBLE_BUFFER * PING_PANG_AR_NUM) * rAligned * inputByte + TEMP_BUF_AR_NUM * rAligned * DTYPE_LEN_FP32;
    int64_t aBuf = (DOUBLE_BUFFER * PING_PANG_A_NUM) * inputByte + TEMP_BUF_A_NUM * DTYPE_LEN_FP32;
    int64_t aNum = arAllUbSize / (arBuf + aBuf);
    if (aNum < perBlock) {
        OP_LOGD(context_, "can not full load r, aNum = %ld", aNum);
        CalTilingDataRSplit();
        schIdNum = SCHID_SPLIT_R;
        return ge::GRAPH_SUCCESS;
    } else {
        aNum = aNum > baseTiling_.blockFactor ? baseTiling_.blockFactor : aNum;
        baseTiling_.aUbNumFactor = aNum > perBlock ? aNum / perBlock * perBlock : perBlock;
        baseTiling_.aLoopTimes = baseTiling_.aUbNumFactor == 0 ? 0 : baseTiling_.blockFactor / baseTiling_.aUbNumFactor;
        baseTiling_.aLoopTail = baseTiling_.blockFactor % baseTiling_.aUbNumFactor;
        baseTiling_.aLoopTimesT =
            baseTiling_.aUbNumFactor == 0 ? 0 : baseTiling_.tailBlockFactor / baseTiling_.aUbNumFactor;
        baseTiling_.aLoopTailT = baseTiling_.tailBlockFactor % baseTiling_.aUbNumFactor;
        schIdNum = SCHID_FULL_LOAD;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxCrossEntropyWithLogitsRegbaseTiling::CalTilingDataRSplit()
{
    auto featuresDtype = context_->GetInputDesc(INPUT_FEATURES_IDX)->GetDataType();
    int64_t useCoreNum = baseTiling_.a > coreNum ? coreNum : baseTiling_.a;
    int64_t blockCalNum = Ops::Base::CeilDiv(baseTiling_.a, useCoreNum);
    baseTiling_.realCoreNum = Ops::Base::CeilDiv(baseTiling_.a, blockCalNum);
    baseTiling_.blockFactor = blockCalNum;
    baseTiling_.tailBlockFactor = baseTiling_.a - blockCalNum * (baseTiling_.realCoreNum - 1);
    int64_t inputByte = featuresDtype == ge::DT_FLOAT ? DTYPE_LEN_FP32 : DTYPE_LEN_FP16;
    int64_t perBlock = Ops::Base::FloorDiv(ubBlockSize, inputByte);
    baseTiling_.aUbNumFactor = perBlock;
    baseTiling_.aLoopTimes =
        baseTiling_.aUbNumFactor == 0 ? 0 : Ops::Base::FloorDiv(baseTiling_.blockFactor, baseTiling_.aUbNumFactor);
    baseTiling_.aLoopTail = GetMod(baseTiling_.blockFactor, baseTiling_.aUbNumFactor);
    baseTiling_.aLoopTimesT =
        baseTiling_.aUbNumFactor == 0 ? 0 : Ops::Base::FloorDiv(baseTiling_.tailBlockFactor, baseTiling_.aUbNumFactor);
    baseTiling_.aLoopTailT = GetMod(baseTiling_.tailBlockFactor, baseTiling_.aUbNumFactor);
    int64_t cacheUbSize = (CONST_CACHE * vfLenB32 + baseTiling_.aUbNumFactor) * DTYPE_LEN_FP32;
    int64_t arAllUbSize =
        static_cast<int64_t>(ubSize) - (baseTiling_.aUbNumFactor * A_UB_NUMS * DTYPE_LEN_FP32) - cacheUbSize;
    int64_t oneRowRNum = static_cast<int64_t>(Ops::Base::FloorDiv(arAllUbSize, (baseTiling_.aUbNumFactor * AR_UB_NUMS_B16)));
    if (featuresDtype == ge::DT_FLOAT) {
        oneRowRNum = static_cast<int64_t>(Ops::Base::FloorDiv(arAllUbSize, (baseTiling_.aUbNumFactor * AR_UB_NUMS_B32)));
    }
    int64_t rOnceNumByte = oneRowRNum < cacheLineSize ? cacheLineSize : Ops::Base::FloorAlign(oneRowRNum, cacheLineSize);
    baseTiling_.rUbNumFactor = baseTiling_.r <= perBlock ? perBlock : Ops::Base::FloorDiv(rOnceNumByte, inputByte);
    baseTiling_.rLoopTime = Ops::Base::CeilDiv(baseTiling_.r, baseTiling_.rUbNumFactor);
    int64_t rTail = GetMod(baseTiling_.r, baseTiling_.rUbNumFactor);
    baseTiling_.rLoopTile = rTail == 0 ? baseTiling_.rUbNumFactor : rTail;
    baseTiling_.rLoopTileAlign = Ops::Base::CeilAlign(baseTiling_.rLoopTile, perBlock);
    int64_t mainTile = baseTiling_.rLoopTile == 0 ? baseTiling_.rLoopTime : baseTiling_.rLoopTime - 1;
    int64_t logK = static_cast<int64_t>(std::floor(std::log(mainTile) / std::log(2)));
    baseTiling_.kTimes = baseTiling_.rLoopTime == 1 ? 1 : static_cast<int64_t>(std::exp2(logK));
    baseTiling_.kTimesTail = baseTiling_.rLoopTime - baseTiling_.kTimes;

    return ge::GRAPH_SUCCESS;
}

void SoftmaxCrossEntropyWithLogitsRegbaseTiling::GetTilingKey()
{
    uint64_t schId;
    if (baseTiling_.r == 0) {
        schId = SCHID_EMPTY;
    } else {
        schId = schIdNum;
    }

    uint64_t db = 0;
    uint64_t featuresBrc = featuresBroc;
    uint64_t labelsBrc = labelsBroc;

    OP_LOGI(
        context_->GetNodeName(), "schId is %lu, featuresBrc is %lu, labelsBrc is %lu, db is %lu", schId, featuresBrc,
        labelsBrc, db);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId, featuresBrc, labelsBrc, db);
    OP_LOGI(context_->GetNodeName(), "tilingKey is %ld", tilingKey);
    context_->SetTilingKey(tilingKey);
}

void SoftmaxCrossEntropyWithLogitsRegbaseTiling::FillTilingData()
{
    tilingData_->realCoreNum = baseTiling_.realCoreNum;
    tilingData_->a = baseTiling_.a;
    tilingData_->r = baseTiling_.r;
    tilingData_->blockFactor = baseTiling_.blockFactor;
    tilingData_->tailBlockFactor = baseTiling_.tailBlockFactor;
    tilingData_->rUbNumFactor = baseTiling_.rUbNumFactor;
    tilingData_->aUbNumFactor = baseTiling_.aUbNumFactor;
    tilingData_->aLoopTimes = baseTiling_.aLoopTimes;
    tilingData_->aLoopTimesT = baseTiling_.aLoopTimesT;
    tilingData_->aLoopTail = baseTiling_.aLoopTail;
    tilingData_->aLoopTailT = baseTiling_.aLoopTailT;
    tilingData_->rLoopTime = baseTiling_.rLoopTime;
    tilingData_->rLoopTile = baseTiling_.rLoopTile;
    tilingData_->rLoopTileAlign = baseTiling_.rLoopTileAlign;
    tilingData_->kTimesTail = baseTiling_.kTimesTail;
    tilingData_->kTimes = baseTiling_.kTimes;
    tilingData_->featuresBrcDim = featuresBrcDim;
    tilingData_->labelsBrcDim = labelsBrcDim;
}

void SoftmaxCrossEntropyWithLogitsRegbaseTiling::PrintTilingData()
{
    auto nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "SoftmaxCrossEntropyWithLogits regbase tiling data begin print.");
    OP_LOGD(nodeName, "realCoreNum = %ld.", tilingData_->realCoreNum);
    OP_LOGD(nodeName, "a = %ld.", tilingData_->a);
    OP_LOGD(nodeName, "r = %ld.", tilingData_->r);
    OP_LOGD(nodeName, "blockFactor = %ld.", tilingData_->blockFactor);
    OP_LOGD(nodeName, "tailBlockFactor = %ld.", tilingData_->tailBlockFactor);
    OP_LOGD(nodeName, "rUbNumFactor = %ld.", tilingData_->rUbNumFactor);
    OP_LOGD(nodeName, "aUbNumFactor = %ld.", tilingData_->aUbNumFactor);
    OP_LOGD(nodeName, "aLoopTimes = %ld.", tilingData_->aLoopTimes);
    OP_LOGD(nodeName, "aLoopTimesT = %ld.", tilingData_->aLoopTimesT);
    OP_LOGD(nodeName, "aLoopTail = %ld.", tilingData_->aLoopTail);
    OP_LOGD(nodeName, "aLoopTailT = %ld.", tilingData_->aLoopTailT);
    OP_LOGD(nodeName, "rLoopTime = %ld.", tilingData_->rLoopTime);
    OP_LOGD(nodeName, "rLoopTile = %ld.", tilingData_->rLoopTile);
    OP_LOGD(nodeName, "rLoopTileAlign = %ld.", tilingData_->rLoopTileAlign);
    OP_LOGD(nodeName, "kTimesTail = %ld.", tilingData_->kTimesTail);
    OP_LOGD(nodeName, "kTimes = %ld.", tilingData_->kTimes);
    OP_LOGD(nodeName, "labelsBrcDim = %ld.", tilingData_->labelsBrcDim);
    OP_LOGD(nodeName, "featuresBrcDim = %ld.", tilingData_->featuresBrcDim);
    OP_LOGD(nodeName, "SoftmaxCrossEntropyWithLogits regbase tiling data end print.");
}

ge::graphStatus SoftmaxCrossEntropyWithLogitsRegbaseTiling::Init()
{
    OP_LOGD(context_->GetNodeName(), "Enter SoftmaxCrossEntropyWithLogitsRegbaseTiling init.");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        coreNum <= 0, OP_LOGE(context_->GetNodeName(), "GetHardwareInfo Failed vectorCoreNum %ld", coreNum),
        return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize == 0UL, OP_LOGE(context_->GetNodeName(), "GetHardwareInfo Failed ubSize %ld", ubSize),
        return ge::GRAPH_FAILED);
    ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    vfLenB32 = static_cast<int64_t>(Ops::Base::GetVRegSize(context_));
    cacheLineSize = static_cast<int64_t>(Ops::Base::GetCacheLineSize(context_));

    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<SoftmaxCrossEntropyWithLogitsTilingData>();
        OP_CHECK_IF(
            tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        (memset_s(
             tilingData_, sizeof(SoftmaxCrossEntropyWithLogitsTilingData), 0,
             sizeof(SoftmaxCrossEntropyWithLogitsTilingData)) != EOK),
        OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "Exit SoftmaxCrossEntropyWithLogitsRegbaseTiling init.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxCrossEntropyWithLogitsRegbaseTiling::DoDimCollapse()
{
    std::vector<gert::Shape> inShapes;
    for (uint64_t i = 0; i < INPUT_NUM; i++) {
        auto input = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, input);
        auto inputShape = input->GetStorageShape();
        inShapes.push_back(inputShape);
    }
    auto output = context_->GetOutputShape(OUTPUT_BACKPROP_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, output);
    auto outputShape = output->GetStorageShape();

    std::vector<std::vector<int64_t>> dims;
    std::vector<std::vector<int64_t>> strides;
    ge::graphStatus status = Ops::Base::DimensionCollapse(inShapes, outputShape, dims, strides);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto collapseOutput = dims.back();
    uint64_t shapeLen = collapseOutput.size();
    OP_CHECK_IF(
        shapeLen != DIM_NUM, OP_LOGE(context_->GetNodeName(), "broadcast only support dim size 2."),
        return ge::GRAPH_FAILED);

    baseTiling_.a = collapseOutput[0];
    baseTiling_.r = collapseOutput[1];

    if (dims[INPUT_FEATURES_IDX][0] == 1 && dims[INPUT_FEATURES_IDX][1] != 1) {
        featuresBrcDim = 0;
    } else if (dims[INPUT_FEATURES_IDX][0] != 1 && dims[INPUT_FEATURES_IDX][1] == 1) {
        featuresBrcDim = 1;
    }

    OP_LOGD(context_, "features broadcast dim is %ld.", featuresBrcDim);
    if (dims[INPUT_LABELS_IDX][0] == 1 && dims[INPUT_LABELS_IDX][1] != 1) {
        labelsBrcDim = 0;
    } else if (dims[INPUT_LABELS_IDX][0] != 1 && dims[INPUT_LABELS_IDX][1] == 1) {
        labelsBrcDim = 1;
    }
    OP_LOGD(context_, "labels broadcast dim is %ld.", labelsBrcDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxCrossEntropyWithLogitsRegbaseTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter SoftmaxCrossEntropyWithLogitsRegbaseTiling DoTiling.v1.1");

    OP_CHECK_IF(
        CheckInputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckInputParams is failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckInputShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckInputShapes is failed"),
        return ge::GRAPH_FAILED);
    if (featuresBroc == NEED_BROC || labelsBroc == NEED_BROC) {
        OP_CHECK_IF(
            DoDimCollapse() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "DoDimCollapse is failed"),
            return ge::GRAPH_FAILED);
    } else {
        auto output = context_->GetOutputShape(OUTPUT_BACKPROP_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, output);
        auto outputShape = output->GetStorageShape();
        auto outputDimNum = outputShape.GetDimNum();
        baseTiling_.r = outputShape.GetDim(outputDimNum - 1);
        baseTiling_.a = baseTiling_.r == 0 ? 0 : outputShape.GetShapeSize() / baseTiling_.r;
    }
    OP_CHECK_IF(
        CalTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Calculate TilingData failed"),
        return ge::GRAPH_FAILED);

    GetTilingKey();
    FillTilingData();
    PrintTilingData();
    context_->SetBlockDim(tilingData_->realCoreNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    OP_LOGD(context_->GetNodeName(), "Exit SoftmaxCrossEntropyWithLogitsRegbaseTiling DoTiling.");
    return ge::GRAPH_SUCCESS;
}

struct SCEWithLogitsCompileInfo {
    bool isAscendC{false};
};

ge::graphStatus Tiling4SoftmaxCrossEntropyWithLogits(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Start Tiling4SoftmaxCrossEntropyWithLogitsRegbase.");
    SoftmaxCrossEntropyWithLogitsRegbaseTiling tilingImpl = SoftmaxCrossEntropyWithLogitsRegbaseTiling(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4SoftmaxCrossEntropyWithLogitsForAscendC init failed.");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4SoftmaxCrossEntropyWithLogitsForAscendC do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "end Tiling4SoftmaxCrossEntropyWithLogitsRegbase");
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingParse4SoftmaxCrossEntropyWithLogits(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SoftmaxCrossEntropyWithLogits)
    .Tiling(Tiling4SoftmaxCrossEntropyWithLogits)
    .TilingParse<SCEWithLogitsCompileInfo>(TilingParse4SoftmaxCrossEntropyWithLogits);
} // namespace optiling