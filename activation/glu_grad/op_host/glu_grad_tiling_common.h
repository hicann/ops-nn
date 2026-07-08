/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GLU_GRAD_TILING_COMMON_H
#define GLU_GRAD_TILING_COMMON_H

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "util/math_util.h"
#include "glu_grad_tiling.h"
#include "../op_kernel/arch35/glu_grad_tiling_key.h"

namespace optiling {
namespace glu_grad_common {

static const uint32_t SELF_INPUT_IDX = 1;
static const uint32_t ATTR_DIM_INDEX = 0;
static const uint32_t FP16_DTYPE_BYTES = 2;
static const uint32_t BF16_DTYPE_BYTES = 2;
static const uint32_t FP32_DTYPE_BYTES = 4;
static const uint32_t SPLIT_FACTOR = 2;
static const uint32_t SPLIT_ERROR_STATUS = 10000;
static const int64_t BYTES_ONE_BLOCK = 32;

static const int64_t FP16_BLOCK_SIZE = 16;
static const int64_t BF16_BLOCK_SIZE = 16;
static const int64_t FP32_BLOCK_SIZE = 8;

constexpr uint32_t BATCH_MODE = 1;

inline ge::graphStatus SetTilingDataForGluGrad(gert::TilingContext* context, GluGradTilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

inline void GetTilingDataSmall(GluGradTilingData& tilingData, GluGradTilingParam& tilingParam, bool isVreduce)
{
    int64_t splitSizeAlign{tilingParam.ny};
    int64_t blockSize = tilingParam.blockSize;
    int64_t group{1};
    if (isVreduce || tilingParam.ny % blockSize == 0) {
        group = tilingParam.bufSize / tilingParam.ny;
    } else {
        splitSizeAlign = (tilingParam.ny + blockSize - 1) / blockSize * blockSize;
        group = tilingParam.bufSize / splitSizeAlign;
    }

    int64_t numPerCore = Ops::Base::CeilDiv(tilingParam.x, tilingParam.coreNum);
    int64_t realCoreNum = Ops::Base::CeilDiv(tilingParam.x, numPerCore);

    tilingData.set_splitSize(tilingParam.ny);
    tilingData.set_group(group);
    tilingData.set_numPerCore(numPerCore);
    tilingData.set_loopNum(numPerCore / group);
    tilingData.set_nLastTailGroup(numPerCore % group);
    tilingData.set_realCoreNum(realCoreNum);

    if (numPerCore * realCoreNum != tilingParam.x) {
        int64_t tailTotalNum = tilingParam.x - numPerCore * (realCoreNum - 1);
        tilingData.set_tailLoopNum(tailTotalNum / group);
        tilingData.set_lastTailGroup(tailTotalNum % group);
    } else {
        tilingData.set_tailLoopNum(0);
        tilingData.set_lastTailGroup(0);
    }
}

inline void GetTilingDataBig(GluGradTilingData& tilingData, GluGradTilingParam& tilingParam)
{
    int64_t group = tilingParam.ny / tilingParam.bufSize;
    int64_t tailNum = tilingParam.ny - group * tilingParam.bufSize;

    tilingData.set_realCoreNum(tilingParam.coreNum);
    tilingData.set_splitSize(tilingParam.bufSize);
    tilingData.set_group(group);
    tilingData.set_tailLoopNum(tailNum);
    if (tailNum == 0) {
        tilingData.set_loopNum(tilingParam.x * group);
    } else {
        tilingData.set_loopNum(tilingParam.x * (group + 1));
    }
}

inline uint64_t GetTilingData(
    ge::DataType dtype, GluGradTilingData& tilingData, GluGradTilingParam& tilingParam, int64_t commonBufSize,
    int64_t singleBufSize)
{
    if (dtype == ge::DT_FLOAT) {
        tilingParam.blockSize = FP32_BLOCK_SIZE;
        tilingData.set_blockSize(FP32_BLOCK_SIZE);
    } else if (dtype == ge::DT_FLOAT16) {
        tilingParam.blockSize = FP16_BLOCK_SIZE;
        tilingData.set_blockSize(FP16_BLOCK_SIZE);
        commonBufSize = commonBufSize * (FP16_BLOCK_SIZE / FP32_BLOCK_SIZE);
        singleBufSize = singleBufSize * (FP16_BLOCK_SIZE / FP32_BLOCK_SIZE);
    } else {
        tilingParam.blockSize = BF16_BLOCK_SIZE;
        tilingData.set_blockSize(BF16_BLOCK_SIZE);
        commonBufSize = commonBufSize * (BF16_BLOCK_SIZE / FP32_BLOCK_SIZE);
        singleBufSize = singleBufSize * (BF16_BLOCK_SIZE / FP32_BLOCK_SIZE);
    }

    uint64_t shapeMode;
    if (tilingParam.ny <= commonBufSize) {
        tilingParam.bufSize = commonBufSize;
        GetTilingDataSmall(tilingData, tilingParam, false);
        shapeMode = TPL_SMALL_SHAPE;
    } else {
        tilingParam.bufSize = commonBufSize;
        GetTilingDataBig(tilingData, tilingParam);
        shapeMode = TPL_BIG_SHAPE;
    }
    return shapeMode;
}

inline ge::graphStatus CheckInputParams(gert::TilingContext* context)
{
    auto inputShape = context->GetInputShape(SELF_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    auto inputDesc = context->GetInputDesc(SELF_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dtype = inputDesc->GetDataType();
    int32_t typeSize = ge::GetSizeByDataType(dtype);

    OP_CHECK_IF(
        dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT,
        OP_LOGE(context, "input dtype only support fp16, fp32, bf16 currently, please check."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (typeSize <= 0), OP_LOGE(context, "typeSize is invalid %d, please check.", typeSize), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

inline size_t GetAttrSplitDim(const gert::TilingContext* context)
{
    auto inputStorageShape = context->GetInputShape(SELF_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    auto inputShape = inputStorageShape->GetStorageShape();
    size_t inputShapeSize = static_cast<size_t>(inputShape.GetDimNum());

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto* attrDim = attrs->GetAttrPointer<int64_t>(ATTR_DIM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDim);
    auto splitDim = static_cast<int64_t>(*attrDim);
    OP_LOGD(context, "splitDim is %ld, inputShapeSize is %zu", splitDim, inputShapeSize);

    int64_t splitDimI = splitDim < 0 ? splitDim + static_cast<int64_t>(inputShapeSize) : splitDim;

    OP_CHECK_IF(
        (splitDimI < 0 || splitDimI >= static_cast<int64_t>(inputShapeSize)),
        OP_LOGE(
            context, "The value of attr [dim] must be in the range [-%zu, %zu], but got [%ld].", inputShapeSize,
            inputShapeSize - 1, splitDim),
        return SPLIT_ERROR_STATUS);

    size_t splitDimU = static_cast<size_t>(splitDimI);

    OP_CHECK_IF(
        (inputShape.GetDim(splitDimU) % SPLIT_FACTOR != 0),
        OP_LOGE(
            context, "The dim of: %zu can not be split with factor: %u on value %ld.", splitDimU, SPLIT_FACTOR,
            inputShape.GetDim(splitDimU)),
        return SPLIT_ERROR_STATUS);

    return splitDimU;
}

inline ge::graphStatus GetTilingParam(const gert::TilingContext* context, GluGradTilingParam& tilingParam)
{
    size_t splitDim = GetAttrSplitDim(context);
    OP_CHECK_IF((splitDim == SPLIT_ERROR_STATUS), OP_LOGE(context, "Get Split Dim Failed."), return ge::GRAPH_FAILED);

    auto inputStorageShape = context->GetInputShape(SELF_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    auto inputShape = inputStorageShape->GetStorageShape();
    int64_t x{1}, y{1}, n{1};
    for (size_t i = 0; i < inputShape.GetDimNum(); i++) {
        if (i < splitDim) {
            x *= inputShape.GetDim(i);
        } else if (i > splitDim) {
            y *= inputShape.GetDim(i);
        } else {
            n = inputShape.GetDim(i) / SPLIT_FACTOR;
        }
    }
    int64_t ny = n * y;
    OP_CHECK_IF((x == 0 || ny == 0), OP_LOGE(context, "Get input tensor is empty."), return ge::GRAPH_FAILED);

    auto compileInfo = context->GetCompileInfo<GluGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    tilingParam.x = x;
    tilingParam.ny = ny;
    tilingParam.coreNum = compileInfo->totalCoreNum;
    tilingParam.ubSize = compileInfo->ubSizePlatForm;

    OP_LOGI(
        context, "tilingParam is x:%ld, coreNum: %ld, ubSize: %lu splitDim: %zu", tilingParam.x, tilingParam.coreNum,
        tilingParam.ubSize, splitDim);

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus RunCommonTiling(
    gert::TilingContext* context, int64_t commonBufSize, int64_t singleBufSize, size_t workspaceSize)
{
    OP_LOGD(context, "RunCommonTiling enter.");
    context->SetScheduleMode(BATCH_MODE);

    OP_CHECK_IF(
        CheckInputParams(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "InputParams not valid."),
        return ge::GRAPH_FAILED);

    GluGradTilingParam tilingParam;
    OP_CHECK_IF(
        GetTilingParam(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Get Tiling Param Failed."),
        return ge::GRAPH_FAILED);

    auto dtypeDesc = context->GetInputDesc(SELF_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeDesc);
    auto dtype = dtypeDesc->GetDataType();
    GluGradTilingData tilingData;

    uint64_t shapeMode = GetTilingData(dtype, tilingData, tilingParam, commonBufSize, singleBufSize);

    tilingData.set_ny(tilingParam.ny);

    OP_CHECK_IF(
        SetTilingDataForGluGrad(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GluGradSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData.get_realCoreNum());
    uint32_t dTypeX = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, shapeMode);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context, "workspaces is null."), return ge::GRAPH_FAILED);

    workspaces[0] = workspaceSize;

    OP_LOGD(
        context,
        "tilingData is splitSize:%ld, group:%ld, realCoreNum:%ld, numPerCore:%ld, loopNum:%ld, \
           tailLoopNum:%ld,nLastTailGroup:%ld, lastTailGroup:%ld, shapeMode:%lu, blockSize:%ld, \
           ny: %ld",
        tilingData.get_splitSize(), tilingData.get_group(), tilingData.get_realCoreNum(), tilingData.get_numPerCore(),
        tilingData.get_loopNum(), tilingData.get_tailLoopNum(), tilingData.get_nLastTailGroup(),
        tilingData.get_lastTailGroup(), shapeMode, tilingData.get_blockSize(), tilingData.get_ny());

    OP_LOGD(context, "RunCommonTiling exit.");
    return ge::GRAPH_SUCCESS;
}

} // namespace glu_grad_common
} // namespace optiling

#endif // GLU_GRAD_TILING_COMMON_H
