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
 * \file glu_tiling.cpp
 * \brief
 */
#include "glu_tiling.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"

namespace optiling {

static const uint32_t INPUT_IDX = 0;
static const uint32_t ATTR_DIM_INDEX = 0;
static const uint32_t FP16_DTYPE_BYTES = 2;
static const uint32_t BF16_DTYPE_BYTES = 2;
static const uint32_t FP32_DTYPE_BYTES = 4;
static const uint32_t WORK_SPACE_SIZE = 16U * 1024U * 1024U;
static const uint32_t SPLIT_FACTOR = 2;
static const uint32_t SPLIT_ERROR_STATUS = 10000;
static const int64_t BYTES_ONE_BLOCK = 32;

static const int64_t FP16_BLOCK_SIZE = 16;
static const int64_t BF16_BLOCK_SIZE = 16;
static const int64_t FP32_BLOCK_SIZE = 8;

static const int64_t COMMON_BUFFER_SIZE = 8 * 1024;
static const int64_t SINGLE_BUFFER_SIZE = 6 * 1024;

constexpr uint32_t BATCH_MODE = 1;

inline static ge::graphStatus SetTilingDataForGlu(gert::TilingContext* context, GluTilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static void GetTilingDataSmall(GluTilingData& tilingData, TilingParam& tilingParam, bool isVreduce)
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

static void GetTilingDataBig(GluTilingData& tilingData, TilingParam& tilingParam)
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

static void GetTilingData(ge::DataType dtype, GluTilingData& tilingData, TilingParam& tilingParam)
{
    int64_t commonBufSize = COMMON_BUFFER_SIZE;
    int64_t singleBufSize = SINGLE_BUFFER_SIZE;

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
    }

    if (tilingParam.ny == 1) {
        // 根据ub大小、存活空间个数计算，32B对齐
        tilingParam.bufSize = singleBufSize;
        GetTilingDataSmall(tilingData, tilingParam, true);
        int64_t tilingkey = static_cast<int64_t>(GluTilingKey::TILINGKEY_SINGLESHAPE);
        tilingData.set_tilingKey(tilingkey);
    } else if (tilingParam.ny <= commonBufSize) {
        tilingParam.bufSize = commonBufSize;
        GetTilingDataSmall(tilingData, tilingParam, false);
        int64_t tilingkey = static_cast<int64_t>(GluTilingKey::TILINGKEY_SMALLSHAPE);
        tilingData.set_tilingKey(tilingkey);
    } else {
        tilingParam.bufSize = commonBufSize;
        GetTilingDataBig(tilingData, tilingParam);
        int64_t tilingkey = static_cast<int64_t>(GluTilingKey::TILINGKEY_BIGSHAPE);
        tilingData.set_tilingKey(tilingkey);
    }
}

static ge::graphStatus CheckInputParams(gert::TilingContext* context)
{
    auto input = context->GetInputTensor(INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);

    auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
    int32_t typeSize = ge::GetSizeByDataType(dtype);

    OP_CHECK_IF(
        dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT,
        OP_LOGE(context, "input dtype only support fp16, fp32, bf16 currently, please check."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (typeSize <= 0), OP_LOGE(context, "typeSize is invalid %d, please check.", typeSize), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Glu(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4Glu enter.");

    auto compileInfo = context->GetCompiledInfo<GluCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "Tiling totalCoreNum: %d", compileInfo->totalCoreNum);
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context, "TilingPrepare4Glu fail to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "TilingPrepare4Glu fail to get ub size."),
        return ge::GRAPH_FAILED);

    OP_LOGD(
        context, "TilingPrepare4Glu exit. coreNum: %d ubSize: %lu", compileInfo->totalCoreNum,
        compileInfo->ubSizePlatForm);
    return ge::GRAPH_SUCCESS;
}

static size_t GetAttrSplitDim(const gert::TilingContext* context)
{
    auto input = context->GetInputTensor(INPUT_IDX);
    auto inputShape = input->GetStorageShape();
    size_t inputShapeSize = static_cast<int64_t>(inputShape.GetDimNum());

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto* attrDim = attrs->GetAttrPointer<int64_t>(ATTR_DIM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDim);
    auto splitDim = static_cast<int64_t>(*attrDim);
    OP_LOGD(context, "splitDim is %ld, inputShapeSize is %zu", splitDim, inputShapeSize);

    size_t splitDimU = splitDim < 0 ? splitDim + inputShapeSize : splitDim;

    OP_CHECK_IF(
        (splitDimU >= inputShapeSize),
        OP_LOGE(
            context, "The value of attr [dim] must be in the range [-%zu, %zu], but got [%zu].", inputShapeSize,
            inputShapeSize, splitDimU),
        return SPLIT_ERROR_STATUS);

    OP_CHECK_IF(
        (inputShape.GetDim(splitDimU) % SPLIT_FACTOR != 0),
        OP_LOGE(
            context, "The dim of: %zu can not be split with factor: %u on value %ld.", splitDimU, SPLIT_FACTOR,
            inputShape.GetDim(splitDimU)),
        return SPLIT_ERROR_STATUS);

    return splitDimU;
}

static ge::graphStatus GetTilingParam(const gert::TilingContext* context, TilingParam& tilingParam)
{
    size_t splitDim = GetAttrSplitDim(context);
    OP_CHECK_IF((splitDim == SPLIT_ERROR_STATUS), OP_LOGE(context, "Get Split Dim Failed."), return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputTensor(INPUT_IDX)->GetStorageShape();
    // fuse dims
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

    auto compileInfo = context->GetCompileInfo<GluCompileInfo>();
    tilingParam.x = x;
    tilingParam.ny = ny;
    tilingParam.coreNum = compileInfo->totalCoreNum;
    tilingParam.ubSize = compileInfo->ubSizePlatForm;

    OP_LOGI(
        context,
        "tilingParm is x:%ld, coreNum: %ld, ubSize: %lu splitDim: %zu",
        tilingParam.x, tilingParam.coreNum, tilingParam.ubSize, splitDim);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Glu(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4Glu enter.");
    context->SetScheduleMode(BATCH_MODE);
    OP_CHECK_IF(
        CheckInputParams(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "InputParams not valid."),
        return ge::GRAPH_FAILED);

    TilingParam tilingParam;
    OP_CHECK_IF(
        GetTilingParam(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Get Tiling Param Failed."),
        return ge::GRAPH_FAILED);

    auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
    GluTilingData tilingData;

    GetTilingData(dtype, tilingData, tilingParam);

    tilingData.set_ny(tilingParam.ny);

    OP_CHECK_IF(
        SetTilingDataForGlu(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GluSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE + tilingParam.coreNum * BYTES_ONE_BLOCK * FP32_DTYPE_BYTES;

    OP_LOGD(
        context,
        "tilingData is splitSize:%ld, group:%ld, realCoreNum:%ld, numPerCore:%ld, loopNum:%ld, \
           tailLoopNum:%ld,nLastTailGroup:%ld, lastTailGroup:%ld, tilingKey:%ld, blockSize:%ld, \
           ny: %ld",
        tilingData.get_splitSize(), tilingData.get_group(), tilingData.get_realCoreNum(), tilingData.get_numPerCore(),
        tilingData.get_loopNum(), tilingData.get_tailLoopNum(), tilingData.get_nLastTailGroup(),
        tilingData.get_lastTailGroup(), tilingData.get_tilingKey(), tilingData.get_blockSize(),
        tilingData.get_ny());

    OP_LOGD(context, "Tiling4Glu exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Glu).Tiling(Tiling4Glu).TilingParse<GluCompileInfo>(TilingPrepare4Glu);
} // namespace optiling