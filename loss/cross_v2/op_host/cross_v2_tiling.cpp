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
 * \file cross_v2_tiling.cc
 * \brief cross_v2 tiling for ascendC impl
 */
#include "cross_v2_tiling.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {
struct CrossV2CompileInfo {
};
const uint32_t X1_IDX = 0;
const uint32_t X2_IDX = 1;
const uint32_t DIM_IDX = 0;
const uint32_t Y_IDX = 0;
const uint32_t TILING_ONE_STEP_MODE = 0;
const uint32_t TILING_DEFAULT_MODE = 1;
const int64_t INPUT_VAR_NUM = 2;
const int64_t MAX_SHAPE_DIM = 8;
const int64_t SPECIFIED_DIM_SIZE = 3;
const int64_t BLOCK_SIZE = 32;
const int64_t BUFFER_NUM = 2;
const int64_t UINT8_ALIGN_SIZE = 256;
const int64_t THRESHOLD_STEP_SIZE = 1;
struct TilingData {
    int64_t batchSize{1};
    int64_t stepSize{1};
    int64_t tileNum{1};
    int64_t tileNumPerBatch{1};
    int64_t tileDataNum{1};
    int64_t tailDataNum{1};
};

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return value;
    }
    return (value + factor - 1) / factor;
}

inline static int64_t RoundUp(int64_t a, int64_t b)
{
    return CeilDiv(a, b) * b;
}

inline int64_t DimProcess(const int64_t shape[], int64_t dimNum, int64_t dimIdx)
{
    int64_t dim = dimIdx;
    if (dim < 0) {
        dim = dimNum + dim;
    }
    if (dim < 0) {
        for (int64_t i = 0; i < dimNum; ++i) {
            if (shape[i] == SPECIFIED_DIM_SIZE) {
                dim = i;
                break;
            }
        }
    }
    return dim;
}

inline void GetBatchAndStepSize(int64_t shape[], int64_t dimNum, int64_t dim, int64_t& batchSize, int64_t& stepSize)
{
    for (int i = 0; i < dimNum; ++i) {
        if (i < dim) {
            batchSize *= shape[i];
        }
        if (i > dim) {
            stepSize *= shape[i];
        }
    }
    return;
}

void GetTilingData(ge::DataType dtype, int64_t inputBytes, uint64_t ubSize, TilingData& tilingData, uint32_t& coreNum)
{
    int64_t& batchSize = tilingData.batchSize;
    int64_t& stepSize = tilingData.stepSize;
    int64_t& tileNum = tilingData.tileNum;
    int64_t& tileNumPerBatch = tilingData.tileNumPerBatch;
    int64_t& tileDataNum = tilingData.tileDataNum;
    int64_t& tailDataNum = tilingData.tailDataNum;
    int64_t ubDataNumber = 11; // 11个包括x1,y1,z1,x2,y2,z2,x3,y3,z3,xa*yb,xb*ya
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_INT32 && dtype != ge::DT_INT16) {
        ubDataNumber = 20; // Cast场景20个是多了9个存放Cast后数据的x1,y1,z1,x2,y2,z2,x3,y3,z3
    }
    int64_t tileAlignSize = BLOCK_SIZE;
    int64_t numPerAlignSize = tileAlignSize / inputBytes;
    if (dtype == ge::DT_UINT8) {
        tileAlignSize = UINT8_ALIGN_SIZE;
        numPerAlignSize = tileAlignSize / inputBytes;
    }
    int64_t tileBlockNum = (static_cast<int64_t>(ubSize) / tileAlignSize / BUFFER_NUM) / ubDataNumber;
    tileDataNum = tileBlockNum * numPerAlignSize;
    tileNumPerBatch = CeilDiv(stepSize, tileDataNum);
    tileNum = batchSize * tileNumPerBatch;
    if (tileNum < static_cast<int64_t>(coreNum)) {
        tileDataNum = batchSize * stepSize / static_cast<int64_t>(coreNum) / numPerAlignSize * numPerAlignSize;
        tileDataNum = (tileDataNum < numPerAlignSize) ? numPerAlignSize : tileDataNum;
        tileNumPerBatch = CeilDiv(stepSize, tileDataNum);
        tileNum = batchSize * tileNumPerBatch;
    }
    tailDataNum = stepSize % tileDataNum;
    tailDataNum = (tailDataNum == 0) ? tileDataNum : tailDataNum;
    coreNum = (tileNum > static_cast<int64_t>(coreNum)) ? coreNum : static_cast<uint32_t>(tileNum);
    return;
}

void GetTilingDataOneStep(
    ge::DataType dtype, int64_t inputBytes, uint64_t ubSize, TilingData& tilingData, uint32_t& coreNum)
{
    int64_t& batchSize = tilingData.batchSize;
    int64_t& tileNum = tilingData.tileNum;
    int64_t& tileDataNum = tilingData.tileDataNum;
    int64_t& tailDataNum = tilingData.tailDataNum;
    int64_t ubDataNumber = 21; // 20个相比stepSize>1场景多10份空间存放未分开的x1,y1,z1,x2,y2,z2,x3,y3,z3和srcOffset
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_INT32 && dtype != ge::DT_INT16) {
        ubDataNumber = 30; // 30相比stepSize>1场景多10份空间存放未分开的x1,y1,z1,x2,y2,z2,x3,y3,z3和srcOffset
    }
    int64_t tileAlignSize = BLOCK_SIZE;
    int64_t numPerAlignSize = tileAlignSize / inputBytes;
    if (dtype == ge::DT_UINT8) {
        tileAlignSize = UINT8_ALIGN_SIZE;
        numPerAlignSize = tileAlignSize / inputBytes;
    }
    int64_t tileBlockNum = (static_cast<int64_t>(ubSize) / tileAlignSize / BUFFER_NUM) / ubDataNumber;
    tileDataNum = tileBlockNum * numPerAlignSize;
    tileNum = CeilDiv(batchSize, tileDataNum);
    if (tileNum < static_cast<int64_t>(coreNum)) {
        tileDataNum = batchSize / static_cast<int64_t>(coreNum) / numPerAlignSize * numPerAlignSize;
        tileDataNum = (tileDataNum < numPerAlignSize) ? numPerAlignSize : tileDataNum;
        tileNum = CeilDiv(batchSize, tileDataNum);
    }
    tailDataNum = batchSize % tileDataNum;
    tailDataNum = (tailDataNum == 0) ? tileDataNum : tailDataNum;
    coreNum = (tileNum > static_cast<int64_t>(coreNum)) ? coreNum : static_cast<uint32_t>(tileNum);
    return;
}

void SetTiling(
    gert::TilingContext* context, const TilingData& tilingData, const uint32_t& coreNum, uint32_t sysWorkspaceSize,
    CrossV2TilingData& tiling)
{
    tiling.set_stepSize(tilingData.stepSize);
    tiling.set_tileNum(tilingData.tileNum);
    tiling.set_tileNumPerBatch(tilingData.tileNumPerBatch);
    tiling.set_tileDataNum(tilingData.tileDataNum);
    tiling.set_tailDataNum(tilingData.tailDataNum);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = sysWorkspaceSize;

    OP_LOGD(context, "coreNum = %u.", coreNum);
    OP_LOGD(context, "batchSize = %lu.", tilingData.batchSize);
    OP_LOGD(context, "stepSize = %lu.", tilingData.stepSize);
    OP_LOGD(context, "tileNum = %lu.", tilingData.tileNum);
    OP_LOGD(context, "tileNumPerBatch = %lu.", tilingData.tileNumPerBatch);
    OP_LOGD(context, "tileDataNum = %lu.", tilingData.tileDataNum);
    OP_LOGD(context, "tailDataNum = %lu.", tilingData.tailDataNum);
    OP_LOGD(context, "CrossV2 tiling end.");
}

ge::graphStatus TilingFuncForCrossV2(gert::TilingContext* context)
{
    CrossV2TilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == static_cast<uint32_t>(0)) {
        OP_LOGE(context, "coreNum should not be 0!");
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* x1StorageShape = context->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1StorageShape);
    auto x1Shape = x1StorageShape->GetStorageShape();
    OP_CHECK_IF(
        x1Shape.GetShapeSize() == 0,
        OP_LOGE(context->GetNodeName(), "the x1Shape of input should not be empty tensor"),
        return ge::GRAPH_FAILED);
    int64_t dimNum = x1Shape.GetDimNum();
    int64_t shape[MAX_SHAPE_DIM];
    TilingData tilingData;
    int64_t* ss = &shape[0];
    for (int i = 0; i < dimNum; ++i) {
        ss[i] = x1Shape.GetDim(i);
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t dim = DimProcess(shape, dimNum, *attrs->GetInt(DIM_IDX));
    if (dim < 0 || dim >= dimNum) {
        OP_LOGE(context, "dim is invalid!");
        return ge::GRAPH_FAILED;
    }
    GetBatchAndStepSize(shape, dimNum, dim, tilingData.batchSize, tilingData.stepSize);
    OP_CHECK_IF(
        tilingData.batchSize < 1 || tilingData.stepSize < 1,
        OP_LOGE(context->GetNodeName(), "batchSize and stepSize should not less than 1!"),
        return ge::GRAPH_FAILED);
    uint32_t typeLength = 0;
    auto x1Desc = context->GetInputDesc(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    ge::DataType dtype = x1Desc->GetDataType();
    ge::TypeUtils::GetDataTypeLength(dtype, typeLength);
    int64_t inputBytes = static_cast<int64_t>(typeLength);
    OP_CHECK_IF(
        inputBytes == 0, OP_LOGE(context->GetNodeName(), "inputBytes should not be 0!"),
        return ge::GRAPH_FAILED);
    if (tilingData.stepSize > THRESHOLD_STEP_SIZE) {
        GetTilingData(dtype, inputBytes, ubSize, tilingData, coreNum);
        context->SetTilingKey(TILING_DEFAULT_MODE);
    } else {
        GetTilingDataOneStep(dtype, inputBytes, ubSize, tilingData, coreNum);
        context->SetTilingKey(TILING_ONE_STEP_MODE);
    }
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    SetTiling(context, tilingData, coreNum, sysWorkspaceSize, tiling);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForCrossV2(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForCrossV2", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CrossV2).Tiling(TilingFuncForCrossV2).TilingParse<CrossV2CompileInfo>(TilingPrepareForCrossV2);
} // namespace optiling
