/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
* \file map_index_tiling_arch35.cpp
* \brief
*/

#include "map_index_tiling_arch35.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_util.h"

using namespace std;
using namespace ge;

namespace optiling {

constexpr int64_t DIGIT_EIGHT = 8;
constexpr int64_t X_MAX_DIM0 = 24000;
constexpr int64_t DATA_SEQ_MAX_DIM0 = 256;
constexpr int64_t VL_NUMS = 64;
constexpr size_t WORKSPACE_SIZE = 32;
constexpr size_t LEVEL_INDEX_INDEX = 2;
constexpr int64_t NUM_TWO_DB = 2;
constexpr int64_t RESERVED_UB_SIZE = static_cast<int64_t>(8) * 1024; // 8k
constexpr int64_t WORKSPACE_BUFFER = static_cast<int64_t>(20) * 1024 * 1024;
constexpr int64_t ATTR_INDEX_TRANSPOSE = 0;
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = { ge::DT_INT32 };

template <class T> T inline CeilDivide(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

static ge::graphStatus CheckDtype(const gert::TilingContext *context, MapIndexTilingParam &tilingParam)
{
    OP_LOGD(context->GetNodeName(), "CheckDtype begin.");
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE(context->GetNodeName(), 
            "Input x's data type is [%s], only supports INT32.",
            Ops::Base::ToString(static_cast<ge::DataType>(xDtype)).c_str()),
        return ge::GRAPH_FAILED);

    auto inputDataSeqPtr = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDataSeqPtr);
    auto dataSeqDtype = inputDataSeqPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(dataSeqDtype) == 0,
        OP_LOGE(context->GetNodeName(), 
            "Input dataSeq's data type is [%s], only supports INT32.",
            Ops::Base::ToString(static_cast<ge::DataType>(dataSeqDtype)).c_str()),
        return ge::GRAPH_FAILED);

    auto levelIndexInput = context->GetOptionalInputDesc(LEVEL_INDEX_INDEX);
    if (levelIndexInput == nullptr) {
        tilingParam.hasLevelIndex = false;
    } else {
        auto levelIndexDtype = levelIndexInput->GetDataType();
        OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(levelIndexDtype) == 0,
        OP_LOGE(context->GetNodeName(), 
            "Input levelIndex's data type is [%s], only supports INT32.",
            Ops::Base::ToString(static_cast<ge::DataType>(levelIndexDtype)).c_str()),
        return ge::GRAPH_FAILED);
    }

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE(context->GetNodeName(),
            "Output y's data type is [%s], only supports INT32.",
            Ops::Base::ToString(static_cast<ge::DataType>(yDtype)).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(const gert::TilingContext *context, MapIndexTilingParam &tilingParam)
{
    OP_LOGD(context->GetNodeName(), "CheckShape begin.");
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    auto dataSeqShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dataSeqShapePtr);
    auto dataSeqShape = dataSeqShapePtr->GetStorageShape();

    auto yShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = Ops::NN::OpTiling::EnsureNotScalar(yShapePtr->GetStorageShape());
    OP_CHECK_IF(yShape.GetDimNum() != 1,
        OP_LOGE(context->GetNodeName(),
        "The shape of output y must be 1D."),
        return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(yShape.GetDim(0) != 1, 
        OP_LOGE(context->GetNodeName(),
        "The shape of output y must be [1]."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(xShape.GetDimNum() != 1,
        OP_LOGE(context->GetNodeName(),
        "The shape of input x must be 1D."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDim(0) > X_MAX_DIM0,
        OP_LOGE(context->GetNodeName(),
        "The shape of input x must be less than 24000."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(dataSeqShape.GetDimNum() != 1,
        OP_LOGE(context->GetNodeName(),
        "The shape of input data_seq must be 1D."),
        return ge::GRAPH_FAILED);

    tilingParam.Dim1Size = xShape.GetDim(0);
    OP_CHECK_IF(dataSeqShape.GetDim(0) % xShape.GetDim(0) != 0,
        OP_LOGE(context->GetNodeName(),
        "the length of data_seq must be multiple of the length of x"),
        return ge::GRAPH_FAILED);

    tilingParam.Dim0Size = dataSeqShape.GetDim(0) / xShape.GetDim(0);

    OP_CHECK_IF(tilingParam.Dim0Size > DATA_SEQ_MAX_DIM0,
        OP_LOGE(context->GetNodeName(),
        "The input length of dataseq, which is a multiple of x, should be less than 256."),
        return ge::GRAPH_FAILED);
    
    if(tilingParam.hasLevelIndex) {
        auto levelIndexShapePtr = context->GetOptionalInputShape(LEVEL_INDEX_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, levelIndexShapePtr);
        auto levelIndexShape = levelIndexShapePtr->GetStorageShape();
        OP_CHECK_IF(levelIndexShape.GetDimNum() != 1,
            OP_LOGE(context->GetNodeName(),
            "The shape of input level_index must be 1D."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(tilingParam.Dim0Size != levelIndexShape.GetDim(0),
            OP_LOGE(context->GetNodeName(),
            "The input levelindex shape should be a multiple of dataseq, which should be a multiple of x."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttr(const gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "checkAttr begin.");

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    
    auto* attrTransPose = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrTransPose);
    OP_CHECK_IF((*attrTransPose),
        OP_LOGE(
            context->GetNodeName(), "The attr transpose should be false on A5, please check"),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatInfo(const gert::TilingContext *context, MapIndexTilingParam &tilingParam)
{
    OP_LOGD(context->GetNodeName(), "GetPlatInfo begin.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParam.totalCoreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParam.ubSize = static_cast<int64_t>(ubSize) - RESERVED_UB_SIZE;
    OP_CHECK_IF((tilingParam.ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    tilingParam.vfLen = Ops::Base::GetVRegSize(context);
    tilingParam.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTiling(const gert::TilingContext *context, MapIndexTilingParam &tilingParam)
{
    OP_LOGD(context->GetNodeName(), "DoTiling begin.");
    tilingParam.normalCoreProcessNum = CeilDivide(tilingParam.Dim0Size, tilingParam.totalCoreNum);
    tilingParam.usedCoreNum = CeilDivide(tilingParam.Dim0Size, tilingParam.normalCoreProcessNum);
    tilingParam.tailCoreProcessNum = tilingParam.Dim0Size - tilingParam.normalCoreProcessNum * (tilingParam.usedCoreNum - 1);
    tilingParam.Dim1SizeAlign = Ops::Base::CeilAlign(tilingParam.Dim1Size, VL_NUMS);

    int64_t OneRowUB = tilingParam.Dim1SizeAlign * sizeof(int32_t);
    int64_t rowsNums = tilingParam.ubSize / OneRowUB;
    int64_t dataSeqNums = rowsNums - 1;

    if (dataSeqNums > tilingParam.normalCoreProcessNum){
        tilingParam.CopyInDim0 = tilingParam.normalCoreProcessNum;
        tilingParam.CopyInDim0Times = 1;
    } else {
        tilingParam.CopyInDim0 = dataSeqNums;
        tilingParam.CopyInDim0Times = CeilDivide(tilingParam.normalCoreProcessNum, dataSeqNums);
        tilingParam.tailCopyInDim0Times = CeilDivide(tilingParam.tailCoreProcessNum, dataSeqNums);
    }

    if (tilingParam.CopyInDim0 < NUM_TWO_DB ){
        tilingParam.doubleBuffNum = 1;
    } else {
        tilingParam.doubleBuffNum = NUM_TWO_DB;
        tilingParam.CopyInDim0 = Ops::Base::CeilAlign(tilingParam.CopyInDim0, tilingParam.doubleBuffNum);
    }

    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus SetTilingData(gert::TilingContext *context,
    const MapIndexTilingParam &tilingParam, MapIndexTilingData &tilingData)
{
    OP_LOGD(context->GetNodeName(), "SetTilingData begin.");
    tilingData.set_totalCoreNum(tilingParam.totalCoreNum);
    tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
    tilingData.set_normalCoreProcessNum(tilingParam.normalCoreProcessNum);
    tilingData.set_tailCoreProcessNum(tilingParam.tailCoreProcessNum);
    tilingData.set_Dim1Size(tilingParam.Dim1Size);
    tilingData.set_Dim1SizeAlign(tilingParam.Dim1SizeAlign);
    tilingData.set_CopyInDim0(tilingParam.CopyInDim0);
    tilingData.set_CopyInDim0Times(tilingParam.CopyInDim0Times);
    tilingData.set_tailCopyInDim0Times(tilingParam.tailCopyInDim0Times);
    tilingData.set_doubleBuffNum(tilingParam.doubleBuffNum);

    OP_CHECK_IF(tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity(),
            OP_LOGE(context->GetNodeName(), "tiling datasize: %zu is bigger than %zu",
                                        tilingData.GetDataSize(), context->GetRawTilingData()->GetCapacity()),
            return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(tilingData.get_totalCoreNum());
    context->SetTilingKey(1);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(WORKSPACE_BUFFER);
    return ge::GRAPH_SUCCESS;
}

inline static void PrintTilingData(const gert::TilingContext *context, MapIndexTilingData &tilingData)
{
    OP_LOGI(context->GetNodeName(), "tilingData is totalCoreNum:%ld, usedCoreNum:%ld,  normalCoreProcessNum:%ld, \
        tailCoreProcessNum:%ld, Dim1Size:%ld, Dim1SizeAlign:%ld, CopyInDim0:%ld, CopyInDim0Times:%ld, tailCopyInDim0Times:%ld, doubleBuffNum:%ld",
        tilingData.get_totalCoreNum(), tilingData.get_usedCoreNum(), tilingData.get_normalCoreProcessNum(),
        tilingData.get_tailCoreProcessNum(), tilingData.get_Dim1Size(), tilingData.get_Dim1SizeAlign(),
        tilingData.get_CopyInDim0(), tilingData.get_CopyInDim0Times(), tilingData.get_tailCopyInDim0Times(), tilingData.get_doubleBuffNum());
}

ge::graphStatus Tiling4MapIndex(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4MapIndex running begin.");

    MapIndexTilingParam tilingParam;

    OP_CHECK_IF(CheckDtype(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The data type check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The shape check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetPlatInfo(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "GetPlatInfo failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(DoTiling(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DoTiling failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckAttr(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "check attr failed."), return ge::GRAPH_FAILED);

    MapIndexTilingData tilingData;
    OP_CHECK_IF(SetTilingData(context, tilingParam, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "SetContext fail."),
        return ge::GRAPH_FAILED);

    PrintTilingData(context, tilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4MapIndex(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4MapIndex entering.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MapIndex)
    .Tiling(Tiling4MapIndex)
    .TilingParse<MapIndexCompileInfo>(TilingPrepare4MapIndex);
}