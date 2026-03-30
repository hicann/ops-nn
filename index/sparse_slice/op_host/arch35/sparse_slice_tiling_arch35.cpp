/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sparse_slice_tiling_arch35.cpp
 * \brief
 */
 
#include <cmath>
#include "sparse_slice_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "error_util.h"

using namespace std;
using namespace ge;
using namespace Ops::NN::OpTiling;

namespace optiling {
const std::set<ge::DataType> INDICES_SUPPORT_DTYPE_SET = {ge::DT_INT64};
const std::set<ge::DataType> VALUE_SUPPORT_DTYPE_SET = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,   ge::DT_UINT8,
                                                        ge::DT_INT8,  ge::DT_INT16,   ge::DT_UINT16, ge::DT_INT32,
                                                        ge::DT_INT64, ge::DT_BOOL};
constexpr int64_t DIGIT_ZERO = 0;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_THREE = 3;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_SIX = 6;
constexpr int64_t DIGIT_SEVEN = 7;
constexpr int64_t DIGIT_TWENTYFOUR = 24;
constexpr int64_t DIGIT_TEN_THOUSAND = 10000;
constexpr int64_t SIZE_OF_INT64 = 8;
constexpr int64_t RESERVED_UB_SIZE = 8 * 1024;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t MAX_ITER_DIM = 32;
constexpr int64_t WORKSPACE_SIZE_ALIGN = 512;
constexpr int64_t SHAPE_IDX = 2;
constexpr int64_t START_IDX = 3;
constexpr int64_t SIZE_IDX = 4;
constexpr uint64_t DCACHE_SIZE = 32UL * 1024UL;
constexpr int64_t INDICES_NUM_MAX_SIMT = 19968;

template <typename T>
static void GetConstValueToShape(const gert::Tensor* tensor, size_t size, gert::Shape* shape)
{
    const T* value = tensor->GetData<T>();
    shape->SetDimNum(size);
    for (size_t i = 0; i < size; i++) {
        shape->SetDim(i, value[i]);
    }
}

bool SparseSliceTiling::UseSIMT()
{
    bool rank2 = tilingParams.rankNumbers > DIGIT_TWO;
    bool dataSizeUB =
        tilingParams.valueNumbers * tilingParams.rankNumbers / tilingParams.totalCoreNum <= INDICES_NUM_MAX_SIMT;
    return rank2 && dataSizeUB;
}

ge::graphStatus SparseSliceTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter SparseSliceTiling GetShapeAttrsInfo.");
    OP_TILING_CHECK(
        CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Check datatype failed. "),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Check shape failed. "),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "End SparseSliceTiling GetShapeAttrsInfo.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSliceTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter SparseSliceTiling GetPlatformInfo.");
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK(
        (tilingParams.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParams.ubSize = static_cast<int64_t>(ubSize) - RESERVED_UB_SIZE;
    OP_TILING_CHECK(
        (tilingParams.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);
    tilingParams.vfLen = Ops::Base::GetVRegSize(context_);
    tilingParams.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context_->GetNodeName(), "End SparseSliceTiling GetPlatformInfo.");

    return ge::GRAPH_SUCCESS;
}

bool SparseSliceTiling::IsCapable()
{
    return true;
}

ge::graphStatus SparseSliceTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter SparseSliceTiling DoOpTiling.");
    ge::graphStatus res = SetTilingParams();
    OP_TILING_CHECK(
        res != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SparseSliceTiling SetTilingParams Failed"),
        return res);

    SetTilingData();
    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSliceTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSliceTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSliceTiling::PostTiling()
{
    if (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        OP_LOGD(context_->GetNodeName(), "Tiling DataSize Greater than capacity, please check.");
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    OP_LOGD(nodeName.c_str(), "Tiling totalCoreNum is %lu.", tilingParams.totalCoreNum);
    context_->SetBlockDim(tilingParams.totalCoreNum);

    if (tilingParams.templateType == DIGIT_FOUR) {
        auto res = context_->SetLocalMemorySize(tilingParams.ubSize - DCACHE_SIZE);
        OP_LOGD(nodeName.c_str(), "SetLocalMemorySize ubSize = %lu, %d.", tilingParams.ubSize, res);
    }

    if (tilingParams.templateType == DIGIT_ONE || tilingParams.templateType == DIGIT_FOUR) {
        context_->SetScheduleMode(DIGIT_ONE);
        OP_LOGD(context_->GetNodeName(), "Set block sync batch mode.");
    }

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    OP_LOGD(nodeName.c_str(), "Tiling workspaceSize is %ld.", tilingParams.workspaceSize);
    auto workspaceSizeAlign =
        ((tilingParams.valueNumbers * sizeof(int8_t) + WORKSPACE_SIZE_ALIGN - 1) / WORKSPACE_SIZE_ALIGN) *
            WORKSPACE_SIZE_ALIGN +
        WORKSPACE_SIZE_ALIGN * 65;
    workspaces[0] = tilingParams.workspaceSize + workspaceSizeAlign;

    return ge::GRAPH_SUCCESS;
}

uint64_t SparseSliceTiling::GetTilingKey() const
{
    int64_t tilingKey = tilingParams.tilingKey;
    OP_LOGD(nodeName.c_str(), "TilingKey is %lu.", tilingKey);
    return tilingKey;
}

// 非override函数
ge::graphStatus SparseSliceTiling::CheckDtype()
{
    auto indicesPtr = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesPtr);
    auto indicesDtype = indicesPtr->GetDataType();
    OP_TILING_CHECK(
        INDICES_SUPPORT_DTYPE_SET.count(indicesDtype) == 0,
        OP_LOGE(context_->GetNodeName(), "Input indices only support INT64 currently, please check. "),
        return ge::GRAPH_FAILED);

    auto valuesPtr = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valuesPtr);
    auto valuesDtype = valuesPtr->GetDataType();
    OP_TILING_CHECK(
        VALUE_SUPPORT_DTYPE_SET.count(valuesDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Input values only support DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT8, DT_INT8, DT_INT16, DT_UINT16,  "
            "DT_INT32, DT_INT64, DT_BOOL currently, please check. "),
        return ge::GRAPH_FAILED);

    auto shapePtr = context_->GetInputDesc(DIGIT_TWO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapePtr);
    auto shapeDtype = shapePtr->GetDataType();
    OP_TILING_CHECK(
        INDICES_SUPPORT_DTYPE_SET.count(shapeDtype) == 0,
        OP_LOGE(context_->GetNodeName(), "Input shape only support INT64 currently, please check. "),
        return ge::GRAPH_FAILED);

    auto startPtr = context_->GetInputDesc(DIGIT_THREE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, startPtr);
    auto startDtype = startPtr->GetDataType();
    OP_TILING_CHECK(
        INDICES_SUPPORT_DTYPE_SET.count(startDtype) == 0,
        OP_LOGE(context_->GetNodeName(), "Input start only support INT64 currently, please check. "),
        return ge::GRAPH_FAILED);

    auto sizePtr = context_->GetInputDesc(DIGIT_FOUR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sizePtr);
    auto sizeDtype = sizePtr->GetDataType();
    OP_TILING_CHECK(
        INDICES_SUPPORT_DTYPE_SET.count(sizeDtype) == 0,
        OP_LOGE(context_->GetNodeName(), "Input size only support INT64 currently, please check. "),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSliceTiling::CheckShape()
{
    auto indicesPtr = context_->GetInputShape(0);
    auto indicesShape = indicesPtr->GetStorageShape();
    OP_TILING_CHECK(
        static_cast<int64_t>(indicesShape.GetDimNum()) != DIGIT_TWO,
        OP_LOGE(
            context_->GetNodeName(), "Input indices has dim number %ld, it should be 2. ",
            static_cast<int64_t>(indicesShape.GetDimNum())),
        return ge::GRAPH_FAILED);
    auto valueNumbers = static_cast<int64_t>(indicesShape.GetDim(0));
    auto rankNumbers = static_cast<int64_t>(indicesShape.GetDim(1));
    OP_TILING_CHECK(
        rankNumbers > DIGIT_TWENTYFOUR || rankNumbers < DIGIT_ONE,
        OP_LOGE(context_->GetNodeName(), "Input indices second dim only support 1-24, current is %ld. ", rankNumbers),
        return ge::GRAPH_FAILED);
    auto valuesPtr = context_->GetInputShape(1);
    auto valuesShape = valuesPtr->GetStorageShape();
    OP_TILING_CHECK(
        static_cast<int64_t>(valuesShape.GetDimNum()) != DIGIT_ONE,
        OP_LOGE(
            context_->GetNodeName(), "Input values has dim number %ld, it should be 1. ",
            static_cast<int64_t>(valuesShape.GetDimNum())),
        return ge::GRAPH_FAILED);
    auto actualValueNumbers = static_cast<int64_t>(valuesShape.GetDim(0));
    OP_TILING_CHECK(
        valueNumbers != actualValueNumbers,
        OP_LOGE(
            context_->GetNodeName(),
            "First dim of input values is %ld, it should be the same as the first dim of indices, which is %ld. ",
            actualValueNumbers, valueNumbers),
        return ge::GRAPH_FAILED);
    auto shapePtr = context_->GetInputShape(DIGIT_TWO);
    auto shapeShape = shapePtr->GetStorageShape();
    OP_TILING_CHECK(
        static_cast<int64_t>(shapeShape.GetDimNum()) != DIGIT_ONE,
        OP_LOGE(
            context_->GetNodeName(), "Input shape has dim number %ld, it should be 1. ",
            static_cast<int64_t>(shapeShape.GetDimNum())),
        return ge::GRAPH_FAILED);
    auto shapeRankNumbers = static_cast<int64_t>(shapeShape.GetDim(0));
    OP_TILING_CHECK(
        rankNumbers != shapeRankNumbers,
        OP_LOGE(
            context_->GetNodeName(),
            "First dim of input rank is %ld, it should be the same as the second dim of indices, which is %ld. ",
            shapeRankNumbers, rankNumbers),
        return ge::GRAPH_FAILED);
    auto startPtr = context_->GetInputShape(DIGIT_THREE);
    auto startShape = startPtr->GetStorageShape();
    OP_TILING_CHECK(
        startShape != shapeShape,
        OP_LOGE(
            context_->GetNodeName(),
            "The shape of input start is not the same with the shape of input shape, they should be the same. "),
        return ge::GRAPH_FAILED);
    auto sizePtr = context_->GetInputShape(DIGIT_FOUR);
    auto sizeShape = sizePtr->GetStorageShape();
    OP_TILING_CHECK(
        sizeShape != shapeShape,
        OP_LOGE(
            context_->GetNodeName(),
            "The shape of input size is not the same with the shape of input shape, they should be the same. "),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSliceTiling::SetTilingParams()
{
    OP_LOGD(context_->GetNodeName(), "Enter SparseSliceTiling SetTilingParams.");
    auto indicesPtr = context_->GetInputShape(0);
    auto indicesShape = indicesPtr->GetStorageShape();
    tilingParams.valueNumbers = static_cast<int64_t>(indicesShape.GetDim(0));
    tilingParams.rankNumbers = static_cast<int64_t>(indicesShape.GetDim(1));

    auto valuesPtr = context_->GetInputDesc(1);
    auto valuesDtype = valuesPtr->GetDataType();
    int64_t valuesDataTypeSize = GetSizeByDataType(valuesDtype);
    OP_LOGD(context_->GetNodeName(), "The data type size of input values is %ld. ", valuesDataTypeSize);

    OP_TILING_CHECK(
        CalcYShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Calc y shape failed. "),
        return ge::GRAPH_FAILED);
    auto sizePerCalc =
        (tilingParams.rankNumbers * SIZE_OF_INT64 * DIGIT_SEVEN + valuesDataTypeSize * DIGIT_TWO) * DOUBLE_BUFFER;

    tilingParams.templateType = DIGIT_ONE;
    if (tilingParams.valueNumbers == DIGIT_ZERO) {
        tilingParams.templateType = DIGIT_TWO;
        OP_LOGD(context_->GetNodeName(), "Enters empty tensor template. (Number of input values is 0)");
        tilingParams.tilingKey = tilingParams.templateType * DIGIT_TEN_THOUSAND;
        tilingParams.valuePerUb = 0;
        tilingParams.valuePerCore = 0;
        tilingParams.usedCoreNum = 1;
        tilingParams.valuePerTail = 0;
        return ge::GRAPH_SUCCESS;
    } else if (tilingParams.IsEmptyYShape == true) {
        tilingParams.templateType = DIGIT_TWO;
        OP_LOGD(context_->GetNodeName(), "Enters empty tensor template. (Output shape implies empty tensor)");
        tilingParams.tilingKey = tilingParams.templateType * DIGIT_TEN_THOUSAND;
        tilingParams.valuePerUb = 0;
        tilingParams.valuePerCore = 0;
        tilingParams.usedCoreNum = 1;
        tilingParams.valuePerTail = 0;
        return ge::GRAPH_SUCCESS;
    } else if (UseSIMT()) {
        tilingParams.templateType = DIGIT_FOUR;
    }

    tilingParams.tilingKey = tilingParams.templateType * DIGIT_TEN_THOUSAND;

    tilingParams.valuePerUb = tilingParams.ubSize / sizePerCalc;
    tilingParams.valuePerCore =
        (tilingParams.valueNumbers + tilingParams.totalCoreNum - DIGIT_ONE) / tilingParams.totalCoreNum;
    tilingParams.usedCoreNum =
        (tilingParams.valueNumbers + tilingParams.valuePerCore - DIGIT_ONE) / tilingParams.valuePerCore;
    tilingParams.valuePerTail = tilingParams.valuePerCore * tilingParams.usedCoreNum == tilingParams.valueNumbers ?
                                    tilingParams.valuePerCore :
                                    tilingParams.valueNumbers % tilingParams.valuePerCore;

    return ge::GRAPH_SUCCESS;
}

void SparseSliceTiling::SetTilingData()
{
    tilingData.set_usedCoreNum(tilingParams.usedCoreNum);
    tilingData.set_valueNumbers(tilingParams.valueNumbers);
    tilingData.set_rankNumbers(tilingParams.rankNumbers);
    tilingData.set_valuePerUb(tilingParams.valuePerUb);
    tilingData.set_valuePerCore(tilingParams.valuePerCore);
    tilingData.set_valuePerTail(tilingParams.valuePerTail);
}

void SparseSliceTiling::PrintTilingData()
{
    OP_LOGD(
        context_->GetNodeName(),
        "PrintTilingData usedCoreNum: %ld, valueNumbers: %ld, rankNumbers: %ld, "
        "valuePerUb: %ld, valuePerCore: %ld, valuePerTail: %ld. ",
        tilingData.get_usedCoreNum(), tilingData.get_valueNumbers(), tilingData.get_rankNumbers(),
        tilingData.get_valuePerUb(), tilingData.get_valuePerCore(), tilingData.get_valuePerTail());
}

ge::graphStatus SparseSliceTiling::CalcYShape()
{
    OP_LOGD(context_->GetNodeName(), "Begin calculate y_shape. ");
    const gert::Tensor* shapeTensor = context_->GetInputTensor(DIGIT_TWO);
    const gert::Tensor* startTensor = context_->GetInputTensor(DIGIT_THREE);
    const gert::Tensor* sizeTensor = context_->GetInputTensor(DIGIT_FOUR);
    if (shapeTensor == nullptr || startTensor == nullptr || sizeTensor == nullptr) {
        OP_LOGD(context_->GetNodeName(), "INPUT TENSOR IS NULLPTR");
        return ge::GRAPH_FAILED;
    }

    const int64_t* shapeValue = shapeTensor->GetData<int64_t>();
    const int64_t* startValue = startTensor->GetData<int64_t>();
    const int64_t* sizeValue = sizeTensor->GetData<int64_t>();
    if (shapeValue == nullptr || startValue == nullptr || sizeValue == nullptr) {
        OP_LOGD(context_->GetNodeName(), "INPUT TENSOR VALUE IS NULLPTR");
        return ge::GRAPH_FAILED;
    }

    GetValueList(DIGIT_TWO, shapeTensor, tilingParams.rankNumbers, tilingParams.shape);
    GetValueList(DIGIT_THREE, startTensor, tilingParams.rankNumbers, tilingParams.start);
    GetValueList(DIGIT_FOUR, sizeTensor, tilingParams.rankNumbers, tilingParams.size);

    for (int64_t i = 0; i < tilingParams.rankNumbers; i++) {
        int64_t tmpShape = tilingParams.shape[i];
        int64_t tmpStart = tilingParams.start[i];
        int64_t tmpSize = tilingParams.size[i];
        int64_t tmpEndValue = tmpStart + tmpSize;
        int64_t tmpYShapeValue = tmpShape;
        if (tmpYShapeValue > tmpEndValue) {
            tmpYShapeValue = tmpEndValue;
        }
        tmpYShapeValue = tmpYShapeValue - tmpStart;
        if (tmpYShapeValue <= 0) {
            tmpYShapeValue = 0;
            tilingParams.IsEmptyYShape = true;
        }
        OP_LOGD(context_->GetNodeName(), "Print curent Y value %ld. ", tmpYShapeValue);
        tilingParams.yShapeOut[i] = tmpYShapeValue;
        tilingParams.sliceStart[i] = tmpStart;
        tilingParams.sliceEnd[i] = tmpEndValue;
    }

    tilingData.set_yShape(tilingParams.yShapeOut);
    tilingData.set_sliceStart(tilingParams.sliceStart);
    tilingData.set_sliceEnd(tilingParams.sliceEnd);
    OP_LOGD(context_->GetNodeName(), "Print Y shape is empty: %d. ", tilingParams.IsEmptyYShape);
    OP_LOGD(context_->GetNodeName(), "End Calculate Y shape. ");
    return ge::GRAPH_SUCCESS;
}

void SparseSliceTiling::GetValueList(size_t idx, const gert::Tensor* tensor, int64_t size, gert::Shape& valueList)
{
    if (size > 0) {
        if (tensor->GetDataType() == ge::DT_INT64) {
            GetConstValueToShape<int64_t>(tensor, size, &valueList);
            OP_LOGD(context_->GetNodeName(), "GetConstValueToShape successfully");
        } else {
            OP_LOGD(context_->GetNodeName(), "input[%zu] data type is invalid: %d", idx, tensor->GetDataType());
        }
    }
}

static ge::graphStatus Tiling4SparseSlice(gert::TilingContext* context_)
{
    OP_TILING_CHECK(
        context_ == nullptr, OP_LOGE("SparseSlice", "context_ should not be nullptr."), return ge::GRAPH_FAILED);

    if (IsRegbaseSocVersion(context_)) {
        SparseSliceTiling tiling(context_);
        ge::graphStatus status = tiling.DoTiling();
        return status;
    }

    return ge::GRAPH_FAILED;
}

ge::graphStatus TilingPrepare4SparseSlice(gert::TilingParseContext* context_)
{
    OP_LOGD(context_->GetNodeName(), "TilingPrepare4SparseSlice entering.");

    auto compileInfo = GetCompileInfoPtr<SparseSliceCompileInfo>(context_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_TILING_CHECK(
        (compileInfo->coreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(
            context_->GetNodeName(), "SparseSlice GetHardwareInfo Failed, coreNum:%d, ubSize:%ld.",
            compileInfo->coreNum, compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "GetCoreNum:%d, ubSize:%lu", compileInfo->coreNum, compileInfo->ubSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseSlice)
    .Tiling(Tiling4SparseSlice)
    .TilingParse<SparseSliceCompileInfo>(TilingPrepare4SparseSlice)
    .TilingInputsDataDependency({SHAPE_IDX, START_IDX, SIZE_IDX});
} // namespace optiling