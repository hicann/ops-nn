/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_dynamic_mx_quant_with_dual_axis_tiling_arch35.cpp
 * \brief
 */

#include <string>
#include "grouped_dynamic_mx_quant_with_dual_axis_tiling_arch35.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "log/log.h"
#include "../../op_kernel/arch35/grouped_dynamic_mx_quant_with_dual_axis_tiling_key.h"
#include "graph/utils/type_utils.h"

namespace optiling {
namespace {
constexpr size_t INDEX_INPUT_X = 0;
constexpr size_t INDEX_INPUT_GROUP_INDEX = 1;
constexpr size_t INDEX_OUTPUT_Y1 = 0;
constexpr size_t INDEX_OUTPUT_SCALE1 = 1;
constexpr size_t INDEX_OUTPUT_Y2 = 2;
constexpr size_t INDEX_OUTPUT_SCALE2 = 3;
constexpr size_t INDEX_ATTR_ROUND_MODE = 0;
constexpr size_t INDEX_ATTR_SCALE_ALG = 1;
constexpr size_t INDEX_ATTR_DST_DTYPE = 2;
constexpr size_t INDEX_ATTR_MAX_DTYPE_VALUE = 3;
constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr int64_t ROW_BLOCK_SIZE = 64;
constexpr int64_t COL_BLOCK_SIZE = 256;
constexpr int64_t SCALE_AXIS_UNIT = 64;
constexpr int64_t SCALE_PAIR = 2;
constexpr int64_t SCALE_ALG_CEIL_ALG = 1;
constexpr int64_t DTYPE_FLOAT8_E5M2 = 35;
constexpr int64_t DTYPE_FLOAT8_E4M3FN = 36;
constexpr float MAX_DTYPE_VALUE_DEFAULT = 0.0f;
constexpr int64_t UB_QUEUE_DEPTH = 2;
constexpr int64_t SCALE2_ROW_SLOT = ROW_BLOCK_SIZE / MX_BLOCK_SIZE;
constexpr int64_t SCALE_COUNT_PER_BLOCK = ROW_BLOCK_SIZE * (COL_BLOCK_SIZE / MX_BLOCK_SIZE);

static int64_t CeilDiv(int64_t lhs, int64_t rhs) { return rhs == 0 ? 0 : (lhs + rhs - 1) / rhs; }

static int64_t Min(int64_t lhs, int64_t rhs) { return lhs < rhs ? lhs : rhs; }

static int64_t CalculateBasicBlockUbSize(int64_t xTypeSize, int64_t ubBlockSize)
{
    // 每行有 COL_BLOCK_SIZE / MX_BLOCK_SIZE 个 reciprocal 元素，先将该行字节数按一个 UB block 对齐，
    // 再乘以 ROW_BLOCK_SIZE，得到完整 reciprocal 缓冲区所需空间。
    const int64_t y1ScaleReciprocalSize = ROW_BLOCK_SIZE *
                                          CeilDiv((COL_BLOCK_SIZE / MX_BLOCK_SIZE) * xTypeSize, ubBlockSize) *
                                          (ubBlockSize / xTypeSize) * xTypeSize;
    return UB_QUEUE_DEPTH * ROW_BLOCK_SIZE * COL_BLOCK_SIZE * xTypeSize         // inQueue (XType): 64 KB
           + UB_QUEUE_DEPTH * ROW_BLOCK_SIZE * COL_BLOCK_SIZE                   // y1OutQueue: 32 KB
           + UB_QUEUE_DEPTH * ROW_BLOCK_SIZE * COL_BLOCK_SIZE                   // y2OutQueue: 32 KB
           + UB_QUEUE_DEPTH * ROW_BLOCK_SIZE * (COL_BLOCK_SIZE / MX_BLOCK_SIZE) // y1ScaleOutQueue: 1 KB
           + UB_QUEUE_DEPTH * SCALE2_ROW_SLOT * COL_BLOCK_SIZE                  // y2ScaleOutQueue : 1 KB
           + SCALE_COUNT_PER_BLOCK * xTypeSize                                  // y1ScaleMaxExpBuf: 1 KB
           + y1ScaleReciprocalSize                                              // y1ScaleReciprocalBuf: 2 KB
           + SCALE2_ROW_SLOT * COL_BLOCK_SIZE * xTypeSize;                      // y2ScaleMaxExpBuf: 1 KB, total: 134 KB
}

static bool IsSupportedInputDtype(ge::DataType dtype) { return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16; }

static bool IsSupportedOutputDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT8_E5M2 || dtype == ge::DT_FLOAT8_E4M3FN;
}

static int64_t GetXTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return static_cast<int64_t>(sizeof(uint16_t));
        default:
            return 0;
    }
}

static bool IsSameShape2D(const gert::Shape& shape, int64_t dim0, int64_t dim1)
{
    return shape.GetDimNum() == 2 && shape.GetDim(0) == dim0 && shape.GetDim(1) == dim1;
}

static bool IsSameShape3D(const gert::Shape& shape, int64_t dim0, int64_t dim1, int64_t dim2)
{
    return shape.GetDimNum() == 3 && shape.GetDim(0) == dim0 && shape.GetDim(1) == dim1 && shape.GetDim(2) == dim2;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context,
                                       GroupedDynamicMxQuantWithDualAxisTilingParam& tilingParam)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(tilingParam.totalCoreNum <= 0,
                OP_LOGE(context, "Failed to get AIV core num, got %ld.", tilingParam.totalCoreNum),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParam.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((tilingParam.ubSize <= 0), OP_LOGE(context, "Failed to get UB size, got %ld.", tilingParam.ubSize),
                return ge::GRAPH_FAILED);
    tilingParam.workspaceSize = static_cast<int64_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape(gert::TilingContext* context,
                                       GroupedDynamicMxQuantWithDualAxisTilingParam& tilingParam)
{
    const gert::StorageShape* xStorageShape = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::StorageShape* groupStorageShape = context->GetInputShape(INDEX_INPUT_GROUP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupStorageShape);
    const gert::Shape& xShape = xStorageShape->GetStorageShape();
    const gert::Shape& groupShape = groupStorageShape->GetStorageShape();
    int64_t xDimNum = xShape.GetDimNum();
    OP_CHECK_IF(xDimNum != 2, OP_LOGE(context, "x dimension count should be 2, but got %ld.", xDimNum),
                return ge::GRAPH_FAILED);
    int64_t groupDimNum = groupShape.GetDimNum();
    OP_CHECK_IF(groupDimNum != 1,
                OP_LOGE(context, "group_index dimension count should be 1, but got %ld.", groupDimNum),
                return ge::GRAPH_FAILED);
    int64_t m = xShape.GetDim(0);
    int64_t n = xShape.GetDim(1);
    int64_t groupNum = groupShape.GetDim(0);
    OP_CHECK_IF(n <= 0, OP_LOGE(context, "x shape last dimension should be positive, but got [M=%ld, N=%ld].", m, n),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupNum <= 0, OP_LOGE(context, "group_index length should be positive, but got %ld.", groupNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(n % SCALE_AXIS_UNIT != 0, OP_LOGE(context, "x.shape[1] should be 64 aligned, but got N=%ld.", n),
                return ge::GRAPH_FAILED);

    tilingParam.m = xShape.GetDim(0);
    tilingParam.n = xShape.GetDim(1);
    tilingParam.groupNum = groupShape.GetDim(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(gert::TilingContext* context,
                                  GroupedDynamicMxQuantWithDualAxisTilingParam& tilingParam)
{
    auto inputXDesc = context->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXDesc);
    auto xDtype = inputXDesc->GetDataType();
    OP_CHECK_IF(!IsSupportedInputDtype(xDtype),
                OP_LOGE(context, "x dtype should be FLOAT16 or BFLOAT16, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                return ge::GRAPH_FAILED);
    auto groupIndexDesc = context->GetInputDesc(INDEX_INPUT_GROUP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupIndexDesc);
    auto groupIndexDtype = groupIndexDesc->GetDataType();
    OP_CHECK_IF(groupIndexDtype != ge::DT_INT64,
                OP_LOGE(context, "group_index dtype should be INT64, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(groupIndexDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto y1Desc = context->GetOutputDesc(INDEX_OUTPUT_Y1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Desc);
    auto y2Desc = context->GetOutputDesc(INDEX_OUTPUT_Y2);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Desc);
    auto y1Dtype = y1Desc->GetDataType();
    auto y2Dtype = y2Desc->GetDataType();
    OP_CHECK_IF(
        !IsSupportedOutputDtype(y1Dtype) || y1Dtype != y2Dtype,
        OP_LOGE(context, "y1 and y2 dtypes should match and be FLOAT8_E5M2 or FLOAT8_E4M3FN, but got y1=%s, y2=%s.",
                ge::TypeUtils::DataTypeToSerialString(y1Dtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(y2Dtype).c_str()),
        return ge::GRAPH_FAILED);
    auto scale1Desc = context->GetOutputDesc(INDEX_OUTPUT_SCALE1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale1Desc);
    auto scale2Desc = context->GetOutputDesc(INDEX_OUTPUT_SCALE2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale2Desc);
    auto scale1Dtype = scale1Desc->GetDataType();
    auto scale2Dtype = scale2Desc->GetDataType();
    OP_CHECK_IF(
        scale1Dtype != ge::DT_FLOAT8_E8M0 || scale2Dtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE(context, "y1_scale and y2_scale dtypes should both be FLOAT8_E8M0, but got y1_scale=%s, y2_scale=%s.",
                ge::TypeUtils::DataTypeToSerialString(scale1Dtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(scale2Dtype).c_str()),
        return ge::GRAPH_FAILED);

    tilingParam.inputDtype = static_cast<int64_t>(xDtype);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(gert::TilingContext* context,
                                  GroupedDynamicMxQuantWithDualAxisTilingParam& tilingParam)
{
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const char* roundMode = attrsPtr->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, roundMode);
    OP_CHECK_IF(std::string(roundMode) != "rint", OP_LOGE(context, "round_mode should be rint, but got %s.", roundMode),
                return ge::GRAPH_FAILED);
    const int64_t* scaleAlg = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleAlg);
    OP_CHECK_IF(*scaleAlg != SCALE_ALG_CEIL_ALG, OP_LOGE(context, "scale_alg should be 1, but got %ld.", *scaleAlg),
                return ge::GRAPH_FAILED);
    const int64_t* dstDtype = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    OP_CHECK_IF(*dstDtype != DTYPE_FLOAT8_E5M2 && *dstDtype != DTYPE_FLOAT8_E4M3FN,
                OP_LOGE(context, "dst_dtype should be FLOAT8_E5M2(35) or FLOAT8_E4M3FN(36), but got %ld.", *dstDtype),
                return ge::GRAPH_FAILED);
    const float* maxDtypeValue = attrsPtr->GetAttrPointer<float>(INDEX_ATTR_MAX_DTYPE_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxDtypeValue);
    OP_CHECK_IF(*maxDtypeValue != MAX_DTYPE_VALUE_DEFAULT,
                OP_LOGE(context, "max_dtype_value should be 0.0, but got %f.", *maxDtypeValue),
                return ge::GRAPH_FAILED);

    auto y1Desc = context->GetOutputDesc(INDEX_OUTPUT_Y1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Desc);
    auto y2Desc = context->GetOutputDesc(INDEX_OUTPUT_Y2);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Desc);
    ge::DataType y1Dtype = y1Desc->GetDataType();
    ge::DataType y2Dtype = y2Desc->GetDataType();
    ge::DataType expectedDtype = static_cast<ge::DataType>(*dstDtype);
    OP_CHECK_IF(y1Dtype != expectedDtype || y2Dtype != expectedDtype,
                OP_LOGE(context, "y1 and y2 dtypes should match dst_dtype, but got dst_dtype=%s, y1=%s, y2=%s.",
                        ge::TypeUtils::DataTypeToSerialString(expectedDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(y1Dtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(y2Dtype).c_str()),
                return ge::GRAPH_FAILED);
    tilingParam.scaleAlg = *scaleAlg;
    tilingParam.dstDtype = *dstDtype;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputShape(gert::TilingContext* context,
                                        const GroupedDynamicMxQuantWithDualAxisTilingParam& tilingParam)
{
    const gert::StorageShape* y1StorageShape = context->GetOutputShape(INDEX_OUTPUT_Y1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1StorageShape);
    const gert::StorageShape* y2StorageShape = context->GetOutputShape(INDEX_OUTPUT_Y2);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2StorageShape);
    const gert::StorageShape* scale1StorageShape = context->GetOutputShape(INDEX_OUTPUT_SCALE1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale1StorageShape);
    const gert::StorageShape* scale2StorageShape = context->GetOutputShape(INDEX_OUTPUT_SCALE2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale2StorageShape);

    const gert::Shape& y1Shape = y1StorageShape->GetStorageShape();
    const gert::Shape& y2Shape = y2StorageShape->GetStorageShape();
    const gert::Shape& scale1Shape = scale1StorageShape->GetStorageShape();
    const gert::Shape& scale2Shape = scale2StorageShape->GetStorageShape();
    OP_CHECK_IF(
        !IsSameShape2D(y1Shape, tilingParam.m, tilingParam.n) || !IsSameShape2D(y2Shape, tilingParam.m, tilingParam.n),
        OP_LOGE(context, "y1 and y2 shapes should both be [M=%ld, N=%ld], but got y1=[%ld,%ld], y2=[%ld,%ld].",
                tilingParam.m, tilingParam.n, y1Shape.GetDim(0), y1Shape.GetDim(1), y2Shape.GetDim(0),
                y2Shape.GetDim(1)),
        return ge::GRAPH_FAILED);

    int64_t expectedScale1Dim1 = tilingParam.n / SCALE_AXIS_UNIT;
    OP_CHECK_IF(!IsSameShape3D(scale1Shape, tilingParam.m, expectedScale1Dim1, SCALE_PAIR),
                OP_LOGE(context, "y1_scale shape should be [M=%ld, N/64=%ld, 2], but got [%ld,%ld,%ld].", tilingParam.m,
                        expectedScale1Dim1, scale1Shape.GetDim(0), scale1Shape.GetDim(1), scale1Shape.GetDim(2)),
                return ge::GRAPH_FAILED);

    int64_t expectedScale2Dim0 = tilingParam.m / SCALE_AXIS_UNIT + tilingParam.groupNum;
    OP_CHECK_IF(
        !IsSameShape3D(scale2Shape, expectedScale2Dim0, tilingParam.n, SCALE_PAIR),
        OP_LOGE(context, "y2_scale shape should be [floor(M/64)+groupNum=%ld, N=%ld, 2], but got [%ld,%ld,%ld].",
                expectedScale2Dim0, tilingParam.n, scale2Shape.GetDim(0), scale2Shape.GetDim(1), scale2Shape.GetDim(2)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeTilingParam(GroupedDynamicMxQuantWithDualAxisTilingParam& tilingParam)
{
    tilingParam.rowBlockSize = ROW_BLOCK_SIZE;
    tilingParam.colBlockSize = COL_BLOCK_SIZE;
    tilingParam.colBlocksNum = CeilDiv(tilingParam.n, tilingParam.colBlockSize);
    tilingParam.scale1ColPairs = tilingParam.n / SCALE_AXIS_UNIT;
    int64_t totalBaseBlocks = (tilingParam.m / ROW_BLOCK_SIZE + tilingParam.groupNum) * tilingParam.colBlocksNum;
    tilingParam.usedCoreNum = Min(tilingParam.totalCoreNum, totalBaseBlocks);
    if (tilingParam.usedCoreNum <= 0) {
        tilingParam.usedCoreNum = 1;
    }
    return ge::GRAPH_SUCCESS;
}

static void SetTilingData(const GroupedDynamicMxQuantWithDualAxisTilingParam& tilingParam,
                          GroupedDynamicMxQuantWithDualAxisTilingData* tilingData)
{
    tilingData->totalCoreNum = tilingParam.totalCoreNum;
    tilingData->usedCoreNum = tilingParam.usedCoreNum;
    tilingData->m = tilingParam.m;
    tilingData->n = tilingParam.n;
    tilingData->groupNum = tilingParam.groupNum;
    tilingData->rowBlockSize = tilingParam.rowBlockSize;
    tilingData->colBlockSize = tilingParam.colBlockSize;
    tilingData->colBlocksNum = tilingParam.colBlocksNum;
    tilingData->scale1ColPairs = tilingParam.scale1ColPairs;
    tilingData->scaleAlg = tilingParam.scaleAlg;
    tilingData->dstDtype = tilingParam.dstDtype;
    tilingData->inputDtype = tilingParam.inputDtype;
}

static void PrintTilingParam(gert::TilingContext* context,
                             const GroupedDynamicMxQuantWithDualAxisTilingParam& tilingParam, int64_t tilingKey)
{
    OP_LOGI(context->GetNodeName(),
            "GroupedDynamicMxQuantWithDualAxis tiling: m=%ld, n=%ld, groupNum=%ld, "
            "rowBlockSize=%ld, colBlockSize=%ld, colBlocksNum=%ld, scale1ColPairs=%ld, "
            "inputDtype=%ld, dstDtype=%ld, scaleAlg=%ld, totalCoreNum=%ld, usedCoreNum=%ld, tilingKey=%ld",
            tilingParam.m, tilingParam.n, tilingParam.groupNum, tilingParam.rowBlockSize, tilingParam.colBlockSize,
            tilingParam.colBlocksNum, tilingParam.scale1ColPairs, tilingParam.inputDtype, tilingParam.dstDtype,
            tilingParam.scaleAlg, tilingParam.totalCoreNum, tilingParam.usedCoreNum, tilingKey);
}
} // namespace

static ge::graphStatus Tiling4GroupedDynamicMxQuantWithDualAxis(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GroupedDynamicMxQuantWithDualAxisTiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Tiling4GroupedDynamicMxQuantWithDualAxis running begin.");
    GroupedDynamicMxQuantWithDualAxisTilingParam tilingParam;
    OP_CHECK_IF(GetPlatformInfo(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShape(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckInputShape failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtype(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckDtype failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAttrs(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckAttrs failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputShape(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "CheckOutputShape failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeTilingParam(tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ComputeTilingParam failed."),
                return ge::GRAPH_FAILED);

    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    OP_CHECK_IF(ubBlockSize <= 0, OP_LOGE(context, "Failed to get UB block size, got %ld.", ubBlockSize),
                return ge::GRAPH_FAILED);
    int64_t allNeedUb = CalculateBasicBlockUbSize(GetXTypeSize(static_cast<ge::DataType>(tilingParam.inputDtype)),
                                                  ubBlockSize);
    OP_CHECK_IF((allNeedUb > tilingParam.ubSize),
                OP_LOGE(context, "Basic split block (%ld, %ld) cannot fit UB, allNeedUb=%ld, ubSize=%ld.",
                        ROW_BLOCK_SIZE, COL_BLOCK_SIZE, allNeedUb, tilingParam.ubSize),
                return ge::GRAPH_FAILED);
    auto* tilingData = context->GetTilingData<GroupedDynamicMxQuantWithDualAxisTilingData>();
    OP_CHECK_IF(tilingData == nullptr, OP_LOGE(context, "Get tiling data pointer failed."), return ge::GRAPH_FAILED);
    SetTilingData(tilingParam, tilingData);

    using namespace GroupedDynamicMxQuantWithDualAxisOp;
    int64_t tilingKey = GET_TPL_TILING_KEY(TPL_RINT, TPL_SCALE_ALG_1);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(tilingData->usedCoreNum);
    PrintTilingParam(context, tilingParam, tilingKey);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(tilingParam.workspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GroupedDynamicMxQuantWithDualAxis(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GroupedDynamicMxQuantWithDualAxisTiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "TilingPrepare4GroupedDynamicMxQuantWithDualAxis entering.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedDynamicMxQuantWithDualAxis)
    .Tiling(Tiling4GroupedDynamicMxQuantWithDualAxis)
    .TilingParse<GroupedDynamicMxQuantWithDualAxisCompileInfo>(TilingPrepare4GroupedDynamicMxQuantWithDualAxis);
} // namespace optiling
