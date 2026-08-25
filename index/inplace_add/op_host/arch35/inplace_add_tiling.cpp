/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_add_tiling.cpp
 * \brief inplace_add_tiling
 */
#include "inplace_add_tiling.h"
#include <algorithm>
#include <limits>
#include <string>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/inplace_add_tiling_data.h"

namespace optiling {
namespace {
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
// 调度模式 1 = batch mode，所有核同时启动，kernel 里的 SyncAll 才有意义。
constexpr uint32_t BATCH_MODE = 1;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;
constexpr int64_t MIN_ELEMENTS_PER_CORE = 1024;
constexpr size_t MIN_DATA_RANK = 1;
constexpr size_t MAX_DATA_RANK = 8;
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t INPUT_INDICES_IDX = 1;
constexpr size_t INPUT_V_IDX = 2;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr int64_t INT32_MAX_VALUE = std::numeric_limits<int32_t>::max();
constexpr int64_t INT64_MAX_VALUE = std::numeric_limits<int64_t>::max();

static bool IsInplaceAddDtypeSupported(ge::DataType dtype)
{
    return dtype == ge::DT_COMPLEX64 || dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_BF16 ||
           dtype == ge::DT_INT8 || dtype == ge::DT_INT16 || dtype == ge::DT_INT32 || dtype == ge::DT_INT64 ||
           dtype == ge::DT_UINT8 || dtype == ge::DT_UINT16 || dtype == ge::DT_UINT32 || dtype == ge::DT_UINT64 ||
           dtype == ge::DT_COMPLEX32;
}

static bool IsComplexDtype(ge::DataType dtype) { return dtype == ge::DT_COMPLEX32 || dtype == ge::DT_COMPLEX64; }

static int64_t GetDataTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return 1;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
        case ge::DT_COMPLEX32:
            return 4;
        case ge::DT_INT64:
        case ge::DT_UINT64:
        case ge::DT_COMPLEX64:
            return 8;
        default:
            return 0;
    }
}

static bool MulOverflow(int64_t lhs, int64_t rhs)
{
    return lhs < 0 || rhs < 0 || (lhs != 0 && rhs > INT64_MAX_VALUE / lhs);
}

static ge::graphStatus CheckInputShape(gert::TilingContext* context, const gert::Shape& xShape,
                                       const gert::Shape& indicesShape, const gert::Shape& vShape,
                                       const gert::Shape& yShape)
{
    OP_CHECK_IF(indicesShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "indices",
                                             std::to_string(indicesShape.GetDimNum()).c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDimNum() < MIN_DATA_RANK || xShape.GetDimNum() > MAX_DATA_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xShape.GetDimNum()).c_str(),
                                             "1D to 8D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vShape.GetDimNum() != xShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "x, v",
                    (std::to_string(xShape.GetDimNum()) + ", " + std::to_string(vShape.GetDimNum())).c_str(),
                    "v rank must equal x rank"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yShape.GetDimNum() != xShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "x, y",
                    (std::to_string(xShape.GetDimNum()) + ", " + std::to_string(yShape.GetDimNum())).c_str(),
                    "y rank must equal x rank"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        vShape.GetDim(0) != indicesShape.GetDim(0),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "indices, v", "mismatched first dimension",
                                               "v.shape[0] must equal indices length"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDim(0) == 0 && indicesShape.GetDim(0) != 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, indices",
                                                       "x.shape[0] is 0 while indices is non-empty",
                                                       "indices must be empty when x.shape[0] is 0"),
                return ge::GRAPH_FAILED);
    for (size_t i = 1; i < xShape.GetDimNum(); ++i) {
        OP_CHECK_IF(vShape.GetDim(i) != xShape.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, v", "mismatched tail dimensions",
                                                           "v.shape[1:] must equal x.shape[1:]"),
                    return ge::GRAPH_FAILED);
    }
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        OP_CHECK_IF(yShape.GetDim(i) != xShape.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y", "mismatched dimensions",
                                                           "y shape must equal x shape"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDtype(gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(INPUT_X_IDX);
    auto indicesDesc = context->GetInputDesc(INPUT_INDICES_IDX);
    auto vDesc = context->GetInputDesc(INPUT_V_IDX);
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_IF(
        indicesDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "indices",
                                  std::to_string(static_cast<int32_t>(indicesDesc->GetDataType())).c_str(), "int32"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xDesc->GetDataType() != vDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, v",
                                               (std::to_string(static_cast<int32_t>(xDesc->GetDataType())) + ", " +
                                                std::to_string(static_cast<int32_t>(vDesc->GetDataType())))
                                                   .c_str(),
                                               "x dtype must equal v dtype"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsInplaceAddDtypeSupported(xDesc->GetDataType()),
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x",
                                          std::to_string(static_cast<int32_t>(xDesc->GetDataType())).c_str(),
                                          "complex64, float16, float32, bfloat16, int8, int16, int32, int64, uint8, "
                                          "uint16, uint32, uint64, complex32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xDesc->GetDataType() != yDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, y",
                                               (std::to_string(static_cast<int32_t>(xDesc->GetDataType())) + ", " +
                                                std::to_string(static_cast<int32_t>(yDesc->GetDataType())))
                                                   .c_str(),
                                               "y dtype must equal x dtype"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTilingDimRange(gert::TilingContext* context, const gert::Shape& xShape,
                                           const gert::Shape& indicesShape)
{
    OP_CHECK_IF(xShape.GetDim(0) < 0 || xShape.GetDim(0) > INT32_MAX_VALUE || indicesShape.GetDim(0) < 0 ||
                    indicesShape.GetDim(0) > INT32_MAX_VALUE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "x.shape[0], indices.shape[0]",
                    (std::to_string(xShape.GetDim(0)) + ", " + std::to_string(indicesShape.GetDim(0))).c_str(),
                    "shape dimensions must be non-negative and must not exceed int32 range"),
                return ge::GRAPH_FAILED);

    for (size_t i = 1; i < xShape.GetDimNum(); ++i) {
        OP_CHECK_IF(xShape.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x.shape tail dimension",
                                                          std::to_string(xShape.GetDim(i)).c_str(),
                                                          "shape dimensions must be non-negative"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcInnerSize(gert::TilingContext* context, const gert::Shape& xShape, int64_t& innerSize)
{
    for (size_t i = 1; i < xShape.GetDimNum(); ++i) {
        if (xShape.GetDim(i) == 0) {
            innerSize = 0;
            return ge::GRAPH_SUCCESS;
        }
    }

    innerSize = 1;
    for (size_t i = 1; i < xShape.GetDimNum(); ++i) {
        int64_t dim = xShape.GetDim(i);
        OP_CHECK_IF(innerSize > INT64_MAX_VALUE / dim,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x.shape tail dimensions", "overflow",
                                                          "innerSize must not exceed int64 range"),
                    return ge::GRAPH_FAILED);
        innerSize *= dim;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckElementCountRange(gert::TilingContext* context, const gert::Shape& xShape,
                                              const gert::Shape& indicesShape, ge::DataType dtype, int64_t innerSize)
{
    int64_t valueFactor = IsComplexDtype(dtype) ? INPLACE_ADD_COMPLEX_COMPONENT_COUNT :
                                                  INPLACE_ADD_SCALAR_COMPONENT_COUNT;
    OP_CHECK_IF(MulOverflow(innerSize, valueFactor),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x.shape tail dimensions", "overflow",
                                                      "row element count must not exceed int64 range"),
                return ge::GRAPH_FAILED);

    int64_t rowSize = innerSize * valueFactor;
    int64_t n = xShape.GetDim(0);
    int64_t k = indicesShape.GetDim(0);
    OP_CHECK_IF(MulOverflow(n, rowSize) || MulOverflow(k, rowSize),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x and v element count", "overflow",
                                                      "tensor element count must not exceed int64 range"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(n > INT64_MAX_VALUE - k || MulOverflow(n + k, innerSize),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "total work", "overflow",
                                                      "(N + K) * innerSize must not exceed int64 range"),
                return ge::GRAPH_FAILED);

    int64_t dtypeSize = GetDataTypeSize(dtype);
    OP_CHECK_IF(dtypeSize <= 0 || MulOverflow(n * innerSize, dtypeSize) || MulOverflow(k * innerSize, dtypeSize) ||
                    MulOverflow(k, static_cast<int64_t>(sizeof(int32_t))),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "GM byte offset", "overflow",
                                                      "tensor byte offset must not exceed int64 range"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetInplaceAddTilingData(gert::TilingContext* context, int32_t n, int32_t k, int64_t innerSize,
                                               int64_t totalWork, int64_t coreNum, InplaceAddTilingData& tiling)
{
    int64_t needCoreNum = 1;
    if (totalWork > 0) {
        int64_t perCoreWork = std::max(Ops::Base::CeilDiv(totalWork, coreNum), MIN_ELEMENTS_PER_CORE);
        needCoreNum = Ops::Base::CeilDiv(totalWork, perCoreWork);
    }
    OP_CHECK_IF(needCoreNum <= 0 || needCoreNum > coreNum || needCoreNum > INT32_MAX_VALUE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "needCoreNum",
                                                      std::to_string(needCoreNum).c_str(),
                                                      "needCoreNum must fit in int32 and not exceed coreNum"),
                return ge::GRAPH_FAILED);

    tiling.needCoreNum = static_cast<int32_t>(needCoreNum);
    tiling.n = n;
    tiling.k = k;
    tiling.innerSize = innerSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BuildTilingData(gert::TilingContext* context, int32_t n, int32_t k, int64_t innerSize,
                                       int64_t totalWork, int64_t coreNum)
{
    auto tiling = context->GetTilingData<InplaceAddTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    if (SetInplaceAddTilingData(context, n, k, innerSize, totalWork, coreNum, *tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(context->SetBlockDim(tiling->needCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to set the block dimension."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetTilingKey(0) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to set the tiling key."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetLocalMemoryAndWorkspace(gert::TilingContext* context,
                                                  const platform_ascendc::PlatformAscendC& ascendcPlatform,
                                                  const InplaceAddCompileInfo* compileInfo)
{
    uint64_t ubSize = 0;
    if (compileInfo != nullptr) {
        OP_CHECK_IF(compileInfo->ubSize <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize",
                                                          std::to_string(compileInfo->ubSize).c_str(),
                                                          "ubSize must be greater than 0"),
                    return ge::GRAPH_FAILED);
        ubSize = static_cast<uint64_t>(compileInfo->ubSize);
    } else {
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }
    OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                                      "ubSize must be greater than reserved local memory"),
                return ge::GRAPH_FAILED);
    uint64_t localMemorySize = ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE;
    OP_CHECK_IF(localMemorySize > std::numeric_limits<uint32_t>::max(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "localMemorySize",
                                                      std::to_string(localMemorySize).c_str(),
                                                      "localMemorySize must fit in uint32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetLocalMemorySize(static_cast<uint32_t>(localMemorySize)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to set the local memory size."), return ge::GRAPH_FAILED);
    auto workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    // kernel 的两个相位之间用 SyncAll 做跨核屏障，必须设为 batch mode 让所有核同时启动；
    // 不设的话核不共驻，SyncAll 直接返回，累加相会读到还没被拷贝相写过的 y。
    OP_CHECK_IF(context->SetScheduleMode(BATCH_MODE) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to set the schedule mode."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 取回四个 Shape 并做形状/维度范围校验，只把后续 tiling 真正要用的 xShape、indicesShape 带出去。
static ge::graphStatus PrepareShapes(gert::TilingContext* context, gert::Shape& xShape, gert::Shape& indicesShape)
{
    auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    auto indicesShapePtr = context->GetInputShape(INPUT_INDICES_IDX);
    auto vShapePtr = context->GetInputShape(INPUT_V_IDX);
    auto yShapePtr = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);

    xShape = xShapePtr->GetStorageShape();
    indicesShape = indicesShapePtr->GetStorageShape();
    auto vShape = vShapePtr->GetStorageShape();
    auto yShape = yShapePtr->GetStorageShape();
    if (CheckInputShape(context, xShape, indicesShape, vShape, yShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckTilingDimRange(context, xShape, indicesShape);
}
} // namespace

static ge::graphStatus TilingPrepare4InplaceAdd(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<InplaceAddCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4InplaceAdd(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "InplaceAdd tiling starts.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto compileInfo = context->GetCompileInfo<InplaceAddCompileInfo>();
    int64_t coreNum = compileInfo == nullptr ? ascendcPlatform.GetCoreNumAiv() : compileInfo->coreNum;
    OP_CHECK_IF(
        coreNum <= 0 || coreNum > INT32_MAX_VALUE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum", std::to_string(coreNum).c_str(),
                                              "coreNum must be greater than 0 and fit in int32"),
        return ge::GRAPH_FAILED);

    if (CheckInputDtype(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape xShape;
    gert::Shape indicesShape;
    if (PrepareShapes(context, xShape, indicesShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int32_t n = static_cast<int32_t>(xShape.GetDim(0));
    int32_t k = static_cast<int32_t>(indicesShape.GetDim(0));
    int64_t innerSize = 0;
    if (CalcInnerSize(context, xShape, innerSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    if (CheckElementCountRange(context, xShape, indicesShape, xDesc->GetDataType(), innerSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t totalWork = (static_cast<int64_t>(n) + static_cast<int64_t>(k)) * innerSize;

    if (BuildTilingData(context, n, k, innerSize, totalWork, coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return SetLocalMemoryAndWorkspace(context, ascendcPlatform, compileInfo);
}

IMPL_OP_OPTILING(InplaceAdd).Tiling(Tiling4InplaceAdd).TilingParse<InplaceAddCompileInfo>(TilingPrepare4InplaceAdd);
} // namespace optiling
