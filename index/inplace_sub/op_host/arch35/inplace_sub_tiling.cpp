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
 * \file inplace_sub_tiling.cpp
 * \brief inplace_sub_tiling
 */
#include "inplace_sub_tiling.h"
#include <algorithm>
#include <limits>
#include <securec.h>
#include <string>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/inplace_sub_tiling_data.h"

namespace optiling {
namespace {
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;
constexpr int64_t MIN_ELEMENTS_PER_CORE = 1024;
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t INPUT_INDICES_IDX = 1;
constexpr size_t INPUT_V_IDX = 2;
constexpr int64_t INT32_MAX_VALUE = std::numeric_limits<int32_t>::max();
constexpr int64_t INT64_MAX_VALUE = std::numeric_limits<int64_t>::max();

static bool IsInplaceSubDtypeSupported(ge::DataType dtype)
{
    return dtype == ge::DT_COMPLEX64 || dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_BF16 ||
           dtype == ge::DT_INT8 || dtype == ge::DT_INT16 || dtype == ge::DT_INT32 || dtype == ge::DT_INT64 ||
           dtype == ge::DT_UINT8 || dtype == ge::DT_UINT16 || dtype == ge::DT_UINT32 || dtype == ge::DT_UINT64 ||
           dtype == ge::DT_COMPLEX32;
}

static bool IsComplexDtype(ge::DataType dtype) { return dtype == ge::DT_COMPLEX32 || dtype == ge::DT_COMPLEX64; }

static ge::graphStatus CheckInputShape(gert::TilingContext* context, const gert::Shape& xShape,
                                       const gert::Shape& indicesShape, const gert::Shape& vShape)
{
    OP_CHECK_IF(indicesShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "indices",
                                             std::to_string(indicesShape.GetDimNum()).c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDimNum() == 0,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", "0D", "greater than 0D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vShape.GetDimNum() != xShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "x, v",
                    (std::to_string(xShape.GetDimNum()) + ", " + std::to_string(vShape.GetDimNum())).c_str(),
                    "v rank must equal x rank"),
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
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDtype(gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(INPUT_X_IDX);
    auto indicesDesc = context->GetInputDesc(INPUT_INDICES_IDX);
    auto vDesc = context->GetInputDesc(INPUT_V_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vDesc);
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
    OP_CHECK_IF(!IsInplaceSubDtypeSupported(xDesc->GetDataType()),
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x",
                                          std::to_string(static_cast<int32_t>(xDesc->GetDataType())).c_str(),
                                          "complex64, float16, float32, bfloat16, int8, int16, int32, int64, uint8, "
                                          "uint16, uint32, uint64, complex32"),
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
    innerSize = 1;
    for (size_t i = 1; i < xShape.GetDimNum(); ++i) {
        int64_t dim = xShape.GetDim(i);
        OP_CHECK_IF(dim != 0 && innerSize > INT64_MAX_VALUE / dim,
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
    int64_t valueFactor = IsComplexDtype(dtype) ? 2 : 1;
    OP_CHECK_IF(innerSize != 0 && innerSize > INT64_MAX_VALUE / valueFactor,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x.shape tail dimensions", "overflow",
                                                      "row element count must not exceed int64 range"),
                return ge::GRAPH_FAILED);

    int64_t rowSize = innerSize * valueFactor;
    int64_t maxRows = std::max(xShape.GetDim(0), indicesShape.GetDim(0));
    OP_CHECK_IF(rowSize != 0 && maxRows > INT64_MAX_VALUE / rowSize,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x and v element count", "overflow",
                                                      "tensor element count must not exceed int64 range"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetInplaceSubTilingData(InplaceSubTilingData* tiling, int32_t n, int32_t k, int64_t innerSize,
                                    int64_t coreNum)
{
    int32_t perCoreN = 0;
    int32_t needCoreNum = 1;
    int64_t totalWork = (static_cast<int64_t>(n) + static_cast<int64_t>(k)) * innerSize;
    if (totalWork > 0) {
        int64_t perCoreWork = std::max(Ops::Base::CeilDiv(totalWork, coreNum), MIN_ELEMENTS_PER_CORE);
        needCoreNum = static_cast<int32_t>(Ops::Base::CeilDiv(totalWork, perCoreWork));
    }
    if (n > 0) {
        perCoreN = static_cast<int32_t>(Ops::Base::CeilDiv(static_cast<int64_t>(n), static_cast<int64_t>(needCoreNum)));
    }
    tiling->needCoreNum = needCoreNum;
    tiling->n = n;
    tiling->k = k;
    tiling->innerSize = innerSize;
    tiling->perCoreN = perCoreN;
}

static ge::graphStatus BuildTilingData(gert::TilingContext* context, int32_t n, int32_t k, int64_t innerSize,
                                       int64_t coreNum)
{
    auto tiling = context->GetTilingData<InplaceSubTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    auto ret = memset_s(tiling, sizeof(InplaceSubTilingData), 0, sizeof(InplaceSubTilingData));
    OP_CHECK_IF(
        ret != EOK,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "memset_s return", std::to_string(ret).c_str(),
                                              "memset_s tiling data should succeed"),
        return ge::GRAPH_FAILED);
    SetInplaceSubTilingData(tiling, n, k, innerSize, coreNum);
    context->SetBlockDim(tiling->needCoreNum);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetLocalMemoryAndWorkspace(gert::TilingContext* context,
                                                  platform_ascendc::PlatformAscendC& ascendcPlatform,
                                                  const InplaceSubCompileInfo* compileInfo)
{
    uint64_t ubSize = 0;
    if (compileInfo != nullptr) {
        ubSize = static_cast<uint64_t>(compileInfo->ub_size);
    } else {
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }
    OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                                      "ubSize must be greater than reserved local memory"),
                return ge::GRAPH_FAILED);
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    auto workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus TilingPrepare4InplaceSub(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<InplaceSubCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ub_size = static_cast<int64_t>(ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4InplaceSub(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "InplaceSubTiling running begin.");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto compileInfo = reinterpret_cast<const InplaceSubCompileInfo*>(context->GetCompileInfo());
    int64_t coreNum = compileInfo == nullptr ? ascendcPlatform.GetCoreNumAiv() : compileInfo->core_num;
    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum", std::to_string(coreNum).c_str(),
                                              "coreNum must be greater than 0"),
        return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    auto indicesShapePtr = context->GetInputShape(INPUT_INDICES_IDX);
    auto vShapePtr = context->GetInputShape(INPUT_V_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShapePtr);

    auto xShape = xShapePtr->GetStorageShape();
    auto indicesShape = indicesShapePtr->GetStorageShape();
    auto vShape = vShapePtr->GetStorageShape();
    if (CheckInputShape(context, xShape, indicesShape, vShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputDtype(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckTilingDimRange(context, xShape, indicesShape) != ge::GRAPH_SUCCESS) {
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

    if (BuildTilingData(context, n, k, innerSize, coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return SetLocalMemoryAndWorkspace(context, ascendcPlatform, compileInfo);
}

IMPL_OP_OPTILING(InplaceSub).Tiling(Tiling4InplaceSub).TilingParse<InplaceSubCompileInfo>(TilingPrepare4InplaceSub);
} // namespace optiling
