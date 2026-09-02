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
 * \file gather_elements_def.cpp
 * \brief
 */
#include <algorithm>
#include "graph/operator.h"
#include "register/op_def_registry.h"

namespace ops {
namespace {
constexpr int64_t INT_MAX_NUM = 2147483647;
constexpr int64_t HALF = 2;
constexpr int64_t DOUBLE_UB = 2;
constexpr int64_t SELF_BOUND = 3000 * 1024;
constexpr int64_t INDEX_BOUND = 2000;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t LEAST_REPEAT_TIME = 1;
constexpr int64_t SMALL_UB_SIZE = 192 * 1024;
constexpr int64_t RESERVED_UB_SIZE = 2 * 1024;
constexpr int64_t ONE_BYTE = 1;
constexpr int64_t TWO_BYTE = 2;
constexpr int64_t FOUR_BYTE = 4;
constexpr int64_t EIGHT_BYTE = 8;

// 与 op_api/gather_elements.cpp 中 910b (非 RegBase) 的 AICORE_DTYPE_SUPPORT_LIST 保持一致
bool IsAicoreSupportedDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_INT32 || dtype == ge::DT_FLOAT16 || dtype == ge::DT_INT8 ||
           dtype == ge::DT_UINT8 || dtype == ge::DT_UINT32 || dtype == ge::DT_INT16 || dtype == ge::DT_UINT16 ||
           dtype == ge::DT_INT64 || dtype == ge::DT_UINT64 || dtype == ge::DT_BF16;
}

int64_t GetDtypeSize(ge::DataType dtype)
{
    if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
        return ONE_BYTE;
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_INT16 || dtype == ge::DT_UINT16 || dtype == ge::DT_BF16) {
        return TWO_BYTE;
    }
    if (dtype == ge::DT_FLOAT || dtype == ge::DT_INT32 || dtype == ge::DT_UINT32) {
        return FOUR_BYTE;
    }
    if (dtype == ge::DT_INT64 || dtype == ge::DT_UINT64) {
        return EIGHT_BYTE;
    }
    return 0;
}

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

int64_t GetTensorSize(const ge::Shape& shape)
{
    int64_t size = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.GetDimNum()); ++i) {
        size *= shape.GetDim(i);
    }
    return size;
}

bool IsDynamicShape(const ge::Shape& shape)
{
    for (int64_t i = 0; i < static_cast<int64_t>(shape.GetDimNum()); ++i) {
        if (shape.GetDim(i) < 0) {
            return true;
        }
    }
    return false;
}

bool IsSameDimValueExceptAxis(const ge::Shape& xShape, const ge::Shape& indexShape, int64_t axis)
{
    const int64_t dims = static_cast<int64_t>(xShape.GetDimNum());
    for (int64_t i = 0; i < dims; ++i) {
        if ((i != axis) && (xShape.GetDim(i) != indexShape.GetDim(i))) {
            return false;
        }
    }
    return true;
}

// 与 op_api/gather_elements.cpp 中 IsLastAxisSupport 保持一致的 UB 占用判断（910b 使用小 UB 192KB）
bool IsLastAxisSupport(const ge::Shape& xShape, const ge::Shape& indexShape, int64_t axis, ge::DataType xDtype,
                       ge::DataType indexDtype)
{
    const int64_t dims = static_cast<int64_t>(xShape.GetDimNum());
    const int64_t indexDsize = GetDtypeSize(indexDtype);
    const int64_t xDsize = GetDtypeSize(xDtype);
    if (std::min(xDsize, indexDsize) == 0) {
        return false;
    }
    const int64_t largeNumPerBlock = BLOCK_SIZE / std::min(xDsize, indexDsize);
    const int64_t indexAxis = indexShape.GetDim(axis);
    const int64_t xAxis = xShape.GetDim(axis);
    int64_t repeatPerCore = LEAST_REPEAT_TIME;
    const bool ifSameDimValueExceptAxis = IsSameDimValueExceptAxis(xShape, indexShape, axis);
    const bool isLastAxis = (axis == dims - 1);
    const int64_t availableUbSize = SMALL_UB_SIZE - RESERVED_UB_SIZE;
    int64_t allDataSize = 0;

    // vgather 分支（910b fp16/bf16）
    const bool vgather910BSupport = (xDtype == ge::DT_FLOAT16 || xDtype == ge::DT_BF16);
    if (isLastAxis && vgather910BSupport && ifSameDimValueExceptAxis) {
        allDataSize = xAxis * xDsize + indexAxis * (xDsize + indexDsize * DOUBLE_UB);
        if (allDataSize < availableUbSize) {
            return true;
        }
    }

    // 将 indices 切片的分支
    const int64_t lastDimSize = xAxis * xDsize + indexAxis * (xDsize + indexDsize);
    const bool cuttingIntoSlicesFlag = lastDimSize >= availableUbSize && ifSameDimValueExceptAxis &&
                                       xAxis * xDsize <= availableUbSize / HALF && indexAxis % largeNumPerBlock == 0;
    if (isLastAxis && cuttingIntoSlicesFlag) {
        return true;
    }

    // normal 分支
    if (isLastAxis && ifSameDimValueExceptAxis) {
        if (indexAxis % largeNumPerBlock != 0) {
            while (repeatPerCore * indexAxis % largeNumPerBlock != 0) {
                ++repeatPerCore;
            }
        }
    } else if (isLastAxis && indexAxis < largeNumPerBlock) {
        return false;
    } else if (!isLastAxis) {
        return false;
    }
    // 与 arch22/gather_elements_tiling.cpp ChooseTilingModeForLastAxis 中 lastAxisUbSize 的 UB 占用判断对齐
    const int64_t blockSizeX = BLOCK_SIZE / xDsize;
    const int64_t blockSizeIdx = BLOCK_SIZE / indexDsize;
    const int64_t blockSizeIdx32 = BLOCK_SIZE / FOUR_BYTE;
    const int64_t xAligned = CeilDiv(xAxis, blockSizeX) * blockSizeX;
    const int64_t idxAligned = CeilDiv(indexAxis, blockSizeIdx) * blockSizeIdx;
    const int64_t resAligned = CeilDiv(indexAxis, blockSizeX) * blockSizeX;
    const int64_t idx32Aligned = CeilDiv(indexAxis, blockSizeIdx32) * blockSizeIdx32;
    const int64_t lastAxisUbSize = repeatPerCore * (xAligned * xDsize + idxAligned * indexDsize + resAligned * xDsize +
                                                    idx32Aligned * 2 * FOUR_BYTE + indexAxis * FOUR_BYTE);
    if (lastAxisUbSize >= availableUbSize) {
        return false;
    }
    return true;
}

// 910b AICore 支持性检查：不支持的场景返回 False，使算子走 AICPU
ge::graphStatus CheckIfAICoreSupported(const ge::Operator& op, ge::AscendString& result)
{
    int64_t dim = 0;
    if (op.GetAttr("dim", dim) != ge::GRAPH_SUCCESS) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "GetAttr dim error."})");
        return ge::GRAPH_FAILED;
    }

    const ge::Shape xShape = op.GetInputDescByName("x").GetShape();
    const ge::Shape indexShape = op.GetInputDescByName("index").GetShape();
    const int64_t dims = static_cast<int64_t>(xShape.GetDimNum());
    if (dims == 0 || indexShape.GetDimNum() != xShape.GetDimNum()) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "x and index rank mismatch or rank is 0."})");
        return ge::GRAPH_FAILED;
    }

    // 动态 shape 交由运行时 tiling 决策，这里不拦截
    if (IsDynamicShape(xShape) || IsDynamicShape(indexShape)) {
        result = ge::AscendString(
            R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "Dynamic shape, defer to runtime tiling."})");
        return ge::GRAPH_SUCCESS;
    }

    const int64_t axis = (dim < 0) ? (dim + dims) : dim;
    if (axis < 0 || axis >= dims) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "dim out of range."})");
        return ge::GRAPH_FAILED;
    }

    const ge::DataType xDtype = op.GetInputDescByName("x").GetDataType();
    const ge::DataType indexDtype = op.GetInputDescByName("index").GetDataType();
    if (!IsAicoreSupportedDtype(xDtype) || GetDtypeSize(indexDtype) == 0) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "Unsupported dtype on aicore."})");
        return ge::GRAPH_FAILED;
    }

    const int64_t xDsize = GetDtypeSize(xDtype);
    const int64_t xSize = GetTensorSize(xShape);
    if (xDsize * xSize > INT_MAX_NUM) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "x is too large for aicore."})");
        return ge::GRAPH_FAILED;
    }
    if (xShape.GetDim(axis) > INT_MAX_NUM / HALF) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "shape range of x axis is too large for aicore."})");
        return ge::GRAPH_FAILED;
    }

    if (IsLastAxisSupport(xShape, indexShape, axis, xDtype, indexDtype)) {
        result = ge::AscendString(
            R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "AICore CheckSupport Passed."})");
        return ge::GRAPH_SUCCESS;
    }

    const int64_t indexSize = GetTensorSize(indexShape);
    if (xDsize * xSize > SELF_BOUND && indexSize > INDEX_BOUND) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "shape is too large for aicore."})");
        return ge::GRAPH_FAILED;
    }

    result = ge::AscendString(
        R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "AICore CheckSupport Passed."})");
    return ge::GRAPH_SUCCESS;
}
} // namespace

class GatherElements : public OpDef {
public:
    explicit GatherElements(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16,   ge::DT_FLOAT16, ge::DT_FLOAT,       ge::DT_UINT8,       ge::DT_INT8,
                       ge::DT_UINT16, ge::DT_INT16,   ge::DT_UINT32,      ge::DT_INT32,       ge::DT_UINT64,
                       ge::DT_INT64,  ge::DT_BOOL,    ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_BF16,   ge::DT_FLOAT16, ge::DT_FLOAT,       ge::DT_UINT8,       ge::DT_INT8,
                       ge::DT_UINT16, ge::DT_INT16,   ge::DT_UINT32,      ge::DT_INT32,       ge::DT_UINT64,
                       ge::DT_INT64,  ge::DT_BOOL,    ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16,   ge::DT_FLOAT16, ge::DT_FLOAT,       ge::DT_UINT8,       ge::DT_INT8,
                       ge::DT_UINT16, ge::DT_INT16,   ge::DT_UINT32,      ge::DT_INT32,       ge::DT_UINT64,
                       ge::DT_INT64,  ge::DT_BOOL,    ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_BF16,   ge::DT_FLOAT16, ge::DT_FLOAT,       ge::DT_UINT8,       ge::DT_INT8,
                       ge::DT_UINT16, ge::DT_INT16,   ge::DT_UINT32,      ge::DT_INT32,       ge::DT_UINT64,
                       ge::DT_INT64,  ge::DT_BOOL,    ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(0);
        this->AICore().SetCheckSupport(CheckIfAICoreSupported);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "gather_elements_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);

        OpAICoreConfig aicoreConfig910b;
        aicoreConfig910b.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(true)
            .ExtendCfgInfo("opInterface.value", "gather_elements")
            .ExtendCfgInfo("opFile.value", "gather_elements");
        aicoreConfig910b.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_UINT8,  ge::DT_INT8,   ge::DT_UINT16,
                       ge::DT_INT16,   ge::DT_UINT32,  ge::DT_INT32,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_UINT8,  ge::DT_INT8,   ge::DT_UINT16, ge::DT_INT16,
                       ge::DT_UINT32,  ge::DT_INT32,   ge::DT_UINT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicoreConfig910b.Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicoreConfig910b.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_UINT8,  ge::DT_INT8,   ge::DT_UINT16,
                       ge::DT_INT16,   ge::DT_UINT32,  ge::DT_INT32,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_UINT8,  ge::DT_INT8,   ge::DT_UINT16, ge::DT_INT16,
                       ge::DT_UINT32,  ge::DT_INT32,   ge::DT_UINT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().AddConfig("ascend910b", aicoreConfig910b);
        this->AICore().AddConfig("ascend910_93", aicoreConfig910b);
    }
};

OP_ADD(GatherElements);
// 手动注册opDef.AICore()里设置的CheckSupport函数
// 需要当前目录下的CMakeLists.txt将本_def.cpp加入${OPHOST_NAME}_tiling_obj编译目标内
static int GatherElements_REGISTERED = [](const char* name) {
    GatherElements opDef(name);
    optiling::OpCheckFuncHelper(FUNC_CHECK_SUPPORTED, name, opDef.AICore().GetCheckSupport());
    return 0;
}("GatherElements");

} // namespace ops
