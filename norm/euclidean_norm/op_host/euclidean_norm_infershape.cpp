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
 * \file euclidean_norm_infershape.cpp
 * \brief EuclideanNorm 算子的 output shape / dtype 推理。
 *
 *   - axes 是 const tensor（IndexNumberType：int32 / int64）；运行时未知 → 输出标 unknown rank
 *   - axes==[]（empty）→ 视为 full reduce，等价 axes=[0..rank-1]（对标 TF / PyTorch；与 NumPy axis=() identity 不同）
 *   - 越界 / 重复 axes → GRAPH_FAILED（与 tiling 端 ParseAxesTensor 同步报错；
 *     重复 axes 在 keep_dims=true 下会让 shape 幂等地正确，但 tiling 仍会拒绝，
 *     infershape 端早拒绝免得后端做无用功）
 *   - keep_dims 是 OPTIONAL attr，默认 false（与 op_def `Attr("keep_dims").AttrType(OPTIONAL).Bool(false)` 对齐）；
 *     GetAttrPointer 返回 nullptr 时按默认值兜底
 *   - 输出 shape 通过 Ops::Base::ReduceDimsWith{Keep,Without}KeepDims helper 推导
 *   - 输入是 unknown rank → 输出标 unknown rank（不再继续推导）
 */

#include <set>
#include <vector>

#include "op_host/infershape_reduce_util.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "log/log.h"

using namespace ge;

namespace ops {

namespace {
constexpr size_t kInputXIdx = 0;
constexpr size_t kInputAxesIdx = 1;
constexpr size_t kOutputYIdx = 0;
constexpr size_t kAttrKeepDimsIdx = 0;
constexpr bool kKeepDimsDefault = false; // op_def: Attr("keep_dims").AttrType(OPTIONAL).Bool(false)
} // namespace

// 解析 axes tensor 内容到 int64 数组：
//   axesNum == 0 时按 "all reduce" 展开 axes = [0..xRank-1]（与 tiling::ParseAxesTensor 一致）；
//   否则按 int32 / int64 dtype 读，归一化负数到 [0, xRank)，重复/越界报错（GRAPH_FAILED）。
static bool LoadAxesFromTensor(
    const gert::Tensor* axesTensor, int64_t xRank, std::vector<int64_t>& axesOut, std::string& errMsg)
{
    const int64_t axesNum = axesTensor->GetShapeSize();
    if (axesNum == 0) {
        axesOut.resize(static_cast<size_t>(xRank));
        for (int64_t i = 0; i < xRank; ++i) {
            axesOut[static_cast<size_t>(i)] = i;
        }
        return true;
    }

    std::set<int64_t> seen;
    auto pushOne = [&](int64_t v, int64_t idx) -> bool {
        if (v < -xRank || v >= xRank) {
            errMsg = "axes[" + std::to_string(idx) + "]=" + std::to_string(v) + " out of range [-" +
                     std::to_string(xRank) + ", " + std::to_string(xRank) + ")";
            return false;
        }
        const int64_t norm = (v < 0) ? (v + xRank) : v;
        if (!seen.insert(norm).second) {
            errMsg = "duplicate axis " + std::to_string(norm) + " in axes (input idx=" + std::to_string(idx) + ")";
            return false;
        }
        axesOut.push_back(norm);
        return true;
    };

    const ge::DataType dt = axesTensor->GetDataType();
    if (dt == ge::DT_INT32) {
        const int32_t* data = axesTensor->GetData<int32_t>();
        if (data == nullptr) {
            errMsg = "axes data ptr is null (int32)";
            return false;
        }
        axesOut.reserve(static_cast<size_t>(axesNum));
        for (int64_t i = 0; i < axesNum; ++i) {
            if (!pushOne(static_cast<int64_t>(data[i]), i)) {
                return false;
            }
        }
    } else if (dt == ge::DT_INT64) {
        const int64_t* data = axesTensor->GetData<int64_t>();
        if (data == nullptr) {
            errMsg = "axes data ptr is null (int64)";
            return false;
        }
        axesOut.reserve(static_cast<size_t>(axesNum));
        for (int64_t i = 0; i < axesNum; ++i) {
            if (!pushOne(data[i], i)) {
                return false;
            }
        }
    } else {
        errMsg = "axes dtype " + std::to_string(static_cast<int>(dt)) + " is not int32/int64";
        return false;
    }
    return true;
}

static ge::graphStatus HandleScalarInput(gert::InferShapeContext* context,
                                         const gert::Shape* xShape, gert::Shape* yShape)
{
    if (xShape->GetDimNum() != 0) {
        return GRAPH_SUCCESS;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* attrKeepDims = attrs->GetAttrPointer<bool>(kAttrKeepDimsIdx);
    const bool keepDims = (attrKeepDims == nullptr) ? kKeepDimsDefault : (*attrKeepDims);

    const gert::Tensor* axesTensor = context->GetInputTensor(kInputAxesIdx);
    if (axesTensor != nullptr && axesTensor->GetShapeSize() > 0) {
        OP_LOGE(context->GetNodeName(), "scalar input: axes must be empty, got %ld axes",
                axesTensor->GetShapeSize());
        return GRAPH_FAILED;
    }

    if (keepDims) {
        *yShape = gert::Shape({1});
    } else {
        *yShape = gert::Shape();
    }
    OP_LOGD(context->GetNodeName(), "scalar input: keepDims=%d, yShape dims=%zu",
            static_cast<int>(keepDims), yShape->GetDimNum());
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4EuclideanNorm(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShape4EuclideanNorm");

    const gert::Shape* xShape = context->GetInputShape(kInputXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(kOutputYIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 1) unknown rank 透传：输入未知 → 输出标未知（reduce_var_infershape 同型）
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        OP_LOGD(context->GetNodeName(), "x is unknown rank; set y as unknown rank");
        return GRAPH_SUCCESS;
    }
    const int64_t xRank = static_cast<int64_t>(xShape->GetDimNum());

    // 2) scalar (0-D) input
    if (xRank == 0) {
        return HandleScalarInput(context, xShape, yShape);
    }

    // 3) keep_dims 是 OPTIONAL attr：GetAttrPointer 可能返回 nullptr，按 op_def 默认值 false 兜底
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* attrKeepDims = attrs->GetAttrPointer<bool>(kAttrKeepDimsIdx);
    const bool keepDims = (attrKeepDims == nullptr) ? kKeepDimsDefault : (*attrKeepDims);

    // 3) axes 是 const tensor at compile time：运行时未知 → 输出标 unknown rank
    const gert::Tensor* axesTensor = context->GetInputTensor(kInputAxesIdx);
    if (axesTensor == nullptr) {
        Ops::Base::SetUnknownRank(*yShape);
        OP_LOGW(context->GetNodeName(), "axes tensor not const at infer time; set output as unknown rank");
        return GRAPH_SUCCESS;
    }

    // 4) axes 解析：empty → all reduce (axes=[0..xRank-1])；否则按 int32/int64 读 + 越界/重复检查
    std::vector<int64_t> axes;
    std::string errMsg;
    if (!LoadAxesFromTensor(axesTensor, xRank, axes, errMsg)) {
        OP_LOGE(
            context->GetNodeName(), "%s; xRank=%ld, axes size=%ld", errMsg.c_str(), xRank, axesTensor->GetShapeSize());
        return GRAPH_FAILED;
    }

    // 5) 调 opbase helper 计算输出 shape（已归一化、无重复，CheckAxisBounds 通过）
    const int32_t axesSize = static_cast<int32_t>(axes.size());
    ge::graphStatus stat = keepDims ?
                               Ops::Base::ReduceDimsWithKeepDims<int64_t>(xShape, axes.data(), axesSize, yShape) :
                               Ops::Base::ReduceDimsWithoutKeepDims<int64_t>(xShape, axes.data(), axesSize, yShape);

    OP_LOGD(
        context->GetNodeName(), "End InferShape: keepDims=%d, xRank=%ld, axes.size=%zu, outDim=%zu, status=%d",
        static_cast<int>(keepDims), xRank, axes.size(), yShape->GetDimNum(), static_cast<int>(stat));
    return stat;
}

static ge::graphStatus InferDataType4EuclideanNorm(gert::InferDataTypeContext* context)
{
    // 输出 dtype 与输入 x 相同（fp16 / bf16 / fp32 / int32 直通）
    const ge::DataType xDt = context->GetInputDataType(kInputXIdx);
    context->SetOutputDataType(kOutputYIdx, xDt);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EuclideanNorm)
    .InferShape(InferShape4EuclideanNorm)
    .InferDataType(InferDataType4EuclideanNorm)
    .InputsDataDependency({1}); // axes（索引 1）值依赖：infershape 阶段需读取张量内容

} // namespace ops
