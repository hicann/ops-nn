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
 *   - axes==[]（empty）→ 无归约轴：y shape = x shape，逐元素 sqrt(x²)=|x|（对标 TF reduce_euclidean_norm
 *     的 axis=[] 行为：不做轴归约；含 0 维的 x 保持 0 元素输出，不得误写标量）
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
#include "../op_kernel/arch35/euclidean_norm_tiling_data.h" // MAX_PATTERN_RANK（与 tiling 门同上限）

using namespace ge;

namespace ops {

namespace {
constexpr size_t kInputXIdx = 0;
constexpr size_t kInputAxesIdx = 1;
constexpr size_t kOutputYIdx = 0;
constexpr size_t kAttrKeepDimsIdx = 0;
constexpr bool kKeepDimsDefault = false; // op_def: Attr("keep_dims").AttrType(OPTIONAL).Bool(false)
constexpr int32_t kMinAxisNum = 2;       // 与 tiling::kMinAxisNum 一致
} // namespace

// 解析 axes tensor 内容到 int64 数组：
//   axesNum == 0 时不归约（axesOut 为空，y shape = x shape，与 tiling::ParseAxesTensor 一致）；
//   否则按 int32 / int64 dtype 读，归一化负数到 [0, xRank)，重复/越界报错（GRAPH_FAILED）。
static bool LoadAxesFromTensor(const gert::Tensor* axesTensor, int64_t xRank, std::vector<int64_t>& axesOut,
                               std::string& errMsg)
{
    const int64_t axesNum = axesTensor->GetShapeSize();
    if (axesNum == 0) {
        // axes=[]（空张量）→ 无归约轴：y shape = x shape（ReduceDims* 对 axesSize=0 恒等），
        // 与 tiling::ParseAxesTensor 语义一致；禁止展开为全归约（防止 0 元素输出被写标量）。
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

static ge::graphStatus HandleScalarInput(gert::InferShapeContext* context, const gert::Shape* xShape,
                                         gert::Shape* yShape)
{
    if (xShape->GetDimNum() != 0) {
        return GRAPH_SUCCESS;
    }

    const gert::Tensor* axesTensor = context->GetInputTensor(kInputAxesIdx);
    if (axesTensor != nullptr && axesTensor->GetShapeSize() > 0) {
        OP_LOGE(context->GetNodeName(), "scalar input: axes must be empty, got %ld axes", axesTensor->GetShapeSize());
        return GRAPH_FAILED;
    }

    *yShape = gert::Shape();
    OP_LOGI(context->GetNodeName(), "EuclideanNorm output shape: %s.", Ops::Base::ToString(*yShape).c_str());
    return GRAPH_SUCCESS;
}

// FR-09 两门一致化：重放 tiling 侧 4 步 pattern 预处理（去 1 / 合轴 / 补 leading A / 补 R 增广，
// 与 euclidean_norm_tiling_arch35.cpp 的 DropSizeOneAxes/FuseAxis/PadLeadingOneA/PadRIfPureA
// 逐步同构），计算预处理后的 pattern 轴数。infershape 此前无该上限校验：rank-10 交替归约
// 被 tiling 门（MAX_PATTERN_RANK）拒绝而 infershape 放行——两门不一致；现收紧到同拒。
// C02：合轴乘法带溢出防护（shape 乘积溢出 int64 不可调度，GRAPH_FAILED），
//   与 tiling 侧 FuseAxis 同款保护、两门行为一致。
static ge::graphStatus CountPatternAxesAfterPreprocess(gert::InferShapeContext* context,
                                                       const std::vector<int64_t>& xDims,
                                                       const std::vector<int64_t>& axes, int64_t& patternAxisNum)
{
    std::vector<int64_t> shape = xDims;
    std::vector<bool> isR(xDims.size(), false);
    for (int64_t a : axes) {
        isR[static_cast<size_t>(a)] = true;
    }
    // 1) 去 1：size==1 的轴整根删除；全删后保留 1 根 A=1 占位
    {
        std::vector<int64_t> newShape;
        std::vector<bool> newIsR;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != 1) {
                newShape.push_back(shape[i]);
                newIsR.push_back(isR[i]);
            }
        }
        if (newShape.empty()) {
            newShape.push_back(1);
            newIsR.push_back(false);
        }
        shape = std::move(newShape);
        isR = std::move(newIsR);
    }
    // 2) 合轴：连续同类型合并为 1 根
    {
        std::vector<int64_t> fused;
        std::vector<bool> fusedIsR;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (!fused.empty() && fusedIsR.back() == isR[i]) {
                int64_t prod = 0;
                if (__builtin_mul_overflow(fused.back(), shape[i], &prod)) {
                    OP_LOGE(context->GetNodeName(), "fused axis shape multiply overflow at dim %zu: %ld * %ld", i,
                            fused.back(), shape[i]);
                    return ge::GRAPH_FAILED;
                }
                fused.back() = prod;
            } else {
                fused.push_back(shape[i]);
                fusedIsR.push_back(isR[i]);
            }
        }
        shape = std::move(fused);
        isR = std::move(fusedIsR);
    }
    // 3) 补 leading A：第 0 轴是 R → 前置一根 A=1
    if (!shape.empty() && isR.front()) {
        shape.insert(shape.begin(), 1);
        isR.insert(isR.begin(), false);
    }
    // 4) 补 R 增广：无 R 轴（纯 A 退化）——A=1 全退化末尾补 R=1；否则前置 [A=1, R=1]
    bool hasR = false;
    for (bool b : isR) {
        if (b) {
            hasR = true;
            break;
        }
    }
    if (!hasR) {
        if (shape.size() == 1 && shape[0] == 1) {
            shape.push_back(1);
            isR.push_back(true);
        } else {
            shape.insert(shape.begin(), {1, 1});
            isR.insert(isR.begin(), {false, true});
        }
    }
    patternAxisNum = static_cast<int64_t>(shape.size());
    return ge::GRAPH_SUCCESS;
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
        OP_LOGI(context->GetNodeName(), "EuclideanNorm output shape: %s.", Ops::Base::ToString(*yShape).c_str());
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
        OP_LOGI(context->GetNodeName(), "EuclideanNorm output shape: %s.", Ops::Base::ToString(*yShape).c_str());
        OP_LOGW(context->GetNodeName(), "axes tensor not const at infer time; set output as unknown rank");
        return GRAPH_SUCCESS;
    }

    // 4) axes 解析：empty → 无归约轴（y shape = x shape）；否则按 int32/int64 读 + 越界/重复检查
    std::vector<int64_t> axes;
    std::string errMsg;
    if (!LoadAxesFromTensor(axesTensor, xRank, axes, errMsg)) {
        OP_LOGE(context->GetNodeName(), "%s; xRank=%ld, axes size=%ld", errMsg.c_str(), xRank,
                axesTensor->GetShapeSize());
        return GRAPH_FAILED;
    }

    // 4b) FR-09 两门一致化：pattern 轴数上限与 tiling 门（MAX_PATTERN_RANK）一致，
    //     消除 rank-10 交替归约 infershape 放行 / tiling 拒绝的两门不一致（收紧方向；
    //     全归约/合轴后 ≤MAX_PATTERN_RANK 的任意 rank 输入不受影响）。
    //     C04（检视修复轮）：镜像 tiling 空 tensor 旁路——tiling 侧 ClassifyEmptyTensor
    //     非 NORMAL 时早退（HandleEmptyTensor）先于 PreprocessPattern 轴数门，空 tensor
    //     完全绕过轴数门；infershape 侧任一维为 0（空 tensor）或 -1（未知维，实际段数
    //     运行时定、由 tiling 运行时门兜底）时同样跳过轴数校验，仅所有维已知且 >0 时执行校验。
    std::vector<int64_t> xDims;
    xDims.reserve(xShape->GetDimNum());
    for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
        xDims.push_back(xShape->GetDim(i));
    }
    bool hasEmptyOrUnknownDim = false;
    for (int64_t d : xDims) {
        if (d == 0 || d == -1) {
            hasEmptyOrUnknownDim = true;
            break;
        }
    }
    if (hasEmptyOrUnknownDim) {
        OP_LOGI(context->GetNodeName(),
                "x has zero-sized dim (empty tensor, mirrors tiling EMPTY_A/EMPTY_R early exit) or unknown dim "
                "(-1); skip pattern axis-count gate");
    } else {
        int64_t patternAxisNum = 0;
        if (CountPatternAxesAfterPreprocess(context, xDims, axes, patternAxisNum) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED; // 合轴乘积溢出 int64（不可调度），已在 helper 内 OP_LOGE
        }
        if (patternAxisNum < kMinAxisNum || patternAxisNum > MAX_PATTERN_RANK) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), "x", Ops::Base::ToString(*xShape).c_str(),
                "the number of alternating reduce/non-reduce pattern segments after preprocessing exceeds the "
                "supported maximum (max 9)");
            return ge::GRAPH_FAILED;
        }
    }

    // 5) 调 opbase helper 计算输出 shape（已归一化、无重复，CheckAxisBounds 通过）
    const int32_t axesSize = static_cast<int32_t>(axes.size());
    ge::graphStatus stat = keepDims ?
                               Ops::Base::ReduceDimsWithKeepDims<int64_t>(xShape, axes.data(), axesSize, yShape) :
                               Ops::Base::ReduceDimsWithoutKeepDims<int64_t>(xShape, axes.data(), axesSize, yShape);

    if (stat == GRAPH_SUCCESS) {
        OP_LOGI(context->GetNodeName(), "EuclideanNorm output shape: %s.", Ops::Base::ToString(*yShape).c_str());
    }
    OP_LOGD(context->GetNodeName(), "End InferShape: keepDims=%d, xRank=%ld, axes.size=%zu, outDim=%zu, status=%d",
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
