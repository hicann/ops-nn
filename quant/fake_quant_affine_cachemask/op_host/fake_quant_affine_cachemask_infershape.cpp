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
 * \file fake_quant_affine_cachemask_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_SCALE = 1;
static constexpr size_t INPUT_IDX_ZERO_POINT = 2;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr size_t OUTPUT_IDX_MASK = 1;
static constexpr int64_t MAX_RANK = 8;

static ge::graphStatus FakeQuantAffineCachemaskInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do FakeQuantAffineCachemaskInferShape");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    // scale/zero_point 形状在极简上下文（如 UT faker 未设置输入）可能为 nullptr，
    // 此时跳过静态校验仅传播输出形状（上游原行为），执行期由 tiling 层同口径拦截
    const gert::Shape* scaleShape = context->GetInputShape(INPUT_IDX_SCALE);
    const gert::Shape* zeroPointShape = context->GetInputShape(INPUT_IDX_ZERO_POINT);

    // 未知秩（-2 占位，DynamicRankSupportFlag(true) 声明）：跳过静态 axis/长度校验，
    // 输出形状原样传播（含未知维度），动态执行时按实际 shape 由 tiling 判定
    const bool xUnknownRank = (xShape->GetDimNum() == 1 && xShape->GetDim(0) == -2);

    if (!xUnknownRank) {
        // ND 格式仅支持 0-8 维
        OP_CHECK_IF(xShape->GetDimNum() > static_cast<size_t>(MAX_RANK),
                    OP_LOGE(context, "x rank %zu exceeds max rank %ld", xShape->GetDimNum(), MAX_RANK),
                    return ge::GRAPH_FAILED);
    }

    // attrs 完备时做静态校验（真实图模式 attrs 恒存在）；极简上下文（attrs 未设置，
    // 如 UT faker/部分图）跳过静态校验仅传播形状——与上游原行为一致，执行期由 tiling 同口径拦截
    auto attrs = context->GetAttrs();
    const int64_t* axisPtr = (attrs != nullptr) ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const int64_t* qMinPtr = (attrs != nullptr) ? attrs->GetAttrPointer<int64_t>(1) : nullptr;
    const int64_t* qMaxPtr = (attrs != nullptr) ? attrs->GetAttrPointer<int64_t>(2) : nullptr;
    if (axisPtr != nullptr && qMinPtr != nullptr && qMaxPtr != nullptr && scaleShape != nullptr &&
        zeroPointShape != nullptr) {
        // scale/zero_point 仅拒 >=2 维；0 维经 EnsureNotScalar 规整为 {1}（PT 标量语义）。
        // aclnn 侧：per-tensor 同此口径（0 维放行）；per-channel 维持上游 1 维原口径
        OP_CHECK_IF(scaleShape->GetDimNum() > 1, OP_LOGE(context, "scale should be a 0-D or 1-D tensor"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(zeroPointShape->GetDimNum() > 1, OP_LOGE(context, "zero_point should be a 0-D or 1-D tensor"),
                    return ge::GRAPH_FAILED);
        // quant 边界值域（与 arch35 tiling 同口径）：qmin<=qmax 且 |边界|<2^24
        OP_CHECK_IF(*qMinPtr > *qMaxPtr,
                    OP_LOGE(context, "quant_min %ld should be less than or equal to quant_max %ld", *qMinPtr, *qMaxPtr),
                    return ge::GRAPH_FAILED);
        constexpr int64_t QUANT_BOUND_LIMIT = 16777215; // 2^24 - 1
        OP_CHECK_IF(*qMinPtr < -QUANT_BOUND_LIMIT || *qMaxPtr > QUANT_BOUND_LIMIT,
                    OP_LOGE(context, "quant bounds out of representable range [-%ld, %ld]", QUANT_BOUND_LIMIT,
                            QUANT_BOUND_LIMIT),
                    return ge::GRAPH_FAILED);
        int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
        int64_t axis = *axisPtr;
        int64_t expectSize = 0;
        int64_t positiveAxis = 0;
        if (xUnknownRank) {
            // 未知秩：axis/长度静态校验整体跳过（含 x 维度未知 -1 的静态可比性丧失）
            expectSize = -1;
        } else if (rank == 0) {
            // 0 维（标量）输入：合法 axis 仅 {0, -1}，scale/zero_point 长度须为 1
            OP_CHECK_IF(axis != 0 && axis != -1, OP_LOGE(context, "axis %ld out of range [-1, 0] for 0-dim x", axis),
                        return ge::GRAPH_FAILED);
            expectSize = 1;
        } else {
            OP_CHECK_IF(axis < -rank || axis >= rank,
                        OP_LOGE(context, "axis %ld out of range [-%ld, %ld]", axis, rank, rank - 1),
                        return ge::GRAPH_FAILED);
            positiveAxis = (axis < 0) ? axis + rank : axis;
            expectSize = xShape->GetDim(positiveAxis);
        }
        if (expectSize >= 0) {
            int64_t scaleNum = scaleShape->GetShapeSize();
            int64_t zpNum = zeroPointShape->GetShapeSize();
            // PT（scaleNum/zpNum<=1 标量语义）豁免：单元素/0 维 scale/zero_point 按 per-tensor 解释，
            // 不要求与 axis 维长度一致（与 tiling_arch35 SelectMode 的 MODE_PT 判定同口径）
            OP_CHECK_IF(scaleNum > 1 && scaleNum != expectSize,
                        OP_LOGE(context, "scale size %ld mismatch with x dim %ld on axis %ld", scaleNum, expectSize,
                                positiveAxis),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(zpNum > 1 && zpNum != expectSize,
                        OP_LOGE(context, "zero_point size %ld mismatch with x dim %ld on axis %ld", zpNum, expectSize,
                                positiveAxis),
                        return ge::GRAPH_FAILED);
        }
    }

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* maskShape = context->GetOutputShape(OUTPUT_IDX_MASK);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShape);

    *yShape = *xShape;
    *maskShape = *xShape;

    OP_LOGI(context->GetNodeName(), "FakeQuantAffineCachemask output y shape: %s, mask shape: %s.",
            Ops::Base::ToString(*yShape).c_str(), Ops::Base::ToString(*maskShape).c_str());
    OP_LOGD(context, "End to do FakeQuantAffineCachemaskInferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus FakeQuantAffineCachemaskInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "FakeQuantAffineCachemaskInferDtype enter");
    auto inputDtype = context->GetInputDataType(INPUT_IDX_X);
    OP_LOGD(context, "x dtype: %s", Ops::Base::ToString(inputDtype).c_str());
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    OP_CHECK_IF(supportedDtype.count(inputDtype) == 0,
                OP_LOGE(context, "x dtype not supported: %s", Ops::Base::ToString(inputDtype).c_str()),
                return ge::GRAPH_FAILED);
    // scale 须与 x 同 dtype（ascend950 注册域每列 x/scale 同型）
    auto scaleDtype = context->GetInputDataType(INPUT_IDX_SCALE);
    OP_CHECK_IF(scaleDtype != inputDtype,
                OP_LOGE(context, "scale dtype must be same as x: scale=%s x=%s",
                        Ops::Base::ToString(scaleDtype).c_str(), Ops::Base::ToString(inputDtype).c_str()),
                return ge::GRAPH_FAILED);
    // zero_point 允许 INT32 或浮点集合（可与 x 交叉组合，对齐 proto 原型与 950 注册矩阵）
    auto zpDtype = context->GetInputDataType(INPUT_IDX_ZERO_POINT);
    OP_CHECK_IF(
        !(zpDtype == ge::DT_INT32 || zpDtype == ge::DT_FLOAT || zpDtype == ge::DT_FLOAT16 || zpDtype == ge::DT_BF16),
        OP_LOGE(context, "zero_point dtype must be INT32 or float family: zp=%s x=%s",
                Ops::Base::ToString(zpDtype).c_str(), Ops::Base::ToString(inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUTPUT_IDX_Y, inputDtype);
    context->SetOutputDataType(OUTPUT_IDX_MASK, DT_BOOL);
    OP_LOGD(context, "y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(OUTPUT_IDX_Y)).c_str());
    OP_LOGD(context, "mask dtype: %s", Ops::Base::ToString(context->GetOutputDataType(OUTPUT_IDX_MASK)).c_str());
    OP_LOGD(context, "FakeQuantAffineCachemaskInferDtype end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FakeQuantAffineCachemask)
    .InferShape(FakeQuantAffineCachemaskInferShape)
    .InferDataType(FakeQuantAffineCachemaskInferDtype);
} // namespace ops
