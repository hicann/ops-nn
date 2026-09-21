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
 * \file mse_loss_grad_v2.cc
 * \brief
 */
#include <algorithm>
#include <cstring>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"
using namespace ge;
namespace {
constexpr uint32_t INPUT_PREDICT_IDX = 0;
constexpr uint32_t INPUT_LABEL_IDX = 1;
constexpr uint32_t INPUT_DOUT_IDX = 2;
constexpr uint32_t OUTPUT_LOSS_GRAD_IDX = 0;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;
constexpr size_t ATTR_REDUCTION_IDX = 0;

// ascend950(regbase) 才走 y = broadcast(dout, predict, label) 的推导与契约校验;
// 其余 SoC 走下方非 arch35 SoC 的基线逻辑(输出 = predict shape), 二者 soc 隔离。
bool IsRegBaseSoc()
{
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    return (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) ==
            ge::GRAPH_SUCCESS) &&
           platform_info.str_info.short_soc_version == "Ascend950";
}

bool IsLegalReduction(const char* reduction)
{
    return reduction == nullptr || std::strcmp(reduction, "none") == 0 || std::strcmp(reduction, "mean") == 0 ||
           std::strcmp(reduction, "sum") == 0;
}

// numpy 右对齐 broadcast。未知维(-1)按通配处理: 一方为 1 则取另一方; 两方相等取该值,
// 否则结果未知(-1); 两个既不等也非 1 的具体维冲突返回 false。
bool BroadcastShapes(const gert::Shape& a, const gert::Shape& b, gert::Shape& out)
{
    const size_t rankA = a.GetDimNum();
    const size_t rankB = b.GetDimNum();
    const size_t rank = std::max(rankA, rankB);
    out.SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t da = (i < rankA) ? a.GetDim(rankA - 1 - i) : 1;
        const int64_t db = (i < rankB) ? b.GetDim(rankB - 1 - i) : 1;
        int64_t dr = 0;
        if (da == 1) {
            dr = db;
        } else if (db == 1) {
            dr = da;
        } else if (da < 0 || db < 0) {
            dr = (da == db) ? da : -1;
        } else if (da == db) {
            dr = da;
        } else {
            return false;
        }
        out.SetDim(rank - 1 - i, dr);
    }
    return true;
}
} // namespace

namespace ops {
// ascend950(regbase) infershape: y.shape = numpy broadcast(dout, predict, label),
// 并校验 reduction 取值与调用方声明的输出 desc。
static ge::graphStatus InferShapeForMseLossGradRegbase(gert::InferShapeContext* context)
{
    const gert::Shape* predictShape = context->GetInputShape(INPUT_PREDICT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    const gert::Shape* labelShape = context->GetInputShape(INPUT_LABEL_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelShape);
    const gert::Shape* doutShape = context->GetInputShape(INPUT_DOUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, doutShape);

    gert::Shape* lossGradShape = context->GetOutputShape(OUTPUT_LOSS_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossGradShape);

    // 覆盖前快照调用方声明的输出 desc, 供具体 shape 冲突校验。
    const gert::Shape declaredShape = *lossGradShape;

    if (Ops::Base::IsUnknownRank(*predictShape)) {
        lossGradShape->SetDim(DIM_0, UNKNOWN_RANK_DIM_VALUE_);
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        if (!IsLegalReduction(attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX))) {
            return ge::GRAPH_FAILED;
        }
    }

    gert::Shape broadcastShape;
    if (!BroadcastShapes(*predictShape, *labelShape, broadcastShape) ||
        !BroadcastShapes(broadcastShape, *doutShape, broadcastShape)) {
        return ge::GRAPH_FAILED;
    }
    *lossGradShape = broadcastShape;

    const size_t declaredRank = declaredShape.GetDimNum();
    if (declaredRank > 0) {
        bool declaredAllConcrete = true;
        for (size_t i = 0; i < declaredRank; ++i) {
            if (declaredShape.GetDim(i) < 0) {
                declaredAllConcrete = false;
                break;
            }
        }
        if (declaredAllConcrete) {
            if (lossGradShape->GetDimNum() != declaredRank) {
                return ge::GRAPH_FAILED;
            }
            for (size_t i = 0; i < declaredRank; ++i) {
                const int64_t computed = lossGradShape->GetDim(i);
                const int64_t declared = declaredShape.GetDim(i);
                if (computed >= 0 && declared >= 0 && computed != declared) {
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForMseLossGrad(gert::InferShapeContext* context)
{
    // ascend950(regbase) 走 broadcast 分支; 非 arch35 SoC 走基线逻辑(下方逐字保留)。
    if (IsRegBaseSoc()) {
        return InferShapeForMseLossGradRegbase(context);
    }
    // input shape
    OP_LOGD(context->GetNodeName(), "MseLossGrad Begin.");
    const gert::Shape* predictShape = context->GetInputShape(INPUT_PREDICT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    const gert::Shape* labelShape = context->GetInputShape(INPUT_LABEL_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelShape);
    const gert::Shape* doutShape = context->GetInputShape(INPUT_DOUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, doutShape);

    // output shape
    gert::Shape* lossGradShape = context->GetOutputShape(OUTPUT_LOSS_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossGradShape);
    if (Ops::Base::IsUnknownRank(*predictShape)) {
        OP_LOGD(context->GetNodeName(), "Input shape is -2, set output shape to (-2,)");
        lossGradShape->SetDim(DIM_0, UNKNOWN_RANK_DIM_VALUE_);
        return ge::GRAPH_FAILED;
    }
    *lossGradShape = *predictShape;
    OP_LOGD(context->GetNodeName(), "InferShapeForMseLossGrad End.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMseLossGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType predictDtype = context->GetInputDataType(INPUT_PREDICT_IDX);
    context->SetOutputDataType(OUTPUT_LOSS_GRAD_IDX, predictDtype);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MseLossGradV2).InferShape(InferShapeForMseLossGrad).InferDataType(InferDataTypeForMseLossGrad);
} // namespace ops
