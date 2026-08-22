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
 * \file bn3d_training_reduce_infershape.cpp
 * \brief BN3DTrainingReduce 的图构建期 Shape 推导。
 *
 * InferDataType 仅图场景使用，单列在 op_graph/bn3d_training_reduce_graph_infer.cpp。
 *
 * 通道轴位置与归约轴完全由 origin format 决定（不是 storage format，也不按 shape 数值猜测）：
 *   - NDHWC   : 仅 rank 5，C = dim4，输出 [C]
 *   - NCDHW   : rank 2~5，C = dim1，输出 [C]
 *   - NDC1HWC0: 仅 rank 6 [N,D,C1,H,W,C0]，输出 [1,1,C1,1,1,C0]
 *   - 其余（含 FORMAT_ND、FORMAT_NCHW）一律失败
 * 该格式集合与 canndev 的 runtime InferShape(ops/built-in/op_proto/runtime/bn_3d_training.cc)
 * 完全一致，不增不减。
 *
 * unknown rank 行为：
 *   - NDHWC / NCDHW  : 输出 [-1]（rank 1、dim 未知），与 canndev runtime 版本一致。
 *     注意 canndev 另有一套 legacy IR InferShape(reduce_ops.cc) 输出 unknown rank [-2] 且按
 *     storage format 分支；GE 实际执行走 runtime 版本，故此处以 runtime 版本为准。
 *   - NDC1HWC0       : 输出 [1,1,-1,1,1,-1]。canndev 的 NDC1HWC0 分支漏写 !is_unknown_rank
 *     短路（NDHWC / NCDHW 两个分支都有），导致 unknown rank 时 dimNum=1 必然失败，与 op_info 中
 *     dynamicRankSupport.flag=true 矛盾。此处在 Ascend950 上修正该缺陷，是相对 canndev 的唯一
 *     一处有意偏离，且不改动 canndev。
 */

#include <string>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t NDHWC_RANK = 5;
constexpr size_t NDC1HWC0_RANK = 6;
constexpr size_t NCDHW_MIN_RANK = 2;
constexpr size_t NCDHW_MAX_RANK = 5;

constexpr size_t NDHWC_C_IDX = 4;
constexpr size_t NCDHW_C_IDX = 1;
constexpr size_t NDC1HWC0_C1_IDX = 2;
constexpr size_t NDC1HWC0_C0_IDX = 5;

constexpr int64_t UNKNOWN_DIM = -1;

// 私有格式的输出形态 [1,1,C1,1,1,C0]：只有 C1、C0 两位携带信息，其余位恒 1。
inline void SetNDC1HWC0OutputShape(gert::Shape* shape, int64_t c1, int64_t c0)
{
    shape->SetDimNum(NDC1HWC0_RANK);
    shape->SetDim(0, 1);
    shape->SetDim(1, 1);
    shape->SetDim(NDC1HWC0_C1_IDX, c1);
    shape->SetDim(3, 1);
    shape->SetDim(4, 1);
    shape->SetDim(NDC1HWC0_C0_IDX, c0);
}

// dense 输出恒为通道向量 [C]。
inline void SetChannelVectorOutputShape(gert::Shape* shape, int64_t c)
{
    shape->SetDimNum(1);
    shape->SetDim(0, c);
}

// 按 origin format 推 sum 的 shape。origin format 唯一确定通道轴位置与允许的 rank；
// 不满足即失败，绝不按 shape 数值猜测通道轴。
static ge::graphStatus InferSumShapeByOriginFormat(gert::InferShapeContext* context, const gert::Shape* xShape,
                                                   ge::Format originFormat, gert::Shape* sumShape)
{
    const bool isUnknownRank = Ops::Base::IsUnknownRank(*xShape);
    const size_t xDimNum = xShape->GetDimNum();

    if (originFormat == FORMAT_NDHWC) {
        OP_CHECK_IF(!isUnknownRank && xDimNum != NDHWC_RANK,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                        "The shape dim of input x must be 5 when the origin format of x is NDHWC"),
                    return ge::GRAPH_FAILED);
        SetChannelVectorOutputShape(sumShape, isUnknownRank ? UNKNOWN_DIM : xShape->GetDim(NDHWC_C_IDX));
    } else if (originFormat == FORMAT_NCDHW) {
        OP_CHECK_IF(!isUnknownRank && (xDimNum < NCDHW_MIN_RANK || xDimNum > NCDHW_MAX_RANK),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                        "The shape dim of input x must be in the range of [2, 5] when the origin format of x is NCDHW"),
                    return ge::GRAPH_FAILED);
        SetChannelVectorOutputShape(sumShape, isUnknownRank ? UNKNOWN_DIM : xShape->GetDim(NCDHW_C_IDX));
    } else if (originFormat == FORMAT_NDC1HWC0) {
        // 与 canndev 的差异点：unknown rank 不再直接失败，而是输出 C1/C0 未知的私有形态。
        if (isUnknownRank) {
            SetNDC1HWC0OutputShape(sumShape, UNKNOWN_DIM, UNKNOWN_DIM);
        } else {
            OP_CHECK_IF(xDimNum != NDC1HWC0_RANK,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            context->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                            "The shape dim of input x must be 6 when the origin format of x is NDC1HWC0"),
                        return ge::GRAPH_FAILED);
            SetNDC1HWC0OutputShape(sumShape, xShape->GetDim(NDC1HWC0_C1_IDX), xShape->GetDim(NDC1HWC0_C0_IDX));
        }
    } else {
        // FORMAT_ND 不携带布局语义，无法确定 C 轴；FORMAT_NCHW 作为 origin 亦不受支持
        // （NCHW 只可能是 origin NCDHW rank 4 的 storage 形态）。二者均不得按 shape 数值猜测放行。
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x",
                                   ge::TypeUtils::FormatToSerialString(originFormat).c_str(),
                                   "NDHWC, NCDHW or NDC1HWC0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus InferShape4BN3DTrainingReduce(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4BN3DTrainingReduce");

    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);

    gert::Shape* sumShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumShape);
    gert::Shape* squareSumShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, squareSumShape);

    const ge::Format originFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetOriginFormat()));
    OP_CHECK_IF(InferSumShapeByOriginFormat(context, xShape, originFormat, sumShape) != ge::GRAPH_SUCCESS,
                OP_LOGD(context->GetNodeName(), "InferSumShapeByOriginFormat failed"), return ge::GRAPH_FAILED);

    // 两个输出的 shape 恒一致。
    *squareSumShape = *sumShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4BN3DTrainingReduce");
    return ge::GRAPH_SUCCESS;
}

// InferDataType 仅图场景使用，已按交付件划分挪到
// op_graph/bn3d_training_reduce_graph_infer.cpp；此处只挂图与单算子共用的 InferShape。
IMPL_OP_INFERSHAPE(BN3DTrainingReduce).InferShape(InferShape4BN3DTrainingReduce);
} // namespace ops
