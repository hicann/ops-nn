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
 * \file roi_pooling_infershape.cpp
 * \brief infer shape for roi_pooling
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

constexpr int32_t ROI_COLS = 5;     // rois 每行列数 [batch_idx, x1, y1, x2, y2]
constexpr int32_t X_DIM_NUM = 4;    // x 维度数 [N, C, H, W]
constexpr int32_t ROIS_DIM_NUM = 2; // rois 维度数 [K, 5]

// K = rois.shape[0], C = x.shape[1], pooled_h/w from attrs
// 动态 rank 保护：图模式下 shape 为 unknown rank {-2} 时 GetDimNum() 返回 1，
// 会被误判为"非 2D/4D"报错；unknown rank 时跳过维度校验，输出对应 dim 设为 UNKNOWN_DIM。
static ge::graphStatus InferShapeRoiPooling(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* roisShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, roisShape);

    if (!Ops::Base::IsUnknownRank(*roisShape)) {
        OP_CHECK_IF(roisShape->GetDimNum() != ROIS_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "rois",
                                                 (std::to_string(roisShape->GetDimNum()) + "D").c_str(), "2D"),
                    return ge::GRAPH_FAILED);
        if (roisShape->GetDim(1) != ROI_COLS && roisShape->GetDim(1) != ge::UNKNOWN_DIM) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "rois.shape[1]",
                                                  std::to_string(roisShape->GetDim(1)).c_str(),
                                                  "rois dim[1] must be 5");
            return ge::GRAPH_FAILED;
        }
    }
    if (!Ops::Base::IsUnknownRank(*xShape)) {
        OP_CHECK_IF(xShape->GetDimNum() != X_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x",
                                                 (std::to_string(xShape->GetDimNum()) + "D").c_str(), "4D"),
                    return ge::GRAPH_FAILED);
    }

    // unknown rank 时输出 dim 设为 UNKNOWN_DIM(-1)，正常时取实际值（含 -1 透传）
    int64_t K = Ops::Base::IsUnknownRank(*roisShape) ? ge::UNKNOWN_DIM : roisShape->GetDim(0);
    int64_t C = Ops::Base::IsUnknownRank(*xShape) ? ge::UNKNOWN_DIM : xShape->GetDim(1);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto pooledHPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledHPtr);
    int64_t pooledH = *pooledHPtr;
    const auto pooledWPtr = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledWPtr);
    int64_t pooledW = *pooledWPtr;
    if (pooledH <= 0 || pooledW <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "pooled_h/pooled_w",
            ("pooledH=" + std::to_string(pooledH) + " pooledW=" + std::to_string(pooledW)).c_str(),
            "pooled_h/w must > 0");
        return ge::GRAPH_FAILED;
    }
    // 输出 [K, C, pooledH, pooledW]
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(4);
    yShape->SetDim(0, K);
    yShape->SetDim(1, C);
    yShape->SetDim(2, pooledH);
    yShape->SetDim(3, pooledW);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ROIPooling).InferShape(InferShapeRoiPooling);

} // namespace ops
