/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4MaxPoolWithArgmaxV3(gert::InferShapeContext* context)
{
    // 获取输入 shape (InferShapeContext 返回的是 gert::Shape* 而非 StorageShape*)
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (inputShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    int64_t N = inputShape->GetDim(0);
    int64_t C = inputShape->GetDim(1);
    int64_t H = inputShape->GetDim(2);
    int64_t W = inputShape->GetDim(3);

    // 获取属性 (通过 GetAttrs() 获取 RuntimeAttrs)
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 属性顺序: 0=kernel_size, 1=strides, 2=pads, 3=dilations, 4=ceil_mode
    const auto* kernelSizeList = attrs->GetListInt(0);
    const auto* stridesList = attrs->GetListInt(1);
    const auto* padsList = attrs->GetListInt(2);
    const auto* dilationsList = attrs->GetListInt(3);
    const bool* ceilModePtr = attrs->GetBool(4);

    if (kernelSizeList == nullptr || stridesList == nullptr ||
        padsList == nullptr || dilationsList == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t* ksData = kernelSizeList->GetData();
    const int64_t* stData = stridesList->GetData();
    const int64_t* pdData = padsList->GetData();
    const int64_t* dlData = dilationsList->GetData();

    int64_t kH = (kernelSizeList->GetSize() >= 1) ? ksData[0] : 1;
    int64_t kW = (kernelSizeList->GetSize() >= 2) ? ksData[1] : kH;
    int64_t sH = (stridesList->GetSize() >= 1) ? stData[0] : 1;
    int64_t sW = (stridesList->GetSize() >= 2) ? stData[1] : sH;
    int64_t padH = (padsList->GetSize() >= 1) ? pdData[0] : 0;
    int64_t padW = (padsList->GetSize() >= 2) ? pdData[1] : padH;
    int64_t dH = (dilationsList->GetSize() >= 1) ? dlData[0] : 1;
    int64_t dW = (dilationsList->GetSize() >= 2) ? dlData[1] : dH;

    bool ceilMode = false;
    if (ceilModePtr != nullptr) {
        ceilMode = *ceilModePtr;
    }

    // 计算输出尺寸
    int64_t hNumerator = H + 2 * padH - dH * (kH - 1) - 1;
    int64_t wNumerator = W + 2 * padW - dW * (kW - 1) - 1;

    int64_t Hout, Wout;
    if (ceilMode) {
        Hout = (hNumerator + sH - 1) / sH + 1;
        Wout = (wNumerator + sW - 1) / sW + 1;
    } else {
        Hout = hNumerator / sH + 1;
        Wout = wNumerator / sW + 1;
    }

    // 设置输出 y shape
    gert::Shape* yShape = context->GetOutputShape(0);
    if (yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    yShape->SetDimNum(4);
    yShape->SetDim(0, N);
    yShape->SetDim(1, C);
    yShape->SetDim(2, Hout);
    yShape->SetDim(3, Wout);

    // 设置输出 argmax shape
    gert::Shape* argmaxShape = context->GetOutputShape(1);
    if (argmaxShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    argmaxShape->SetDimNum(4);
    argmaxShape->SetDim(0, N);
    argmaxShape->SetDim(1, C);
    argmaxShape->SetDim(2, Hout);
    argmaxShape->SetDim(3, Wout);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolWithArgmaxV3).InferShape(InferShape4MaxPoolWithArgmaxV3);

} // namespace ops
