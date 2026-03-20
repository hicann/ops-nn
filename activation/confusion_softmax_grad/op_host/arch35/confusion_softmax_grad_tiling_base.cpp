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
 * \file confusion_softmax_grad_tiling_base.cpp
 * \brief
 */

#include "confusion_softmax_grad_tiling.h"

namespace optiling
{
ge::graphStatus ConfusionSoftmaxGradTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("SoftmaxV2TilingBase", "context is nullptr."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAndCheckDtypes() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetDimsAndCheckShapeValid() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    reduceAxes_ = xShapeSize_ - 1;
    // 合轴(a1_, r_, a0_)
    a1_ = DIM_NUM_ONE;
    r_ = xShape_[reduceAxes_];
    a0_ = DIM_NUM_ONE;
    for (int i = 0; i < xShapeSize_; i++) {
        if (i < reduceAxes_) {
            a1_ *= xShape_[i];
        } else if (i > reduceAxes_) {
            a0_ *= xShape_[i];
        }
    }

    OP_LOGD(context_->GetNodeName(), "input original shape is:(%s), axes is:%ld, fused shape is: (%ld, %ld, %ld)\n",
            VectorToString(xShape_).c_str(), reduceAxes_, a1_, r_, a0_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForConfusionSoftmaxGrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("ConfusionSoftmaxGradTilingBase", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "ConfusionSoftmaxGradTilingBase Ascendc enter");
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(ConfusionSoftmaxGrad)
    .Tiling(TilingForConfusionSoftmaxGrad)
    .TilingParse<SoftmaxGradCompileInfo>(TilingPrepareForSoftmaxGrad);

}  // namespace optiling
