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
 * \file confusion_softmax_grad_tiling_ar_small_r.cpp
 * \brief
 */
#include "confusion_softmax_grad_tiling.h"

using namespace ge;

namespace optiling
{
// A5 64核场景特化门限值
static constexpr int64_t A_CHANGE_SHAPE = 200000;
static constexpr int64_t MAX_A_CHANGE_SHAPE = 6000000;
class ConfusionSoftmaxGradTilingARSmallR : virtual public ConfusionSoftmaxGradTilingBase, public SoftmaxGradTilingARSmallR
{
public:
    explicit ConfusionSoftmaxGradTilingARSmallR(gert::TilingContext* context)
        : Ops::NN::Optiling::TilingBaseClass(context),
        SoftmaxGradTilingBase(context),
        ConfusionSoftmaxGradTilingBase(context),
        SoftmaxGradTilingARSmallR(context)
    {
    }
    ~ConfusionSoftmaxGradTilingARSmallR() override = default;

protected:
    bool IsCapable() override
    {
        if (r_ == CONST_ONE && a1_ > CONST_ONE) {
            return true;
        }
        if (yDtype_ == ge::DT_FLOAT && r_ <= CONST_FOUR && a1_ > A_CHANGE_SHAPE && a1_ < MAX_A_CHANGE_SHAPE) {
            return true;
        } 
        if (yDtype_ != ge::DT_FLOAT && r_ <= CONST_EIGHT && a1_ > A_CHANGE_SHAPE) {
            return true;
        }
        OP_LOGI("AR small r template is not capable.");
        return false;
    }
};

REGISTER_TILING_TEMPLATE("ConfusionSoftmaxGrad", ConfusionSoftmaxGradTilingARSmallR, TEMPLATE_AR_SMALL_R_PRIORITY);
}  // namespace optiling
