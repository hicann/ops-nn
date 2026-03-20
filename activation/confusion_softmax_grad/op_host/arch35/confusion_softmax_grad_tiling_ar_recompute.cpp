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
 * \file confusion_softmax_grad_tiling_ar_recompute.cpp
 * \brief
 */

#include "confusion_softmax_grad_tiling.h"

using namespace ge;

namespace optiling
{
class ConfusionSoftmaxGradTilingARRecompute : virtual public ConfusionSoftmaxGradTilingBase, public SoftmaxGradTilingARRecompute
{
public:
    explicit ConfusionSoftmaxGradTilingARRecompute(gert::TilingContext* context)
        : Ops::NN::Optiling::TilingBaseClass(context),
        SoftmaxGradTilingBase(context),
        ConfusionSoftmaxGradTilingBase(context),
        SoftmaxGradTilingARRecompute(context)
    {
    }
    ~ConfusionSoftmaxGradTilingARRecompute() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
};

REGISTER_OPS_TILING_TEMPLATE(ConfusionSoftmaxGrad, ConfusionSoftmaxGradTilingARRecompute, TEMPLATE_AR_RECOMPUTE_PRIORITY);
}  // namespace optiling
