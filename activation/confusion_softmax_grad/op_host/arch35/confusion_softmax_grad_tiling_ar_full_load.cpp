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
 * \file confusion_softmax_grad_tiling_ar_full_load.cpp
 * \brief
 */
#include "confusion_softmax_grad_tiling.h"

using namespace ge;

namespace optiling
{
static constexpr int64_t R_MAX_VALUE = 16384;
class ConfusionSoftmaxGradTilingAR : virtual public ConfusionSoftmaxGradTilingBase, public SoftmaxGradTilingAR
{
public:
    explicit ConfusionSoftmaxGradTilingAR(gert::TilingContext* context)
        : Ops::NN::Optiling::TilingBaseClass(context),
        SoftmaxGradTilingBase(context),
        ConfusionSoftmaxGradTilingBase(context),
        SoftmaxGradTilingAR(context)
    {
    }
    ~ConfusionSoftmaxGradTilingAR() override = default;
protected:
    bool IsCapable() override
    {
        OP_TILING_CHECK(r_ > R_MAX_VALUE,
                        OP_LOGI(context_->GetNodeName(),
                                "AR full load template is not capable. actual r is %ld, larger than %ld", r_, R_MAX_VALUE),
                        return false);
        return true;
    }
};

REGISTER_OPS_TILING_TEMPLATE(ConfusionSoftmaxGrad, ConfusionSoftmaxGradTilingAR, TEMPLATE_AR_FULL_LOAD_PRIORITY);
}  // namespace optiling
