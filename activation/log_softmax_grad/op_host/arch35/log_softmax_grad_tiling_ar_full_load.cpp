/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file log_softmax_grad_tiling_ar_full_load.cpp
 * \brief
 */

#include "tiling_base/tiling_templates_registry.h"
#include "log_softmax_grad_tiling.h"

using namespace ge;

namespace optiling
{

class LogSoftmaxGradTilingAR : public LogSoftmaxGradTilingBase, public SoftmaxGradTilingAR
{
public:
    explicit LogSoftmaxGradTilingAR(gert::TilingContext* context)
        : TilingBaseClass(context),
          SoftmaxGradTilingBase(context),
          LogSoftmaxGradTilingBase(context),
          SoftmaxGradTilingAR(context)
    {
    }
    ~LogSoftmaxGradTilingAR() override = default;
};

REGISTER_OPS_TILING_TEMPLATE(LogSoftmaxGrad, LogSoftmaxGradTilingAR, TEMPLATE_AR_FULL_LOAD_PRIORITY);

}  // namespace optiling
