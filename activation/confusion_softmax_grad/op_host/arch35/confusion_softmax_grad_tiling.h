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
 * \file confusion_softmax_grad_tiling.h
 * \brief
 */

#ifndef NORM_CONFUSION_SOFTMAX_GRAD_TILING_H_
#define NORM_CONFUSION_SOFTMAX_GRAD_TILING_H_

#include "log/log.h"
#include "error_util.h"
#include "util/math_util.h"
#include "op_api/runtime2_util.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "activation/softmax_grad/op_host/arch35/softmax_grad_tiling.h"

namespace optiling
{
constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NEGATIVE_ONE = -1;
constexpr int64_t NEGATIVE_TWO = -2;
constexpr int64_t NEGATIVE_FOUR = -4;

class ConfusionSoftmaxGradTilingBase : virtual public SoftmaxGradTilingBase
{
public:
    explicit ConfusionSoftmaxGradTilingBase(gert::TilingContext* context)
        : Ops::NN::Optiling::TilingBaseClass(context), SoftmaxGradTilingBase(context)
    {
    }
    ~ConfusionSoftmaxGradTilingBase() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
};

}  // namespace optiling

#endif  // NORM_CONFUSION_SOFTMAX_GRAD_TILING_H_