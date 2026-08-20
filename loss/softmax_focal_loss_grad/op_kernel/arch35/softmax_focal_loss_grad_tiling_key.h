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
 * \file softmax_focal_loss_grad_tiling_key.h
 * \brief softmax_focal_loss_grad tiling key declare
 */

#ifndef SOFTMAX_FOCAL_LOSS_GRAD_TILING_KEY_H
#define SOFTMAX_FOCAL_LOSS_GRAD_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

// hasWeight: 0 = weight 缺省(按全 1 语义, A5 相对 A2 的功能补齐), 1 = weight 传入
#define SOFTMAX_FOCAL_LOSS_GRAD_TPL_KEY_DECL() \
    ASCENDC_TPL_UINT_DECL(hasWeight, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1)

#define SOFTMAX_FOCAL_LOSS_GRAD_TPL_KEY_SEL() ASCENDC_TPL_UINT_SEL(hasWeight, ASCENDC_TPL_UI_LIST, 0, 1)

ASCENDC_TPL_ARGS_DECL(SoftmaxFocalLossGrad, SOFTMAX_FOCAL_LOSS_GRAD_TPL_KEY_DECL());

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(SOFTMAX_FOCAL_LOSS_GRAD_TPL_KEY_SEL()));
#endif // SOFTMAX_FOCAL_LOSS_GRAD_TILING_KEY_H
