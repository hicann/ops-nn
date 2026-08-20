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
 * \file softmax_focal_loss_tiling_key.h
 * \brief softmax_focal_loss tiling key declare
 */

#ifndef SOFTMAX_FOCAL_LOSS_TILING_KEY_H
#define SOFTMAX_FOCAL_LOSS_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

// hasWeight:    0 = weight 缺省(按全 1 语义), 1 = weight 传入
// weightIsHalf: 0 = weight 为 float32, 1 = weight 为 float16
//
// weight 的 dtype 独立于 pred(A2 ini 声明四种组合), 但二进制匹配用的 simplifiedKey
// 会把 optional 输入的 dtype 槽填成 pred 的 dtype, 四条 binary 会塌成两组键而选错 .o。
// 故不依赖 binary 匹配区分 weight dtype, 改由 host tiling 按实际 dtype 下发本模板参数。
#define SOFTMAX_FOCAL_LOSS_TPL_KEY_DECL()                                          \
    ASCENDC_TPL_UINT_DECL(hasWeight, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1), \
        ASCENDC_TPL_UINT_DECL(weightIsHalf, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1)

#define SOFTMAX_FOCAL_LOSS_TPL_KEY_SEL()                        \
    ASCENDC_TPL_UINT_SEL(hasWeight, ASCENDC_TPL_UI_LIST, 0, 1), \
        ASCENDC_TPL_UINT_SEL(weightIsHalf, ASCENDC_TPL_UI_LIST, 0, 1)

ASCENDC_TPL_ARGS_DECL(SoftmaxFocalLoss, SOFTMAX_FOCAL_LOSS_TPL_KEY_DECL());

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(SOFTMAX_FOCAL_LOSS_TPL_KEY_SEL()));
#endif // SOFTMAX_FOCAL_LOSS_TILING_KEY_H
