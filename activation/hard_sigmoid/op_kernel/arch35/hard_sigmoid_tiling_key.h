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
 * \file hard_sigmoid_tiling_key.h
 * \brief HardSigmoid TilingKey 声明。
 *
 * dtype 由底层编译链路按 def 输入名 input_x 注入 DTYPE_INPUT_X，不放入 tiling key。
 * 当前算子只有一个固定调度模式，tiling key 仅用于匹配 kernel 模板入口。
 */

#ifndef HARD_SIGMOID_TILING_KEY_H
#define HARD_SIGMOID_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define HARD_SIGMOID_SCH_MODE_DEFAULT 0

ASCENDC_TPL_ARGS_DECL(HardSigmoid,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, HARD_SIGMOID_SCH_MODE_DEFAULT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          HARD_SIGMOID_SCH_MODE_DEFAULT)));

#endif // HARD_SIGMOID_TILING_KEY_H
