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
 * \file arg_max_grad_tiling_key.h
 * \brief ArgMaxGrad TilingKey 定义
 *
 * DTYPE_VAR 由 arg_max_grad_def.cpp 的输入 dtype profile 驱动, 框架按 dtype 自动注入并实例化,
 * 不进入 TilingKey。唯一的算法分支是 inner(dimension 之后各维的乘积)是否为 1:
 * inner>1 沿 inner 方向向量化, inner==1 时改沿被选轴向量化并把 indices/updates 退化成标量。
 */
#ifndef ARG_MAX_GRAD_TILING_KEY_H
#define ARG_MAX_GRAD_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define ARG_MAX_GRAD_TPL_KEY_DECL() ASCENDC_TPL_UINT_DECL(innerIsOne, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1)

#define ARG_MAX_GRAD_TPL_KEY_SEL() ASCENDC_TPL_UINT_SEL(innerIsOne, ASCENDC_TPL_UI_LIST, 0, 1)

ASCENDC_TPL_ARGS_DECL(ArgMaxGrad, ARG_MAX_GRAD_TPL_KEY_DECL());

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ARG_MAX_GRAD_TPL_KEY_SEL()));

#endif // ARG_MAX_GRAD_TILING_KEY_H
