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
 * \file sgd_tiling_key.h
 * \brief SGD TilingKey 模板参数定义
 *
 * 模板参数（四维，全部是编译期分支）：
 *   - schMode         : 框架调度模式，由 ElewiseBaseTiling 自动决定（0 / 1）
 *   - useNesterov     : 属性 nesterov（0 / 1）
 *   - hasWeightDecay  : 属性 weight_decay != 0（0 / 1）
 *   - hasDampening    : 属性 dampening != 0（0 / 1）
 *
 * 组合数核算：
 *   业务 TilingKey  = 2(nesterov) × 2(wd) × 2(damp) - 2(非法) = 6  （K0~K5）
 *   TPL_SEL 展开    = 6 × 2(schMode)                          = 12
 *   Kernel binary   = 12 × 3(dtype，来自 binary.json)          = 36
 *
 * 非法组合 useNesterov == 1 && hasDampening == 1 由下方两组 ARGS_SEL 剪掉，
 * 【不生成对应 binary】；Host 侧 InferShape / Tiling 亦对该组合报
 * attribute_value_out_of_range（对齐 A2：nesterov == true 时 dampening 必须为 0）。
 *
 * ⛔ `momentum == 0` 掩码【不是】TilingKey 维度：momentum 是 Device 侧 [1] 张量，
 *    Host Tiling 收不到张量数据（$ATV/elewise/elewise_tiling.h:216-245），
 *    其值在 Tiling 阶段不可见 → 只能做运行期分支。两套 DAG 同时存在于同一个
 *    binary 内、由 sgd.cpp 的运行期 if 选择，binary 数量不变。
 */

#ifndef SGD_TILING_KEY_H
#define SGD_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define SGD_TPL_FALSE 0
#define SGD_TPL_TRUE 1
#define SGD_TPL_BIT_WIDTH 1

ASCENDC_TPL_ARGS_DECL(
    SGD, ASCENDC_TPL_UINT_DECL(schMode, SGD_TPL_BIT_WIDTH, ASCENDC_TPL_UI_LIST, SGD_TPL_FALSE, SGD_TPL_TRUE),
    ASCENDC_TPL_UINT_DECL(useNesterov, SGD_TPL_BIT_WIDTH, ASCENDC_TPL_UI_LIST, SGD_TPL_FALSE, SGD_TPL_TRUE),
    ASCENDC_TPL_UINT_DECL(hasWeightDecay, SGD_TPL_BIT_WIDTH, ASCENDC_TPL_UI_LIST, SGD_TPL_FALSE, SGD_TPL_TRUE),
    ASCENDC_TPL_UINT_DECL(hasDampening, SGD_TPL_BIT_WIDTH, ASCENDC_TPL_UI_LIST, SGD_TPL_FALSE, SGD_TPL_TRUE));

ASCENDC_TPL_SEL(
    // 组一：useNesterov == 0 —— dampening 可 0 可非 0，共 2×2×2 = 8 个组合（K0~K3 × schMode）
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SGD_TPL_FALSE, SGD_TPL_TRUE),
                         ASCENDC_TPL_UINT_SEL(useNesterov, ASCENDC_TPL_UI_LIST, SGD_TPL_FALSE),
                         ASCENDC_TPL_UINT_SEL(hasWeightDecay, ASCENDC_TPL_UI_LIST, SGD_TPL_FALSE, SGD_TPL_TRUE),
                         ASCENDC_TPL_UINT_SEL(hasDampening, ASCENDC_TPL_UI_LIST, SGD_TPL_FALSE, SGD_TPL_TRUE)),
    // 组二：useNesterov == 1 —— dampening 必为 0（非法组合已剪），共 2×2 = 4 个组合（K4~K5 × schMode）
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SGD_TPL_FALSE, SGD_TPL_TRUE),
                         ASCENDC_TPL_UINT_SEL(useNesterov, ASCENDC_TPL_UI_LIST, SGD_TPL_TRUE),
                         ASCENDC_TPL_UINT_SEL(hasWeightDecay, ASCENDC_TPL_UI_LIST, SGD_TPL_FALSE, SGD_TPL_TRUE),
                         ASCENDC_TPL_UINT_SEL(hasDampening, ASCENDC_TPL_UI_LIST, SGD_TPL_FALSE)), );

#endif // SGD_TILING_KEY_H
