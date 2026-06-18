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
 * \file fake_quant_with_min_max_vars_per_channel_tiling_key.h
 * \brief FakeQuantWithMinMaxVarsPerChannel TilingKey 模板参数
 *
 * 四维 TilingKey (DESIGN.md v2.1 §3.3):
 *   - D_T_X        : 输入 x 数据类型槽位（占位，仅 FQ_TPL_DT_FP32 = 1）
 *   - MODE         : PER_MODE (2 = Native PC) — 占位
 *   - HAS_ZP       : 0 (无 zero_point) — 占位
 *   - ROUND_MODE   : 0 (floor + 0.5, TF 兼容) — 占位
 *
 * 说明：D_T_X 原本使用 ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, ...)，
 *      但 C_DT_FLOAT 仅在算子正式编译路径上有定义，CPU 仿真 UT 编译时未声明，
 *      导致 kernel UT 编译失败。参考 ascend_anti_quant_struct.h 的写法，
 *      改用 ASCENDC_TPL_UINT_DECL + 自定义整数字面量占位，host 端
 *      tiling.cpp 传给 ASCENDC_TPL_SEL_PARAM 时本就是 uint32_t（xDtype 强转），
 *      语义完全等价。
 */
#ifndef _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_TILING_KEY_H_
#define _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define FQ_TPL_DT_FP32 1   // 占位：表示输入 x 为 float32。预留扩展为 fp16/bf16 时再加常量。

ASCENDC_TPL_ARGS_DECL(FakeQuantWithMinMaxVarsPerChannel,
    ASCENDC_TPL_UINT_DECL(D_T_X,      8, ASCENDC_TPL_UI_LIST, FQ_TPL_DT_FP32),
    ASCENDC_TPL_UINT_DECL(MODE,       8, ASCENDC_TPL_UI_LIST, 2),
    ASCENDC_TPL_UINT_DECL(HAS_ZP,     8, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_UINT_DECL(ROUND_MODE, 8, ASCENDC_TPL_UI_LIST, 0)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(D_T_X,      ASCENDC_TPL_UI_LIST, FQ_TPL_DT_FP32),
        ASCENDC_TPL_UINT_SEL(MODE,       ASCENDC_TPL_UI_LIST, 2),
        ASCENDC_TPL_UINT_SEL(HAS_ZP,     ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROUND_MODE, ASCENDC_TPL_UI_LIST, 0)
    ),
);

#endif  // _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_TILING_KEY_H_
