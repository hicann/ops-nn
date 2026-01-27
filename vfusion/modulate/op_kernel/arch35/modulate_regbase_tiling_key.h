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
 * \file modulate_regbase_tiling_key.h
 * \brief modulate_regbase_tiling_key.h declare 
 */

#ifndef MODULATE_REGBASE_TILING_KEY_H
#define MODULATE_REGBASE_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

#define MODULATE_TILING_L 0
#define MODULATE_TILING_D 1

// 模板参数
ASCENDC_TPL_ARGS_DECL(Modulate,
    ASCENDC_TPL_UINT_DECL(tilingStrategy, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, \
                          MODULATE_TILING_L, MODULATE_TILING_D), // LIST模式, 穷举
    ASCENDC_TPL_BOOL_DECL(isScale, 0, 1),
    ASCENDC_TPL_BOOL_DECL(isShift, 0, 1)
);

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(tilingStrategy, ASCENDC_TPL_UI_RANGE, 1, MODULATE_TILING_L, MODULATE_TILING_D), // RANGE模式, 第一个值1表示范围个数1，后两个值表示范围起、终位置
        ASCENDC_TPL_BOOL_SEL(isScale, 0, 1),
        ASCENDC_TPL_BOOL_SEL(isShift, 0, 1)
    ),
);

#endif // MODULATE_REGBASE_TILING_KEY_H