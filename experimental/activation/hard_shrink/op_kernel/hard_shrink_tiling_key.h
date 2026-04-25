/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
 
/*!
 * \file hard_shrink_tiling_key.h
 * \brief HardShrink TilingKey 模板参数定义
 *
 * 模板参数：
 *   - D_T: 计算数据类型 (half/float)，bf16 场景用 float 计算
 *   - BUFFER_MODE: 缓冲模式 (0=单缓冲, 1=双缓冲)
 *   - IS_BF16: 是否为 bf16 输入（0=否, 1=是，需 Cast bf16<->float）
 */

#ifndef __HARD_SHRINK_TILING_KEY_H__
#define __HARD_SHRINK_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(HardShrink,
    ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(IS_BF16, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    // fp16 单缓冲
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(IS_BF16, ASCENDC_TPL_UI_LIST, 0)
    ),
    // fp16 双缓冲
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(IS_BF16, ASCENDC_TPL_UI_LIST, 0)
    ),
    // fp32 单缓冲
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(IS_BF16, ASCENDC_TPL_UI_LIST, 0)
    ),
    // fp32 双缓冲
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(IS_BF16, ASCENDC_TPL_UI_LIST, 0)
    ),
    // bf16 单缓冲 (D_T=bf16, IS_BF16=1 触发 cast 到 float 计算)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(IS_BF16, ASCENDC_TPL_UI_LIST, 1)
    ),
    // bf16 双缓冲 (D_T=bf16, IS_BF16=1 触发 cast 到 float 计算)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(IS_BF16, ASCENDC_TPL_UI_LIST, 1)
    ),
);

#endif
