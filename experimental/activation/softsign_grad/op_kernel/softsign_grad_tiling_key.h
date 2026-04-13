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
 * \file softsign_grad_tiling_key.h
 * \brief SoftsignGrad Tiling 模板参数定义
 *
 * 模板参数:
 *   - D_T: 数据类型 (FP16 / FP32 / BF16)
 *   - BUFFER_MODE: 缓冲模式 (0=单缓冲, 1=双缓冲)
 *
 * 全量覆盖（迭代三完成）：6 个 TilingKey 组合
 *   - FP16 单/双缓冲 + FP32 单/双缓冲 + BF16 单/双缓冲
 */

#ifndef __SOFTSIGN_GRAD_TILING_KEY_H__
#define __SOFTSIGN_GRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(SoftsignGrad,
    ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    // FP16: 单缓冲 + 双缓冲（迭代二启用双缓冲）
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    // FP32: 单缓冲 + 双缓冲（迭代二启用）
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    // BF16: 单缓冲 + 双缓冲（迭代三完成全量覆盖）
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
);

#endif
