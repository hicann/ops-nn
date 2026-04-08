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

/**
 * \file hard_swish_tiling_key.h
 * \brief HardSwish TilingKey 定义
 *
 * TilingKey 映射：
 *   HARDSWISH_TPL_SCH_MODE_FP32 (0) : float32 计算路径
 *   HARDSWISH_TPL_SCH_MODE_FP16 (1) : float16 计算路径
 *   HARDSWISH_TPL_SCH_MODE_BF16 (2) : bfloat16 计算路径
 */

#ifndef __HARD_SWISH_TILING_KEY_H__
#define __HARD_SWISH_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define HARDSWISH_TPL_SCH_MODE_FP32 0
#define HARDSWISH_TPL_SCH_MODE_FP16 1
#define HARDSWISH_TPL_SCH_MODE_BF16 2

ASCENDC_TPL_ARGS_DECL(
    HardSwish,
    ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST,
                          HARDSWISH_TPL_SCH_MODE_FP32,
                          HARDSWISH_TPL_SCH_MODE_FP16,
                          HARDSWISH_TPL_SCH_MODE_BF16));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                         HARDSWISH_TPL_SCH_MODE_FP32,
                         HARDSWISH_TPL_SCH_MODE_FP16,
                         HARDSWISH_TPL_SCH_MODE_BF16)));

#endif // __HARD_SWISH_TILING_KEY_H__
