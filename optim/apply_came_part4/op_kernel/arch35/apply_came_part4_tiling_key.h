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
 * \file apply_came_part4_tiling_key.h
 * \brief ApplyCamePart4 TilingKey template parameter declaration (arch35)
 *
 * Template parameters:
 *   - D_T_X: Data type of param/m/r/c (C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16)
 */

#ifndef _APPLY_CAME_PART4_TILING_KEY_H_
#define _APPLY_CAME_PART4_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// Kernel UT host 编译时 ASCENDC_CPU_DEBUG 被定义，ASCENDC_TPL_DATATYPE_DECL
// 宏展开为 ParamStruct{...} 构造函数，要求 C_DT_* 为有效的 C++ 标识符。
#ifdef ASCENDC_CPU_DEBUG
#include "graph/c_types.h"
#endif

ASCENDC_TPL_ARGS_DECL(ApplyCamePart4,
                      ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16)));

#endif // _APPLY_CAME_PART4_TILING_KEY_H_
