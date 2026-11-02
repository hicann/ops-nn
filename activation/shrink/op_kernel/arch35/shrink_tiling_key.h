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
 * @file shrink_tiling_key.h
 * @brief Shrink Tiling template parameter definition
 *
 * def 驱动 dtype 模式：dtype 由 _def.cpp 的 DataType 列表经构建系统注入
 * DTYPE_INPUT_X 编译宏覆盖，tiling_key 只编码 def 未覆盖的调度维度，
 * 不再以 ASCENDC_TPL_DATATYPE_DECL 编码运算 dtype（对齐 hard_shrink/relu6
 * 的工程规范改造：TilingKey 仅编码整型调度维度，不耦合算子语义 dtype）。
 *
 * Template parameters:
 *   - BUFFER_MODE: buffer mode (0=single, 1=double)
 */
#ifndef __SHRINK_TILING_KEY_H__
#define __SHRINK_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Shrink, ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    // 单缓冲
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
    // 双缓冲
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)), );

#endif // __SHRINK_TILING_KEY_H__
