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
 * \file relu6_tiling_key.h
 * \brief Relu6 TilingKey 模板参数定义（arch35 = Ascend950）
 *
 * 迭代一：仅 float16 单 dtype 骨架，预留 float/int32/bfloat16 扩展位置
 * 模板参数类型参考：
 *   - DATATYPE: 原生数据类型（C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_BF16）
 * 参考：ascendc/host_api/tiling/template_argument.h
 */

#ifndef __RELU6_TILING_KEY_H__
#define __RELU6_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// 迭代一：仅 float16；迭代二扩展全部 4 种 dtype
ASCENDC_TPL_ARGS_DECL(Relu6,
    ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_BF16, ASCENDC_TPL_INPUT(0))
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_INT32)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BF16)
    ),
);

#endif // __RELU6_TILING_KEY_H__
