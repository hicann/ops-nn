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
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/*!
 * \file bn_infer_grad_tiling_key.h
 * \brief BnInferGrad TilingKey 模板参数定义
 *
 * 模板参数:
 *   D_T_X: 输入数据类型 (float, half, bfloat16)
 *   SCH_MODE: 调度模式 (0=CONTIGUOUS, 1=NC1HWC0)
 */

#ifndef __BN_INFER_GRAD_TILING_KEY_H__
#define __BN_INFER_GRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(BnInferGrad,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(SCH_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    // float + 2 种调度模式
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    // float16 + 2 种调度模式
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    // bfloat16 + 2 种调度模式
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
);

#endif // __BN_INFER_GRAD_TILING_KEY_H__
