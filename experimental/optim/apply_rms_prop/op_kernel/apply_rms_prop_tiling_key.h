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
 * \file apply_rms_prop_tiling_key.h
 * \brief ApplyRmsProp Tiling 模板参数定义（按 var dtype 分发）
 */

#ifndef _APPLY_RMS_PROP_TILING_KEY_H_
#define _APPLY_RMS_PROP_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(ApplyRmsProp,
    ASCENDC_TPL_DATATYPE_DECL(D_T_VAR, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0))
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_VAR, C_DT_FLOAT)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_VAR, C_DT_FLOAT16)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_VAR, C_DT_BF16)
    )
);

#endif  // _APPLY_RMS_PROP_TILING_KEY_H_
