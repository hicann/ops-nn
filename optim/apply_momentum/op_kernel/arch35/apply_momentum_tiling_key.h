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
 * \file apply_momentum_tiling_key.h
 * \brief
 */

#ifndef APPLY_MOMENTUM_TILING_KEY_H
#define APPLY_MOMENTUM_TILING_KEY_H

#define TILING_KEY_SEL_0 0 // specifying schMode is 0 or useNesterov is false
#define TILING_KEY_SEL_1 1 // specifying schMode is 1 or useNesterov is true

#define BIT_WIDTH 1

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(ApplyMomentum, 
    ASCENDC_TPL_UINT_DECL(schMode, BIT_WIDTH, ASCENDC_TPL_UI_LIST, TILING_KEY_SEL_0, TILING_KEY_SEL_1),
    ASCENDC_TPL_UINT_DECL(useNesterov, BIT_WIDTH, ASCENDC_TPL_UI_LIST, TILING_KEY_SEL_0, TILING_KEY_SEL_1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TILING_KEY_SEL_0, TILING_KEY_SEL_1),
        ASCENDC_TPL_UINT_SEL(useNesterov, ASCENDC_TPL_UI_LIST, TILING_KEY_SEL_0, TILING_KEY_SEL_1)
        )
);

#endif // APPLY_MOMENTUM_TILING_KEY_H