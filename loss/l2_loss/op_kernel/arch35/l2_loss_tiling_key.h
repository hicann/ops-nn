/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file l2_loss_tiling_key.h
 * \brief l2loss tiling key
 */

#ifndef _L2_LOSS_TILING_KEY_H_
#define _L2_LOSS_TILING_KEY_H_
#include "atvoss/reduce/reduce_tiling_key_decl.h"

ASCENDC_TPL_ARGS_DECL(L2Loss, REDUCE_TPL_KEY_DECL());

ASCENDC_TPL_SEL(
    // empty
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY()),
    // A
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_A()),
    // AR
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_AR_NORMAL()),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_AR_GROUP())
    );

#endif