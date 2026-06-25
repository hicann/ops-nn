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
 * \file gru_tiling_key.h
 * \brief gru tiling key declare
 */

#ifndef __GRU_TILING_KEY_H__
#define __GRU_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define GRU_TPL_MM_FP16_SPLIT 0
#define GRU_TPL_MM_FP32_SPLIT 1

ASCENDC_TPL_ARGS_DECL(Gru,
    ASCENDC_TPL_UINT_DECL(mmSplit, 1, ASCENDC_TPL_UI_LIST, GRU_TPL_MM_FP16_SPLIT, GRU_TPL_MM_FP32_SPLIT)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(mmSplit, ASCENDC_TPL_UI_LIST, GRU_TPL_MM_FP16_SPLIT, GRU_TPL_MM_FP32_SPLIT)));

#endif