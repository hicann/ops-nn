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
 * \file clipped_swiglu_tiling_key.h
 * \brief TPL tiling key for ClippedSwiglu (Ascend 950)
 */

#ifndef CLIPPED_SWIGLU_TILING_KEY_H
#define CLIPPED_SWIGLU_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_INTERLEAVED_FALSE 0
#define TPL_INTERLEAVED_TRUE 1

#define TPL_NO_GROUP_INDEX 0
#define TPL_GROUP_INDEX 1

namespace ClippedSwigluOp {
ASCENDC_TPL_ARGS_DECL(ClippedSwiglu,
                      ASCENDC_TPL_UINT_DECL(isInterleaved, 1, ASCENDC_TPL_UI_LIST, TPL_INTERLEAVED_FALSE,
                                            TPL_INTERLEAVED_TRUE),
                      ASCENDC_TPL_UINT_DECL(isGroup, 1, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX, TPL_GROUP_INDEX));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(isInterleaved, ASCENDC_TPL_UI_LIST, TPL_INTERLEAVED_FALSE, TPL_INTERLEAVED_TRUE),
    ASCENDC_TPL_UINT_SEL(isGroup, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX, TPL_GROUP_INDEX, TPL_FLOOR)));
} // namespace ClippedSwigluOp

#endif // CLIPPED_SWIGLU_TILING_KEY_H
