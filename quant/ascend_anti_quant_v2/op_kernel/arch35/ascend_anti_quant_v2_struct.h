/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ascend_anti_quant_v2_struct.h
 * \brief
 */
#ifndef ASCEND_ANTI_QUANT_V2_STRUCT_H_
#define ASCEND_ANTI_QUANT_V2_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef TPL_PER_TENSOR

#define TPL_PER_TENSOR 0
#define TPL_PER_CHANNEL 1
#define TPL_PER_HEAD 2
#define TPL_PER_CHANNEL_NDDMA 3
#define TPL_PER_HEAD_NDDMA 4
#define TPL_NO_OFFSET 0
#define TPL_HAS_OFFSET 1
#define TPL_NO_SQRT_MODE 0
#define TPL_SQRT_MODE 1

#endif

namespace AscendAntiQuantV2Op {
ASCENDC_TPL_ARGS_DECL(
    AscendAntiQuantV2,
    ASCENDC_TPL_UINT_DECL(perMode, 3, ASCENDC_TPL_UI_LIST, TPL_PER_TENSOR, TPL_PER_CHANNEL, TPL_PER_HEAD, TPL_PER_CHANNEL_NDDMA, TPL_PER_HEAD_NDDMA),
    ASCENDC_TPL_UINT_DECL(zeroPointsType, 1, ASCENDC_TPL_UI_LIST, TPL_NO_OFFSET, TPL_HAS_OFFSET),
    ASCENDC_TPL_UINT_DECL(sqrtMode, 1, ASCENDC_TPL_UI_LIST, TPL_NO_SQRT_MODE, TPL_SQRT_MODE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(perMode, ASCENDC_TPL_UI_LIST, TPL_PER_TENSOR, TPL_PER_CHANNEL, TPL_PER_HEAD, TPL_PER_CHANNEL_NDDMA, TPL_PER_HEAD_NDDMA),
    ASCENDC_TPL_UINT_SEL(zeroPointsType, ASCENDC_TPL_UI_LIST, TPL_NO_OFFSET, TPL_HAS_OFFSET),
    ASCENDC_TPL_UINT_SEL(sqrtMode, ASCENDC_TPL_UI_LIST, TPL_NO_SQRT_MODE, TPL_SQRT_MODE)));

} // namespace AscendAntiQuantV2Op

#endif // _ASCEND_ANTI_QUANT_V2_STRUCT_H_