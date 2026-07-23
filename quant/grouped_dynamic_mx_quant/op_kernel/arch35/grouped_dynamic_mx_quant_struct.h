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
 * \file grouped_dynamic_mx_quant_struct.h
 * \brief
 */

#ifndef _GROUPED_DYNAMIC_MX_QUANT_STRUCT_H_
#define _GROUPED_DYNAMIC_MX_QUANT_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_SCALE_ALG_0 0
#define TPL_SCALE_ALG_1 1
#define TPL_SCALE_ALG_2 2
#define TPL_DST_TYPE_MAX_0 0 // normal
#define TPL_DST_TYPE_MAX_1 1 // 0,6
#define TPL_DST_TYPE_MAX_2 2 // 7
#define TPL_DST_TYPE_MAX_3 3 // 1.875
#define TPL_DST_TYPE_0 0     // fp8
#define TPL_DST_TYPE_1 1     // fp4_e2m1
#define TPL_DST_TYPE_2 2     // fp4_e1m2
#define TPL_ROUND_MODE_ROUND 0
#define TPL_ROUND_MODE_FLOOR 1
#define TPL_ROUND_MODE_RINT 4

namespace GroupedDynamicMxQuantOp {
ASCENDC_TPL_ARGS_DECL(GroupedDynamicMxQuant,
                      ASCENDC_TPL_UINT_DECL(scaleAlg, 2, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_1,
                                            TPL_SCALE_ALG_2),
                      ASCENDC_TPL_UINT_DECL(dstTypeMax, 3, ASCENDC_TPL_UI_LIST, TPL_DST_TYPE_MAX_0, TPL_DST_TYPE_MAX_1,
                                            TPL_DST_TYPE_MAX_2, TPL_DST_TYPE_MAX_3),
                      ASCENDC_TPL_UINT_DECL(dstType, 2, ASCENDC_TPL_UI_LIST, TPL_DST_TYPE_0, TPL_DST_TYPE_1,
                                            TPL_DST_TYPE_2),
                      ASCENDC_TPL_UINT_DECL(roundMode, 3, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_FLOOR,
                                            TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_RINT));

ASCENDC_TPL_SEL(
    // fp8
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(scaleAlg, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_1),
                         ASCENDC_TPL_UINT_SEL(dstTypeMax, ASCENDC_TPL_UI_LIST, TPL_DST_TYPE_MAX_0, TPL_DST_TYPE_MAX_1,
                                              TPL_DST_TYPE_MAX_2, TPL_DST_TYPE_MAX_3),
                         ASCENDC_TPL_UINT_SEL(dstType, ASCENDC_TPL_UI_LIST, TPL_DST_TYPE_0),
                         ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_RINT)),
    // fp4_e2m1
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(scaleAlg, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_2),
                         ASCENDC_TPL_UINT_SEL(dstTypeMax, ASCENDC_TPL_UI_LIST, TPL_DST_TYPE_MAX_0, TPL_DST_TYPE_MAX_1,
                                              TPL_DST_TYPE_MAX_2),
                         ASCENDC_TPL_UINT_SEL(dstType, ASCENDC_TPL_UI_LIST, TPL_DST_TYPE_1),
                         ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_FLOOR,
                                              TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_RINT)),
    // fp4_e1m2
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(scaleAlg, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_2),
                         ASCENDC_TPL_UINT_SEL(dstTypeMax, ASCENDC_TPL_UI_LIST, TPL_DST_TYPE_MAX_0, TPL_DST_TYPE_MAX_3),
                         ASCENDC_TPL_UINT_SEL(dstType, ASCENDC_TPL_UI_LIST, TPL_DST_TYPE_2),
                         ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_FLOOR,
                                              TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_RINT)));

} // namespace GroupedDynamicMxQuantOp

#endif // _GROUPED_DYNAMIC_MX_QUANT_STRUCT_H_
