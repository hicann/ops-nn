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
 * \file apply_proximal_adagrad_tiling_key.h
 * \brief ApplyProximalAdagrad TilingKey template-argument declaration (arch35).
 *
 * Iteration-2 introduces three TilingKey routes (DESIGN s5.5):
 *   10001 - aligned tail block        + L1 >= 0 general path
 *   10002 - non-aligned tail block    + L1 >= 0 general path  (DataCopyPad write-back)
 *   10003 - any tail block            + L1 == 0 fast path     (skip sign/soft-threshold)
 *
 * They are encoded via two boolean template parameters (PAD_TAIL, HAS_L1):
 *   (PAD_TAIL=0, HAS_L1=1)  -> 10001
 *   (PAD_TAIL=1, HAS_L1=1)  -> 10002
 *   (PAD_TAIL=0|1, HAS_L1=0) -> 10003
 *
 * - PAD_TAIL is derived in host tiling from (totalElements % ubBlockSize != 0).
 * - HAS_L1 is normally driven from host = 1 (lr/l1/l2 sit in Device GM and we
 *   cannot synchronously inspect them at tiling time).  The kernel additionally
 *   performs a cheap runtime fast-path check when l1Scalar == 0.  We still
 *   register the HAS_L1 = 0 binary so UT (and a future L0-API hint) can drive
 *   the dedicated TilingKey 10003.
 *
 * Uses ASCENDC_TPL_ARGS_DECL template-argument mechanism.
 * TILING_KEY_IS macro is forbidden (see REQUIREMENTS s8.4).
 */

#ifndef __APPLY_PROXIMAL_ADAGRAD_TILING_KEY_H__
#define __APPLY_PROXIMAL_ADAGRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(ApplyProximalAdagrad,
    ASCENDC_TPL_DATATYPE_DECL(D_T_VAR, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(PAD_TAIL, 8, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(HAS_L1,   8, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_VAR, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(PAD_TAIL, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HAS_L1,   ASCENDC_TPL_UI_LIST, 0, 1)
    ),
);

#endif // __APPLY_PROXIMAL_ADAGRAD_TILING_KEY_H__
