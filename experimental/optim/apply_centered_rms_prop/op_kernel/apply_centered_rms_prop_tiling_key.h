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

/**
 * \file apply_centered_rms_prop_tiling_key.h
 * \brief ApplyCenteredRMSProp TilingKey template-argument declaration (arch35).
 *
 * DESIGN §5.4:
 *   TilingKey = 1 -> float32 path (direct fp32 compute, no Cast)
 *   TilingKey = 2 -> float16 path (Cast fp16->fp32, compute, Cast fp32->fp16)
 *
 * They are encoded via D_T_VAR (dtype template arg).
 *   D_T_VAR = C_DT_FLOAT   -> TilingKey 1
 *   D_T_VAR = C_DT_FLOAT16 -> TilingKey 2
 *
 * Iteration-1 skeleton registers both fp16 and fp32 in the template-argument
 * declaration so the autogen pipeline produces both binary slots.  The fp16
 * branch is fully implemented; the fp32 branch is a stub dispatched via
 * if constexpr in the kernel entry (fp32 body will land in iteration-2).
 *
 * Uses ASCENDC_TPL_ARGS_DECL template-argument mechanism.
 * TILING_KEY_IS macro is forbidden per DESIGN §5.4.
 */

#ifndef __APPLY_CENTERED_RMS_PROP_TILING_KEY_H__
#define __APPLY_CENTERED_RMS_PROP_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(ApplyCenteredRMSProp,
    ASCENDC_TPL_DATATYPE_DECL(D_T_VAR, C_DT_FLOAT16, C_DT_FLOAT, ASCENDC_TPL_INPUT(0))
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_VAR, C_DT_FLOAT16)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_VAR, C_DT_FLOAT)
    ),
);

#endif // __APPLY_CENTERED_RMS_PROP_TILING_KEY_H__
