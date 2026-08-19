/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file situ_mx_quant_tiling_key.h
 * \brief TPL tiling key template argument declarations for SituMxQuant
 */

#ifndef SITU_MX_QUANT_TILING_KEY_H
#define SITU_MX_QUANT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_NO_LINEAR_BETA 0
#define TPL_HAS_LINEAR_BETA 1

#define TPL_RINT 4
#define TPL_ROUND 0
#define TPL_FLOOR 1

namespace SituMxQuantOp {
ASCENDC_TPL_ARGS_DECL(SituMxQuant,
                      ASCENDC_TPL_UINT_DECL(hasLinearBeta, 2, ASCENDC_TPL_UI_LIST, TPL_NO_LINEAR_BETA,
                                            TPL_HAS_LINEAR_BETA),
                      ASCENDC_TPL_UINT_DECL(roundMode, 3, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_ROUND, TPL_FLOOR));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(hasLinearBeta, ASCENDC_TPL_UI_LIST, TPL_NO_LINEAR_BETA, TPL_HAS_LINEAR_BETA),
    ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_ROUND, TPL_FLOOR)));
} // namespace SituMxQuantOp

#endif // SITU_MX_QUANT_TILING_KEY_H
