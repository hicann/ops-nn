/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hinge_loss_grad_tiling_key.h
 * \brief hinge_loss_grad tiling key declare
 */

#ifndef __HINGE_LOSS_GRAD_TILING_KEY_H__
#define __HINGE_LOSS_GRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define HINGE_LOSS_GRAD_TPL_SCH_MODE_0 0

ASCENDC_TPL_ARGS_DECL(HingeLossGrad,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, HINGE_LOSS_GRAD_TPL_SCH_MODE_0)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, HINGE_LOSS_GRAD_TPL_SCH_MODE_0)));

#endif
