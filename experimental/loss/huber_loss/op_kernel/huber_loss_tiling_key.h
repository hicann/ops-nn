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
 * \file huber_loss_tiling_key.h
 * \brief HuberLoss tiling key declare
 *
 * The schedule modes are declared here once and included by both the kernel
 * and host tiling. Three things must agree or the failure surfaces at link
 * time as a missing mangled name: the set declared in ASCENDC_TPL_SEL, the
 * values host tiling can pass to SetTilingKey, and the template instantiations
 * the kernel provides. Sharing one definition makes two of the three
 * structural rather than a matter of discipline.
 */

#ifndef __HUBER_LOSS_TILING_KEY_H__
#define __HUBER_LOSS_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"
// The mode values themselves live in the tiling data header, which carries no
// framework dependency -- that keeps the tiling arithmetic unit-testable on a
// plain host compiler while still sharing one definition with this file.
#include "huber_loss_tiling_data.h"

/* The second argument is the bit width, not the value count: 2 values need
 * width 1, 3 need 2. Over-allocating is legal; under-allocating has no
 * guaranteed diagnostic. Widen this when adding a mode.
 */
ASCENDC_TPL_ARGS_DECL(HuberLoss, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, HUBER_LOSS_SCH_MODE_NONE,
                                                       HUBER_LOSS_SCH_MODE_REDUCE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, HUBER_LOSS_SCH_MODE_NONE,
                                                          HUBER_LOSS_SCH_MODE_REDUCE)));

#endif // __HUBER_LOSS_TILING_KEY_H__
