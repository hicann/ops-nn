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
 * \file kl_div_target_loss_grad_tiling_key.h
 * \brief kl_div_target_loss_grad tiling key
 */

#ifndef _KL_DIV_TARGET_LOSS_GRAD_TILING_KEY_H_
#define _KL_DIV_TARGET_LOSS_GRAD_TILING_KEY_H_

#define KDTLG_FALSE 0
#define KDTLG_TRUE 1

#include "atvoss/broadcast/broadcast_base_struct.h"

ASCENDC_TPL_ARGS_DECL(klDivTargetLossGrad, BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
                      ASCENDC_TPL_UINT_DECL(logTarget, 1, ASCENDC_TPL_UI_LIST, KDTLG_TRUE, KDTLG_FALSE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
                                     ASCENDC_TPL_UINT_SEL(logTarget, ASCENDC_TPL_UI_LIST, KDTLG_TRUE, KDTLG_FALSE)));

#endif