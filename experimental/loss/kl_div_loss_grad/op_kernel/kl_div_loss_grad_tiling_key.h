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
 * \file kl_div_loss_grad_tiling_key.h
 * \brief kl_div_loss_grad tiling key declare
 */

#ifndef __KL_DIV_LOSS_GRAD_TILING_KEY_H__
#define __KL_DIV_LOSS_GRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

/* 模板参数 */
ASCENDC_TPL_ARGS_DECL(klDivLossGrad, ASCENDC_TPL_BOOL_DECL(logTarget, 0, 1), ASCENDC_TPL_BOOL_DECL(broadcast, 0, 1), );

/* 模板参数组合 */
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(logTarget, 0, 1), ASCENDC_TPL_BOOL_SEL(broadcast, 0, 1)), );

#endif
