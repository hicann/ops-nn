/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mse_loss_grad_v2_tiling_key.h
 * \brief arch35 template-parameter (TPL) declarations: RANK ∈ {4, 8} drives the
 *        kernel TilingKey dispatch (4 -> tilingKey 0, 8 -> tilingKey 1).
 */

#ifndef MSE_LOSS_GRAD_V2_TILING_KEY_H_
#define MSE_LOSS_GRAD_V2_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h" // ASCENDC_TPL macros

// RANK value constants: RANK=4 covers effective rank 1-4 (TilingData4), RANK=8
// covers effective rank 5-8 (TilingData8).
#define MSE_LOSS_GRAD_V2_RANK_4 4
#define MSE_LOSS_GRAD_V2_RANK_8 8

ASCENDC_TPL_ARGS_DECL(MseLossGradV2, ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST, MSE_LOSS_GRAD_V2_RANK_4,
                                                           MSE_LOSS_GRAD_V2_RANK_8));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, MSE_LOSS_GRAD_V2_RANK_4)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, MSE_LOSS_GRAD_V2_RANK_8)));

#endif
