/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ctc_loss_v2_tiling_key.h
 * \brief ctc_loss_v2_tiling_key.h declare 
 */

#ifndef _CTC_LOSS_V2_TILING_KEY_DECL_H_
#define _CTC_LOSS_V2_TILING_KEY_DECL_H_
#include "ascendc/host_api/tiling/template_argument.h"

#define CTC_LOSS_V2_TPL_KEY_TRUE 1
#define CTC_LOSS_V2_TPL_KEY_FALSE 0 

#define CTC_LOSS_V2_TPL_KEY_DECL()                                                                   \
    ASCENDC_TPL_UINT_DECL(isFP32, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1),                      \
    ASCENDC_TPL_UINT_DECL(threadTypeInt32, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1)
    

#define CTC_LOSS_V2_FP32_KEY_SEL()                                                                   \
    ASCENDC_TPL_UINT_SEL(isFP32, ASCENDC_TPL_UI_RANGE, 1, 0, 1),                                     \
    ASCENDC_TPL_UINT_SEL(threadTypeInt32, ASCENDC_TPL_UI_RANGE, 1, 0, 1)

ASCENDC_TPL_ARGS_DECL(CTCLossV2, CTC_LOSS_V2_TPL_KEY_DECL());
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(CTC_LOSS_V2_FP32_KEY_SEL()));

#endif