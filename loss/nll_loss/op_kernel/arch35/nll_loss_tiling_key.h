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
 * \file nll_loss_tiling_key.h
 * \brief nll_loss_tiling_key
 */

#ifndef _NLL_LOSS_TILING_KEY_DECL_H_
#define _NLL_LOSS_TILING_KEY_DECL_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_SCH_ID_SIMT 0
#define TPL_SCH_ID_MIX 1
#define TPL_SCH_ID_EMPTY 2

#define TPL_XDIM_1 1 // 输入为1维
#define TPL_XDIM_2 2 // 输入为2维
#define TPL_XDIM_4 4 // 输入为4维

#define TPL_REDUCTION_NONE 0 // reduce模式为none
#define TPL_REDUCTION_MEAN 1 // reduce模式为mean
#define TPL_REDUCTION_SUM 2  // reduce模式为sum

#define NLL_LOSS_TPL_KEY_DECL()                                                                                  \
    ASCENDC_TPL_UINT_DECL(schId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_SCH_ID_SIMT, TPL_SCH_ID_MIX,         \
                          TPL_SCH_ID_EMPTY),                                                                     \
        ASCENDC_TPL_UINT_DECL(xDims, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_XDIM_1, TPL_XDIM_2, TPL_XDIM_4), \
        ASCENDC_TPL_UINT_DECL(reduction, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_REDUCTION_NONE,              \
                              TPL_REDUCTION_MEAN, TPL_REDUCTION_SUM)

#define NLL_LOSS_TPL_KEY_SEL()                                                                           \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, TPL_SCH_ID_SIMT, TPL_SCH_ID_MIX, TPL_SCH_ID_EMPTY), \
        ASCENDC_TPL_UINT_SEL(xDims, ASCENDC_TPL_UI_LIST, TPL_XDIM_1, TPL_XDIM_2, TPL_XDIM_4),            \
        ASCENDC_TPL_UINT_SEL(reduction, ASCENDC_TPL_UI_LIST, TPL_REDUCTION_NONE, TPL_REDUCTION_MEAN,     \
                             TPL_REDUCTION_SUM)

ASCENDC_TPL_ARGS_DECL(NLLLoss, NLL_LOSS_TPL_KEY_DECL());
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(NLL_LOSS_TPL_KEY_SEL()));

#endif
