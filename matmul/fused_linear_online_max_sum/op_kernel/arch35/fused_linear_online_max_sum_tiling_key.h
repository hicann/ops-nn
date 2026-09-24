/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_linear_online_max_sum_tiling_key.h
 * \brief
 */
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

#define FUSED_LINEAR_ONLINE_MAX_SUM_MM_OUT_FLAG_LM 0
#define FUSED_LINEAR_ONLINE_MAX_SUM_MM_OUT_FLAG_HP 1

// 模板参数
ASCENDC_TPL_ARGS_DECL(FusedLinearOnlineMaxSum, ASCENDC_TPL_UINT_DECL(MM_OUT_FLAG, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                                                     FUSED_LINEAR_ONLINE_MAX_SUM_MM_OUT_FLAG_LM,
                                                                     FUSED_LINEAR_ONLINE_MAX_SUM_MM_OUT_FLAG_HP));

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(MM_OUT_FLAG, ASCENDC_TPL_UI_LIST,
                                                          FUSED_LINEAR_ONLINE_MAX_SUM_MM_OUT_FLAG_LM,
                                                          FUSED_LINEAR_ONLINE_MAX_SUM_MM_OUT_FLAG_HP),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(FusedLinearOnlineMaxSumTilingData)), );
