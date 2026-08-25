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
 * \file cross_entropy_sum_exp_and_index_logit_tiling_key.h
 * \brief CrossEntropySumExpAndIndexLogit TilingKey 声明（单默认调度模式）。
 *
 * dtype 由底层编译链路按 def 输入名 vocab_parallel_logits 注入 DTYPE_VOCAB_PARALLEL_LOGITS，
 * 不放入 tiling key。当前算子只有 RegBase 一种调度模式。
 */

#ifndef CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_TILING_KEY_H_
#define CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define CE_SCH_MODE_DEFAULT 0

ASCENDC_TPL_ARGS_DECL(CrossEntropySumExpAndIndexLogit,
                      ASCENDC_TPL_UINT_DECL(MODE, 1, ASCENDC_TPL_UI_LIST, CE_SCH_MODE_DEFAULT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, CE_SCH_MODE_DEFAULT)));

#endif
