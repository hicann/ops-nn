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
 * \file median_tiling_key.h
 * \brief median tiling key declare。
 *        计算路径(schMode)作为 tilingkey 编译期区分：0=MAIN 1=SMALL 2=BIGSEL 3=BIG 4=HEAP 5=BIGSORT
 *        （与 median_tiling_data.h 的 MedianPath 枚举一致）；host 侧按 shape 选定，kernel 侧 if constexpr 分发，
 *        无运行期分支。dtype 由编译期 DTYPE_INPUT 决定，框架按 input dtype 各编一份，与本维度正交。
 */

#ifndef __MEDIAN_TILING_KEY_H__
#define __MEDIAN_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Median, ASCENDC_TPL_UINT_DECL(schMode, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5)));

#endif
