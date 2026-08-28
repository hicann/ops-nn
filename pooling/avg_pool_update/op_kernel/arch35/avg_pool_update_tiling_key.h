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
 * \file avg_pool_update_tiling_key.h
 * \brief Tiling key declare for avg_pool_update operator
 *
 * Single template parameter:
 *   schMode (UINT 1-bit): scene mode
 *     0 = ELEMWISE (element-wise division, the only scene)
 */

#ifndef AVG_POOL_UPDATE_TILING_KEY_H_
#define AVG_POOL_UPDATE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define AVG_POOL_UPDATE_SCH_MODE_ELEMWISE 0

ASCENDC_TPL_ARGS_DECL(AvgPoolUpdate,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, AVG_POOL_UPDATE_SCH_MODE_ELEMWISE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          AVG_POOL_UPDATE_SCH_MODE_ELEMWISE),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(AvgPoolUpdateTilingData)));

#endif // AVG_POOL_UPDATE_TILING_KEY_H_
