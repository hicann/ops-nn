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
 * \file foreach_sub_scalar_list_tiling_key.h
 * \brief foreach_sub_scalar_list tiling key declare
 */

#ifndef FOREACH_SUB_SCALAR_LIST_TILING_KEY_H
#define FOREACH_SUB_SCALAR_LIST_TILING_KEY_H

#include <cstdint>
#include "ascendc/host_api/tiling/template_argument.h"

enum class ForeachSubScalarListTilingKey : uint32_t {
    TILING_KEY_FLOAT = 0,
    TILING_KEY_FLOAT16 = 1,
    TILING_KEY_INT32 = 2,
    TILING_KEY_BF16 = 3,
};

#define FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_FLOAT 0
#define FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_FLOAT16 1
#define FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_INT32 2
#define FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_BF16 3

ASCENDC_TPL_ARGS_DECL(ForeachSubScalarList,
                      ASCENDC_TPL_UINT_DECL(schMode, 4, ASCENDC_TPL_UI_LIST, FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_FLOAT,
                                            FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_FLOAT16,
                                            FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_INT32,
                                            FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_BF16));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_FLOAT,
                                                          FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_FLOAT16,
                                                          FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_INT32,
                                                          FOREACH_SUB_SCALAR_LIST_TPL_SCH_MODE_BF16)));

#endif // FOREACH_SUB_SCALAR_LIST_TILING_KEY_H
