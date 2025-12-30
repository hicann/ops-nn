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
 * \file swish_struct.h
 * \brief swish_struct
 */
#ifndef SWISH_STRUCT_H_
#define SWISH_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

namespace SwishOp {
#define TPL_SCALE_NEG_ONE 1
#define TPL_SCALE_ZERO 2
#define TPL_SCALE_OTHER 3

#define TPL_SCH_MODE_0 0
#define TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(Swish,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_DECL(dType, TPL_SCALE_NEG_ONE, TPL_SCALE_ZERO, TPL_SCALE_OTHER)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
        ASCENDC_TPL_DTYPE_SEL(dType, TPL_SCALE_NEG_ONE, TPL_SCALE_ZERO, TPL_SCALE_OTHER)
    )
);
} // namespace SwishOp
#endif  // SWISH_STRUCT_H_