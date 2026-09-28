/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_RNN_DYNAMIC_AUGRU_TILING_KEY_H_
#define OPS_RNN_DYNAMIC_AUGRU_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define AUGRU_SEQUENCE_NONE 0
#define AUGRU_SEQUENCE_LENGTH 1
#define AUGRU_SEQUENCE_MASK 2

ASCENDC_TPL_ARGS_DECL(DynamicAUGRU, ASCENDC_TPL_UINT_DECL(SEQUENCE_MODE, 2, ASCENDC_TPL_UI_LIST, AUGRU_SEQUENCE_NONE,
                                                          AUGRU_SEQUENCE_LENGTH, AUGRU_SEQUENCE_MASK));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_1),
                                     ASCENDC_TPL_UINT_SEL(SEQUENCE_MODE, ASCENDC_TPL_UI_LIST, AUGRU_SEQUENCE_NONE,
                                                          AUGRU_SEQUENCE_LENGTH, AUGRU_SEQUENCE_MASK)));

#endif
