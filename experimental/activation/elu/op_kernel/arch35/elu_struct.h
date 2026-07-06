/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_ACTIVATION_ELU_OP_KERNEL_ARCH35_ELU_STRUCT_H
#define OPS_NN_ACTIVATION_ELU_OP_KERNEL_ARCH35_ELU_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace EluOp {

#define ELU_TPL_FP16 1
#define ELU_TPL_BF16 2
#define ELU_TPL_FP32 3

#define ELU_TPL_SCH_MODE_0 0
#define ELU_TPL_SCH_MODE_1 1

#define ELU_TPL_KEY_FP16_SCH0 2
#define ELU_TPL_KEY_FP16_SCH1 3
#define ELU_TPL_KEY_BF16_SCH0 4
#define ELU_TPL_KEY_BF16_SCH1 5
#define ELU_TPL_KEY_FP32_SCH0 6
#define ELU_TPL_KEY_FP32_SCH1 7

ASCENDC_TPL_ARGS_DECL(Elu, ASCENDC_TPL_UINT_DECL(tilingKey, 3, ASCENDC_TPL_UI_LIST, ELU_TPL_KEY_FP16_SCH0,
                                                 ELU_TPL_KEY_FP16_SCH1, ELU_TPL_KEY_BF16_SCH0, ELU_TPL_KEY_BF16_SCH1,
                                                 ELU_TPL_KEY_FP32_SCH0, ELU_TPL_KEY_FP32_SCH1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(tilingKey, ASCENDC_TPL_UI_LIST, ELU_TPL_KEY_FP16_SCH0,
                                                          ELU_TPL_KEY_FP16_SCH1, ELU_TPL_KEY_BF16_SCH0,
                                                          ELU_TPL_KEY_BF16_SCH1, ELU_TPL_KEY_FP32_SCH0,
                                                          ELU_TPL_KEY_FP32_SCH1)));

} // namespace EluOp

#endif // OPS_NN_ACTIVATION_ELU_OP_KERNEL_ARCH35_ELU_STRUCT_H
