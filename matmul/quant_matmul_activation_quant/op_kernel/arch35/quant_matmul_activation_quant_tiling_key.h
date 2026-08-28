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
 * \file quant_matmul_activation_quant_tiling_key.h
 * \brief
 */
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

namespace QuantMatmulActivationQuantArch35TilingKey {

// Kernel Type
#define TPL_NO_FULLLOAD 0
#define TPL_FULLLOAD 1
#define TPL_B_FORMAT_ND 0
#define TPL_B_FORMAT_NZ 1

ASCENDC_TPL_ARGS_DECL(
    QuantMatmulActivationQuant, ASCENDC_TPL_UINT_DECL(A_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(B_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(KERNELTYPE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TPL_NO_FULLLOAD, TPL_FULLLOAD),
    ASCENDC_TPL_UINT_DECL(B_FORMAT, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TPL_B_FORMAT_ND, TPL_B_FORMAT_NZ));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(A_TRANS, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(B_TRANS, ASCENDC_TPL_UI_LIST, 0, 1)),
                ASCENDC_TPL_UINT_SEL(KERNELTYPE, ASCENDC_TPL_UI_LIST, TPL_NO_FULLLOAD, TPL_FULLLOAD),
                ASCENDC_TPL_UINT_SEL(B_FORMAT, ASCENDC_TPL_UI_LIST, TPL_B_FORMAT_ND, TPL_B_FORMAT_NZ));
} // namespace QuantMatmulActivationQuantArch35TilingKey
