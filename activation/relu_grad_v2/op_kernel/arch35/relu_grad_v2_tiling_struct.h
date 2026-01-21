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
 * \file relu_grad_v2_tiling_struct.h
 * \brief
 */

#ifndef RELU_GRAD_V2_TILING_STRUCT_H
#define RELU_GRAD_V2_TILING_STRUCT_H

#include "atvoss/elewise/elewise_base_struct.h"

using namespace Ops::Base;

namespace ReluGradV2Ns {

#define TPL_FP16 1
#define TPL_BF16 2
#define TPL_FP32 3
#define TPL_INT8 4
#define TPL_UINT8 5
#define TPL_INT32 6

#define TPL_SCH_MODE_0 0
#define TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    ReluGradV2, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_DECL(dType, TPL_FP16, TPL_BF16, TPL_FP32, TPL_INT8, TPL_UINT8, TPL_INT32));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_SEL(dType, TPL_FP16, TPL_BF16, TPL_FP32, TPL_INT8, TPL_UINT8, TPL_INT32)));

} // namespace ReluGradV2Ns

#endif // RELU_GRAD_V2_TILING_STRUCT_H