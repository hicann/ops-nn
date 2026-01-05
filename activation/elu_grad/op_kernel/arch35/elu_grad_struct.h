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
 * \file elu_grad_struct.h
 * \brief
 */
#ifndef CANN_CUSTOM_OPS_ELU_GRAD_STRUCT_H_
#define CANN_CUSTOM_OPS_ELU_GRAD_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

namespace EluGradOp
{
#define TPL_FP16 1
#define TPL_BF16 2
#define TPL_FP32 3
 
#define TPL_SCH_MODE_0 0
#define TPL_SCH_MODE_1 1


ASCENDC_TPL_ARGS_DECL(EluGrad,
    ASCENDC_TPL_UINT_DECL(scheMode, 1, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_DECL(dType, TPL_FP16, TPL_BF16, TPL_FP32)
);
 
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(scheMode, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
        ASCENDC_TPL_DTYPE_SEL(dType, TPL_FP16, TPL_BF16, TPL_FP32)
    )
);
}   //namespace EluGradOp


#endif //  CANN_CUSTOM_OPS_ELU_GRAD_STRUCT_H_