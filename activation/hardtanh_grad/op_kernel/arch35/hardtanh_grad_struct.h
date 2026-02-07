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
 * \file hardtanh_grad_struct.h
 * \brief hardtanh_grad_struct
 */

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef _HARDTANH_GRAD_STRUCT_H_
#define _HARDTANH_GRAD_STRUCT_H_

namespace HardtanhGradOp
{
#define HardtanhGrad_TPL 1

#define HardtanhGrad_TPL_SCH_MODE_0 0
#define HardtanhGrad_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    HardtanhGrad,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, HardtanhGrad_TPL_SCH_MODE_0, HardtanhGrad_TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_DECL(dType, HardtanhGrad_TPL)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, HardtanhGrad_TPL_SCH_MODE_0, HardtanhGrad_TPL_SCH_MODE_1),
        ASCENDC_TPL_DTYPE_SEL(dType, HardtanhGrad_TPL)
    )
);
}  // namespace HardtanhGradOp

#endif  // _HARDTANH_GRAD_STRUCT_H_