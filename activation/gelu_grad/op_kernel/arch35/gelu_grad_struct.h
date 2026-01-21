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
 * \file gelu_grad_struct.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_GELU_GRAD_STRUCT_H_
#define CANN_CUSTOM_OPS_GELU_GRAD_STRUCT_H_

#include "atvoss/broadcast/broadcast_base_struct.h"

namespace GeluGradOp {
ASCENDC_TPL_ARGS_DECL(GeluGrad, BRC_TEMP_SCH_MODE_KEY_DECL(schMode));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode)));
} // namespace GeluGradOp

#endif //  CANN_CUSTOM_OPS_GELU_GRAD_STRUCT_H_